"""
mnist_models.py

Trains ConvNeXt-Tiny, MobileNetV2, and ResNet18 on MNIST under two data regimes:
  - IID:     full training set, randomly shuffled
  - Non-IID: Dirichlet-sampled (alpha=0.5) label distribution across 10 clients,
             then concatenated — severely skewed class counts per "client shard"

Six training runs total (3 models x 2 splits). After each run the model is
evaluated on the full MNIST test set and a confusion matrix PNG is saved.
A combined 2x3 grid of all six matrices is also saved.

Output directory: ./confusion_matrices/
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED        = 42
BATCH_SIZE  = 64
EPOCHS      = 5
LR          = 1e-3
NUM_CLASSES = 10
NUM_CLIENTS = 10          # used for Non-IID Dirichlet split
ALPHA       = 0.5         # Dirichlet concentration — lower = more skewed
OUTPUT_DIR  = Path("/home/user/Sign-Hand/confusion_matrices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
# Most torchvision backbones expect at least 32x32; ConvNeXt works fine at 32.
# We keep a single 32x32 transform for speed.
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset_full = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# ---------------------------------------------------------------------------
# IID split — just use the entire training set (already random via DataLoader)
# ---------------------------------------------------------------------------
def make_iid_loader(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2)

# ---------------------------------------------------------------------------
# Non-IID split via Dirichlet sampling
# Assigns each sample to one of NUM_CLIENTS clients with a label distribution
# drawn from Dir(alpha). We then concatenate all client shards — the combined
# dataset has a badly skewed global label distribution that reflects the per-
# client imbalance.
# ---------------------------------------------------------------------------
def make_noniid_loader(dataset):
    targets = np.array(dataset.targets)
    n = len(targets)

    # For each class, get indices
    class_indices = {c: np.where(targets == c)[0] for c in range(NUM_CLASSES)}

    # Draw Dirichlet proportions: shape (NUM_CLASSES, NUM_CLIENTS)
    proportions = np.random.dirichlet(
        alpha=[ALPHA] * NUM_CLIENTS, size=NUM_CLASSES
    )  # proportions[c][k] = fraction of class c going to client k

    client_indices = [[] for _ in range(NUM_CLIENTS)]
    for c in range(NUM_CLASSES):
        idxs = class_indices[c].copy()
        np.random.shuffle(idxs)
        # Compute cumulative split points
        splits = (proportions[c] * len(idxs)).astype(int)
        splits[-1] = len(idxs) - splits[:-1].sum()  # ensure exact count
        splits = np.maximum(splits, 0)
        parts = np.split(idxs, np.cumsum(splits)[:-1])
        for k, part in enumerate(parts):
            client_indices[k].extend(part.tolist())

    # Concatenate all clients — this preserves skew while covering all samples
    all_indices = []
    for k in range(NUM_CLIENTS):
        all_indices.extend(client_indices[k])

    print(f"  Non-IID: total samples = {len(all_indices)}")
    # Show per-class count to confirm skew at client 0
    c0 = client_indices[0]
    c0_labels = targets[c0]
    counts = np.bincount(c0_labels, minlength=NUM_CLASSES)
    print(f"  Client-0 class distribution: {counts}")

    return DataLoader(Subset(dataset, all_indices), batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2)

# ---------------------------------------------------------------------------
# Model factories — all adapted for 1-channel input, 10-class output
# ---------------------------------------------------------------------------

def build_convnext_tiny():
    model = models.convnext_tiny(weights=None)
    # Replace first Conv2d: 3→1 channel, keep everything else
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_mobilenet_v2():
    model = models.mobilenet_v2(weights=None)
    # First layer is features[0][0] (Conv2d 3→32)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    # Replace classifier
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_resnet18():
    model = models.resnet18(weights=None)
    # Replace first conv: 3→1 channel
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


MODEL_FACTORIES = {
    "ConvNeXt-Tiny": build_convnext_tiny,
    "MobileNetV2":   build_mobilenet_v2,
    "ResNet18":      build_resnet18,
}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model, loader, epochs=EPOCHS):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        iterator = tqdm(loader, desc=f"  Epoch {epoch}/{epochs}", leave=False) \
            if HAS_TQDM else loader

        for images, labels in iterator:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        print(f"    Epoch {epoch}/{epochs}  loss={total_loss/total:.4f}  train_acc={acc:.2f}%")

    return model

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader):
    model.to(DEVICE)
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return acc, cm

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(range(NUM_CLASSES)),
        yticklabels=list(range(NUM_CLASSES)),
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined(results):
    """results: list of (model_name, split_name, cm, acc)"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes_flat = axes.flatten()

    for idx, (model_name, split_name, cm, acc) in enumerate(results):
        ax = axes_flat[idx]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(range(NUM_CLASSES)),
            yticklabels=list(range(NUM_CLASSES)),
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} | {split_name}\nAcc={acc:.2f}%", fontsize=11)

    fig.suptitle("Confusion Matrices — MNIST (IID vs Non-IID)", fontsize=15, y=1.01)
    plt.tight_layout()
    combined_path = OUTPUT_DIR / "all_confusion_matrices.png"
    fig.savefig(combined_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined figure saved: {combined_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    splits = {
        "IID":     make_iid_loader(train_dataset_full),
        "Non-IID": make_noniid_loader(train_dataset_full),
    }

    results = []  # (model_name, split_name, cm, acc)

    for split_name, train_loader in splits.items():
        for model_name, factory in MODEL_FACTORIES.items():
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}   Split: {split_name}")
            print(f"{'='*60}")

            model = factory()
            model = train_model(model, train_loader)

            acc, cm = evaluate_model(model, test_loader)
            print(f"  Test accuracy: {acc:.2f}%")

            safe_model = model_name.replace(" ", "_").replace("-", "_")
            fname = OUTPUT_DIR / f"{safe_model}_{split_name}.png"
            plot_confusion_matrix(cm, f"{model_name} | {split_name}  (acc={acc:.2f}%)", fname)

            results.append((model_name, split_name, cm, acc))

            # Free GPU memory between runs
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Reorder results so rows = IID / Non-IID, cols = models (2x3 grid)
    iid_results    = [r for r in results if r[1] == "IID"]
    noniid_results = [r for r in results if r[1] == "Non-IID"]
    ordered = iid_results + noniid_results

    plot_combined(ordered)

    print("\nAll done. Summary:")
    print(f"{'Model':<20} {'Split':<10} {'Test Acc':>10}")
    print("-" * 44)
    for model_name, split_name, _, acc in results:
        print(f"{model_name:<20} {split_name:<10} {acc:>9.2f}%")


if __name__ == "__main__":
    main()
