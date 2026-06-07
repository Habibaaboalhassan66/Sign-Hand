"""
mnist_models.py
Train ConvNeXt-Tiny, MobileNetV2, and ResNet18 on MNIST under both
IID and Non-IID data splits, then evaluate and save confusion matrices.

Non-IID method: Dirichlet distribution (alpha=0.5) across 5 virtual clients.
The training set is the union of all client shards — what differs is the
class-distribution skew each client's shard introduces.

Outputs
-------
confusion_matrices/
    cm_ConvNeXt_IID.png
    cm_ConvNeXt_NonIID.png
    cm_MobileNetV2_IID.png
    cm_MobileNetV2_NonIID.png
    cm_ResNet18_IID.png
    cm_ResNet18_NonIID.png
    all_confusion_matrices.png
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import seaborn as sns

HAS_TQDM = False  # disabled for clean output on CPU

# ──────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
EPOCHS      = 2
BATCH_SIZE  = 128
LR          = 1e-3
NUM_CLIENTS = 5       # virtual federated clients for Non-IID sharding
ALPHA       = 0.5     # Dirichlet concentration (lower → more skewed)
IMG_SIZE    = 32      # resize target (ConvNeXt needs >=32; keeps training fast)
NUM_CLASSES = 10
OUT_DIR     = "/home/user/Sign-Hand/confusion_matrices"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST channel stats
])

# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
def get_mnist(train: bool):
    return torchvision.datasets.MNIST(
        root="/tmp/mnist_data",
        train=train,
        download=True,
        transform=transform,
    )

def make_iid_loader(dataset, batch_size=BATCH_SIZE):
    """Shuffle entire training set — classic IID."""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

def make_noniid_loader(dataset, num_clients=NUM_CLIENTS,
                       alpha=ALPHA, batch_size=BATCH_SIZE):
    """
    Non-IID via Dirichlet sharding (classic FL benchmark).

    Steps:
      1. For each class c, draw a Dirichlet(alpha) proportion vector of length
         num_clients.  This determines how many samples of class c each client
         receives.
      2. Build per-client index lists accordingly.
      3. Concatenate all client shards into one dataset — the global class
         distribution is now severely skewed (some clients barely see certain
         classes, mimicking real-world Non-IID federated data).
    """
    targets = np.array(dataset.targets)

    # Indices sorted by class
    class_indices = [np.where(targets == c)[0].tolist() for c in range(NUM_CLASSES)]
    for idx_list in class_indices:
        random.shuffle(idx_list)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(NUM_CLASSES):
        idxs = class_indices[c]
        # Dirichlet proportions for this class across clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to actual counts (must sum to len(idxs))
        counts = (proportions * len(idxs)).astype(int)
        # Fix rounding so counts sum exactly to len(idxs)
        counts[-1] = len(idxs) - counts[:-1].sum()

        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(idxs[start:start + count])
            start += count

    # Print per-client class distribution for transparency
    print("\n  Non-IID client class distributions (sample counts per class):")
    for cid, idxs in enumerate(client_indices):
        client_targets = targets[idxs]
        dist = {c: int((client_targets == c).sum()) for c in range(NUM_CLASSES)}
        print(f"    Client {cid}: {dist}")

    # Union of all client shards (preserves Non-IID distribution skew)
    all_indices = [i for client in client_indices for i in client]
    combined = Subset(dataset, all_indices)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

# ──────────────────────────────────────────────────────────
# Model factory — adapt pretrained architectures to 1-channel MNIST
# ──────────────────────────────────────────────────────────
def build_convnext():
    """
    ConvNeXt-Tiny.
    Stem: Conv2d(3, 96, 4, 4) -> Conv2d(1, 96, 4, 4)  (1 in-channel).
    Head: Linear(768, 10).
    """
    m = models.convnext_tiny(weights=None)
    # features[0][0] is the patchify stem Conv2d
    orig = m.features[0][0]
    m.features[0][0] = nn.Conv2d(
        1, orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
        bias=(orig.bias is not None),
    )
    # Final classifier head
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return m

def build_mobilenetv2():
    """
    MobileNetV2.
    First Conv2d: (3, 32, 3, 2, 1) -> (1, 32, 3, 2, 1).
    Classifier head: Linear(1280, 10).
    """
    m = models.mobilenet_v2(weights=None)
    orig = m.features[0][0]
    m.features[0][0] = nn.Conv2d(
        1, orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
        bias=(orig.bias is not None),
    )
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return m

def build_resnet18():
    """
    ResNet18.
    First conv: (3, 64, 7, 2, 3) -> (1, 64, 7, 2, 3).
    FC head: Linear(512, 10).
    """
    m = models.resnet18(weights=None)
    orig = m.conv1
    m.conv1 = nn.Conv2d(
        1, orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
        bias=(orig.bias is not None),
    )
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

MODEL_FACTORIES = {
    "ConvNeXt":    build_convnext,
    "MobileNetV2": build_mobilenetv2,
    "ResNet18":    build_resnet18,
}

# ──────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if HAS_TQDM:
        it = tqdm(loader, desc=f"  Epoch {epoch}/{total_epochs}", leave=False)
    else:
        it = loader

    for images, labels in it:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if HAS_TQDM:
            it.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def train(model, loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion,
                                    epoch, epochs)
        print(f"    Epoch {epoch}/{epochs} — loss: {loss:.4f}  acc: {acc:.4f}")
    return model

# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total   += images.size(0)

    accuracy = correct / total
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return accuracy, cm

# ──────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

def plot_cm(cm, title, save_path, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {save_path}")

def plot_all(cms_dict, save_path):
    """
    cms_dict: dict with key=(model_name, split) and value=cm ndarray.
    Layout: rows=splits (IID, NonIID), cols=models (ConvNeXt, MobileNetV2, ResNet18).
    """
    splits      = ["IID", "NonIID"]
    model_names = list(MODEL_FACTORIES.keys())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Confusion Matrices — MNIST (rows: data split, cols: model)",
                 fontsize=15, fontweight="bold", y=1.01)

    for row, split in enumerate(splits):
        for col, mname in enumerate(model_names):
            cm  = cms_dict[(mname, split)]
            acc = accuracies_global[(mname, split)]
            ax  = axes[row][col]
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.3, linecolor="gray",
                cbar=False,
            )
            ax.set_title(f"{mname} — {split}  (acc={acc*100:.1f}%)",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Combined figure saved: {save_path}")

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

# Module-level dicts so plot_all can access accuracies
cms_global        = {}
accuracies_global = {}

def main():
    print("=" * 60)
    print("Loading MNIST ...")
    train_dataset = get_mnist(train=True)
    test_dataset  = get_mnist(train=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test  samples: {len(test_dataset)}")

    print("\nBuilding IID loader ...")
    iid_loader = make_iid_loader(train_dataset)

    print("\nBuilding Non-IID loader ...")
    noniid_loader = make_noniid_loader(train_dataset)

    splits = {
        "IID":    iid_loader,
        "NonIID": noniid_loader,
    }

    for split_name, loader in splits.items():
        print(f"\n{'='*60}")
        print(f"DATA SPLIT: {split_name}")
        print(f"{'='*60}")

        for model_name, factory in MODEL_FACTORIES.items():
            print(f"\n  Model: {model_name}")
            model = factory()
            print(f"  Training ({EPOCHS} epochs, lr={LR}) ...")
            model = train(model, loader)

            print(f"  Evaluating on full test set ...")
            acc, cm = evaluate(model, test_loader)
            accuracies_global[(model_name, split_name)] = acc
            cms_global[(model_name, split_name)] = cm

            print(f"  Test accuracy: {acc:.4f}  ({acc*100:.2f}%)")

            # Individual confusion matrix PNG
            title = f"{model_name} — {split_name}  (acc={acc*100:.1f}%)"
            fname = f"cm_{model_name}_{split_name}.png"
            plot_cm(cm, title, os.path.join(OUT_DIR, fname))

            # Free memory between runs
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # Combined 2x3 figure
    print("\nGenerating combined confusion matrix figure ...")
    plot_all(cms_global, os.path.join(OUT_DIR, "all_confusion_matrices.png"))

    # Summary table
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    header = f"{'Model':<15} {'IID':>10} {'NonIID':>10}"
    print(header)
    print("-" * len(header))
    for mname in MODEL_FACTORIES:
        iid_acc    = accuracies_global.get((mname, "IID"),    float("nan"))
        noniid_acc = accuracies_global.get((mname, "NonIID"), float("nan"))
        print(f"{mname:<15} {iid_acc*100:>9.2f}% {noniid_acc*100:>9.2f}%")
    print("=" * 60)
    print(f"\nAll confusion matrices saved to: {OUT_DIR}/")

if __name__ == "__main__":
    main()
