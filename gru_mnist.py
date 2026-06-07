import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS      = 3
BATCH_SIZE  = 128
LR          = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
NUM_CLASSES = 10
INPUT_SIZE  = 28   # features per timestep (one pixel row)
SEQ_LEN     = 28   # timesteps (one per row)
ALPHA       = 0.5  # Dirichlet concentration
NUM_CLIENTS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUT_DIR = "/home/user/Sign-Hand/confusion_matrices"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset  (ToTensor only — no resize)
# ---------------------------------------------------------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="/home/user/Sign-Hand/data", train=True,
    download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="/home/user/Sign-Hand/data", train=False,
    download=True, transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# IID split: randomly shuffle full 60 k training set
# ---------------------------------------------------------------------------
def make_iid_loader(dataset, batch_size=BATCH_SIZE, seed=SEED):
    indices = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------
# Non-IID split: Dirichlet(alpha=0.5) across NUM_CLIENTS virtual clients,
# then union all shards into one loader
# ---------------------------------------------------------------------------
def make_noniid_loader(dataset, num_clients=NUM_CLIENTS, alpha=ALPHA,
                       batch_size=BATCH_SIZE, seed=SEED):
    rng    = np.random.default_rng(seed)
    labels = np.array(dataset.targets)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(NUM_CLASSES):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts      = (proportions * len(class_idx)).astype(int)
        # Fix rounding so every sample is assigned
        deficit = len(class_idx) - counts.sum()
        for i in range(deficit):
            counts[i % num_clients] += 1

        splits = np.split(class_idx, np.cumsum(counts)[:-1])
        for cid, shard in enumerate(splits):
            client_indices[cid].extend(shard.tolist())

    # Union all shards
    all_indices = []
    for cid in range(num_clients):
        all_indices.extend(client_indices[cid])

    return DataLoader(Subset(dataset, all_indices), batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        # x: (B, 28, 28)  — batch of 28-step sequences of 28 features
        out, _ = self.gru(x)     # (B, 28, HIDDEN_SIZE)
        last   = out[:, -1, :]   # take last hidden state: (B, HIDDEN_SIZE)
        return self.fc(last)     # (B, NUM_CLASSES)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # images: (B, 1, 28, 28) → squeeze dim-1 → (B, 28, 28)
        x = images.squeeze(1).to(DEVICE)
        y = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch}  Step [{batch_idx+1}/{len(loader)}]  "
                  f"loss={total_loss/total:.4f}  acc={correct/total*100:.2f}%")

    return total_loss / total, correct / total

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            x = images.squeeze(1).to(DEVICE)
            logits = model(x)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    cm  = confusion_matrix(all_labels, all_preds)
    return acc, cm

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(tag, train_loader):
    print(f"\n{'='*60}")
    print(f"  Training GRU  —  {tag}")
    print(f"{'='*60}")

    # Fresh model + optimizer for each experiment
    torch.manual_seed(SEED)
    model     = GRUClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        print(f"  => Epoch {epoch} done  loss={loss:.4f}  train_acc={train_acc*100:.2f}%")

    print(f"\nEvaluating on MNIST test set ({tag})...")
    test_acc, cm = evaluate(model, test_loader)
    print(f"  Test Accuracy [{tag}]: {test_acc*100:.2f}%")
    return test_acc, cm

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_cm(cm, title, filepath):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(range(NUM_CLASSES)),
        yticklabels=list(range(NUM_CLASSES)),
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_combined(cm_iid, cm_noniid, filepath):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    titles = ["GRU — IID Split", f"GRU — Non-IID Split (Dirichlet α={ALPHA})"]

    for ax, cm, title in zip(axes, [cm_iid, cm_noniid], titles):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(range(NUM_CLASSES)),
            yticklabels=list(range(NUM_CLASSES)),
            ax=ax,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    plt.suptitle("GRU on MNIST — Confusion Matrices", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building IID data loader...")
    iid_train_loader = make_iid_loader(train_dataset)

    print("Building Non-IID (Dirichlet) data loader...")
    noniid_train_loader = make_noniid_loader(train_dataset)

    # Run experiments
    iid_acc,    iid_cm    = run_experiment("GRU_IID",    iid_train_loader)
    noniid_acc, noniid_cm = run_experiment("GRU_NonIID", noniid_train_loader)

    # Save confusion matrices
    print("\nSaving confusion matrices...")
    plot_cm(
        iid_cm,
        "GRU on MNIST — IID Split",
        os.path.join(OUT_DIR, "cm_GRU_IID.png"),
    )
    plot_cm(
        noniid_cm,
        f"GRU on MNIST — Non-IID Split (Dirichlet α={ALPHA})",
        os.path.join(OUT_DIR, "cm_GRU_NonIID.png"),
    )
    plot_combined(iid_cm, noniid_cm, os.path.join(OUT_DIR, "cm_GRU_combined.png"))

    # Accuracy summary
    print("\n" + "="*60)
    print("  ACCURACY SUMMARY")
    print("="*60)
    print(f"  GRU IID     test accuracy : {iid_acc*100:.2f}%")
    print(f"  GRU Non-IID test accuracy : {noniid_acc*100:.2f}%")
    print("="*60)
