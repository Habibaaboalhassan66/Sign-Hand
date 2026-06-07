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

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS     = 3
BATCH_SIZE = 128
LR         = 1e-3
HIDDEN     = 128
NUM_LAYERS = 2
DROPOUT    = 0.3
INPUT_SIZE = 28   # one row = 28 pixels
SEQ_LEN    = 28   # 28 rows per image
NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUTPUT_DIR = "/home/user/Sign-Hand/confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Model ────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.fc = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x):
        # x: (B, 1, 28, 28) → squeeze → (B, 28, 28)
        x = x.squeeze(1)
        out, _ = self.lstm(x)        # out: (B, 28, 128)
        last = out[:, -1, :]         # (B, 128)
        return self.fc(last)         # (B, 10)


# ── Data helpers ─────────────────────────────────────────────────────────────
def get_datasets():
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(root="/tmp/mnist", train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=tf)
    return train_ds, test_ds


def iid_indices(train_ds):
    """Shuffle all 60k indices — effectively IID."""
    idx = list(range(len(train_ds)))
    random.shuffle(idx)
    return idx


def noniid_dirichlet_indices(train_ds, num_clients=5, alpha=0.5):
    """
    Dirichlet(alpha) split across num_clients virtual clients.
    Returns the union of all client shards (same data, different ordering).
    """
    targets = np.array(train_ds.targets)
    n = len(targets)
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(NUM_CLASSES):
        cls_idx = np.where(targets == cls)[0]
        np.random.shuffle(cls_idx)
        # Draw proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert to counts that sum to len(cls_idx)
        counts = (proportions * len(cls_idx)).astype(int)
        # Fix rounding so we don't lose samples
        counts[-1] = len(cls_idx) - counts[:-1].sum()
        splits = np.split(cls_idx, np.cumsum(counts[:-1]))
        for c, shard in enumerate(splits):
            client_indices[c].extend(shard.tolist())

    # Union of all shards
    all_idx = []
    for c in client_indices:
        all_idx.extend(c)
    # Shuffle so batches are mixed across clients
    random.shuffle(all_idx)
    return all_idx


# ── Training / Evaluation ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
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

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | batch {batch_idx+1}/{len(loader)} "
                  f"| loss {loss.item():.4f}")

    avg_loss = total_loss / total
    acc = correct / total
    print(f"  Epoch {epoch} TRAIN — loss: {avg_loss:.4f}  acc: {acc*100:.2f}%")


def evaluate(model, loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    avg_loss = total_loss / total
    cm = confusion_matrix(all_labels, all_preds)
    return acc, avg_loss, cm


def save_cm(cm, title, filepath):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")


def run_experiment(label, train_indices, train_ds, test_loader, criterion):
    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"{'='*60}")

    subset = Subset(train_ds, train_indices)
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE.type == "cuda"))

    model = LSTMClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion, epoch)

    acc, loss, cm = evaluate(model, test_loader, criterion)
    print(f"\n  TEST — loss: {loss:.4f}  acc: {acc*100:.2f}%")
    return acc, cm


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_ds, test_ds = get_datasets()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    # IID
    iid_idx = iid_indices(train_ds)
    iid_acc, iid_cm = run_experiment("IID", iid_idx, train_ds, test_loader, criterion)

    # Non-IID
    noniid_idx = noniid_dirichlet_indices(train_ds, num_clients=5, alpha=0.5)
    noniid_acc, noniid_cm = run_experiment("Non-IID (Dirichlet α=0.5)",
                                           noniid_idx, train_ds, test_loader, criterion)

    # Save individual confusion matrices
    save_cm(iid_cm,
            f"LSTM on MNIST — IID (acc={iid_acc*100:.2f}%)",
            os.path.join(OUTPUT_DIR, "cm_LSTM_IID.png"))
    save_cm(noniid_cm,
            f"LSTM on MNIST — Non-IID Dirichlet α=0.5 (acc={noniid_acc*100:.2f}%)",
            os.path.join(OUTPUT_DIR, "cm_LSTM_NonIID.png"))

    # Side-by-side combined figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, cm, title in zip(
        axes,
        [iid_cm, noniid_cm],
        [f"IID  (acc={iid_acc*100:.2f}%)",
         f"Non-IID α=0.5  (acc={noniid_acc*100:.2f}%)"]
    ):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(10), yticklabels=range(10), ax=ax)
        ax.set_title(f"LSTM MNIST — {title}", fontsize=13)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, "cm_LSTM_combined.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {combined_path}")

    # Summary
    print("\n" + "="*60)
    print("  ACCURACY SUMMARY")
    print("="*60)
    print(f"  IID      test accuracy: {iid_acc*100:.2f}%")
    print(f"  Non-IID  test accuracy: {noniid_acc*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
