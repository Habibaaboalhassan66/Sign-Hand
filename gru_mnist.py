import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
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

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS     = 3
BATCH_SIZE = 128
LR         = 1e-3
HIDDEN     = 128
LAYERS     = 2
DROPOUT    = 0.3
ALPHA      = 0.5
NUM_CLIENTS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUT_DIR = "/home/user/Sign-Hand/confusion_matrices"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="/home/user/Sign-Hand/data", train=True,  download=True, transform=transform
)
test_dataset  = torchvision.datasets.MNIST(
    root="/home/user/Sign-Hand/data", train=False, download=True, transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── IID split ─────────────────────────────────────────────────────────────────
def make_iid_loader(dataset, batch_size, seed=SEED):
    indices = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)

# ── Non-IID split (Dirichlet) ─────────────────────────────────────────────────
def make_noniid_loader(dataset, num_clients, alpha, batch_size, seed=SEED):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        proportions = (proportions * len(class_idx)).astype(int)
        # Adjust rounding so all indices are assigned
        deficit = len(class_idx) - proportions.sum()
        for i in range(deficit):
            proportions[i % num_clients] += 1
        splits = np.split(class_idx, np.cumsum(proportions)[:-1])
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    # Union all client shards
    all_indices = []
    for cid in range(num_clients):
        all_indices.extend(client_indices[cid])

    subset = Subset(dataset, all_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, dropout=0.3, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, 28, 28)
        out, _ = self.gru(x)          # out: (B, 28, hidden)
        last    = out[:, -1, :]       # (B, hidden)
        return self.fc(last)          # (B, num_classes)

# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for batch_idx, (images, labels) in enumerate(loader):
        # images: (B, 1, 28, 28) → squeeze → (B, 28, 28)
        images = images.squeeze(1).to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Step [{batch_idx+1}/{len(loader)}]  "
                  f"loss={total_loss/total:.4f}  acc={correct/total*100:.2f}%")

    return total_loss / total, correct / total

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.squeeze(1).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    cm  = confusion_matrix(all_labels, all_preds)
    return acc, cm

# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_cm(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

# ── Run one full experiment ───────────────────────────────────────────────────
def run_experiment(tag, loader):
    print(f"\n{'='*60}")
    print(f"Experiment: {tag}")
    print(f"{'='*60}")

    model     = GRUClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        loss, acc = train(model, loader, optimizer, criterion)
        print(f"  => Train loss={loss:.4f}  acc={acc*100:.2f}%")

    print("\nEvaluating on test set...")
    test_acc, cm = evaluate(model, test_loader)
    print(f"Test accuracy ({tag}): {test_acc*100:.2f}%")
    return test_acc, cm

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # IID
    iid_loader    = make_iid_loader(train_dataset, BATCH_SIZE)
    iid_acc, iid_cm = run_experiment("GRU_IID", iid_loader)

    # Non-IID
    noniid_loader        = make_noniid_loader(train_dataset, NUM_CLIENTS, ALPHA, BATCH_SIZE)
    noniid_acc, noniid_cm = run_experiment("GRU_NonIID", noniid_loader)

    # Save individual CMs
    plot_cm(iid_cm,    "GRU – IID Split",     os.path.join(OUT_DIR, "cm_GRU_IID.png"))
    plot_cm(noniid_cm, "GRU – Non-IID Split", os.path.join(OUT_DIR, "cm_GRU_NonIID.png"))

    # Side-by-side combined figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, cm, title in zip(
        axes,
        [iid_cm, noniid_cm],
        ["GRU – IID Split", "GRU – Non-IID Split"]
    ):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(range(10)),
            yticklabels=list(range(10)),
            ax=ax
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("GRU on MNIST – Confusion Matrices", fontsize=15, y=1.01)
    plt.tight_layout()
    combined_path = os.path.join(OUT_DIR, "cm_GRU_combined.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")

    # Summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    print(f"  GRU IID     test accuracy: {iid_acc*100:.2f}%")
    print(f"  GRU Non-IID test accuracy: {noniid_acc*100:.2f}%")
    print("="*60)
