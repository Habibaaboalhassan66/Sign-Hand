"""
Vision Transformer (ViT) trained on MNIST with IID and Non-IID data splits.
Outputs confusion matrices as PNG files.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-3
IMG_SIZE = 32
PATCH_SIZE = 4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_DIM = 256
DROPOUT = 0.1
NUM_CLASSES = 10
NUM_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
OUTPUT_DIR = "/home/user/Sign-Hand/confusion_matrices"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = torchvision.datasets.MNIST(
    root="/tmp/mnist_data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="/tmp/mnist_data", train=False, download=True, transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


def get_iid_loader():
    """Shuffle the full 60k training set randomly (IID)."""
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    subset = Subset(train_dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)


def get_noniid_loader():
    """
    Dirichlet (alpha=0.5) Non-IID split across NUM_CLIENTS virtual clients.
    All shards are unioned into a single loader.
    """
    labels = np.array([train_dataset.targets[i].item() for i in range(len(train_dataset))])
    num_classes = NUM_CLASSES

    # Group indices by class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(NUM_CLIENTS)]

    rng = np.random.default_rng(SEED)
    for c in range(num_classes):
        idx = class_indices[c].tolist()
        rng.shuffle(idx)
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(DIRICHLET_ALPHA, NUM_CLIENTS))
        # Convert proportions to counts
        splits = (proportions * len(idx)).astype(int)
        # Fix rounding so sum == len(idx)
        splits[-1] = len(idx) - splits[:-1].sum()
        splits = np.maximum(splits, 0)

        start = 0
        for k in range(NUM_CLIENTS):
            end = start + splits[k]
            client_indices[k].extend(idx[start:end])
            start = end

    # Union all client shards
    all_indices = []
    for k in range(NUM_CLIENTS):
        all_indices.extend(client_indices[k])

    print(f"  Non-IID total samples (after union): {len(all_indices)}")
    subset = Subset(train_dataset, all_indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------------------------------------------------------
# Model: Vision Transformer from scratch
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and linearly project."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        # Conv with kernel=stride=patch_size extracts patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)                     # (B, E, H/P, W/P)
        x = x.flatten(2)                     # (B, E, num_patches)
        x = x.transpose(1, 2)               # (B, num_patches, E)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                          # (B, N, E)

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, E)
        x = torch.cat([cls, x], dim=1)                   # (B, N+1, E)
        x = x + self.pos_embed                           # add positional embedding
        x = self.dropout(x)

        x = self.transformer(x)                          # (B, N+1, E)
        x = self.norm(x[:, 0])                           # CLS token -> (B, E)
        x = self.head(x)                                 # (B, num_classes)
        return x


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, epoch):
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

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                f"| Loss: {total_loss/total:.4f} "
                f"| Train Acc: {100.*correct/total:.2f}%"
            )

    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return acc, cm


def train_and_evaluate(split_name, loader):
    print(f"\n{'='*60}")
    print(f"Training with {split_name} split")
    print(f"{'='*60}")

    model = ViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=1,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, epoch)
        print(f"  => Epoch {epoch} summary | Loss: {train_loss:.4f} | Train Acc: {100.*train_acc:.2f}%")

    print(f"\nEvaluating on test set...")
    test_acc, cm = evaluate(model, test_loader)
    print(f"Test Accuracy ({split_name}): {100.*test_acc:.2f}%")

    return test_acc, cm


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------
def plot_cm(cm, title, save_path, ax=None):
    """Plot a single confusion matrix as a heatmap."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize row-wise for readability
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=list(range(NUM_CLASSES)),
        yticklabels=list(range(NUM_CLASSES)),
        ax=ax,
        cbar=standalone,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    if standalone:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_combined(cm_iid, cm_noniid, save_path):
    """Plot IID and Non-IID confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_cm(cm_iid, "ViT — IID Split", None, ax=axes[0])
    plot_cm(cm_noniid, "ViT — Non-IID Split (Dirichlet α=0.5)", None, ax=axes[1])
    fig.suptitle("MNIST Confusion Matrices: ViT (IID vs Non-IID)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Preparing data loaders...")
    iid_loader = get_iid_loader()
    print(f"IID loader: {len(iid_loader.dataset)} samples, {len(iid_loader)} batches")

    noniid_loader = get_noniid_loader()
    print(f"Non-IID loader: {len(noniid_loader.dataset)} samples, {len(noniid_loader)} batches")

    # Train IID
    torch.manual_seed(SEED)
    acc_iid, cm_iid = train_and_evaluate("IID", iid_loader)

    # Train Non-IID
    torch.manual_seed(SEED)
    acc_noniid, cm_noniid = train_and_evaluate("Non-IID", noniid_loader)

    # Save individual confusion matrices
    print("\nSaving confusion matrices...")
    plot_cm(cm_iid, "ViT — IID Split", os.path.join(OUTPUT_DIR, "cm_ViT_IID.png"))
    plot_cm(cm_noniid, "ViT — Non-IID Split (Dirichlet α=0.5)", os.path.join(OUTPUT_DIR, "cm_ViT_NonIID.png"))
    plot_combined(cm_iid, cm_noniid, os.path.join(OUTPUT_DIR, "cm_ViT_combined.png"))

    # Accuracy summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    print(f"  ViT  IID   Test Accuracy : {100.*acc_iid:.2f}%")
    print(f"  ViT  Non-IID Test Accuracy: {100.*acc_noniid:.2f}%")
    print(f"  Delta (IID - Non-IID)     : {100.*(acc_iid - acc_noniid):+.2f}%")
    print("="*60)
    print(f"\nConfusion matrix PNGs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
