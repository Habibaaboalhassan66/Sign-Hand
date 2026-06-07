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
from torch.utils.data import DataLoader, Subset
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

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-3
IMG_SIZE = 32          # resize 28x28 -> 32x32
PATCH_SIZE = 4         # 4x4 patches -> (32/4)^2 = 64 patches
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_DIM = 256
DROPOUT = 0.1
NUM_CLASSES = 10
IN_CHANNELS = 1        # grayscale

# Non-IID
NUM_CLIENTS = 5
ALPHA = 0.5            # Dirichlet concentration

OUTPUT_DIR = "/home/user/Sign-Hand/confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def load_mnist(train=True):
    return torchvision.datasets.MNIST(
        root="./data", train=train, download=True,
        transform=get_transforms()
    )


def iid_loader(dataset, batch_size):
    """Shuffle the full training set and return a single DataLoader."""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    shuffled = Subset(dataset, indices)
    return DataLoader(shuffled, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=False)


def noniid_loader(dataset, num_clients, alpha, batch_size):
    """
    Dirichlet Non-IID split across `num_clients` virtual clients.
    Returns a DataLoader over the union of all client shards.
    """
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    # For each class, collect indices
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.default_rng(SEED)

    for c in range(num_classes):
        idx = class_indices[c].copy()
        rng.shuffle(idx)
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        # Convert to cumulative split points
        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, split_points)
        for client_id, shard in enumerate(splits):
            client_indices[client_id].extend(shard.tolist())

    # Union all client shards
    all_indices = []
    for ci in client_indices:
        all_indices.extend(ci)

    combined = Subset(dataset, all_indices)
    return DataLoader(combined, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=False)


# ---------------------------------------------------------------------------
# Vision Transformer — built from scratch using nn.TransformerEncoderLayer
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim."""

    def __init__(self, in_channels, patch_size, img_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        # Extract patches using unfold: (B, C, H//p, W//p, p, p)
        x = x.unfold(2, p, p).unfold(3, p, p)
        # Reorder to (B, num_patches, C, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # Flatten each patch: (B, num_patches, C*p*p)
        x = x.view(B, self.num_patches, -1)
        return self.projection(x)


class ViT(nn.Module):
    """
    Lightweight Vision Transformer for MNIST classification.

    Architecture:
        - PatchEmbedding: splits 32x32 image into 64 patches of size 4x4
        - CLS token prepended to patch sequence
        - Learnable positional embeddings
        - Stack of TransformerEncoderLayer blocks (Pre-LN, GELU)
        - Classification head on CLS token output
    """

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(in_channels, patch_size, img_size, embed_dim)

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,      # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # Classification head applied to CLS token
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
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

        # Patch embedding: (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, num_patches+1, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.embed_dropout(x)

        # Transformer encoder
        x = self.transformer(x)

        # Extract CLS token and classify
        cls_out = x[:, 0]
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            running_acc = 100.0 * correct / total
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{num_batches} "
                f"| Loss: {total_loss / (batch_idx + 1):.4f} "
                f"| Acc: {running_acc:.2f}%"
            )

    return total_loss / num_batches, 100.0 * correct / total


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return accuracy, cm


def train_model(loader, split_name):
    print(f"\n{'=' * 60}")
    print(f"Training ViT  —  {split_name} split")
    print(f"{'=' * 60}")

    model = ViT().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, epoch)
        print(
            f"  -- Epoch {epoch} done  |  "
            f"Avg Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%"
        )

    return model


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, title, save_path, ax=None):
    """Plot a single confusion matrix heatmap.

    If `ax` is provided the plot is drawn into that axes and NOT saved
    individually (used for the combined figure). Otherwise a standalone
    figure is created and saved to `save_path`.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(range(NUM_CLASSES)),
        yticklabels=list(range(NUM_CLASSES)),
        ax=ax,
        cbar=standalone,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    if standalone:
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_combined(cm_iid, acc_iid, cm_noniid, acc_noniid, save_path):
    """Side-by-side confusion matrices in a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "ViT on MNIST — IID vs Non-IID Confusion Matrices",
        fontsize=15, fontweight="bold",
    )

    plot_confusion_matrix(
        cm_iid,
        f"IID Split  (Test Acc: {acc_iid:.2f}%)",
        save_path=None,
        ax=axes[0],
    )
    plot_confusion_matrix(
        cm_noniid,
        f"Non-IID Split  (Test Acc: {acc_noniid:.2f}%)",
        save_path=None,
        ax=axes[1],
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading MNIST dataset...")
    train_dataset = load_mnist(train=True)
    test_dataset = load_mnist(train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # ------------------------------------------------------------------ IID
    print("\nPreparing IID data loader (shuffled full 60k)...")
    iid_train_loader = iid_loader(train_dataset, BATCH_SIZE)

    model_iid = train_model(iid_train_loader, "IID")

    print("\nEvaluating IID model on test set...")
    acc_iid, cm_iid = evaluate(model_iid, test_loader)
    print(f"IID Test Accuracy: {acc_iid:.2f}%")

    cm_iid_path = os.path.join(OUTPUT_DIR, "cm_ViT_IID.png")
    plot_confusion_matrix(cm_iid, f"ViT — IID Split  (Acc: {acc_iid:.2f}%)", cm_iid_path)

    # --------------------------------------------------------------- Non-IID
    print(f"\nPreparing Non-IID data loader "
          f"(Dirichlet alpha={ALPHA}, {NUM_CLIENTS} clients)...")
    noniid_train_loader = noniid_loader(train_dataset, NUM_CLIENTS, ALPHA, BATCH_SIZE)

    model_noniid = train_model(noniid_train_loader, "Non-IID")

    print("\nEvaluating Non-IID model on test set...")
    acc_noniid, cm_noniid = evaluate(model_noniid, test_loader)
    print(f"Non-IID Test Accuracy: {acc_noniid:.2f}%")

    cm_noniid_path = os.path.join(OUTPUT_DIR, "cm_ViT_NonIID.png")
    plot_confusion_matrix(
        cm_noniid,
        f"ViT — Non-IID Split  (Acc: {acc_noniid:.2f}%)",
        cm_noniid_path,
    )

    # ------------------------------------------------------------ Combined
    combined_path = os.path.join(OUTPUT_DIR, "cm_ViT_combined.png")
    plot_combined(cm_iid, acc_iid, cm_noniid, acc_noniid, combined_path)

    # ------------------------------------------------------------ Summary
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    print(f"  ViT  IID     Test Accuracy : {acc_iid:.2f}%")
    print(f"  ViT  Non-IID Test Accuracy : {acc_noniid:.2f}%")
    print(f"  Delta (IID - Non-IID)       : {acc_iid - acc_noniid:+.2f}%")
    print("=" * 60)
    print(f"\nOutput directory : {OUTPUT_DIR}")
    print("  cm_ViT_IID.png")
    print("  cm_ViT_NonIID.png")
    print("  cm_ViT_combined.png")


if __name__ == "__main__":
    main()
