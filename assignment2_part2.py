# =============================================================================
# DA Assignment 2 - Part 2: Task 1 – VGG11 with Custom Regularization
# =============================================================================

# ── Cell 1: Custom Dataset ───────────────────────────────────────────────────
"""
JUSTIFICATION – Data Pipeline:
We use albumentations for augmentation because it natively handles images as
numpy arrays and supports synchronized transforms on both images and masks.
RandomHorizontalFlip and RandomCrop improve generalization; Normalize with
ImageNet stats is standard practice for VGG-style models.
"""
import sys, csv, random, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
warnings.filterwarnings("ignore")

SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR  = Path("data")
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(exist_ok=True)
NUM_CLASSES = 37

BREED_NAMES = [
    "Abyssinian","Bengal","Birman","Bombay","British Shorthair",
    "Egyptian Mau","Maine Coon","Persian","Ragdoll","Russian Blue",
    "Siamese","Sphynx","american_bulldog","american_pit_bull_terrier",
    "basset_hound","beagle","boxer","chihuahua","english_cocker_spaniel",
    "english_setter","german_shorthaired","great_pyrenees","havanese",
    "japanese_chin","keeshond","leonberger","miniature_pinscher",
    "newfoundland","pomeranian","pug","saint_bernard","samoyed",
    "scottish_terrier","shiba_inu","staffordshire_bull_terrier",
    "wheaten_terrier","yorkshire_terrier"
]

IMG_SIZE = 224

train_transform = None
val_transform = None

class PetDataset(Dataset):
    """Oxford-IIIT Pet classification dataset."""
    def __init__(self, split: str, transform=None):
        self.img_dir   = DATA_DIR / split / "images"
        self.transform = transform
        self.samples   = []
        csv_path = DATA_DIR / f"{split}.csv"
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                img_path = self.img_dir / f"{r['filename']}.jpg"
                if img_path.exists():
                    self.samples.append({
                        "path"  : img_path,
                        "label" : int(r["breed_label"]),
                        "xmin"  : int(r["xmin"]), "ymin": int(r["ymin"]),
                        "xmax"  : int(r["xmax"]), "ymax": int(r["ymax"]),
                    })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = np.array(Image.open(s["path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        bbox = torch.tensor([s["xmin"], s["ymin"], s["xmax"], s["ymax"]],
                             dtype=torch.float32)
        return img, s["label"], bbox

train_ds = PetDataset("train", train_transform)
val_ds   = PetDataset("val",   val_transform)
test_ds  = PetDataset("test",  val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

# Visualise augmented batch
def plot_augmented_batch():
    imgs, labels, _ = next(iter(train_loader))
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    fig.suptitle("Augmented Training Batch (un-normalised for display)",
                 fontsize=12, fontweight="bold")
    for ax, img, lbl in zip(axes.flat, imgs, labels):
        img_show = (img * std + mean).permute(1,2,0).clamp(0,1).numpy()
        ax.imshow(img_show)
        ax.set_title(BREED_NAMES[lbl].replace("_"," "), fontsize=5.5)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"06_augmented_batch.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[06] Augmented batch saved.")

plot_augmented_batch()


# ── Cell 2: Custom Dropout Layer ────────────────────────────────────────────
"""
JUSTIFICATION – Custom Dropout:
Standard nn.Dropout is a thin wrapper around F.dropout. We re-implement it
from scratch using torch.rand_like to draw a Bernoulli mask:

  mask ~ Bernoulli(1 - p)

During training we zero masked activations and rescale remaining ones by
1/(1-p) (inverted dropout) so that the expected value is unchanged at test
time. During eval, we simply pass through unchanged.

WHY AFTER BatchNorm AND ReLU in the classifier:
  - BN normalises the pre-activation distribution → important for stable BN
    statistics. Dropping AFTER BN avoids corrupting the running mean/var.
  - ReLU is applied first so we only drop genuinely activated (positive)
    neurons, which has been empirically shown to be more effective.
"""

class CustomDropout(nn.Module):
    """
    Inverted dropout implemented from first principles.
    Does NOT use nn.Dropout or F.dropout.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0,1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x                                 # identity at inference
        # Bernoulli mask: 1 with prob (1-p), 0 with prob p
        mask  = (torch.rand_like(x) > self.p).float()
        scale = 1.0 / (1.0 - self.p)                # inverted scaling
        return x * mask * scale

    def extra_repr(self):
        return f"p={self.p}"

# Quick sanity check
def verify_dropout():
    drop = CustomDropout(p=0.5)
    x    = torch.ones(10000)
    drop.train()
    out_train = drop(x)
    drop.eval()
    out_eval  = drop(x)
    train_zeros = (out_train == 0).float().mean().item()
    print(f"[CustomDropout] Training   zero-rate: {train_zeros:.3f}  (expected ~0.500)")
    print(f"[CustomDropout] Eval       mean     : {out_eval.mean():.4f}  (expected 1.000)")
    assert abs(train_zeros - 0.5) < 0.03, "Dropout rate out of range"
    assert torch.allclose(out_eval, x),   "Eval should be identity"
    print("[CustomDropout] All checks passed.\n")

verify_dropout()


# ── Cell 3: VGG11 Architecture ───────────────────────────────────────────────
"""
JUSTIFICATION – VGG11 with Batch Normalization:

Original VGG11 conv layout (groups): 1-1-2-2-2 (8 conv layers total)
Each conv block: Conv2d → BatchNorm2d → ReLU → (MaxPool at block end)

WHY BatchNorm2d AFTER Conv, BEFORE ReLU:
  Batch Normalisation normalises pre-activation outputs (z = Wx+b), stabilising
  the distribution before the nonlinearity is applied. This is the post-conv,
  pre-activation convention (Ioffe & Szegedy, 2015) and is universally used in
  modernised VGG reproductions. It allows:
    • Higher learning rates (faster convergence)
    • Less dependence on weight initialisation
    • Acts as a mild regulariser (reduces need for dropout in conv blocks)

WHY BatchNorm1d IN CLASSIFIER:
  The dense layers also suffer internal covariate shift. BN1d after each Linear
  layer normalises across the batch dimension (N), keeping activations stable.

DROPOUT PLACEMENT:
  CustomDropout(p=0.5) is applied AFTER BN1d → ReLU in each hidden fc layer.
  This order ensures:
    1. BN statistics are computed on the full (undropped) activation distribution
    2. ReLU is applied before dropping so only positive activations are masked
    3. The expected value at test time is preserved by the inverted scaling
"""

def conv_block(in_c, out_c):
    """Conv2d → BatchNorm2d → ReLU (no pooling; pooling added at block end)."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class VGG11BN(nn.Module):
    """
    VGG11 built entirely from scratch using torch.nn primitives.
    Architecture matches the original paper with two modernisations:
      1. BatchNorm2d after every Conv2d
      2. BatchNorm1d + CustomDropout in the classifier head

    Feature extractor: 8 conv layers in 5 groups  (1+1+2+2+2)
    Classifier: 3 fully-connected layers (4096 → 4096 → num_classes)
    """
    def __init__(self, num_classes: int = 37, drop_p: float = 0.5):
        super().__init__()

        # ── Feature extractor ──────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1: 224→224 then pool→112
            conv_block(3,   64),
            nn.MaxPool2d(2, 2),

            # Block 2: 112→112 then pool→56
            conv_block(64,  128),
            nn.MaxPool2d(2, 2),

            # Block 3: 56→56 then pool→28
            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2, 2),

            # Block 4: 28→28 then pool→14
            conv_block(256, 512),
            conv_block(512, 512),
            nn.MaxPool2d(2, 2),

            # Block 5: 14→14 then pool→7
            conv_block(512, 512),
            conv_block(512, 512),
            nn.MaxPool2d(2, 2),
        )

        # ── Adaptive pool ──────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))   # → (B, 512, 7, 7)

        # ── Classifier ────────────────────────────────────────────────────
        # Each hidden layer: Linear → BN1d → ReLU → CustomDropout
        self.classifier = nn.Sequential(
            nn.Flatten(),                              # (B, 25088)

            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(drop_p),

            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(drop_p),

            nn.Linear(4096, num_classes),              # logits
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming He initialisation for Conv2d; normal for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def feature_maps(self, x):
        """Return intermediate feature maps for visualisation."""
        maps = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                maps[f"pool_{i}"] = x.detach()
        return maps

# Instantiate & verify
model = VGG11BN(num_classes=NUM_CLASSES, drop_p=0.5).to(DEVICE)
dummy = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
out   = model(dummy)
assert out.shape == (2, NUM_CLASSES), f"Expected (2,37) got {out.shape}"
print(model)
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params   : {total_params:,}")
print(f"Trainable params: {train_params:,}")
print(f"Output shape   : {out.shape}  [OK]")


# ── Cell 4: Architecture Diagram (text) ─────────────────────────────────────
def plot_architecture_summary():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    blocks = [
        ("Input\n224×224×3", "#E74C3C"),
        ("Block1\nConv(64)→BN→ReLU\nMaxPool → 112×112", "#8E44AD"),
        ("Block2\nConv(128)→BN→ReLU\nMaxPool → 56×56",  "#2980B9"),
        ("Block3\n2×Conv(256)→BN→ReLU\nMaxPool → 28×28", "#16A085"),
        ("Block4\n2×Conv(512)→BN→ReLU\nMaxPool → 14×14", "#27AE60"),
        ("Block5\n2×Conv(512)→BN→ReLU\nMaxPool → 7×7",  "#F39C12"),
        ("AvgPool\n7×7 → 25088",                        "#D35400"),
        ("FC1: 4096\nBN1d→ReLU→Dropout", "#C0392B"),
        ("FC2: 4096\nBN1d→ReLU→Dropout", "#7D3C98"),
        ("Output\n37 classes",             "#1ABC9C"),
    ]

    n = len(blocks)
    xs = np.linspace(0.05, 0.95, n)
    for i, (label, color) in enumerate(blocks):
        ax.add_patch(mpatches.FancyBboxPatch(
            (xs[i]-0.045, 0.3), 0.09, 0.4,
            boxstyle="round,pad=0.01", linewidth=1.5,
            edgecolor="white", facecolor=color, alpha=0.85,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(xs[i], 0.50, label, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold",
                transform=ax.transAxes, wrap=True)
        if i < n-1:
            ax.annotate("", xy=(xs[i+1]-0.045, 0.50),
                        xytext=(xs[i]+0.045, 0.50),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    ax.set_title("VGG11-BN Architecture (from scratch) – Oxford-IIIT Pet",
                 color="white", fontsize=13, fontweight="bold", pad=20)
    note = ("Design rationale:\n"
            "• BatchNorm2d after Conv stabilises gradients, allows higher LR\n"
            "• BatchNorm1d in FC layers reduces covariate shift in dense activations\n"
            "• CustomDropout placed after BN→ReLU to preserve BN statistics and drop only active neurons")
    ax.text(0.5, 0.08, note, ha="center", va="top", fontsize=9,
            color="#ECF0F1", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2C3E50", alpha=0.8))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"07_vgg11_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("[07] Architecture diagram saved.")

plot_architecture_summary()

print("\n[Part 2 complete] Model defined and verified.")
