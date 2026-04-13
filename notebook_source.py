# ===========================================================================
# DA Assignment 2 - Part 1: Setup, Dataset, EDA Visualizations
# ===========================================================================
# Cell 1: Imports & Config ─────────────────────────────────────────────────
import os, sys, random, csv, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR  = Path(".")
DATA_DIR  = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"; PLOTS_DIR.mkdir(exist_ok=True)

# 37 Oxford-IIIT Pet breed names (class 0-36)
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
NUM_CLASSES = 37

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
# Cell 2: Load CSVs ────────────────────────────────────────────────────────
def load_csv(split):
    path = DATA_DIR / f"{split}.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "filename"   : r["filename"],
                "breed_label": int(r["breed_label"]),
                "species"    : int(r["species"]),
                "xmin": int(r["xmin"]), "ymin": int(r["ymin"]),
                "xmax": int(r["xmax"]), "ymax": int(r["ymax"]),
            })
    return rows

train_data = load_csv("train")
val_data   = load_csv("val")
test_data  = load_csv("test")
print(f"Loaded  train:{len(train_data)}  val:{len(val_data)}  test:{len(test_data)}")
# Cell 3: EDA – Class Distribution ────────────────────────────────────────
def plot_class_distribution():
    counts = np.zeros(NUM_CLASSES, dtype=int)
    for e in train_data:
        counts[e["breed_label"]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle("Oxford-IIIT Pet – Train Set Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    bars = ax.bar(range(NUM_CLASSES), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([n.replace("_"," ") for n in BREED_NAMES],
                       rotation=90, fontsize=6.5)
    ax.set_ylabel("# Samples")
    ax.set_title("Samples per Breed (Train)")
    ax.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                str(cnt), ha="center", va="bottom", fontsize=5)

    # Species pie
    cats = sum(1 for e in train_data if e["species"] == 0)
    dogs = len(train_data) - cats
    axes[1].pie([cats, dogs], labels=[f"Cats\n{cats}", f"Dogs\n{dogs}"],
                autopct="%1.1f%%", colors=["#FF9999","#66B2FF"],
                startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[1].set_title("Cat vs Dog Split (Train)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Min samples: {counts.min()}  Max: {counts.max()}  Mean: {counts.mean():.1f}")
    print(f"  Dataset is {'imbalanced' if counts.max()/counts.min() > 2 else 'roughly balanced'}")

plot_class_distribution()
# Cell 4: EDA – Sample Images Grid ────────────────────────────────────────
def plot_sample_grid(split="train", n_cols=6, n_rows=4):
    data   = {"train":train_data,"val":val_data,"test":test_data}[split]
    img_dir = DATA_DIR / split / "images"
    samples = random.sample(data, min(n_cols*n_rows, len(data)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    fig.suptitle(f"Sample Images – {split.capitalize()} Set", fontsize=13, fontweight="bold")
    for ax, e in zip(axes.flat, samples):
        img_path = img_dir / f"{e['filename']}.jpg"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB").resize((128,128))
            ax.imshow(img)
        ax.set_title(BREED_NAMES[e["breed_label"]].replace("_"," "), fontsize=6.5)
        ax.axis("off")
    for ax in axes.flat[len(samples):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"02_sample_grid_{split}.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_sample_grid("train")
# Cell 5: EDA – Image Size Distribution ───────────────────────────────────
def plot_image_stats():
    img_dir = DATA_DIR / "train" / "images"
    widths, heights, aspects = [], [], []
    for e in train_data[:500]:   # sample 500 for speed
        p = img_dir / f"{e['filename']}.jpg"
        if p.exists():
            w, h = Image.open(p).size
            widths.append(w); heights.append(h)
            aspects.append(w/h)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Image Statistics (Train, n=500 sample)", fontsize=13, fontweight="bold")
    for ax, data, label, color in zip(
        axes,
        [widths, heights, aspects],
        ["Width (px)", "Height (px)", "Aspect Ratio (W/H)"],
        ["#4ECDC4","#FF6B6B","#95E1D3"]
    ):
        ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(data), color="navy", linestyle="--", linewidth=1.5,
                   label=f"Mean={np.mean(data):.1f}")
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_image_stats.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_image_stats()
# Cell 6: EDA – Bounding Box Distribution ─────────────────────────────────
def plot_bbox_stats():
    valid_bb = [(e["xmax"]-e["xmin"], e["ymax"]-e["ymin"])
                for e in train_data if e["xmin"] != -1]
    bw = [v[0] for v in valid_bb]
    bh = [v[1] for v in valid_bb]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Bounding Box Size Distribution (Train)", fontsize=13, fontweight="bold")
    for ax, vals, label, color in zip(
        axes, [bw, bh],
        ["Box Width (px)", "Box Height (px)"],
        ["#F7DC6F","#82E0AA"]
    ):
        ax.hist(vals, bins=40, color=color, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--",
                   label=f"Mean={np.mean(vals):.0f}")
        ax.set_xlabel(label); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_bbox_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Samples with bbox: {len(valid_bb)} / {len(train_data)}")

plot_bbox_stats()
# Cell 7: EDA – Trimap Mask Samples ───────────────────────────────────────
def plot_mask_samples(n=6):
    """Show image + trimap overlay side-by-side for n samples."""
    img_dir  = DATA_DIR / "train" / "images"
    mask_dir = DATA_DIR / "train" / "masks"
    samples  = [e for e in train_data
                if (mask_dir / f"{e['filename']}.png").exists()][:n]

    fig, axes = plt.subplots(n, 2, figsize=(6, n*2.5))
    fig.suptitle("Trimap Segmentation Masks\n(1=Foreground, 2=Background, 3=Uncertain)",
                 fontsize=11, fontweight="bold")
    cmap = plt.cm.colors.ListedColormap(["black","green","gray"])

    for i, e in enumerate(samples):
        img  = np.array(Image.open(img_dir  / f"{e['filename']}.jpg").resize((192,192)))
        mask = np.array(Image.open(mask_dir / f"{e['filename']}.png").resize((192,192)))

        axes[i,0].imshow(img); axes[i,0].axis("off")
        axes[i,0].set_title(BREED_NAMES[e["breed_label"]].replace("_"," "), fontsize=8)

        im = axes[i,1].imshow(mask, cmap="tab10", vmin=1, vmax=3)
        axes[i,1].axis("off"); axes[i,1].set_title("Trimap", fontsize=8)

    patches = [mpatches.Patch(color=plt.cm.tab10(v/10), label=l)
               for v, l in zip([0.1, 0.2, 0.3],
                               ["Foreground (1)", "Background (2)", "Uncertain (3)"])]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_trimap_samples.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_mask_samples()

print("\n[Part 1 complete] All EDA plots saved to:", PLOTS_DIR)
# ===========================================================================
# DA Assignment 2 - Part 2: Task 1 – VGG11 with Custom Regularization
# ===========================================================================
# Cell 1: Custom Dataset ───────────────────────────────────────────────────
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torchvision.transforms as T
from torchvision.transforms import functional as F
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

import torchvision.transforms as T
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

val_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

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
        img = Image.open(s["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
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
    plt.show()

plot_augmented_batch()
# Cell 2: Custom Dropout Layer ────────────────────────────────────────────
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
# Cell 3: VGG11 Architecture ───────────────────────────────────────────────
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
# Cell 4: Architecture Diagram (text) ─────────────────────────────────────
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
    plt.show()

plot_architecture_summary()

print("\n[Part 2 complete] Model defined and verified.")
# ===========================================================================
# DA Assignment 2 - Part 3: Training Loop, Evaluation & Visualizations
# ===========================================================================

# Run parts 1 & 2 first, or copy their definitions here.
# This file assumes: model, train_loader, val_loader, test_loader,
#                    DEVICE, NUM_CLASSES, BREED_NAMES, PLOTS_DIR are defined.

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from pathlib import Path

PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
# Cell 1: Hyperparameters & Optimizer ──────────────────────────────────────
"""
JUSTIFICATION – Optimiser Choice:
  - Adam (lr=3e-4) converges faster than SGD for VGG-style nets from scratch.
  - weight_decay=1e-4 (L2 regularisation) further prevents overfitting
    alongside our Custom Dropout.
  - CosineAnnealingLR gently decays the LR to near-zero, allowing the model
    to settle into sharp minima rather than oscillating.
  - We clip gradients at max_norm=2.0 to stabilise early unstable training.
"""
EPOCHS    = 25
LR        = 3e-4
WD        = 1e-4
PATIENCE  = 7    # early stopping

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-7
)

print(f"Optimizer : Adam(lr={LR}, wd={WD})")
print(f"Scheduler : CosineAnnealingLR(T_max={EPOCHS})")
print(f"Loss      : CrossEntropyLoss(label_smoothing=0.1)")
print(f"Epochs    : {EPOCHS}  |  Early-stopping patience: {PATIENCE}")
# Cell 2: Training Loop ────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, max_norm=2.0):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "lr":[]}
best_val_loss = float("inf")
patience_ctr  = 0

print("\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Train Acc':>10} {'Val Acc':>9} {'LR':>10}")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    history["lr"].append(current_lr)

    elapsed = time.time() - t0
    print(f"{epoch:>5} {tr_loss:>11.4f} {va_loss:>9.4f} "
          f"{tr_acc*100:>9.2f}% {va_acc*100:>8.2f}% {current_lr:>10.2e}  [{elapsed:.0f}s]")

    # Save best checkpoint
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        patience_ctr  = 0
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state"  : optimizer.state_dict(),
            "val_loss"   : va_loss,
            "val_acc"    : va_acc,
        }, MODEL_DIR / "vgg11_best.pth")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  [Early stopping] No improvement for {PATIENCE} epochs.")
            break

print(f"\nBest Val Loss: {best_val_loss:.4f}  |  Saved to models/vgg11_best.pth")
# Cell 3: Training Curves ──────────────────────────────────────────────────
def plot_training_curves(history):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes  = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("VGG11-BN Training Curves – Oxford Pet Classification",
                 fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs_ran, history["train_loss"], "o-", label="Train", color="#E74C3C")
    axes[0].plot(epochs_ran, history["val_loss"],   "s-", label="Val",   color="#3498DB")
    axes[0].set_title("Cross-Entropy Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_ran, [a*100 for a in history["train_acc"]], "o-",
                 label="Train", color="#27AE60")
    axes[1].plot(epochs_ran, [a*100 for a in history["val_acc"]], "s-",
                 label="Val",   color="#F39C12")
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # LR schedule
    axes[2].plot(epochs_ran, history["lr"], "^-", color="#8E44AD")
    axes[2].set_title("Learning Rate Schedule"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR"); axes[2].yaxis.set_major_formatter(
        ticker.FormatStrFormatter("%.1e")); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"08_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_training_curves(history)
# Cell 4: Test Evaluation ──────────────────────────────────────────────────
# Load best checkpoint
checkpoint = torch.load(MODEL_DIR / "vgg11_best.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded best checkpoint (epoch {checkpoint['epoch']}, "
      f"val_acc={checkpoint['val_acc']*100:.2f}%)")

@torch.no_grad()
def get_preds(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds)

y_true, y_pred = get_preds(model, test_loader, DEVICE)
test_acc = (y_true == y_pred).mean()
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print("\nPer-class Report:")
print(classification_report(y_true, y_pred,
                             target_names=[n.replace("_"," ") for n in BREED_NAMES],
                             zero_division=0))
# Cell 5: Confusion Matrix ─────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt="d", cmap="YlOrRd",
                xticklabels=[n.replace("_"," ") for n in BREED_NAMES],
                yticklabels=[n.replace("_"," ") for n in BREED_NAMES],
                linewidths=0.3, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(f"Confusion Matrix – Test Set  (Acc={test_acc*100:.1f}%)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"09_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_confusion_matrix(y_true, y_pred)
# Cell 6: Per-class F1 bar chart ──────────────────────────────────────────
def plot_per_class_f1(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)

    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.suptitle("Per-Class Metrics on Test Set", fontsize=14, fontweight="bold")
    metrics = [("Precision", p, "#3498DB"), ("Recall", r, "#27AE60"),
               ("F1-Score",  f1,"#E74C3C")]
    for ax, (name, vals, color) in zip(axes, metrics):
        bars = ax.bar(range(NUM_CLASSES), vals, color=color, alpha=0.8,
                      edgecolor="white", linewidth=0.4)
        ax.axhline(vals.mean(), color="navy", linestyle="--", linewidth=1.2,
                   label=f"Mean={vals.mean():.3f}")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels([n.replace("_"," ") for n in BREED_NAMES],
                           rotation=90, fontsize=6.5)
        ax.set_ylabel(name); ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"10_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_per_class_f1(y_true, y_pred)
# Cell 7: Prediction Samples (correct & wrong) ────────────────────────────
def plot_predictions(model, loader, device, n_correct=8, n_wrong=8):
    model.eval()
    correct_imgs, wrong_imgs = [], []
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    with torch.no_grad():
        for imgs, labels, _ in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            for img, lbl, pred in zip(imgs, labels, preds):
                img_show = (img * std + mean).permute(1,2,0).clamp(0,1).numpy()
                entry    = (img_show, lbl.item(), pred.item())
                if lbl == pred and len(correct_imgs) < n_correct:
                    correct_imgs.append(entry)
                elif lbl != pred and len(wrong_imgs) < n_wrong:
                    wrong_imgs.append(entry)
            if len(correct_imgs) >= n_correct and len(wrong_imgs) >= n_wrong:
                break

    fig, axes = plt.subplots(2, max(n_correct, n_wrong), figsize=(20, 7))
    fig.suptitle("Model Predictions – Green=Correct, Red=Wrong",
                 fontsize=13, fontweight="bold")

    for ax, (img, lbl, pred) in zip(axes[0], correct_imgs):
        ax.imshow(img)
        ax.set_title(f"True: {BREED_NAMES[lbl].replace('_',' ')[:12]}", fontsize=6,
                     color="green", fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("green"); spine.set_linewidth(3)

    for ax, (img, lbl, pred) in zip(axes[1], wrong_imgs):
        ax.imshow(img)
        ax.set_title(f"T:{BREED_NAMES[lbl].replace('_',' ')[:9]}\n"
                     f"P:{BREED_NAMES[pred].replace('_',' ')[:9]}",
                     fontsize=6, color="red")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"11_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_predictions(model, test_loader, DEVICE)

print("\n[Part 3 complete] Training done. All plots saved to:", PLOTS_DIR)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
# ===========================================================================
# DA Assignment 2 - Part 4: Object Detection (BBox) & Semantic Segmentation
# ===========================================================================

import csv, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
warnings.filterwarnings("ignore")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR  = Path("data")
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
IMG_SIZE  = 224

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
2.1 Detection Dataset ─────────────────────────────────────────────────────
"""
JUSTIFICATION – Bounding Box Normalisation:
  Raw pixel coordinates are image-size dependent. We normalise to [0,1] by
  dividing by the resized image dimension so the regression target is
  scale-invariant, making loss values comparable across batches.
"""

class PetDetectionDataset(Dataset):
    def __init__(self, split, transform=None):
        self.img_dir   = DATA_DIR / split / "images"
        self.transform = transform
        self.samples   = []
        with open(DATA_DIR / f"{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                xi, yi, xa, ya = int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])
                if xi == -1:
                    continue
                p = self.img_dir / f"{r['filename']}.jpg"
                if p.exists():
                    self.samples.append({
                        "path": p,
                        "label": int(r["breed_label"]),
                        "bbox_raw": [xi, yi, xa, ya],
                    })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        ow, oh = img.size
        if self.transform:
            img = self.transform(img)
        # Normalise bbox to [0,1] relative to original image dims
        xi, yi, xa, ya = s["bbox_raw"]
        bbox = torch.tensor([xi/ow, yi/oh, xa/ow, ya/oh], dtype=torch.float32)
        return img, s["label"], bbox

det_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])
det_val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

det_train = DataLoader(PetDetectionDataset("train", det_transform),  batch_size=32, shuffle=True)
det_val   = DataLoader(PetDetectionDataset("val",   det_val_transform), batch_size=32)
det_test  = DataLoader(PetDetectionDataset("test",  det_val_transform), batch_size=32)
print(f"Detection dataset  train:{len(det_train.dataset)}  val:{len(det_val.dataset)}")
2.2 Detection Head on VGG11 Features ─────────────────────────────────────
"""
JUSTIFICATION – Detection Head Design:
  We reuse the frozen VGG11 feature extractor trained in Task 1 as a backbone.
  A lightweight regression head (two FC layers → Sigmoid) is added on top.
  - Sigmoid output constrains predictions to [0,1] (normalised bbox space)
  - SmoothL1 loss (Huber) is less sensitive to outliers than MSE while still
    being differentiable at zero, making it the standard choice for bbox regression.
"""

class VGG11Detector(nn.Module):
    def __init__(self, vgg_backbone: nn.Module):
        super().__init__()
        # Freeze backbone weights
        self.features = vgg_backbone.features
        self.avgpool  = vgg_backbone.avgpool
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.avgpool.parameters():
            p.requires_grad = False

        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 4),
            nn.Sigmoid(),      # normalised [x1,y1,x2,y2] in [0,1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.bbox_head(x)

# Load pretrained VGG11 backbone
from assignment2_part2 import VGG11BN, CustomDropout  # noqa
classifier = VGG11BN(num_classes=37).to(DEVICE)
ckpt = torch.load(MODEL_DIR / "vgg11_best.pth", map_location=DEVICE)
classifier.load_state_dict(ckpt["model_state"])

detector = VGG11Detector(classifier).to(DEVICE)
det_opt  = optim.Adam(filter(lambda p: p.requires_grad, detector.parameters()),
                      lr=1e-4, weight_decay=1e-4)
smooth_l1 = nn.SmoothL1Loss()
2.3 IoU Metric ───────────────────────────────────────────────────────────
"""
JUSTIFICATION – IoU Metric:
  Intersection-over-Union measures the overlap between predicted and ground-truth
  boxes. It is scale-invariant and the most widely used metric for localisation.
    IoU = |pred ∩ gt| / |pred ∪ gt|
"""
def iou_batch(pred, gt):
    """pred, gt: (N, 4) normalised [x1,y1,x2,y2] tensors."""
    xi1 = torch.max(pred[:,0], gt[:,0])
    yi1 = torch.max(pred[:,1], gt[:,1])
    xi2 = torch.min(pred[:,2], gt[:,2])
    yi2 = torch.min(pred[:,3], gt[:,3])
    inter = (xi2-xi1).clamp(0) * (yi2-yi1).clamp(0)
    area_p = (pred[:,2]-pred[:,0]).clamp(0) * (pred[:,3]-pred[:,1]).clamp(0)
    area_g = (gt[:,2]-gt[:,0]).clamp(0)   * (gt[:,3]-gt[:,1]).clamp(0)
    union  = area_p + area_g - inter
    return (inter / (union + 1e-6)).mean().item()
2.4 Training – Detection Head ────────────────────────────────────────────
DET_EPOCHS = 1
det_hist   = {"train_loss":[], "val_loss":[], "val_iou":[]}

print("\nTraining detection head ...")
print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} {'Val IoU':>9}")
for epoch in range(1, DET_EPOCHS+1):
    detector.train()
    tr_loss = 0.
    for imgs, _, bboxes in det_train:
        imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
        det_opt.zero_grad()
        pred  = detector(imgs)
        loss  = smooth_l1(pred, bboxes)
        loss.backward()
        det_opt.step()
        tr_loss += loss.item()

    detector.eval()
    va_loss, va_iou = 0., 0.
    with torch.no_grad():
        for imgs, _, bboxes in det_val:
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            pred   = detector(imgs)
            va_loss += smooth_l1(pred, bboxes).item()
            va_iou  += iou_batch(pred, bboxes)
    n_v = len(det_val)
    det_hist["train_loss"].append(tr_loss / len(det_train))
    det_hist["val_loss"].append(va_loss / n_v)
    det_hist["val_iou"].append(va_iou / n_v)
    print(f"{epoch:>5} {tr_loss/len(det_train):>11.4f} "
          f"{va_loss/n_v:>10.4f} {va_iou/n_v:>9.4f}")
    torch.save(detector.state_dict(), MODEL_DIR/"detector_best.pth")
2.5 Detection Curves + BBox Visualisation ────────────────────────────────
def plot_detection_curves(hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Object Detection Training Curves", fontsize=12, fontweight="bold")
    ep = range(1, len(hist["train_loss"])+1)
    axes[0].plot(ep, hist["train_loss"], "o-", label="Train", color="#E74C3C")
    axes[0].plot(ep, hist["val_loss"],   "s-", label="Val",   color="#3498DB")
    axes[0].set_title("SmoothL1 Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, hist["val_iou"], "^-", color="#27AE60")
    axes[1].set_title("Validation IoU"); axes[1].set_ylabel("Mean IoU"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"12_detection_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_detection_curves(det_hist)

def plot_bbox_predictions(n=6):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    detector.eval()
    fig, axes = plt.subplots(2, n, figsize=(n*3, 6))
    fig.suptitle("BBox Predictions vs Ground Truth\n"
                 "(Green=GT, Red=Predicted)", fontsize=11, fontweight="bold")

    with torch.no_grad():
        for imgs, labels, bboxes in det_test:
            imgs_d = imgs[:n].to(DEVICE)
            pred   = detector(imgs_d).cpu()
            for i in range(min(n, len(imgs))):
                img_show = (imgs[i] * std + mean).permute(1,2,0).clamp(0,1).numpy()
                # --- original image row ---
                ax = axes[0, i]
                ax.imshow(img_show); ax.axis("off")
                ax.set_title(BREED_NAMES[labels[i]].replace("_"," "), fontsize=7)
                # --- image with boxes row ---
                ax2 = axes[1, i]
                ax2.imshow(img_show); ax2.axis("off")
                W = H = IMG_SIZE
                gx1,gy1,gx2,gy2 = bboxes[i].numpy() * [W,H,W,H]
                px1,py1,px2,py2  = pred[i].numpy()   * [W,H,W,H]
                ax2.add_patch(patches.Rectangle(
                    (gx1,gy1), gx2-gx1, gy2-gy1,
                    linewidth=2, edgecolor="lime", facecolor="none"))
                ax2.add_patch(patches.Rectangle(
                    (px1,py1), px2-px1, py2-py1,
                    linewidth=2, edgecolor="red", facecolor="none"))
                iou = iou_batch(pred[i:i+1], bboxes[i:i+1])
                ax2.set_title(f"IoU={iou:.2f}", fontsize=8)
            break
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"13_bbox_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_bbox_predictions()


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3: SEMANTIC SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════
"""
JUSTIFICATION – Segmentation Architecture:
  We use a lightweight encoder-decoder (U-Net style) where the encoder is the
  VGG11 feature pyramid and the decoder uses bilinear upsampling + Conv layers
  to recover spatial resolution.

  WHY NOT FULLY TRANSPOSED CONV:
    Bilinear + Conv avoids checkerboard artefacts common with transposed conv.

  LOSS = CrossEntropy + Dice:
    CrossEntropy handles per-pixel class distribution; Dice maximises overlap
    between predicted and true segments – combination is standard in medical/
    general segmentation and handles class imbalance better than CE alone.
"""

class PetSegDataset(Dataset):
    def __init__(self, split, transform=None):
        self.img_dir  = DATA_DIR / split / "images"
        self.mask_dir = DATA_DIR / split / "masks"
        self.transform = transform
        self.samples = []
        with open(DATA_DIR / f"{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                ip = self.img_dir  / f"{r['filename']}.jpg"
                mp = self.mask_dir / f"{r['filename']}.png"
                if ip.exists() and mp.exists():
                    self.samples.append({"img": ip, "mask": mp})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        img = Image.open(s["img"]).convert("RGB")
        mask = Image.open(s["mask"])
        import numpy as np
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img = T.ToTensor()(img)
        img = T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))(img)
        mask = np.array(mask)
        mask = np.clip(mask - 1, 0, 2).astype(np.int64)
        return img, torch.tensor(mask, dtype=torch.long)

seg_transform = None

seg_tr = DataLoader(PetSegDataset("train", seg_transform), batch_size=16, shuffle=True)
seg_va = DataLoader(PetSegDataset("val",   seg_transform), batch_size=16)
seg_te = DataLoader(PetSegDataset("test",  seg_transform), batch_size=16)
print(f"\nSegmentation dataset  train:{len(seg_tr.dataset)}  val:{len(seg_va.dataset)}")


class VGG11Segmenter(nn.Module):
    """Encoder-decoder segmenter built on frozen VGG11 features."""
    def __init__(self, backbone: nn.Module, num_seg_classes=3):
        super().__init__()
        # Split backbone into 5 blocks for skip connections
        feats = list(backbone.features.children())
        self.enc1 = nn.Sequential(*feats[0:2])    # pool → 112
        self.enc2 = nn.Sequential(*feats[2:4])    # pool → 56
        self.enc3 = nn.Sequential(*feats[4:7])    # pool → 28
        self.enc4 = nn.Sequential(*feats[7:10])   # pool → 14
        self.enc5 = nn.Sequential(*feats[10:])    # pool → 7

        for enc in [self.enc1,self.enc2,self.enc3,self.enc4,self.enc5]:
            for p in enc.parameters():
                p.requires_grad = False

        def up_block(in_c, skip_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c + skip_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        self.up4 = up_block(512, 512, 256)
        self.up3 = up_block(256, 256, 128)
        self.up2 = up_block(128, 128, 64)
        self.up1 = up_block(64,  64,  32)
        self.head = nn.Conv2d(32, num_seg_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)          # (B,64,112,112)
        s2 = self.enc2(s1)         # (B,128,56,56)
        s3 = self.enc3(s2)         # (B,256,28,28)
        s4 = self.enc4(s3)         # (B,512,14,14)
        s5 = self.enc5(s4)         # (B,512,7,7)

        def up(x, skip):
            x = nn.functional.interpolate(x, size=skip.shape[2:],
                                          mode="bilinear", align_corners=False)
            return torch.cat([x, skip], dim=1)

        import torch.nn.functional as F
        x = self.up4(up(s5, s4))
        x = self.up3(up(x,  s3))
        x = self.up2(up(x,  s2))
        x = self.up1(up(x,  s1))
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False)
        return self.head(x)


segmenter = VGG11Segmenter(classifier, num_seg_classes=3).to(DEVICE)
seg_opt   = optim.Adam(filter(lambda p: p.requires_grad, segmenter.parameters()),
                       lr=1e-3, weight_decay=1e-4)


def dice_loss(pred, target, num_classes=3, smooth=1.):
    pred   = torch.softmax(pred, dim=1)
    target = nn.functional.one_hot(target, num_classes).permute(0,3,1,2).float()
    inter  = (pred * target).sum(dim=(2,3))
    union  = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    return (1 - (2*inter+smooth)/(union+smooth)).mean()

ce_loss = nn.CrossEntropyLoss()

SEG_EPOCHS = 1
seg_hist   = {"train_loss":[], "val_loss":[], "val_miou":[]}

def seg_miou(pred_logits, masks, num_classes=3):
    preds = pred_logits.argmax(1)
    ious  = []
    for c in range(num_classes):
        inter = ((preds==c) & (masks==c)).float().sum()
        union = ((preds==c) | (masks==c)).float().sum()
        if union > 0:
            ious.append((inter/(union+1e-6)).item())
    return np.mean(ious) if ious else 0.

print("\nTraining segmentation head ...")
print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} {'Val mIoU':>10}")
for epoch in range(1, SEG_EPOCHS+1):
    segmenter.train()
    tr_loss = 0.
    for imgs, masks in seg_tr:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        seg_opt.zero_grad()
        out  = segmenter(imgs)
        loss = ce_loss(out, masks) + dice_loss(out, masks)
        loss.backward(); seg_opt.step()
        tr_loss += loss.item()

    segmenter.eval()
    va_loss = va_iou = 0.
    with torch.no_grad():
        for imgs, masks in seg_va:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = segmenter(imgs)
            va_loss += (ce_loss(out, masks) + dice_loss(out, masks)).item()
            va_iou  += seg_miou(out, masks)
    n_v = len(seg_va)
    seg_hist["train_loss"].append(tr_loss/len(seg_tr))
    seg_hist["val_loss"].append(va_loss/n_v)
    seg_hist["val_miou"].append(va_iou/n_v)
    print(f"{epoch:>5} {tr_loss/len(seg_tr):>11.4f} "
          f"{va_loss/n_v:>10.4f} {va_iou/n_v:>10.4f}")
    torch.save(segmenter.state_dict(), MODEL_DIR/"segmenter_best.pth")


def plot_segmentation_curves(hist):
    ep = range(1, len(hist["train_loss"])+1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Segmentation Training Curves (CE + Dice)", fontsize=12, fontweight="bold")
    axes[0].plot(ep, hist["train_loss"], "o-", label="Train", color="#E74C3C")
    axes[0].plot(ep, hist["val_loss"],   "s-", label="Val",   color="#3498DB")
    axes[0].set_title("Combined Loss (CE+Dice)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, hist["val_miou"], "^-", color="#8E44AD")
    axes[1].set_title("Validation mIoU"); axes[1].set_ylabel("mIoU"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"14_segmentation_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_segmentation_curves(seg_hist)


def plot_segmentation_results(n=4):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    segmenter.eval()
    COLOR_MAP = np.array([[139,0,0], [0,139,0], [128,128,128]], dtype=np.uint8)

    fig, axes = plt.subplots(n, 3, figsize=(9, n*3))
    fig.suptitle("Segmentation: Original | Ground Truth | Prediction\n"
                 "(Red=Foreground, Green=Background, Grey=Uncertain)",
                 fontsize=11, fontweight="bold")

    with torch.no_grad():
        for imgs, masks in seg_te:
            imgs_d = imgs[:n].to(DEVICE)
            preds  = segmenter(imgs_d).argmax(1).cpu().numpy()
            for i in range(min(n, len(imgs))):
                img_show = (imgs[i]*std + mean).permute(1,2,0).clamp(0,1).numpy()
                gt_col   = COLOR_MAP[masks[i].numpy()]
                pr_col   = COLOR_MAP[preds[i]]
                axes[i,0].imshow(img_show);       axes[i,0].axis("off")
                axes[i,1].imshow(gt_col);         axes[i,1].axis("off")
                axes[i,2].imshow(pr_col);         axes[i,2].axis("off")
                if i == 0:
                    axes[i,0].set_title("Original",     fontsize=9)
                    axes[i,1].set_title("Ground Truth",  fontsize=9)
                    axes[i,2].set_title("Prediction",    fontsize=9)
            break
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"15_segmentation_results.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_segmentation_results()

print("\n[Part 4 complete] Detection + Segmentation done. All plots saved.")