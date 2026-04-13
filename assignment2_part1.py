# =============================================================================
# DA Assignment 2 - Part 1: Setup, Dataset, EDA Visualizations
# =============================================================================

# ── Cell 1: Imports & Config ─────────────────────────────────────────────────
import os, sys, random, csv, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
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

print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Plots  : {PLOTS_DIR.resolve()}")


# ── Cell 2: Load CSVs ────────────────────────────────────────────────────────
def load_csv(split):
    rows = []
    with open(DATA_DIR / f"{split}.csv", newline="") as f:
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
print(f"\nLoaded  train:{len(train_data)}  val:{len(val_data)}  test:{len(test_data)}")


# ── Cell 3: EDA – Class Distribution ────────────────────────────────────────
def plot_class_distribution():
    counts = np.zeros(NUM_CLASSES, dtype=int)
    for e in train_data:
        counts[e["breed_label"]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle("Oxford-IIIT Pet – Train Set Class Distribution",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    bars = axes[0].bar(range(NUM_CLASSES), counts, color=colors,
                       edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_xticklabels([n.replace("_"," ") for n in BREED_NAMES],
                             rotation=90, fontsize=6.5)
    axes[0].set_ylabel("# Samples")
    axes[0].set_title("Samples per Breed (Train)")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                     str(cnt), ha="center", va="bottom", fontsize=5)

    cats = sum(1 for e in train_data if e["species"] == 0)
    dogs = len(train_data) - cats
    axes[1].pie([cats, dogs],
                labels=[f"Cats\n{cats}", f"Dogs\n{dogs}"],
                autopct="%1.1f%%", colors=["#FF9999","#66B2FF"],
                startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[1].set_title("Cat vs Dog Split (Train)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[01] Class distribution saved.  "
          f"Min={counts.min()}  Max={counts.max()}  Mean={counts.mean():.1f}")

plot_class_distribution()


# ── Cell 4: EDA – Sample Images Grid ────────────────────────────────────────
def plot_sample_grid(split="train", n_cols=6, n_rows=4):
    data    = {"train":train_data, "val":val_data, "test":test_data}[split]
    img_dir = DATA_DIR / split / "images"
    samples = random.sample(data, min(n_cols*n_rows, len(data)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    fig.suptitle(f"Sample Images – {split.capitalize()} Set",
                 fontsize=13, fontweight="bold")
    for ax, e in zip(axes.flat, samples):
        img_path = img_dir / f"{e['filename']}.jpg"
        if img_path.exists():
            ax.imshow(Image.open(img_path).convert("RGB").resize((128,128)))
        ax.set_title(BREED_NAMES[e["breed_label"]].replace("_"," "), fontsize=6.5)
        ax.axis("off")
    for ax in axes.flat[len(samples):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"02_sample_grid_{split}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[02] Sample grid ({split}) saved.")

plot_sample_grid("train")


# ── Cell 5: EDA – Image Size Distribution ───────────────────────────────────
def plot_image_stats():
    img_dir = DATA_DIR / "train" / "images"
    widths, heights, aspects = [], [], []
    for e in train_data:
        p = img_dir / f"{e['filename']}.jpg"
        if p.exists():
            w, h = Image.open(p).size
            widths.append(w); heights.append(h); aspects.append(w/h)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Image Statistics (Train set)", fontsize=13, fontweight="bold")
    for ax, data_vals, label, color in zip(
        axes,
        [widths, heights, aspects],
        ["Width (px)", "Height (px)", "Aspect Ratio (W/H)"],
        ["#4ECDC4","#FF6B6B","#95E1D3"]
    ):
        ax.hist(data_vals, bins=30, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(data_vals), color="navy", linestyle="--", linewidth=1.5,
                   label=f"Mean={np.mean(data_vals):.1f}")
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_image_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[03] Image stats saved.  Avg size: {np.mean(widths):.0f}x{np.mean(heights):.0f}")

plot_image_stats()


# ── Cell 6: EDA – Bounding Box Distribution ─────────────────────────────────
def plot_bbox_stats():
    valid_bb = [(e["xmax"]-e["xmin"], e["ymax"]-e["ymin"])
                for e in train_data if e["xmin"] != -1]
    if not valid_bb:
        print("[04] No bbox data found, skipping.")
        return
    bw = [v[0] for v in valid_bb]
    bh = [v[1] for v in valid_bb]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Bounding Box Size Distribution (Train)", fontsize=13, fontweight="bold")
    for ax, vals, label, color in zip(
        axes, [bw, bh], ["Box Width (px)", "Box Height (px)"], ["#F7DC6F","#82E0AA"]
    ):
        ax.hist(vals, bins=40, color=color, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--",
                   label=f"Mean={np.mean(vals):.0f}")
        ax.set_xlabel(label); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_bbox_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[04] BBox distribution saved.  Samples with bbox: {len(valid_bb)}/{len(train_data)}")

plot_bbox_stats()


# ── Cell 7: EDA – Trimap Mask Samples ───────────────────────────────────────
def plot_mask_samples(n=6):
    img_dir  = DATA_DIR / "train" / "images"
    mask_dir = DATA_DIR / "train" / "masks"
    samples  = [e for e in train_data
                if (mask_dir / f"{e['filename']}.png").exists()][:n]
    if not samples:
        print("[05] No masks found, skipping."); return

    fig, axes = plt.subplots(n, 2, figsize=(6, n*2.5))
    fig.suptitle("Trimap Segmentation Masks\n(1=Foreground  2=Background  3=Uncertain)",
                 fontsize=10, fontweight="bold")
    for i, e in enumerate(samples):
        img  = np.array(Image.open(img_dir  / f"{e['filename']}.jpg").resize((192,192)))
        mask = np.array(Image.open(mask_dir / f"{e['filename']}.png").resize((192,192)))
        axes[i,0].imshow(img);  axes[i,0].axis("off")
        axes[i,0].set_title(BREED_NAMES[e["breed_label"]].replace("_"," "), fontsize=8)
        axes[i,1].imshow(mask, cmap="tab10", vmin=1, vmax=3)
        axes[i,1].axis("off"); axes[i,1].set_title("Trimap", fontsize=8)

    patches = [mpatches.Patch(color=plt.cm.tab10(v), label=l)
               for v, l in [(0.1,"Foreground (1)"),(0.2,"Background (2)"),(0.3,"Uncertain (3)")]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_trimap_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[05] Trimap samples saved.")

plot_mask_samples()

print(f"\n[Part 1 DONE] EDA plots saved to: {PLOTS_DIR.resolve()}")
