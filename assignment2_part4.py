# =============================================================================
# DA Assignment 2 - Part 4: Object Detection (BBox) & Semantic Segmentation
# =============================================================================

import csv, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
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

# ── 2.1 Detection Dataset ─────────────────────────────────────────────────────
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


# ── 2.2 Detection Head on VGG11 Features ─────────────────────────────────────
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
from models import VGG11BN, CustomDropout  # noqa
classifier = VGG11BN(num_classes=37).to(DEVICE)
ckpt = torch.load(MODEL_DIR / "vgg11_best.pth", map_location=DEVICE)
classifier.load_state_dict(ckpt["model_state"])

detector = VGG11Detector(classifier).to(DEVICE)
det_opt  = optim.Adam(filter(lambda p: p.requires_grad, detector.parameters()),
                      lr=1e-4, weight_decay=1e-4)
smooth_l1 = nn.SmoothL1Loss()


# ── 2.3 IoU Metric ───────────────────────────────────────────────────────────
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


# ── 2.4 Training – Detection Head ────────────────────────────────────────────
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

# ── 2.5 Detection Curves + BBox Visualisation ────────────────────────────────
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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close()

plot_segmentation_results()

print("\n[Part 4 complete] Detection + Segmentation done. All plots saved.")
