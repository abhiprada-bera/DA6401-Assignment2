"""
Train and save three model checkpoint files:
  saved_models/classifier.pth   — VGG11BN classification head
  saved_models/localizer.pth    — VGG11BN + bbox regression head
  saved_models/unet.pth         — VGG11 encoder + U-Net decoder (segmentation)

Each file contains:
  {
    'epoch': N,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'val_loss': float,
    'val_metric': float,   # accuracy / mean-IoU / dice
    'config': {...}
  }
"""
import os, sys, csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, ".")
from models import VGG11BN, CustomDropout, CustomIoULoss, MultiTaskPerceptionModel

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR    = "saved_models"
EPOCHS      = 5
LR          = 1e-3
BATCH       = 16
NUM_CLASSES = 37

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}  |  Save dir: {SAVE_DIR}/")

# ──────────────────────────────────────────────────────────────────────────────
# Shared transforms
# ──────────────────────────────────────────────────────────────────────────────
norm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────
class ClsDS(Dataset):
    """Classification dataset (image, breed_label)."""
    def __init__(self, split, n=None):
        self.rows = []
        with open(f"{DATA_DIR}/{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                p = f"{DATA_DIR}/{split}/images/{r['filename']}.jpg"
                if os.path.exists(p):
                    self.rows.append((p, int(r["breed_label"])))
        if n: self.rows = self.rows[:n]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, lbl = self.rows[i]
        return norm(Image.open(p).convert("RGB")), lbl

class LocDS(Dataset):
    """Localisation dataset (image, bbox [x1,y1,x2,y2] normalised)."""
    def __init__(self, split, n=None):
        self.rows = []
        with open(f"{DATA_DIR}/{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                p = f"{DATA_DIR}/{split}/images/{r['filename']}.jpg"
                if os.path.exists(p) and int(r["xmin"]) != -1:
                    self.rows.append((p, r))
        if n: self.rows = self.rows[:n]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, r = self.rows[i]
        img = Image.open(p).convert("RGB")
        ow, oh = img.size
        bbox = torch.tensor([int(r["xmin"])/ow, int(r["ymin"])/oh,
                              int(r["xmax"])/ow, int(r["ymax"])/oh], dtype=torch.float32)
        return norm(img), bbox

class SegDS(Dataset):
    """Segmentation dataset (image, 3-class mask)."""
    def __init__(self, split, n=None):
        self.rows = []
        self.split = split
        with open(f"{DATA_DIR}/{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                mp = f"{DATA_DIR}/{split}/masks/{r['filename']}.png"
                ip = f"{DATA_DIR}/{split}/images/{r['filename']}.jpg"
                if os.path.exists(mp) and os.path.exists(ip):
                    self.rows.append(r)
        if n: self.rows = self.rows[:n]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r  = self.rows[i]
        nm = r["filename"]
        img  = norm(Image.open(f"{DATA_DIR}/{self.split}/images/{nm}.jpg").convert("RGB"))
        mask = Image.open(f"{DATA_DIR}/{self.split}/masks/{nm}.png")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask_t = torch.tensor((np.array(mask)-1).clip(0,2), dtype=torch.long)
        return img, mask_t

# ──────────────────────────────────────────────────────────────────────────────
# U-Net Segmenter using VGG11BN backbone
# ──────────────────────────────────────────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c+skip_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.shape[2]-x.shape[2]; dx = skip.shape[3]-x.shape[3]
        x  = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x, skip], 1))

class UNet(nn.Module):
    def __init__(self, num_seg_classes=3):
        super().__init__()
        bb   = VGG11BN(num_classes=37)
        feats = list(bb.features.children())
        self.enc1 = nn.Sequential(*feats[0:2])   # 64  ch, /2
        self.enc2 = nn.Sequential(*feats[2:4])   # 128 ch, /4
        self.enc3 = nn.Sequential(*feats[4:7])   # 256 ch, /8
        self.enc4 = nn.Sequential(*feats[7:10])  # 512 ch, /16
        self.enc5 = nn.Sequential(*feats[10:])   # 512 ch, /32
        self.up4  = UpBlock(512, 512, 256)
        self.up3  = UpBlock(256, 256, 128)
        self.up2  = UpBlock(128, 128, 64)
        self.up1  = UpBlock(64,  64,  32)
        self.head = nn.Conv2d(32, num_seg_classes, 1)
    def forward(self, x):
        s1 = self.enc1(x);  s2 = self.enc2(s1)
        s3 = self.enc3(s2); s4 = self.enc4(s3); s5 = self.enc5(s4)
        d  = self.up4(s5, s4); d = self.up3(d, s3)
        d  = self.up2(d,  s2); d = self.up1(d, s1)
        return F.interpolate(self.head(d), (IMG_SIZE, IMG_SIZE),
                             mode="bilinear", align_corners=False)

# ──────────────────────────────────────────────────────────────────────────────
# Generic training loop
# ──────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, loss_fn, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            X = batch[0].to(DEVICE)
            target = batch[1].to(DEVICE)
            pred = model(X)
            loss = loss_fn(pred, target)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)

# ──────────────────────────────────────────────────────────────────────────────
# 1. classifier.pth  — VGG11BN classification model
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[1/3]  Training CLASSIFIER  ->  saved_models/classifier.pth")
print("="*60)

tr_cls = DataLoader(ClsDS("train", 256), BATCH, shuffle=True,  num_workers=0)
va_cls = DataLoader(ClsDS("val",    64), BATCH, shuffle=False, num_workers=0)

classifier = VGG11BN(num_classes=NUM_CLASSES, drop_p=0.5).to(DEVICE)
opt_cls    = optim.Adam(classifier.parameters(), lr=LR, weight_decay=1e-4)
sch_cls    = optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=EPOCHS)
ce         = nn.CrossEntropyLoss()

best_val_cls = float("inf")
best_acc     = 0.0

for ep in range(1, EPOCHS+1):
    tr_loss = run_epoch(classifier, tr_cls, opt_cls, ce, train=True)
    va_loss = run_epoch(classifier, va_cls, opt_cls, ce, train=False)
    sch_cls.step()
    # Accuracy
    classifier.eval(); correct = total = 0
    with torch.no_grad():
        for X, y in va_cls:
            preds = classifier(X.to(DEVICE)).argmax(1).cpu()
            correct += (preds == y).sum().item(); total += len(y)
    acc = correct / total
    print(f"  Epoch {ep}/{EPOCHS}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  acc={acc:.4f}")
    if va_loss < best_val_cls:
        best_val_cls = va_loss; best_acc = acc
        torch.save({
            "epoch": ep,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": opt_cls.state_dict(),
            "val_loss": va_loss,
            "val_metric": acc,      # top-1 accuracy
            "config": {
                "model": "VGG11BN",
                "num_classes": NUM_CLASSES,
                "drop_p": 0.5,
                "img_size": IMG_SIZE,
                "task": "classification"
            }
        }, f"{SAVE_DIR}/classifier.pth")
        print(f"  -> Saved classifier.pth (val_loss={va_loss:.4f}, acc={acc:.4f})")

print(f"  Best: val_loss={best_val_cls:.4f}  accuracy={best_acc:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. localizer.pth  — VGG11BN + bbox regression head
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[2/3]  Training LOCALIZER  ->  saved_models/localizer.pth")
print("="*60)

class VGG11Localizer(nn.Module):
    """VGG11BN backbone with a 4-output regression head for bounding boxes."""
    def __init__(self):
        super().__init__()
        backbone    = VGG11BN(num_classes=NUM_CLASSES, drop_p=0.5)
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool
        # Shared feature extractor
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256),    nn.ReLU(),
            nn.Linear(256, 4),       nn.Sigmoid()  # normalised [0,1]
        )
    def forward(self, x):
        f = self.avgpool(self.features(x))
        return self.bbox_head(f)

tr_loc = DataLoader(LocDS("train", 256), BATCH, shuffle=True,  num_workers=0)
va_loc = DataLoader(LocDS("val",    64), BATCH, shuffle=False, num_workers=0)

localizer = VGG11Localizer().to(DEVICE)
opt_loc   = optim.Adam(localizer.parameters(), lr=LR, weight_decay=1e-4)
sch_loc   = optim.lr_scheduler.CosineAnnealingLR(opt_loc, T_max=EPOCHS)
iou_loss  = CustomIoULoss()

best_val_loc = float("inf"); best_iou = 0.0

for ep in range(1, EPOCHS+1):
    localizer.train(); tl = 0
    for X, bbox in tr_loc:
        X, bbox = X.to(DEVICE), bbox.to(DEVICE)
        opt_loc.zero_grad()
        pred = localizer(X)
        loss = iou_loss(pred, bbox)
        loss.backward(); opt_loc.step()
        tl += loss.item()

    localizer.eval(); vl = miou = 0
    with torch.no_grad():
        for X, bbox in va_loc:
            X, bbox = X.to(DEVICE), bbox.to(DEVICE)
            pred = localizer(X)
            vl   += iou_loss(pred, bbox).item()
            miou += (1 - iou_loss(pred, bbox)).clamp(0).item()

    tr_loss = tl  / len(tr_loc)
    va_loss = vl  / len(va_loc)
    mean_iou= miou/ len(va_loc)
    sch_loc.step()
    print(f"  Epoch {ep}/{EPOCHS}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  mean_iou={mean_iou:.4f}")

    if va_loss < best_val_loc:
        best_val_loc = va_loss; best_iou = mean_iou
        torch.save({
            "epoch": ep,
            "model_state_dict": localizer.state_dict(),
            "optimizer_state_dict": opt_loc.state_dict(),
            "val_loss": va_loss,
            "val_metric": mean_iou,   # mean IoU
            "config": {
                "model": "VGG11BN_Localizer",
                "img_size": IMG_SIZE,
                "task": "bbox_regression",
                "loss": "CustomIoULoss"
            }
        }, f"{SAVE_DIR}/localizer.pth")
        print(f"  -> Saved localizer.pth (val_loss={va_loss:.4f}, mean_iou={mean_iou:.4f})")

print(f"  Best: val_loss={best_val_loc:.4f}  mean_iou={best_iou:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. unet.pth  — UNet segmentation model
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[3/3]  Training UNET  ->  saved_models/unet.pth")
print("="*60)

tr_seg = DataLoader(SegDS("train", 128), BATCH, shuffle=True,  num_workers=0)
va_seg = DataLoader(SegDS("val",    64), BATCH, shuffle=False, num_workers=0)

unet      = UNet(num_seg_classes=3).to(DEVICE)
opt_seg   = optim.Adam(unet.parameters(), lr=5e-4, weight_decay=1e-4)
sch_seg   = optim.lr_scheduler.CosineAnnealingLR(opt_seg, T_max=EPOCHS)
ce_seg    = nn.CrossEntropyLoss()

def dice_score(logits, masks, n=3):
    p = logits.argmax(1); scores = []
    for c in range(n):
        inter = ((p==c) & (masks==c)).float().sum()
        union = (p==c).float().sum() + (masks==c).float().sum()
        if union > 0: scores.append((2*inter+1e-6)/(union+1e-6))
    return float(torch.stack(scores).mean()) if scores else 0.0

best_val_seg = float("inf"); best_dice = 0.0

for ep in range(1, EPOCHS+1):
    unet.train(); tl = 0
    for X, mask in tr_seg:
        X, mask = X.to(DEVICE), mask.to(DEVICE)
        opt_seg.zero_grad()
        loss = ce_seg(unet(X), mask)
        loss.backward(); opt_seg.step()
        tl += loss.item()

    unet.eval(); vl = di = 0
    with torch.no_grad():
        for X, mask in va_seg:
            X, mask = X.to(DEVICE), mask.to(DEVICE)
            out  = unet(X)
            vl  += ce_seg(out, mask).item()
            di  += dice_score(out, mask)

    tr_loss  = tl / len(tr_seg)
    va_loss  = vl / len(va_seg)
    mean_dice= di / len(va_seg)
    sch_seg.step()
    print(f"  Epoch {ep}/{EPOCHS}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  dice={mean_dice:.4f}")

    if va_loss < best_val_seg:
        best_val_seg = va_loss; best_dice = mean_dice
        torch.save({
            "epoch": ep,
            "model_state_dict": unet.state_dict(),
            "optimizer_state_dict": opt_seg.state_dict(),
            "val_loss": va_loss,
            "val_metric": mean_dice,   # Dice Score
            "config": {
                "model": "VGG11BN_UNet",
                "num_seg_classes": 3,
                "img_size": IMG_SIZE,
                "task": "semantic_segmentation",
                "classes": ["background", "foreground", "boundary"]
            }
        }, f"{SAVE_DIR}/unet.pth")
        print(f"  -> Saved unet.pth (val_loss={va_loss:.4f}, dice={mean_dice:.4f})")

print(f"  Best: val_loss={best_val_seg:.4f}  dice={best_dice:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CHECKPOINT FILES SAVED")
print("="*60)
for fn in ["classifier.pth", "localizer.pth", "unet.pth"]:
    fp = os.path.join(SAVE_DIR, fn)
    if os.path.exists(fp):
        ckpt = torch.load(fp, map_location="cpu", weights_only=False)
        size_mb = os.path.getsize(fp) / 1e6
        print(f"  {fn:<20}  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}"
              f"  val_metric={ckpt['val_metric']:.4f}  size={size_mb:.1f} MB")
    else:
        print(f"  {fn:<20}  NOT FOUND")
print(f"\nAll files are in:  {os.path.abspath(SAVE_DIR)}/")
