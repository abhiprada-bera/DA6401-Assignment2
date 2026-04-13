# =============================================================================
# DA Assignment 2 - Part 3: Training, Evaluation & Visualizations (Task 1)
# =============================================================================
import time, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, precision_recall_fscore_support
warnings.filterwarnings("ignore")

# Import shared definitions
from models import (VGG11BN, PetDataset, get_loaders, DEVICE, NUM_CLASSES,
                    BREED_NAMES, MODEL_DIR, PLOTS_DIR, DATA_DIR)

PLOTS_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)

# ── Loaders -------------------------------------------------------------------
train_loader, val_loader, test_loader = get_loaders(batch_size=32)
print(f"Train:{len(train_loader.dataset)}  Val:{len(val_loader.dataset)}  Test:{len(test_loader.dataset)}")

# ── Model + Optimizer ---------------------------------------------------------
"""
JUSTIFICATION – Optimizer and Scheduler:
  Adam(lr=3e-4, wd=1e-4): converges faster than SGD for deep nets from scratch.
  L2 weight decay (1e-4) penalises large weights alongside CustomDropout.
  CosineAnnealingLR: smoothly decays LR to near-zero, helping the model settle
  into sharp minima rather than oscillating around them.
  Label smoothing (0.1): prevents overconfident predictions on the small dataset.
  Gradient clipping (max_norm=2): stabilises early training instability.
"""
model = VGG11BN(num_classes=NUM_CLASSES, drop_p=0.5).to(DEVICE)
EPOCHS   = 25; LR = 3e-4; WD = 1e-4; PATIENCE = 7

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

print(f"Optimizer: Adam(lr={LR}, wd={WD})  |  Scheduler: CosineAnnealingLR  |  Epochs: {EPOCHS}")

# ── Training Loop -------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss = correct = total = 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        tot_loss += loss.item() * imgs.size(0)
        correct  += (model(imgs).argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return tot_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss = correct = total = 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        tot_loss += criterion(out, labels).item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return tot_loss / total, correct / total

history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "lr":[]}
best_val_loss = float("inf"); patience_ctr = 0

print(f"\n{'Ep':>4} {'TrLoss':>8} {'VaLoss':>8} {'TrAcc':>7} {'VaAcc':>7} {'LR':>9}")
print("-"*50)
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_l, tr_a = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    va_l, va_a = evaluate(model, val_loader, criterion, DEVICE)
    cur_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    history["train_loss"].append(tr_l); history["val_loss"].append(va_l)
    history["train_acc"].append(tr_a);  history["val_acc"].append(va_a)
    history["lr"].append(cur_lr)
    print(f"{epoch:>4} {tr_l:>8.4f} {va_l:>8.4f} {tr_a*100:>6.1f}% "
          f"{va_a*100:>6.1f}% {cur_lr:>9.2e}  [{time.time()-t0:.0f}s]")
    if va_l < best_val_loss:
        best_val_loss = va_l; patience_ctr = 0
        torch.save({"epoch":epoch,"model_state":model.state_dict(),
                    "val_loss":va_l,"val_acc":va_a}, MODEL_DIR/"vgg11_best.pth")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n[Early stopping] patience={PATIENCE} reached at epoch {epoch}.")
            break

ckpt = torch.load(MODEL_DIR/"vgg11_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
print(f"\nBest checkpoint: epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']*100:.2f}%")

# ── Training Curves -----------------------------------------------------------
def plot_training_curves(history):
    ep = range(1, len(history["train_loss"])+1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("VGG11-BN Training Curves – Oxford Pet Classification",
                 fontsize=13, fontweight="bold")
    axes[0].plot(ep, history["train_loss"], "o-", label="Train", color="#E74C3C")
    axes[0].plot(ep, history["val_loss"],   "s-", label="Val",   color="#3498DB")
    axes[0].set_title("Cross-Entropy Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, [a*100 for a in history["train_acc"]], "o-",
                 label="Train", color="#27AE60")
    axes[1].plot(ep, [a*100 for a in history["val_acc"]], "s-",
                 label="Val",   color="#F39C12")
    axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(ep, history["lr"], "^-", color="#8E44AD")
    axes[2].set_title("LR Schedule (CosineAnnealing)")
    axes[2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"08_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[08] Training curves saved.")

plot_training_curves(history)

# ── Test Evaluation -----------------------------------------------------------
@torch.no_grad()
def get_preds(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    for imgs, labels, _ in loader:
        all_p.extend(model(imgs.to(device)).argmax(1).cpu().tolist())
        all_l.extend(labels.tolist())
    return np.array(all_l), np.array(all_p)

y_true, y_pred = get_preds(model, test_loader, DEVICE)
test_acc = (y_true == y_pred).mean()
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=[n.replace("_"," ") for n in BREED_NAMES], zero_division=0))

# ── Confusion Matrix ----------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap="YlOrRd",
                xticklabels=[n.replace("_"," ") for n in BREED_NAMES],
                yticklabels=[n.replace("_"," ") for n in BREED_NAMES],
                linewidths=0.3, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix – Test Set  (Acc={test_acc*100:.1f}%)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7); plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"09_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[09] Confusion matrix saved.")

plot_confusion_matrix(y_true, y_pred)

# ── Per-Class F1 Chart --------------------------------------------------------
def plot_per_class_f1(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.suptitle("Per-Class Metrics on Test Set", fontsize=14, fontweight="bold")
    for ax, (name, vals, color) in zip(axes, [
        ("Precision", p, "#3498DB"), ("Recall", r, "#27AE60"), ("F1-Score", f1, "#E74C3C")
    ]):
        ax.bar(range(NUM_CLASSES), vals, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(vals.mean(), color="navy", linestyle="--",
                   label=f"Mean={vals.mean():.3f}")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels([n.replace("_"," ") for n in BREED_NAMES],
                           rotation=90, fontsize=6.5)
        ax.set_ylabel(name); ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_title(name)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"10_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[10] Per-class F1 chart saved.")

plot_per_class_f1(y_true, y_pred)

# ── Prediction Samples --------------------------------------------------------
def plot_predictions(model, loader, device, n=8):
    model.eval()
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    correct_s, wrong_s = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            for img, lbl, pred in zip(imgs, labels, preds):
                show = (img*std+mean).permute(1,2,0).clamp(0,1).numpy()
                if lbl==pred and len(correct_s)<n: correct_s.append((show,lbl.item(),pred.item()))
                elif lbl!=pred and len(wrong_s)<n:  wrong_s.append((show,lbl.item(),pred.item()))
            if len(correct_s)>=n and len(wrong_s)>=n: break

    fig, axes = plt.subplots(2, n, figsize=(n*2.5, 6))
    fig.suptitle("Correct (top) vs Wrong (bottom) Predictions", fontsize=12, fontweight="bold")
    for ax,(img,lbl,pred) in zip(axes[0], correct_s):
        ax.imshow(img); ax.axis("off")
        ax.set_title(BREED_NAMES[lbl].replace("_"," ")[:12], fontsize=6, color="green")
    for ax,(img,lbl,pred) in zip(axes[1], wrong_s):
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"T:{BREED_NAMES[lbl].replace('_',' ')[:8]}\nP:{BREED_NAMES[pred].replace('_',' ')[:8]}",
                     fontsize=5.5, color="red")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"11_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[11] Prediction samples saved.")

plot_predictions(model, test_loader, DEVICE)

print(f"\n[Part 3 DONE] Test Acc={test_acc*100:.2f}%  |  Plots saved to {PLOTS_DIR.resolve()}")
