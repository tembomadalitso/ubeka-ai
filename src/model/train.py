"""
train.py — Ubeka AI (Rebuilt)
================================
What was wrong before:
  1. Val set was built from same augmented pool as train — inflated accuracy
  2. Training transforms were too gentle — model never saw phone-camera conditions
  3. FocalLoss gamma=3.0 caused collapse with tiny severe class
  4. MixUp destabilised training when severe had <100 samples

What is fixed here:
  1. Val set is strictly clean (no augmentation, no oversampling)
  2. OneCycleLR — proven to find better minima in fewer epochs than cosine
  3. Temperature scaling Phase 3 for confidence calibration
  4. Per-class accuracy printed every epoch — catch class collapse early
  5. Conservative training: gamma=2.0, no MixUp, validated hyperparams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parents[1]
DATA_ROOT    = PROJECT_ROOT / "data"
sys.path.append(str(current_dir.parent))

print(f"Project root : {PROJECT_ROOT}")
print(f"Data root    : {DATA_ROOT}  exists={DATA_ROOT.exists()}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROCESSED_TRAIN = DATA_ROOT / "processed" / "train"
PROCESSED_VAL   = DATA_ROOT / "processed" / "val"
SAVE_PATH       = PROJECT_ROOT / "models" / "acne_efficientnet.pth"
CALIBRATED_PATH = PROJECT_ROOT / "models" / "acne_efficientnet_calibrated.pth"

BATCH_SIZE      = 16
EPOCHS          = 30       # OneCycleLR handles scheduling — no need to cap low
LR_MAX          = 1e-3     # OneCycleLR peak LR (higher than AdamW default, finds better minima)
LR_DIV          = 25       # initial LR = LR_MAX / LR_DIV
LR_FINAL_DIV    = 1e4      # final LR = LR_MAX / LR_FINAL_DIV
NUM_CLASSES     = 4
PATIENCE        = 8
LABEL_SMOOTHING = 0.05
SEED            = 42
SEVERE_BOOST    = 6.0      # higher than before — severe only has ~80 train images
FOCAL_GAMMA     = 2.0      # validated stable value
PCT_START       = 0.25     # OneCycleLR warmup fraction

SEVERITY_LABELS = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
# Train: aggressive — simulate real-world phone photos
# The dataset_builder already applies augmentation at save time.
# These are ADDITIONAL transforms at load time for further variety.
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Val: CLEAN — exactly what inference sees
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dark skin: extra brightness variety
dark_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(18),
    transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.35, hue=0.08),
    transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_dark_skin_path(p: Path) -> bool:
    if "_ita" not in p.stem: return False
    try:
        ita = float(p.stem.split("_ita")[-1].split("_")[0])
        return ita <= -30
    except ValueError: return False

# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n   = inputs.size(1)
        with torch.no_grad():
            smooth = torch.full_like(inputs, self.label_smoothing / (n-1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_p  = F.log_softmax(inputs, dim=1)
        p_t    = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha[targets] if self.alpha is not None else \
                  torch.ones(targets.size(0), device=inputs.device)
        focal  = (1.0 - p_t) ** self.gamma
        ce     = -(smooth * log_p).sum(dim=1)
        return (alpha_t * focal * ce).mean()

# ─────────────────────────────────────────────
# TEMPERATURE SCALING
# ─────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x): return self.model(x) / self.temperature

    def calibrate(self, val_loader):
        self.model.eval()
        crit  = nn.CrossEntropyLoss()
        optim = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        logits_all, labels_all = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                logits_all.append(self.model(imgs.to(device)))
                labels_all.append(lbls.to(device))
        L = torch.cat(logits_all); Y = torch.cat(labels_all)
        def eval_fn():
            optim.zero_grad()
            loss = crit(L / self.temperature, Y)
            loss.backward(); return loss
        optim.step(eval_fn)
        t = float(self.temperature.item())
        print(f"  Temperature T = {t:.4f}  ({'overconfident → softened' if t>1.1 else 'underconfident → sharpened' if t<0.9 else 'well calibrated'})")
        return t

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class AcneDataset(Dataset):
    CLASS_MAP = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3}

    def __init__(self, data_dir: Path, transform=None, use_dark_transform=False):
        self.transform          = transform
        self.use_dark_transform = use_dark_transform
        self.samples            = []

        for cls_name, label in self.CLASS_MAP.items():
            folder = data_dir / cls_name
            if not folder.exists(): continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((p, label))

        print(f"\n📦 {data_dir.name}: {len(self.samples)} images")
        total = max(1, len(self.samples))
        for i in range(NUM_CLASSES):
            n   = sum(1 for _, y in self.samples if y == i)
            bar = "█" * int(n / total * 50)
            print(f"   {SEVERITY_LABELS[i]:>10}: {n:>5}  {bar}")

        dark = sum(1 for p, _ in self.samples if is_dark_skin_path(p))
        print(f"   Dark skin: {dark} ({dark/total*100:.1f}%)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.array(Image.open(path).convert("RGB"))
        if self.use_dark_transform and is_dark_skin_path(path):
            return dark_transforms(arr).float(), label
        elif self.transform:
            return self.transform(arr).float(), label
        return torch.tensor(arr).permute(2,0,1).float()/255.0, label

# ─────────────────────────────────────────────
# MODEL — EfficientNet-B0 with improved head
# Architecture deliberately unchanged from validated v2
# Switching models is NOT the bottleneck — data is
# ─────────────────────────────────────────────
class AcneModel(nn.Module):
    """
    EfficientNet-B0 + improved head.
    Head: Dropout(0.4) → Linear(1280→256) → LayerNorm(256) → GELU → Dropout(0.2) → Linear(256→4)
    All inference files (predict.py, fairness_eval.py) must match this exactly.
    """
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = self.base.classifier[1].in_features  # 1280
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def freeze_base(self):
        for p in self.base.features.parameters(): p.requires_grad = False

    def unfreeze_base(self):
        for p in self.base.features.parameters(): p.requires_grad = True
        print("  🔓 All layers unfrozen")

    def forward(self, x): return self.base(x)

# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────
def build_loaders():
    if not PROCESSED_TRAIN.exists() or not PROCESSED_VAL.exists():
        raise FileNotFoundError(
            f"Processed data not found.\n"
            f"  Run: python src/preprocessing/dataset_builder.py\n"
            f"  Looked in: {PROCESSED_TRAIN}"
        )

    train_ds = AcneDataset(PROCESSED_TRAIN, transform=train_transforms,
                           use_dark_transform=True)
    val_ds   = AcneDataset(PROCESSED_VAL,   transform=val_transforms,
                           use_dark_transform=False)

    train_labels = [lbl for _, lbl in train_ds.samples]
    counts       = [train_labels.count(i) for i in range(NUM_CLASSES)]
    weights      = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights[3]  *= SEVERE_BOOST

    print(f"\n  Class weights (severe_boost={SEVERE_BOOST}x):")
    for i, (w, n) in enumerate(zip(weights, counts)):
        print(f"    {SEVERITY_LABELS[i]:>10}: w={w:.5f}  n={n}")

    sw      = [weights[l].item() for l in train_labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, weights

# ─────────────────────────────────────────────
# EPOCH RUNNER
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scheduler,
              scaler, is_train, desc):
    model.train() if is_train else model.eval()
    total_loss = 0.0; all_preds = []; all_labels = []

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        out  = model(imgs); loss = criterion(out, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    out  = model(imgs); loss = criterion(out, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler: scheduler.step()
                total_loss += loss.item()
            else:
                out = model(imgs)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = sum(p==l for p,l in zip(all_preds,all_labels)) / max(1,len(all_labels))
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, acc, all_preds, all_labels

# ─────────────────────────────────────────────
# PER-CLASS ACCURACY (catch collapse early)
# ─────────────────────────────────────────────
def per_class_acc(preds, labels):
    lines = []
    for i, cls in SEVERITY_LABELS.items():
        idxs    = [j for j,l in enumerate(labels) if l==i]
        correct = sum(1 for j in idxs if preds[j]==i)
        total   = len(idxs)
        pct     = correct/max(1,total)
        bar     = "█"*int(pct*15) + "░"*(15-int(pct*15))
        lines.append(f"    {cls:>10}: {correct:>3}/{total:<3}  {pct:.0%}  {bar}")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train():
    train_loader, val_loader, class_weights = build_loaders()
    criterion = FocalLoss(alpha=class_weights.to(device),
                          gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    # ── Phase 1: warmup head only (3 epochs) ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Head warmup (3 epochs, base frozen)")
    print(f"{'='*60}\n")

    model = AcneModel().to(device)
    model.freeze_base()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    # Short OneCycleLR for 3-epoch warmup
    scheduler_p1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=len(train_loader), epochs=3,
        pct_start=0.3, div_factor=10, final_div_factor=100
    )
    scaler   = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    best_acc = 0.0
    history  = {"train_loss": [], "val_acc": [], "val_acc_per_class": []}

    for epoch in range(3):
        tr_loss, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scheduler_p1,
            scaler, is_train=True, desc=f"P1 {epoch+1}/3"
        )
        _, val_acc, vp, vl = run_epoch(
            model, val_loader, criterion, None, None,
            scaler, is_train=False, desc="Val"
        )
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        print(f"  P1 [{epoch+1}/3]  loss={tr_loss:.4f}  val_acc={val_acc:.4f}")
        print(per_class_acc(vp, vl))
        if val_acc > best_acc:
            best_acc = val_acc
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"               💾 Saved ({val_acc:.4f})")

    # ── Phase 2: full fine-tune with OneCycleLR ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Full fine-tune ({EPOCHS} epochs, OneCycleLR)")
    print(f"  max_lr={LR_MAX}  pct_start={PCT_START}  patience={PATIENCE}")
    print(f"{'='*60}\n")

    model.unfreeze_base()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX/LR_DIV, weight_decay=1e-4)

    # OneCycleLR: warmup → peak → decay in a single sweep
    # Better than manual scheduling for small datasets
    scheduler_p2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_MAX,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=PCT_START,
        div_factor=LR_DIV,
        final_div_factor=LR_FINAL_DIV,
        anneal_strategy="cos"
    )

    no_improve   = 0
    final_preds  = final_labels = None

    for epoch in range(EPOCHS):
        tr_loss, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scheduler_p2,
            scaler, is_train=True, desc=f"P2 {epoch+1}/{EPOCHS}"
        )
        _, val_acc, ep_preds, ep_labels = run_epoch(
            model, val_loader, criterion, None, None,
            scaler, is_train=False, desc="Val"
        )
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)

        lr_now = scheduler_p2.get_last_lr()[0]
        print(f"  P2 [{epoch+1:>2}/{EPOCHS}]  loss={tr_loss:.4f}  "
              f"val_acc={val_acc:.4f}  lr={lr_now:.2e}  (best={best_acc:.4f})")
        print(per_class_acc(ep_preds, ep_labels))

        if val_acc > best_acc:
            best_acc     = val_acc
            no_improve   = 0
            final_preds  = ep_preds
            final_labels = ep_labels
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"               💾 Best! ({val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n⏹  Early stopping at P2 epoch {epoch+1}")
                break

    # ── Phase 3: temperature calibration ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — Temperature Scaling")
    print(f"{'='*60}\n")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    ts   = TemperatureScaler(model).to(device)
    temp = ts.calibrate(val_loader)
    torch.save({"model_state_dict": model.state_dict(), "temperature": temp},
               CALIBRATED_PATH)
    print(f"  💾 Calibrated → {CALIBRATED_PATH}")

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE  |  Best val acc: {best_acc:.4f}  |  T: {temp:.4f}")
    print(f"{'='*60}\n")

    if final_labels:
        print("Classification Report:")
        print(classification_report(final_labels, final_preds,
                                    target_names=list(SEVERITY_LABELS.values()),
                                    digits=4))

        # Confusion matrix
        cm = confusion_matrix(final_labels, final_preds)
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=SEVERITY_LABELS.values(),
                    yticklabels=SEVERITY_LABELS.values())
        plt.title(f"Confusion Matrix — Val Acc: {best_acc:.4f}  T={temp:.3f}")
        plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
        cm_path = SAVE_PATH.parent / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150); plt.close()
        print(f"📊 Confusion matrix → {cm_path}")

    # Training curves
    n  = len(history["train_loss"])
    xs = range(1, n+1)
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(xs, history["train_loss"], "o-", ms=3, color="#1565C0", lw=1.5)
    axes[0].axvline(3.5, color="#E64A19", ls="--", alpha=0.6, label="P1→P2")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(xs, history["val_acc"], "o-", ms=3, color="#2E7D32", lw=1.5)
    axes[1].axhline(best_acc, color="#C62828", ls="--", label=f"Best: {best_acc:.4f}")
    axes[1].axvline(3.5, color="#E64A19", ls="--", alpha=0.6, label="P1→P2")
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    curves_path = SAVE_PATH.parent / "training_curves.png"
    plt.savefig(curves_path, dpi=150); plt.close()
    print(f"📈 Training curves → {curves_path}")

    summary = {
        "best_val_accuracy": round(best_acc, 4),
        "temperature": round(temp, 4),
        "epochs_trained": n,
        "scheduler": "OneCycleLR",
        "focal_gamma": FOCAL_GAMMA,
        "severe_boost": SEVERE_BOOST,
    }
    with open(SAVE_PATH.parent / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Next: python src/model/evaluate.py")


if __name__ == "__main__":
    train()