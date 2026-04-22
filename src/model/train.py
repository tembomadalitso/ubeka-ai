"""
train.py — Ubeka AI FINAL
===========================
Strategy: Keep EXACTLY what achieved 95.77% (v2 core), add only
temperature scaling (Phase 3) which is risk-free post-training calibration.

Dataset from build_dataset.py (confirmed paths):
  PROJECT_ROOT / data / processed / train/val
  Train: clear=833  mild=1050  moderate=305  severe=97  + dark augmentation
  Val:   clear=204  mild=221   moderate=45   severe=24

Key design decisions (from notebook analysis):
  - FocalLoss gamma=2.0: VALIDATED in v2, stable on CPU + small dataset
  - No MixUp: caused collapse when severe class too small
  - No differential LR: unnecessary complexity, single LR works
  - Severe boost=5.0: raised from 4.0 — severe still underrepresented
  - Epochs 25 + patience=8: more time than v2's 20/6
  - Temperature scaling Phase 3: safe, closes skin tone confidence gap
  - Dark skin transform: applied automatically via ITA in filename
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
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

# ── Paths — confirmed correct from build_dataset.py output ────────────────────
current_dir  = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parents[1]            # ubeka-ai/
DATA_ROOT    = PROJECT_ROOT / "data"             # ubeka-ai/data/  ← confirmed exists
sys.path.append(str(current_dir.parent))

from preprocessing.transforms import train_transforms, val_transforms, SEVERITY_LABELS

print(f"Project root  : {PROJECT_ROOT}")
print(f"Data root     : {DATA_ROOT}  exists={DATA_ROOT.exists()}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
USE_PROCESSED   = True
PROCESSED_TRAIN = DATA_ROOT / "processed" / "train"
PROCESSED_VAL   = DATA_ROOT / "processed" / "val"
RAW_DIR         = DATA_ROOT / "raw" / "acne" / "acne_1024"  # fallback only

SAVE_PATH       = PROJECT_ROOT / "models" / "acne_efficientnet.pth"
CALIBRATED_PATH = PROJECT_ROOT / "models" / "acne_efficientnet_calibrated.pth"

# ── Hyperparameters — validated config ────────────────────────────────────────
BATCH_SIZE      = 16
EPOCHS_FROZEN   = 5        # Phase 1: head only, fast convergence
EPOCHS_FULL     = 25       # Phase 2: full fine-tune
LR_FROZEN       = 3e-4     # head-only LR
LR_FULL         = 5e-5     # full fine-tune LR
NUM_CLASSES     = 4
VAL_SPLIT       = 0.2      # only used if raw fallback
PATIENCE        = 8        # early stopping patience
LABEL_SMOOTHING = 0.05
SEED            = 42
SEVERE_BOOST    = 5.0      # class weight multiplier for severe
FOCAL_GAMMA     = 2.0      # validated: stable on CPU + small dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# FOCAL LOSS — validated gamma=2.0
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2.0: validated stable value for CPU training on ~1000 images per class.
    alpha: per-class weights from WeightedRandomSampler + SEVERE_BOOST.
    label_smoothing: prevents extreme over-confidence on easy examples.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_cls = inputs.size(1)
        with torch.no_grad():
            smooth = torch.full_like(inputs, self.label_smoothing / (n_cls - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p = F.log_softmax(inputs, dim=1)
        p     = log_p.exp()

        alpha_t      = self.alpha[targets] if self.alpha is not None else \
                       torch.ones(targets.size(0), device=inputs.device)
        p_t          = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        ce           = -(smooth * log_p).sum(dim=1)
        return (alpha_t * focal_weight * ce).mean()


# ─────────────────────────────────────────────
# TEMPERATURE SCALING — safe post-training calibration
# ─────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Single learnable scalar T divides all logits before softmax.
    Trained on val set ONLY — model weights never change.
    T > 1 → softens overconfident predictions
    T < 1 → sharpens underconfident predictions
    Why this helps fairness: dark skin images tend to get lower confidence
    because the model has seen less of them. T calibration applies
    globally and partially corrects this systematic bias.
    """
    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        return self.model(x) / self.temperature

    def calibrate(self, val_loader):
        """Optimise T using LBFGS on NLL loss over full val set."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        # Collect all logits in one pass (efficient — no gradient through model)
        logits_all, labels_all = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                logits_all.append(self.model(imgs.to(device)))
                labels_all.append(lbls.to(device))
        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)

        def eval_fn():
            optimizer.zero_grad()
            loss = criterion(logits_all / self.temperature, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_fn)

        t = float(self.temperature.item())
        print(f"  Temperature T = {t:.4f}")
        if t > 1.1:
            print(f"  → Model overconfident — softened {t:.2f}x")
        elif t < 0.9:
            print(f"  → Model underconfident — sharpened {1/t:.2f}x")
        else:
            print(f"  → Well calibrated (T ≈ 1.0)")
        return t


# ─────────────────────────────────────────────
# DARK SKIN TRANSFORM
# ─────────────────────────────────────────────
dark_skin_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def is_dark_skin_path(path: Path) -> bool:
    """ITA <= -30 means dark skin (group3). Embedded in filename by build_dataset.py."""
    stem = path.stem
    if "_ita" not in stem:
        return False
    try:
        ita_val = float(stem.split("_ita")[-1].split("_")[0])
        return ita_val <= -30
    except ValueError:
        return False


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class AcneDataset(Dataset):
    """
    Reads folder-per-class structure:
      Processed: clear/ mild/ moderate/ severe/
      Raw:       acne0_1024/ acne1_1024/ acne2_1024/ acne3_1024/

    Dark skin images (ITA <= -30 in filename) automatically get
    the stronger dark_skin_transforms augmentation.
    """
    PROCESSED_MAP = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3}
    RAW_MAP       = {"acne0_1024": 0, "acne1_1024": 1,
                     "acne2_1024": 2, "acne3_1024": 3, "acne3_512_selection": 3}

    def __init__(self, data_dir: Path, transform=None, is_processed=True,
                 use_dark_transform=True):
        self.transform          = transform
        self.use_dark_transform = use_dark_transform
        self.samples            = []

        folder_map = self.PROCESSED_MAP if is_processed else self.RAW_MAP
        for fname, label in folder_map.items():
            folder = data_dir / fname
            if not folder.exists():
                continue
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
        print(f"   Dark skin images: {dark} ({dark/total*100:.1f}%)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        arr = np.array(Image.open(img_path).convert("RGB"))
        if self.use_dark_transform and is_dark_skin_path(img_path):
            return dark_skin_transforms(arr).float(), label
        elif self.transform:
            return self.transform(arr).float(), label
        return torch.tensor(arr).permute(2, 0, 1).float() / 255.0, label


class TransformSubset(Dataset):
    """Used only in raw fallback mode — wraps Subset with its own transform."""
    def __init__(self, subset, transform, use_dark_transform=True):
        self.subset             = subset
        self.transform          = transform
        self.use_dark_transform = use_dark_transform

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        arr = np.array(Image.open(img_path).convert("RGB"))
        if self.use_dark_transform and is_dark_skin_path(img_path):
            return dark_skin_transforms(arr).float(), label
        return self.transform(arr).float(), label


# ─────────────────────────────────────────────
# MODEL — EfficientNet-B0 with improved head
# ─────────────────────────────────────────────
class AcneModel(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet.
    Classifier head:
      Dropout(0.4) → Linear(1280→256) → LayerNorm(256) → GELU → Dropout(0.2) → Linear(256→4)

    LayerNorm: more stable than BatchNorm with small batches (BATCH_SIZE=16)
    GELU: smoother gradient flow than ReLU
    Dropout 0.4/0.2: prevents head overfitting on small dataset

    IMPORTANT: fairness_eval.py and predict.py must define identical architecture.
    """
    def __init__(self, freeze_base=False):
        super().__init__()
        self.base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        for p in self.base.features.parameters():
            p.requires_grad = not freeze_base

        in_features = self.base.classifier[1].in_features  # 1280
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def unfreeze(self):
        """Phase 2: unfreeze all EfficientNet layers for full fine-tuning."""
        for p in self.base.features.parameters():
            p.requires_grad = True
        print("  🔓 All EfficientNet layers unfrozen")

    def forward(self, x):
        return self.base(x)


# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────
def build_loaders():
    ok = USE_PROCESSED and PROCESSED_TRAIN.exists() and PROCESSED_VAL.exists()

    if ok:
        print("\n✅ Preprocessed dataset found")
        train_ds = AcneDataset(PROCESSED_TRAIN, transform=train_transforms,
                               is_processed=True, use_dark_transform=True)
        val_ds   = AcneDataset(PROCESSED_VAL,   transform=val_transforms,
                               is_processed=True, use_dark_transform=False)
        train_labels = [lbl for _, lbl in train_ds.samples]
    else:
        print(f"\n⚠️  Processed data NOT found at {PROCESSED_TRAIN}")
        print("   Run: python src/preprocessing/build_dataset.py")
        print("   Falling back to raw data...\n")
        base = AcneDataset(RAW_DIR, transform=None, is_processed=False)
        vs   = int(len(base) * VAL_SPLIT)
        tr_sub, va_sub = random_split(base, [len(base)-vs, vs],
                                      generator=torch.Generator().manual_seed(SEED))
        train_ds     = TransformSubset(tr_sub, train_transforms, use_dark_transform=True)
        val_ds       = TransformSubset(va_sub, val_transforms)
        train_labels = [base.samples[i][1] for i in tr_sub.indices]

    print(f"\n  ✂️  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    counts        = [train_labels.count(i) for i in range(NUM_CLASSES)]
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    class_weights[3] *= SEVERE_BOOST   # severe: ~4x fewer samples → boost

    print(f"\n  Effective class weights (severe_boost={SEVERE_BOOST}x):")
    max_n = max(counts)
    for i, (w, n) in enumerate(zip(class_weights, counts)):
        bar = "█" * int(n / max_n * 25)
        print(f"    {SEVERITY_LABELS[i]:>10}: w={w:.5f}  n={n}  {bar}")

    sw      = [class_weights[l].item() for l in train_labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, class_weights


# ─────────────────────────────────────────────
# EPOCH RUNNER
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, is_train, desc):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        out  = model(imgs)
                        loss = criterion(out, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out  = model(imgs)
                    loss = criterion(out, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item()
            else:
                out = model(imgs)

            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / max(1, len(all_labels))
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, acc, all_preds, all_labels


# ─────────────────────────────────────────────
# MAIN TRAIN
# ─────────────────────────────────────────────
def train():
    train_loader, val_loader, class_weights = build_loaders()
    criterion = FocalLoss(alpha=class_weights.to(device),
                          gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    # ══════════════════════════════════════════
    # PHASE 1 — Head only, frozen backbone
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Head-only training")
    print(f"  Epochs: {EPOCHS_FROZEN}  |  LR: {LR_FROZEN}  |  Backbone: FROZEN")
    print(f"  FocalLoss gamma={FOCAL_GAMMA}  |  LabelSmoothing={LABEL_SMOOTHING}")
    print(f"{'='*60}\n")

    model     = AcneModel(freeze_base=True).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FROZEN, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler   = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    best_acc = 0.0
    history  = {"train_loss": [], "val_acc": [], "phase": []}

    for epoch in range(EPOCHS_FROZEN):
        tr_loss, tr_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            is_train=True, desc=f"P1 Train {epoch+1}/{EPOCHS_FROZEN}"
        )
        _, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            is_train=False, desc=f"P1 Val"
        )
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        history["phase"].append(1)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  P1 [{epoch+1:>2}/{EPOCHS_FROZEN}]  "
              f"loss={tr_loss:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.2e}")
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"               💾 Saved  ({val_acc:.4f})")

    # ══════════════════════════════════════════
    # PHASE 2 — Full fine-tune, all layers
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Full fine-tune")
    print(f"  Epochs: up to {EPOCHS_FULL}  |  LR: {LR_FULL}  |  Patience: {PATIENCE}")
    print(f"  All EfficientNet layers + head train together")
    print(f"{'='*60}\n")

    model.unfreeze()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    # Cosine annealing: smooth LR decay avoids abrupt drops that can destabilise
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FULL, eta_min=1e-6
    )

    no_improve   = 0
    final_preds  = final_labels = None

    for epoch in range(EPOCHS_FULL):
        tr_loss, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            is_train=True, desc=f"P2 Train {epoch+1}/{EPOCHS_FULL}"
        )
        _, val_acc, ep_preds, ep_labels = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            is_train=False, desc=f"P2 Val"
        )
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        history["phase"].append(2)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  P2 [{epoch+1:>2}/{EPOCHS_FULL}]  "
              f"loss={tr_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={lr_now:.2e}  (best={best_acc:.4f})")
        scheduler.step()

        if val_acc > best_acc:
            best_acc     = val_acc
            no_improve   = 0
            final_preds  = ep_preds
            final_labels = ep_labels
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"               💾 Best!  ({val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n⏹  Early stopping — no improvement for {PATIENCE} epochs")
                break

    # ══════════════════════════════════════════
    # PHASE 3 — Temperature scaling calibration
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — Temperature Scaling Calibration")
    print(f"  Learns single scalar T on val set. Weights unchanged.")
    print(f"{'='*60}\n")

    # Load the BEST checkpoint for calibration
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    ts   = TemperatureScaler(model).to(device)
    temp = ts.calibrate(val_loader)

    torch.save({"model_state_dict": model.state_dict(), "temperature": temp},
               CALIBRATED_PATH)
    print(f"  💾 Calibrated checkpoint → {CALIBRATED_PATH}")

    # ══════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val accuracy : {best_acc:.4f}")
    print(f"  Temperature T     : {temp:.4f}")
    print(f"{'='*60}\n")

    if final_labels:
        print("Classification Report:")
        print(classification_report(
            final_labels, final_preds,
            target_names=list(SEVERITY_LABELS.values()),
            digits=4
        ))

        print("Per-class accuracy:")
        for i, cls in SEVERITY_LABELS.items():
            idxs    = [j for j, l in enumerate(final_labels) if l == i]
            correct = sum(1 for j in idxs if final_preds[j] == i)
            total   = len(idxs)
            pct     = correct / max(1, total)
            bar     = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            print(f"  {cls:>10}: {correct:>3}/{total:<3}  {pct:.1%}  {bar}")

        # Confusion matrix
        cm = confusion_matrix(final_labels, final_preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(SEVERITY_LABELS.values()),
                    yticklabels=list(SEVERITY_LABELS.values()))
        plt.title(f"Confusion Matrix — Val Acc: {best_acc:.4f}  (T={temp:.3f})")
        plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
        cm_path = SAVE_PATH.parent / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150); plt.close()
        print(f"\n📊 Confusion matrix → {cm_path}")

    # Training curves with phase boundary
    n_epochs = len(history["train_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    xs = range(1, n_epochs + 1)
    axes[0].plot(xs, history["train_loss"], "o-", ms=3, color="#1565C0", lw=1.5)
    axes[0].axvline(EPOCHS_FROZEN + 0.5, color="#E64A19", ls="--",
                    alpha=0.7, label="Phase 1→2")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(xs, history["val_acc"], "o-", ms=3, color="#2E7D32", lw=1.5)
    axes[1].axhline(best_acc, color="#C62828", ls="--",
                    label=f"Best: {best_acc:.4f}")
    axes[1].axvline(EPOCHS_FROZEN + 0.5, color="#E64A19", ls="--",
                    alpha=0.7, label="Phase 1→2")
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curves_path = SAVE_PATH.parent / "training_curves.png"
    plt.savefig(curves_path, dpi=150); plt.close()
    print(f"📈 Training curves → {curves_path}")

    # Save training summary JSON
    summary = {
        "best_val_accuracy": round(best_acc, 4),
        "temperature": round(temp, 4),
        "epochs_trained": n_epochs,
        "phase1_epochs": EPOCHS_FROZEN,
        "phase2_epochs": n_epochs - EPOCHS_FROZEN,
        "focal_gamma": FOCAL_GAMMA,
        "severe_boost": SEVERE_BOOST,
        "history": {
            "val_acc": [round(v, 4) for v in history["val_acc"]],
            "train_loss": [round(v, 4) for v in history["train_loss"]],
        }
    }
    summary_path = SAVE_PATH.parent / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📋 Training summary → {summary_path}")
    print(f"\n  Next: python src/inference/fairness_eval.py")


if __name__ == "__main__":
    train()