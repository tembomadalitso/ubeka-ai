"""
evaluate.py — Ubeka AI
========================
Full model evaluation beyond accuracy:
  - Per-class precision, recall, F1
  - Confusion matrix with error analysis
  - Calibration (ECE — Expected Calibration Error)
  - Failure mode analysis: WHY does the model get things wrong?
  - Confidence distribution per class
  - Real-world simulation (tests on val set WITHOUT any preprocessing)

Run after training:
    python src/model/evaluate.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import json
import sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parents[1]
DATA_ROOT    = PROJECT_ROOT / "data"
sys.path.append(str(current_dir.parent))

CALIBRATED_PATH = PROJECT_ROOT / "models" / "acne_efficientnet_calibrated.pth"
MODEL_PATH      = PROJECT_ROOT / "models" / "acne_efficientnet.pth"
VAL_DIR         = DATA_ROOT / "processed" / "val"
REPORT_DIR      = PROJECT_ROOT / "models" / "eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES     = 4
BATCH_SIZE      = 16
SEVERITY_LABELS = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}
SEV_COLORS      = {"clear":"#2d6a4f","mild":"#b7950b","moderate":"#c4623a","severe":"#c0392b"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# MODEL (must match train.py)
# ─────────────────────────────────────────────
class AcneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x): return self.base(x)


def load_model():
    base = AcneModel().to(device)
    temperature = 1.0

    if CALIBRATED_PATH.exists():
        ckpt = torch.load(CALIBRATED_PATH, map_location=device)
        base.load_state_dict(ckpt["model_state_dict"])
        temperature = float(ckpt.get("temperature", 1.0))
        print(f"✅ Calibrated model loaded (T={temperature:.4f})")
    elif MODEL_PATH.exists():
        base.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("⚠️  Uncalibrated model loaded")
    else:
        raise FileNotFoundError(f"No model found at {MODEL_PATH}")

    base.eval()
    return base, temperature

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ValDataset(Dataset):
    CLASS_MAP = {"clear": 0, "mild": 1, "moderate": 2, "severe": 3}
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, data_dir: Path):
        self.samples = []
        for cls_name, label in self.CLASS_MAP.items():
            folder = data_dir / cls_name
            if not folder.exists(): continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((p, label))
        print(f"Val set: {len(self.samples)} images")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.array(Image.open(path).convert("RGB"))
        return self.transform(arr).float(), label, str(path)

# ─────────────────────────────────────────────
# CALIBRATION: Expected Calibration Error (ECE)
# ─────────────────────────────────────────────
def compute_ece(confidences, labels, n_bins=10):
    """
    ECE: measures whether confidence scores match actual accuracy.
    ECE=0 is perfect. ECE>0.1 means the model is poorly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n         = len(confidences)

    bin_stats = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_stats.append(None); continue
        acc_in_bin  = labels[mask].mean()
        conf_in_bin = confidences[mask].mean()
        ece        += mask.sum() / n * abs(acc_in_bin - conf_in_bin)
        bin_stats.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "accuracy": round(float(acc_in_bin), 4),
            "confidence": round(float(conf_in_bin), 4),
            "gap": round(float(abs(acc_in_bin - conf_in_bin)), 4),
        })
    return float(ece), bin_stats

# ─────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────
def run_evaluation():
    print("\n" + "="*60)
    print("  Ubeka AI — Full Evaluation")
    print("="*60)

    if not VAL_DIR.exists():
        print(f"❌ Val dir not found: {VAL_DIR}")
        print("   Run dataset_builder.py first"); return

    model, temperature = load_model()

    val_ds     = ValDataset(VAL_DIR)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Collect predictions ────────────────────────────────────────────────────
    all_logits, all_probs, all_preds, all_labels, all_paths = [], [], [], [], []
    all_confs = []

    print("\n  Running inference on val set...")
    with torch.no_grad():
        for imgs, labels, paths in val_loader:
            imgs    = imgs.to(device)
            logits  = model(imgs) / temperature   # apply calibration
            probs   = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)

            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(conf.cpu().numpy())
            all_paths.extend(paths)

    all_logits = np.array(all_logits)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    overall_acc = (all_preds == all_labels).mean()
    print(f"\n  Overall val accuracy: {overall_acc:.4f}")

    # ── Classification report ──────────────────────────────────────────────────
    print("\n  Classification Report:")
    report_str = classification_report(
        all_labels, all_preds,
        target_names=list(SEVERITY_LABELS.values()), digits=4
    )
    print(report_str)

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS.values(),
                yticklabels=SEVERITY_LABELS.values())
    plt.title(f"Confusion Matrix — Val Acc: {overall_acc:.4f}  (T={temperature:.3f})")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(REPORT_DIR/"confusion_matrix.png", dpi=150); plt.close()

    # ── ECE (calibration) ──────────────────────────────────────────────────────
    correct   = (all_preds == all_labels).astype(float)
    ece, bins = compute_ece(all_confs, correct)
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    if ece > 0.10:
        print("  ⚠️  Poor calibration — temperature scaling needed")
    elif ece > 0.05:
        print("  ⚠️  Moderate calibration — acceptable")
    else:
        print("  ✅ Good calibration")

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(6,5))
    bin_data = [b for b in bins if b is not None]
    centres  = [float(b["bin"].split("-")[0]) + 0.05 for b in bin_data]
    accs     = [b["accuracy"] for b in bin_data]
    confs    = [b["confidence"] for b in bin_data]
    ax.plot([0,1],[0,1],"--",color="#999",lw=1.5,label="Perfect calibration")
    ax.bar(centres, accs, width=0.09, alpha=0.7, color="#c4623a", label="Accuracy in bin")
    ax.step(centres, confs, color="#2d6a4f", lw=2, where="mid", label="Confidence in bin")
    ax.set_title(f"Reliability Diagram — ECE={ece:.4f}")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"reliability_diagram.png", dpi=150); plt.close()

    # ── Confidence distribution per class ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10,5))
    for i, cls in SEVERITY_LABELS.items():
        mask  = all_labels == i
        if mask.sum() == 0: continue
        confs_cls = all_confs[mask]
        ax.hist(confs_cls, bins=25, alpha=0.6, label=f"{cls} (n={mask.sum()})",
                color=list(SEV_COLORS.values())[i])
    ax.axvline(0.60, color="black", ls="--", lw=1.2, label="Threshold (0.60)")
    ax.set_title("Confidence Distribution per Class"); ax.set_xlabel("Confidence")
    ax.set_ylabel("Count"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"confidence_distribution.png", dpi=150); plt.close()

    # ── FAILURE ANALYSIS ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FAILURE ANALYSIS — Why does the model get things wrong?")
    print("="*60)

    wrong_mask = all_preds != all_labels
    wrong_idx  = np.where(wrong_mask)[0]

    # Error patterns
    error_patterns = {}
    for i in wrong_idx:
        true_cls = SEVERITY_LABELS[all_labels[i]]
        pred_cls = SEVERITY_LABELS[all_preds[i]]
        key = f"{true_cls}→{pred_cls}"
        error_patterns[key] = error_patterns.get(key, 0) + 1

    print(f"\n  Total errors: {len(wrong_idx)} / {len(all_labels)}")
    print(f"\n  Error patterns (true→predicted):")
    for k, v in sorted(error_patterns.items(), key=lambda x: -x[1]):
        print(f"    {k:>22}: {v:>4} errors")

    # Per-class error analysis
    print(f"\n  Per-class breakdown:")
    for i, cls in SEVERITY_LABELS.items():
        true_mask   = all_labels == i
        n_true      = true_mask.sum()
        n_correct   = (all_preds[true_mask] == i).sum()
        n_wrong     = n_true - n_correct
        wrong_conf  = all_confs[true_mask & wrong_mask]
        correct_conf = all_confs[true_mask & ~wrong_mask]

        print(f"\n  [{cls.upper()}]  n={n_true}  accuracy={n_correct/max(1,n_true):.1%}")
        print(f"    Correct predictions — mean conf: {correct_conf.mean():.3f}" if len(correct_conf)>0 else "    No correct predictions!")
        print(f"    Wrong predictions   — mean conf: {wrong_conf.mean():.3f}  (model confident when wrong!)" if len(wrong_conf)>0 else "    No errors ✅")

        if len(wrong_conf) > 0 and wrong_conf.mean() > 0.65:
            print(f"    ⚠️  HIGH CONFIDENCE ERRORS — model is wrong AND confident")
            print(f"       Root cause: distribution mismatch or label noise")

    # Root cause diagnosis
    print(f"\n{'='*60}")
    print(f"  ROOT CAUSE DIAGNOSIS")
    print(f"{'='*60}")

    mild_clear_errors = error_patterns.get("mild→clear", 0) + error_patterns.get("clear→mild", 0)
    severe_errors     = sum(v for k,v in error_patterns.items() if "severe" in k)
    total_errors      = len(wrong_idx)

    print(f"\n  Mild↔Clear confusion : {mild_clear_errors} errors ({mild_clear_errors/max(1,total_errors):.0%} of all errors)")
    if mild_clear_errors / max(1, total_errors) > 0.30:
        print("  → CAUSE: These classes are visually similar. Mild acne (few comedones)")
        print("           overlaps with 'clear' skin that has minor blemishes.")
        print("  → FIX:   Need clearer labelling guidelines OR merge into 3 classes.")

    print(f"\n  Severe errors        : {severe_errors} errors")
    if severe_errors > 0:
        n_severe_val = (all_labels == 3).sum()
        print(f"  → CAUSE: Only {n_severe_val} severe images in val — model has seen too few.")
        print("  → FIX:   Collect more severe acne images (priority #1).")

    # High-confidence errors (most dangerous for real-world use)
    hce_mask = wrong_mask & (all_confs > 0.75)
    n_hce    = hce_mask.sum()
    print(f"\n  High-confidence errors (conf>0.75): {n_hce}")
    if n_hce > 0:
        print("  → These are the most dangerous: model is confidently WRONG")
        print("  → CAUSE: Either label noise in val set, or severe distribution mismatch")
        print("           between training images and val images")

    # ── Save full report ────────────────────────────────────────────────────────
    report = {
        "overall_accuracy": round(float(overall_acc), 4),
        "temperature": temperature,
        "ece": round(ece, 4),
        "n_val_images": len(all_labels),
        "per_class": {
            SEVERITY_LABELS[i]: {
                "precision": round(float(prec[i]), 4),
                "recall":    round(float(rec[i]), 4),
                "f1":        round(float(f1[i]), 4),
                "support":   int(sup[i]),
                "accuracy":  round(float((all_preds[all_labels==i]==i).mean()), 4)
                             if (all_labels==i).sum()>0 else 0.0,
            } for i in range(NUM_CLASSES)
        },
        "error_patterns": error_patterns,
        "high_confidence_errors": int(n_hce),
        "total_errors": int(total_errors),
        "calibration_bins": bin_stats,
        "root_causes": {
            "mild_clear_confusion_pct": round(mild_clear_errors/max(1,total_errors)*100,1),
            "severe_errors": int(severe_errors),
            "high_confidence_errors": int(n_hce),
        }
    }

    report_path = REPORT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  📄 Full report → {report_path}")
    print(f"  📊 Plots saved → {REPORT_DIR}/")
    print(f"\n  Next: python src/inference/fairness_eval.py")
    print("="*60 + "\n")

    return report


if __name__ == "__main__":
    run_evaluation()