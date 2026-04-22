"""
fairness_eval.py — Ubeka AI FINAL
===================================
Evaluates the trained model on Fitzpatrick skin tone groups.
Addresses brief requirement: "Bias toward darker skin tones should be considered"

Architecture MUST match train.py exactly:
  Dropout(0.4) → Linear(1280→256) → LayerNorm(256) → GELU → Dropout(0.2) → Linear(256→4)

Improvements over v2:
  - Auto-loads calibrated model (acne_efficientnet_calibrated.pth)
  - TTA (5 passes) for stable confidence estimates on dark skin images
  - Entropy metric: mathematically cleaner uncertainty measure
  - Improvement delta vs v2 baseline printed and saved
  - Fixed paths anchored to __file__
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parents[1]       # ubeka-ai/
DATA_ROOT    = PROJECT_ROOT / "data"        # ubeka-ai/data/

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CALIBRATED_PATH = PROJECT_ROOT / "models" / "acne_efficientnet_calibrated.pth"
MODEL_PATH      = PROJECT_ROOT / "models" / "acne_efficientnet.pth"
FITZPATRICK_DIR = DATA_ROOT / "processed" / "fitzpatrick"
REPORT_PATH     = PROJECT_ROOT / "models" / "fairness_report.json"

NUM_CLASSES = 4
TTA_PASSES  = 5
CONF_THRESH = 0.60

# v2 baselines from actual run output — used to measure improvement
V2_CONF_RANGE = 0.0633
V2_UNC_RANGE  = 12.3

SEVERITY_LABELS = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}
GROUPS = {
    "group1_light":  "Light Skin (Group 1)",
    "group2_medium": "Medium Skin (Group 2)",
    "group3_dark":   "Dark Skin (Group 3)",
}
GROUP_COLORS = {
    "group1_light":  "#f4c89b",
    "group2_medium": "#c8855a",
    "group3_dark":   "#3d1a00",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Project root    : {PROJECT_ROOT}")
print(f"Data root       : {DATA_ROOT}")
print(f"Fitzpatrick dir : {FITZPATRICK_DIR}  exists={FITZPATRICK_DIR.exists()}")
print(f"Calibrated model: {CALIBRATED_PATH}  exists={CALIBRATED_PATH.exists()}")


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tta_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# MODEL — must match train.py EXACTLY
# ─────────────────────────────────────────────
class AcneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = self.base.classifier[1].in_features  # 1280
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),             # ← must match train.py
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),           # ← must match train.py
            nn.GELU(),                   # ← must match train.py
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.base(x)


class CalibratedModel(nn.Module):
    """Wraps model with temperature for calibrated probabilities."""
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.t     = temperature

    def forward(self, x):
        return self.model(x) / self.t


def load_model():
    base = AcneModel().to(device)

    if CALIBRATED_PATH.exists():
        ckpt = torch.load(CALIBRATED_PATH, map_location=device)
        base.load_state_dict(ckpt["model_state_dict"])
        temp  = float(ckpt.get("temperature", 1.0))
        model = CalibratedModel(base, temp).to(device)
        print(f"\n  ✅ Calibrated model (T={temp:.4f})")
        if temp > 1.1:
            print(f"     Temperature softens overconfident predictions")
    elif MODEL_PATH.exists():
        base.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = base
        temp  = 1.0
        print(f"\n  ⚠️  Uncalibrated model loaded (T=1.0)")
        print(f"     Run full train.py for temperature calibration")
    else:
        raise FileNotFoundError(
            f"No model found.\n"
            f"  Checked: {CALIBRATED_PATH}\n"
            f"  Checked: {MODEL_PATH}\n"
            f"  Run: python src/model/train.py"
        )

    model.eval()
    return model, temp


# ─────────────────────────────────────────────
# TTA PREDICTION
# ─────────────────────────────────────────────
def predict_tta(model, image_path, n=TTA_PASSES):
    """
    Test-time augmentation: 1 clean pass + (n-1) augmented passes.
    Averages probability distributions for more stable estimates.
    5 passes is a good tradeoff — beyond 8 passes gains are marginal.
    """
    img   = Image.open(image_path).convert("RGB")
    probs = []

    with torch.no_grad():
        # Pass 1: clean
        t = base_transform(img).unsqueeze(0).to(device)
        probs.append(F.softmax(model(t), dim=1))

        # Passes 2-N: augmented
        for _ in range(n - 1):
            t = tta_transform(img).unsqueeze(0).to(device)
            probs.append(F.softmax(model(t), dim=1))

    avg              = torch.stack(probs).mean(0).squeeze()
    conf, pred_class = avg.max(0)

    # Shannon entropy: H = -sum(p * log(p)) — 0 = certain, log(4)≈1.39 = max uncertain
    entropy = float(-(avg * (avg + 1e-8).log()).sum())

    return {
        "pred_class": int(pred_class),
        "confidence": float(conf),
        "uncertain":  float(conf) < CONF_THRESH,
        "entropy":    entropy,
        "probs":      avg.cpu().numpy().tolist(),
    }


# ─────────────────────────────────────────────
# GROUP EVALUATION
# ─────────────────────────────────────────────
def evaluate_group(model, group_dir: Path, group_name: str):
    imgs = sorted(list(group_dir.glob("*.jpg")) + list(group_dir.glob("*.png")))
    if not imgs:
        print(f"    ⚠️  No images found in {group_dir}")
        return None

    confidences, predictions, entropies, ita_scores = [], [], [], []
    uncertain_count = 0
    errors          = 0

    for img_path in imgs:
        try:
            r = predict_tta(model, img_path)
        except Exception as e:
            errors += 1
            continue

        confidences.append(r["confidence"])
        predictions.append(r["pred_class"])
        entropies.append(r["entropy"])
        if r["uncertain"]:
            uncertain_count += 1

        # Extract ITA from filename (e.g. face_ita-15.3.jpg → -15.3)
        if "_ita" in img_path.stem:
            try:
                ita = float(img_path.stem.split("_ita")[-1].split("_")[0])
                ita_scores.append(ita)
            except ValueError:
                pass

    n = len(confidences)
    if n == 0:
        return None

    class_dist = {SEVERITY_LABELS[i]: predictions.count(i) for i in range(4)}
    dom_class  = max(class_dist, key=class_dist.get)

    return {
        "group":             group_name,
        "n_images":          n,
        "errors":            errors,
        "mean_confidence":   round(float(np.mean(confidences)), 4),
        "std_confidence":    round(float(np.std(confidences)),  4),
        "uncertain_pct":     round(uncertain_count / n * 100,   2),
        "mean_entropy":      round(float(np.mean(entropies)),   4),
        "std_entropy":       round(float(np.std(entropies)),    4),
        "mean_ita":          round(float(np.mean(ita_scores)),  2) if ita_scores else None,
        "dominant_prediction": dom_class,
        "prediction_distribution": class_dist,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  UBEKA AI — Fairness Evaluation (Final)")
    print(f"  TTA passes: {TTA_PASSES}  |  Confidence threshold: {CONF_THRESH}")
    print("="*60)

    model, temperature = load_model()
    results = []

    for folder_name, display_name in GROUPS.items():
        gdir = FITZPATRICK_DIR / folder_name
        if not gdir.exists():
            print(f"\n  ⚠️  Group folder not found: {gdir}")
            continue

        n_imgs = len(list(gdir.glob("*.jpg"))) + len(list(gdir.glob("*.png")))
        print(f"\n  [{folder_name}] {display_name} — {n_imgs} images")

        r = evaluate_group(model, gdir, display_name)
        if r:
            results.append(r)
            print(f"    Mean confidence : {r['mean_confidence']:.4f}  "
                  f"(±{r['std_confidence']:.4f})")
            print(f"    Uncertain       : {r['uncertain_pct']:.1f}%")
            print(f"    Mean entropy    : {r['mean_entropy']:.4f}  "
                  f"(±{r['std_entropy']:.4f})")
            print(f"    Mean ITA score  : {r['mean_ita']}")
            print(f"    Dominant pred   : {r['dominant_prediction']}")
            print(f"    Distribution    : {r['prediction_distribution']}")

    if not results:
        print("\n  ❌ No results. Check that Fitzpatrick data was processed.")
        return

    # ── Bias metrics ──────────────────────────────────────────────────────────
    confs     = [r["mean_confidence"] for r in results]
    uncs      = [r["uncertain_pct"]   for r in results]
    entropies = [r["mean_entropy"]    for r in results]

    conf_range    = max(confs)     - min(confs)
    unc_range     = max(uncs)      - min(uncs)
    entropy_range = max(entropies) - min(entropies)

    delta_conf = V2_CONF_RANGE - conf_range   # positive = improvement
    delta_unc  = V2_UNC_RANGE  - unc_range

    flag = ("high"     if conf_range > 0.10 else
            "moderate" if conf_range > 0.05 else "low")

    print("\n" + "="*60)
    print("  BIAS ANALYSIS — FINAL REPORT")
    print("="*60)
    print(f"\n  {'Metric':<25} {'v2 baseline':>12}  →  {'v3 final':>10}  {'Change':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Confidence range':<25} {V2_CONF_RANGE:>12.4f}  →  "
          f"{conf_range:>10.4f}  "
          f"{'✅ -'+str(round(abs(delta_conf),4)) if delta_conf>0 else '❌ +'+str(round(abs(delta_conf),4)):>12}")
    print(f"  {'Uncertainty range':<25} {V2_UNC_RANGE:>11.1f}%  →  "
          f"{unc_range:>9.1f}%  "
          f"{'✅ -'+str(round(abs(delta_unc),1))+'%' if delta_unc>0 else '❌ +'+str(round(abs(delta_unc),1))+'%':>12}")
    print(f"  {'Entropy range':<25} {'—':>12}  →  {entropy_range:>10.4f}")
    print(f"  {'Bias flag':<25} {'moderate':>12}  →  {flag:>10}")

    if flag == "low":
        print(f"\n  ✅ EXCELLENT — model performs consistently across all skin tones")
    elif flag == "moderate":
        print(f"\n  ⚠️  MODERATE — some variation remains, acceptable for trial build")
    else:
        print(f"\n  ❌ HIGH — significant skin tone bias, needs more dark skin data")

    # ── Per-group table ───────────────────────────────────────────────────────
    print(f"\n  Per-group confidence:")
    for r in results:
        name  = r["group"].split(" (")[0]
        conf  = r["mean_confidence"]
        unc   = r["uncertain_pct"]
        bar   = "█" * int(conf * 20)
        print(f"    {name:<15}: conf={conf:.4f}  uncertain={unc:.1f}%  {bar}")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "version":     "final",
        "temperature": temperature,
        "tta_passes":  TTA_PASSES,
        "summary": {
            "confidence_range":    round(conf_range,    4),
            "uncertainty_range":   round(unc_range,     2),
            "entropy_range":       round(entropy_range, 4),
            "bias_flag":           flag,
            "v2_confidence_range": V2_CONF_RANGE,
            "v2_uncertainty_range": V2_UNC_RANGE,
            "delta_conf":  round(delta_conf, 4),
            "delta_unc":   round(delta_unc,  2),
            "improved":    delta_conf > 0 and delta_unc > 0,
        },
        "groups": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  📄 Fairness report → {REPORT_PATH}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    group_short = [r["group"].split(" (")[0] for r in results]
    colors      = [GROUP_COLORS[k] for k in list(GROUPS.keys())[:len(results)]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Ubeka AI — Final Fairness Evaluation  "
                 f"(T={temperature:.3f}, TTA={TTA_PASSES})",
                 fontsize=12, fontweight="bold")

    metrics = [
        (axes[0], confs,     "Mean Confidence by Skin Tone",        "Confidence",    (0, 1)),
        (axes[1], uncs,      "Uncertain Predictions (%) by Tone",   "Uncertain (%)", None),
        (axes[2], entropies, "Mean Entropy (↓ better)",              "Entropy",       None),
    ]

    for ax, vals, title, ylabel, ylim in metrics:
        bars = ax.bar(group_short, vals, color=colors, edgecolor="#111", lw=0.8, width=0.55)
        ax.axhline(np.mean(vals), color="#E53935", ls="--", lw=1.3, label=f"Mean={np.mean(vals):.3f}")
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = REPORT_PATH.parent / "fairness_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  📊 Fairness chart → {chart_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()