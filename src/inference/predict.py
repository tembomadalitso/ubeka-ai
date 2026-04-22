"""
predict.py — Ubeka AI FINAL
=============================
CLI and importable inference module.

Architecture matches train.py FINAL exactly:
  Dropout(0.4) → Linear(1280→256) → LayerNorm(256) → GELU → Dropout(0.2) → Linear(256→4)

Auto-loads calibrated model (acne_efficientnet_calibrated.pth) when available.
Use --tta flag for more accurate predictions via test-time augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import argparse
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parents[1]       # ubeka-ai/

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CALIBRATED_PATH = PROJECT_ROOT / "models" / "acne_efficientnet_calibrated.pth"
MODEL_PATH      = PROJECT_ROOT / "models" / "acne_efficientnet.pth"

IMAGE_SIZE  = 224
NUM_CLASSES = 4

SEVERITY_LABELS = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}
CONFIDENCE_THRESHOLD = 0.60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# MODEL — must match train.py FINAL exactly
# ─────────────────────────────────────────────
class AcneModel(nn.Module):
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

    def forward(self, x):
        return self.base(x)


class CalibratedModel(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.t     = temperature

    def forward(self, x):
        return self.model(x) / self.t


def load_model(model_path=None):
    """
    Load model. Priority:
      1. calibrated checkpoint (acne_efficientnet_calibrated.pth)
      2. uncalibrated checkpoint (acne_efficientnet.pth)
      3. explicit path via model_path argument
    """
    base = AcneModel().to(device)

    if model_path:
        path = Path(model_path)
        if path.suffix == ".pth":
            ckpt = torch.load(path, map_location=device)
            if "model_state_dict" in ckpt:
                base.load_state_dict(ckpt["model_state_dict"])
                temp  = float(ckpt.get("temperature", 1.0))
                model = CalibratedModel(base, temp).to(device)
            else:
                base.load_state_dict(ckpt)
                model = base
        else:
            raise ValueError(f"Expected .pth file, got: {path}")
    elif CALIBRATED_PATH.exists():
        ckpt = torch.load(CALIBRATED_PATH, map_location=device)
        base.load_state_dict(ckpt["model_state_dict"])
        temp  = float(ckpt.get("temperature", 1.0))
        model = CalibratedModel(base, temp).to(device)
        print(f"Loaded calibrated model (T={temp:.4f})")
    elif MODEL_PATH.exists():
        base.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = base
        print("Loaded uncalibrated model")
    else:
        raise FileNotFoundError(
            f"No model found.\n"
            f"  {CALIBRATED_PATH}\n"
            f"  {MODEL_PATH}\n"
            f"Run: python src/model/train.py"
        )

    model.eval()
    return model


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tta_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict_image(model, image_path):
    """Single-pass inference."""
    image  = Image.open(image_path).convert("RGB")
    tensor = base_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs                  = F.softmax(model(tensor), dim=1).squeeze()
        confidence, pred_class = probs.max(0)

    return _build_result(probs, int(pred_class), float(confidence))


def predict_with_tta(model, image_path, n_augments=8):
    """
    Test-time augmentation: 1 clean + (n-1) augmented passes.
    Average of probability distributions → more stable, especially
    for dark skin images and borderline cases.
    """
    image      = Image.open(image_path).convert("RGB")
    probs_list = []

    with torch.no_grad():
        # Pass 1: always clean
        t = base_transform(image).unsqueeze(0).to(device)
        probs_list.append(F.softmax(model(t), dim=1))

        # Passes 2-N: augmented
        for _ in range(n_augments - 1):
            t = tta_transform(image).unsqueeze(0).to(device)
            probs_list.append(F.softmax(model(t), dim=1))

    avg_probs              = torch.stack(probs_list).mean(0).squeeze()
    confidence, pred_class = avg_probs.max(0)

    result = _build_result(avg_probs, int(pred_class), float(confidence))
    result["tta_passes"] = n_augments
    return result


def _build_result(probs, pred_class, confidence):
    return {
        "skin_condition":     "acne",
        "severity":           SEVERITY_LABELS[pred_class],
        "severity_level":     pred_class,
        "confidence":         round(confidence, 4),
        "uncertain":          confidence < CONFIDENCE_THRESHOLD,
        "all_probabilities":  {
            SEVERITY_LABELS[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)
        }
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ubeka AI — Acne Severity Predictor")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--model",  type=str, help="Path to model checkpoint (optional)")
    parser.add_argument("--tta",    action="store_true",
                        help="Use TTA — slower but more accurate (recommended)")
    args = parser.parse_args()

    model      = load_model(args.model)
    predict_fn = (lambda m, p: predict_with_tta(m, p)) if args.tta else predict_image

    if args.image:
        result = predict_fn(model, args.image)
        print(f"\n{'='*45}")
        print(f"  Ubeka AI — Acne Analysis")
        print(f"{'='*45}")
        print(f"  Image      : {Path(args.image).name}")
        print(f"  Severity   : {result['severity'].upper()}")
        print(f"  Confidence : {result['confidence']:.4f}")
        if result["uncertain"]:
            print(f"  ⚠️  Low confidence — result may be uncertain")
        if args.tta:
            print(f"  TTA passes : {result.get('tta_passes', '—')}")
        print(f"\n  Probabilities:")
        for cls, prob in result["all_probabilities"].items():
            filled = int(prob * 30)
            bar    = "█" * filled + "░" * (30 - filled)
            marker = " ◄" if cls == result["severity"] else ""
            print(f"    {cls:>10}: {prob:.4f}  {bar}{marker}")
        print(f"{'='*45}\n")

    elif args.folder:
        folder = Path(args.folder)
        images = sorted([p for p in folder.glob("*")
                         if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        if not images:
            print(f"❌ No images found in {folder}")
            return

        print(f"\n📂 Testing {len(images)} images  "
              f"(TTA={'on' if args.tta else 'off'})\n")

        counts = {v: 0 for v in SEVERITY_LABELS.values()}
        for img_path in images:
            try:
                result = predict_fn(model, img_path)
                counts[result["severity"]] += 1
                flag = "⚠️ " if result["uncertain"] else "   "
                print(f"  {flag}{img_path.name:<35} → "
                      f"{result['severity']:<10} ({result['confidence']:.4f})")
            except Exception as e:
                print(f"  ❌ {img_path.name}: {e}")

        print(f"\n{'='*45}")
        print(f"  Summary ({len(images)} images):")
        for label, count in counts.items():
            bar = "█" * count
            print(f"    {label:>10}: {count:>3}  {bar}")
        print(f"{'='*45}\n")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python predict.py --image face.jpg --tta")
        print("  python predict.py --folder ./test_images/ --tta")


if __name__ == "__main__":
    main()