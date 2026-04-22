"""
inference.py — Ubeka AI (Production)
=======================================
Production-grade inference pipeline.

Key fixes over previous predict.py:
  1. Architecture matches train.py EXACTLY
  2. Auto-loads calibrated model (temperature applied)
  3. Real-world preprocessing: handles raw phone photos,
     not just pre-cropped clinical images
  4. TTA (8 passes) for stable predictions
  5. Confidence thresholding with "uncertain" handling
  6. Detailed output: all class probabilities, uncertainty flag

CLI usage:
  python src/inference/inference.py --image face.jpg
  python src/inference/inference.py --image face.jpg --tta
  python src/inference/inference.py --folder ./test_images/ --tta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import argparse
import cv2
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
current_dir  = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parents[1]
DATA_ROOT    = PROJECT_ROOT / "data"

CALIBRATED_PATH = PROJECT_ROOT / "models" / "acne_efficientnet_calibrated.pth"
MODEL_PATH      = PROJECT_ROOT / "models" / "acne_efficientnet.pth"

NUM_CLASSES          = 4
CONFIDENCE_THRESHOLD = 0.60
SEVERITY_LABELS      = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_real_world(image_path: str, target_size=(224, 224)):
    """
    Real-world preprocessing pipeline.
    Handles raw phone camera images — not just pre-cropped clinical shots.

    Steps:
      1. Read image (handles rotations via EXIF)
      2. Face detection — crop to face region if found
      3. Fallback to full image if no face (Acne04-style images)
      4. Resize to 224×224
    """
    # Handle EXIF rotation
    try:
        pil_img = Image.open(image_path)
        from PIL import ExifTags
        exif = pil_img._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    rotations = {3: 180, 6: 270, 8: 90}
                    if val in rotations:
                        pil_img = pil_img.rotate(rotations[val], expand=True)
                    break
        img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        img_bgr = cv2.imread(str(image_path))

    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Face detection
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        pad = int(0.1 * min(w, h))
        x1  = max(0, x-pad);          y1 = max(0, y-pad)
        x2  = min(img_bgr.shape[1], x+w+pad)
        y2  = min(img_bgr.shape[0], y+h+pad)
        face = img_bgr[y1:y2, x1:x2]
        method = "face_detected"
    else:
        face   = img_bgr
        method = "full_image_fallback"

    face_rgb = cv2.cvtColor(cv2.resize(face, target_size), cv2.COLOR_BGR2RGB)
    return face_rgb, method

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tta_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.88, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.12, contrast=0.12),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# MODEL (must match train.py exactly)
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


class CalibratedModel(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.t     = temperature
    def forward(self, x): return self.model(x) / self.t


def load_model(model_path=None):
    """
    Load model. Priority: calibrated → uncalibrated → explicit path.
    """
    base = AcneModel().to(device)

    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        if "model_state_dict" in ckpt:
            base.load_state_dict(ckpt["model_state_dict"])
            temp = float(ckpt.get("temperature", 1.0))
        else:
            base.load_state_dict(ckpt)
            temp = 1.0
    elif CALIBRATED_PATH.exists():
        ckpt = torch.load(CALIBRATED_PATH, map_location=device)
        base.load_state_dict(ckpt["model_state_dict"])
        temp = float(ckpt.get("temperature", 1.0))
        print(f"  Calibrated model loaded (T={temp:.4f})")
    elif MODEL_PATH.exists():
        base.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        temp = 1.0
        print("  Uncalibrated model loaded")
    else:
        raise FileNotFoundError(f"No model found at {MODEL_PATH}")

    model = CalibratedModel(base, temp).to(device) if temp != 1.0 else base
    model.eval()
    return model

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict_single(model, image_path: str) -> dict:
    """Single-pass inference with real-world preprocessing."""
    face_rgb, preprocess_method = preprocess_real_world(image_path)
    tensor = base_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs                  = F.softmax(model(tensor), dim=1).squeeze()
        confidence, pred_class = probs.max(0)

    return _build_result(probs, int(pred_class), float(confidence), preprocess_method)


def predict_tta(model, image_path: str, n_passes: int = 8) -> dict:
    """
    Test-time augmentation: 1 clean + (n-1) augmented passes.
    Recommended for production use.
    """
    face_rgb, preprocess_method = preprocess_real_world(image_path)
    face_arr = np.array(face_rgb)
    probs_list = []

    with torch.no_grad():
        # Pass 1: clean
        t = base_transform(Image.fromarray(face_arr)).unsqueeze(0).to(device)
        probs_list.append(F.softmax(model(t), dim=1))

        # Passes 2-N: augmented
        for _ in range(n_passes - 1):
            t = tta_transform(face_arr).unsqueeze(0).to(device)
            probs_list.append(F.softmax(model(t), dim=1))

    avg_probs              = torch.stack(probs_list).mean(0).squeeze()
    confidence, pred_class = avg_probs.max(0)

    result = _build_result(avg_probs, int(pred_class),
                           float(confidence), preprocess_method)
    result["tta_passes"] = n_passes
    return result


def _build_result(probs, pred_class: int, confidence: float,
                  preprocess_method: str) -> dict:
    """Consistent API response format matching brief specification."""
    uncertain = confidence < CONFIDENCE_THRESHOLD
    return {
        # Brief-specified format
        "skin_condition":    "acne",
        "severity":          SEVERITY_LABELS[pred_class],
        "confidence":        round(confidence, 4),
        # Extended
        "label":             SEVERITY_LABELS[pred_class],  # UI compat
        "severity_level":    pred_class,
        "uncertain":         uncertain,
        "preprocess_method": preprocess_method,
        "all_probabilities": {
            SEVERITY_LABELS[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)
        },
    }

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def print_result(result: dict, image_name: str = ""):
    SEV_EMOJI = {"clear": "✅", "mild": "🟡", "moderate": "🟠", "severe": "🔴"}
    sev = result["severity"]
    print(f"\n{'='*50}")
    if image_name: print(f"  Image     : {image_name}")
    print(f"  Severity  : {SEV_EMOJI.get(sev,'')} {sev.upper()}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    if result["uncertain"]:
        print(f"  ⚠️  Low confidence — result uncertain")
    print(f"  Method    : {result['preprocess_method']}")
    if result.get("tta_passes"):
        print(f"  TTA passes: {result['tta_passes']}")
    print(f"\n  All probabilities:")
    for cls, prob in result["all_probabilities"].items():
        filled = int(prob * 30)
        bar    = "█" * filled + "░" * (30 - filled)
        marker = " ◄" if cls == sev else ""
        print(f"    {cls:>10}: {prob*100:5.1f}%  {bar}{marker}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Ubeka AI — Acne Severity Inference")
    parser.add_argument("--image",  type=str, help="Path to image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--model",  type=str, help="Path to model checkpoint")
    parser.add_argument("--tta",    action="store_true",
                        help="Test-time augmentation (recommended, slower)")
    parser.add_argument("--passes", type=int, default=8,
                        help="TTA passes (default=8)")
    args = parser.parse_args()

    model      = load_model(args.model)
    predict_fn = (lambda m, p: predict_tta(m, p, args.passes)) if args.tta \
                  else predict_single

    if args.image:
        result = predict_fn(model, args.image)
        print_result(result, Path(args.image).name)

    elif args.folder:
        folder = Path(args.folder)
        images = sorted([p for p in folder.glob("*")
                         if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        if not images: print(f"❌ No images in {folder}"); return

        print(f"\n📂 Analysing {len(images)} images (TTA={'on' if args.tta else 'off'})")
        counts = {v: 0 for v in SEVERITY_LABELS.values()}
        errors = 0

        for img_path in images:
            try:
                result = predict_fn(model, str(img_path))
                counts[result["severity"]] += 1
                flag = "⚠️ " if result["uncertain"] else "   "
                print(f"  {flag}{img_path.name:<35} → "
                      f"{result['severity']:<10} {result['confidence']*100:.1f}%")
            except Exception as e:
                errors += 1
                print(f"  ❌ {img_path.name}: {e}")

        print(f"\n  Summary:")
        for cls, n in counts.items():
            bar = "█" * n
            print(f"    {cls:>10}: {n:>3}  {bar}")
        if errors: print(f"    Errors    : {errors}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python src/inference/inference.py --image face.jpg --tta")
        print("  python src/inference/inference.py --folder ./photos/ --tta")


if __name__ == "__main__":
    main()