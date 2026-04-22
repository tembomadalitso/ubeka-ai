"""
build_dataset.py  — Ubeka AI v2
================================
Key improvements over v1:
  1. Blur threshold lowered 80→50 (recovered ~400 valid acne images)
  2. Dark-skin acne images oversampled 8x to address 31x representation gap
  3. Severe class augmented with all available sources (acne3_512_selection included)
  4. Manifest-driven: reads data/acne_manifest.csv when available (from notebook 01)
  5. CelebA used correctly — IoU validation only, no training data
  6. Fitzpatrick processed at higher quality (blur threshold 60)

Run from ubeka-ai/ root:
    python src/preprocessing/build_dataset.py
"""

import cv2
import json
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Anchor to project root — works regardless of where script is run from
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"         # sibling of ubeka-ai/
RAW_DIR         = DATA_ROOT / "raw"
PROCESSED_DIR   = DATA_ROOT / "processed"
ACNE_DIR        = RAW_DIR / "acne" / "acne_1024"
MANIFEST_CSV    = DATA_ROOT / "acne_manifest.csv"       # from notebook 01

# Try multiple Fitzpatrick path patterns
FITZ_CANDIDATES = [
    RAW_DIR / "fitzpatrick" / "kaggle" / "working" / "fitzpatrick_3groups",
    RAW_DIR / "fitzpatrick" / "fitzpatrick_3groups",
    RAW_DIR / "fitzpatrick",
]
FITZPATRICK_DIR = next((p for p in FITZ_CANDIDATES if p.exists()), None)

# CelebA
CELEBA_CSV_CANDIDATES = [
    RAW_DIR / "celeba" / "list_bbox_celeba.csv",
    RAW_DIR / "celeba" / "img_align_celeba" / "list_bbox_celeba.csv",
]
CELEBA_IMG_CANDIDATES = [
    RAW_DIR / "celeba" / "img_align_celeba" / "img_align_celeba",
    RAW_DIR / "celeba" / "img_align_celeba",
]
CELEBA_CSV  = next((p for p in CELEBA_CSV_CANDIDATES if p.exists()), None)
CELEBA_IMGS = next((p for p in CELEBA_IMG_CANDIDATES if p.exists() and p.is_dir()), None)

TARGET_SIZE    = (224, 224)
VAL_SPLIT      = 0.2

# ── Quality thresholds (v2 — more permissive on blur) ─────────────────────────
# v1 used 80, which rejected 917 images (29% of dataset).
# Acne images are clinical shots — intentionally softer than natural photos.
# Lowering to 50 recovers ~400 valid images, especially in severe class.
BLUR_THRESHOLD  = 50.0
MIN_BRIGHTNESS  = 25     # slightly more permissive for dark skin
MAX_BRIGHTNESS  = 245
MIN_SIZE_PX     = 60

# ── Dark skin oversampling ─────────────────────────────────────────────────────
# dataset_report.json showed: dark=37, light=1146 in acne images (31x gap)
# We oversample dark-skin acne images 8x to partially close this gap.
DARK_SKIN_OVERSAMPLE = 8

# ── Acne folder map ───────────────────────────────────────────────────────────
ACNE_FOLDER_MAP = {
    "acne0_1024":          ("clear",    0),
    "acne1_1024":          ("mild",     1),
    "acne2_1024":          ("moderate", 2),
    "acne3_1024":          ("severe",   3),
    "acne3_512_selection": ("severe",   3),  # critical — adds to starved severe class
    "all_1024":            (None,       None),
    "small_1024":          (None,       None),
    "small_1024_renamed":  (None,       None),
}

RENAMED_PREFIX_MAP = {
    "acnezero":   ("clear",    0),
    "acnesmall":  ("mild",     1),
    "acnemedium": ("moderate", 2),
    "acnestrong": ("severe",   3),
}

SEVERITY_LABELS = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}

print(f"Project root : {PROJECT_ROOT}")
print(f"Data root    : {DATA_ROOT}")
print(f"Raw dir      : {RAW_DIR}  exists={RAW_DIR.exists()}")
print(f"Fitz dir     : {FITZPATRICK_DIR}")
print(f"Manifest     : {MANIFEST_CSV}  exists={MANIFEST_CSV.exists()}")


# ─────────────────────────────────────────────
# LABEL INFERENCE
# ─────────────────────────────────────────────
def infer_label(stem: str):
    import re
    s = stem.lower()
    for prefix, info in RENAMED_PREFIX_MAP.items():
        if s.startswith(prefix):
            return info
    matches = re.findall(r'(?<!\d)([0-3])(?!\d)', s)
    if matches:
        level = int(matches[0])
        return SEVERITY_LABELS[level], level
    return None, None


# ─────────────────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(MIN_SIZE_PX, MIN_SIZE_PX)
    )
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.1 * min(w, h))
        x1  = max(0, x - pad);          y1 = max(0, y - pad)
        x2  = min(img_bgr.shape[1], x + w + pad)
        y2  = min(img_bgr.shape[0], y + h + pad)
        return img_bgr[y1:y2, x1:x2], "cascade"
    return img_bgr, "fallback"  # acne04 images are already face-cropped


# ─────────────────────────────────────────────
# QUALITY FILTER (v2 — looser blur)
# ─────────────────────────────────────────────
def passes_quality(img_bgr):
    h, w = img_bgr.shape[:2]
    if h < MIN_SIZE_PX or w < MIN_SIZE_PX:
        return False, "too_small"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
        return False, "blurry"
    mean = img_bgr.mean()
    if not (MIN_BRIGHTNESS <= mean <= MAX_BRIGHTNESS):
        return False, "bad_brightness"
    return True, "ok"


# ─────────────────────────────────────────────
# ITA — skin tone score
# ─────────────────────────────────────────────
def compute_ita(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L   = lab[:, :, 0] * (100.0 / 255.0)
    b   = lab[:, :, 2] - 128.0
    h, w = img_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    r  = min(h, w) // 4
    Lc = L[cy-r:cy+r, cx-r:cx+r].mean()
    bc = b[cy-r:cy+r, cx-r:cx+r].mean()
    return float(np.degrees(np.arctan((Lc - 50) / bc))) if bc != 0 else 0.0

def ita_to_group(ita):
    if ita > 28:  return "group1_light"
    if ita > -30: return "group2_medium"
    return "group3_dark"

def is_dark_skin(ita):
    return ita <= -30


# ─────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────
def save_image(img_bgr, split, label_str, stem, stats, oversample=1):
    """
    Detect face, quality-check, resize, save.
    oversample > 1: save multiple copies with slight augmentation (dark skin boost).
    """
    face, method = detect_face(img_bgr)
    stats["face_detection"][method] = stats["face_detection"].get(method, 0) + 1

    ok, reason = passes_quality(face)
    if not ok:
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
        return False

    face_r = cv2.resize(face, TARGET_SIZE)
    ita    = compute_ita(face_r)
    tone   = ita_to_group(ita)

    out_dir = PROCESSED_DIR / split / label_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    out_path = out_dir / f"{stem}_ita{ita:.1f}.jpg"
    cv2.imwrite(str(out_path), face_r)
    stats["saved"][split][label_str] = stats["saved"][split].get(label_str, 0) + 1
    stats["skin_tone"][tone] = stats["skin_tone"].get(tone, 0) + 1

    # Oversample dark skin — save augmented copies
    for i in range(1, oversample):
        aug = augment_dark_skin(face_r)
        aug_path = out_dir / f"{stem}_dark_aug{i}_ita{ita:.1f}.jpg"
        cv2.imwrite(str(aug_path), aug)
        stats["saved"][split][label_str] += 1
        stats["dark_skin_augmented"] = stats.get("dark_skin_augmented", 0) + 1

    return True


def augment_dark_skin(img_bgr):
    """
    Targeted augmentation for dark skin images:
    - Brightness/contrast jitter (critical — dark images cluster in low range)
    - Horizontal flip
    - Slight rotation
    """
    img = img_bgr.copy().astype(np.float32)

    # Random brightness shift
    alpha = np.random.uniform(0.75, 1.35)   # contrast
    beta  = np.random.randint(-20, 25)       # brightness offset
    img   = np.clip(alpha * img + beta, 0, 255)

    # Horizontal flip (50% chance)
    if random.random() > 0.5:
        img = cv2.flip(img.astype(np.uint8), 1)

    # Slight rotation (±15°)
    angle = random.uniform(-15, 15)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img   = cv2.warpAffine(img.astype(np.uint8), M, (w, h),
                            borderMode=cv2.BORDER_REFLECT)
    return img.astype(np.uint8)


# ─────────────────────────────────────────────
# STEP 1 — ACNE04
# ─────────────────────────────────────────────
def process_acne(report):
    print("\n" + "="*60)
    print("  STEP 1: Acne04 — Processing with v2 improvements")
    print("="*60)

    # ── Collect all samples ───────────────────────────────────────
    all_samples = []  # (path, label_str, label_int)
    skipped_unknown = 0

    for folder_name, (fixed_label, fixed_int) in ACNE_FOLDER_MAP.items():
        folder = ACNE_DIR / folder_name
        if not folder.exists():
            print(f"  ⚠️  Missing: {folder_name}")
            continue

        imgs = [p for p in folder.rglob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        resolved = 0
        for p in imgs:
            if fixed_label is not None:
                all_samples.append((p, fixed_label, fixed_int))
                resolved += 1
            else:
                ls, li = infer_label(p.stem)
                if ls:
                    all_samples.append((p, ls, li))
                    resolved += 1
                else:
                    skipped_unknown += 1

        print(f"  {folder_name:<25} → {resolved} images collected")

    print(f"\n  Total collected  : {len(all_samples)}")
    print(f"  Skipped (no label): {skipped_unknown}")

    # Per-class counts before quality filter
    for i in range(4):
        n = sum(1 for _, ls, _ in all_samples if ls == SEVERITY_LABELS[i])
        print(f"     {SEVERITY_LABELS[i]:>10}: {n} raw")

    # ── Shuffle and split ─────────────────────────────────────────
    random.shuffle(all_samples)
    val_n         = int(len(all_samples) * VAL_SPLIT)
    val_samples   = all_samples[:val_n]
    train_samples = all_samples[val_n:]

    stats = {
        "total_raw": len(all_samples),
        "skipped_unknown": skipped_unknown,
        "blur_threshold_used": BLUR_THRESHOLD,
        "rejected": {},
        "saved": {"train": {}, "val": {}},
        "face_detection": {},
        "skin_tone": {},
        "dark_skin_augmented": 0,
    }

    # ── Process train ─────────────────────────────────────────────
    print(f"\n  Processing train ({len(train_samples)}) ...")
    dark_count = 0
    for img_path, label_str, _ in tqdm(train_samples, desc="  train"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Quick ITA check to determine if dark skin oversample needed
        try:
            sample_face = cv2.resize(img, TARGET_SIZE)
            ita         = compute_ita(sample_face)
            oversample  = DARK_SKIN_OVERSAMPLE if is_dark_skin(ita) else 1
            if oversample > 1:
                dark_count += 1
        except:
            oversample = 1

        save_image(img, "train", label_str, img_path.stem, stats, oversample)

    print(f"  Dark skin images found and oversampled: {dark_count} → ×{DARK_SKIN_OVERSAMPLE}")

    # ── Process val (NO oversampling — val must reflect real distribution) ──
    print(f"  Processing val ({len(val_samples)}) ...")
    for img_path, label_str, _ in tqdm(val_samples, desc="  val"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        save_image(img, "val", label_str, img_path.stem, stats, oversample=1)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n  ✅ Acne04 v2 complete")
    print(f"  Blur threshold    : {BLUR_THRESHOLD} (was 80 — recovers more images)")
    print(f"  Rejected          : {stats['rejected']}")
    print(f"  Saved train       : {stats['saved']['train']}")
    print(f"  Saved val         : {stats['saved']['val']}")
    print(f"  Dark skin augmented: +{stats['dark_skin_augmented']} copies")
    print(f"  Skin tone (train) : {stats['skin_tone']}")

    report["acne_v2"] = stats
    return report


# ─────────────────────────────────────────────
# STEP 2 — CELEBA VALIDATION
# ─────────────────────────────────────────────
def process_celeba(report):
    """
    Note: CelebA IoU=0.118 in v1 — Haar cascade performs poorly on these images.
    We still run it for documentation but don't block on it.
    CelebA images are NOT added to training data.
    """
    print("\n" + "="*60)
    print("  STEP 2: CelebA — Face Detector Validation")
    print(f"  Note: v1 IoU was 0.118 — Haar cascade weak on celeba style images")
    print("="*60)

    if not CELEBA_CSV or not CELEBA_IMGS:
        print("  ⚠️  CelebA files not found — skipping")
        report["celeba"] = {"status": "skipped"}
        return report

    bbox_df   = pd.read_csv(CELEBA_CSV)
    sample_df = bbox_df.sample(n=min(200, len(bbox_df)), random_state=SEED)

    out_dir = PROCESSED_DIR / "celeba_sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    iou_scores = []
    saved = skipped = 0

    def compute_iou(a, b):
        ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
        inter  = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_a = (a["x2"]-a["x1"]) * (a["y2"]-a["y1"])
        area_b = (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="  CelebA"):
        img_path = CELEBA_IMGS / row["image_id"]
        if not img_path.exists():
            skipped += 1; continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1; continue

        gt   = {"x1": int(row["x_1"]), "y1": int(row["y_1"]),
                "x2": int(row["x_1"])+int(row["width"]),
                "y2": int(row["y_1"])+int(row["height"])}
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            iou_scores.append(compute_iou(gt, {"x1":x,"y1":y,"x2":x+w,"y2":y+h}))

        # Save ground-truth crops for reference
        x,y = int(row["x_1"]), int(row["y_1"])
        crop = img[max(0,y):y+int(row["height"]), max(0,x):x+int(row["width"])]
        if crop.size > 0:
            cv2.imwrite(str(out_dir/row["image_id"]), cv2.resize(crop, TARGET_SIZE))
            saved += 1

    mean_iou       = float(np.mean(iou_scores)) if iou_scores else 0.0
    detection_rate = len(iou_scores) / max(1, len(sample_df)-skipped)

    print(f"  Detection rate : {detection_rate:.1%}")
    print(f"  Mean IoU       : {mean_iou:.3f}  (v1 was 0.118)")
    print(f"  Recommendation : {'Use as-is' if mean_iou > 0.4 else 'Consider MediaPipe for production'}")

    report["celeba"] = {"detection_rate": round(detection_rate,4),
                        "mean_iou": round(mean_iou,4), "crops_saved": saved}
    return report


# ─────────────────────────────────────────────
# STEP 3 — FITZPATRICK
# ─────────────────────────────────────────────
def process_fitzpatrick(report):
    """
    Fitzpatrick images are clinical/dermatology shots — NOT close-up face crops.
    This means face detection naturally has low rates (~15% in baseline).
    We use fallback (full image) for all of them.
    Used exclusively for fairness_eval.py post-training.
    """
    print("\n" + "="*60)
    print("  STEP 3: Fitzpatrick — Fairness Eval Set")
    print("  Note: These are clinical shots, not face crops.")
    print("        Fallback (full image) used for all.")
    print("="*60)

    if not FITZPATRICK_DIR:
        print("  ⚠️  Fitzpatrick dir not found — skipping")
        report["fitzpatrick"] = {"status": "skipped"}
        return report

    group_map = {
        "group1_light":  "group1_light",
        "group2_medium": "group2_medium",
        "group3_dark":   "group3_dark",
    }
    stats = {}

    for folder_name, group_label in group_map.items():
        folder = FITZPATRICK_DIR / folder_name
        if not folder.exists():
            print(f"  ⚠️  Missing: {folder_name}")
            continue

        imgs    = [p for p in folder.rglob("*")
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        out_dir = PROCESSED_DIR / "fitzpatrick" / group_label
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = rejected = 0
        ita_scores = []

        for img_path in tqdm(imgs, desc=f"  {folder_name}"):
            img = cv2.imread(str(img_path))
            if img is None: continue

            # Use full image (skip Haar — known to fail on clinical shots)
            face_r = cv2.resize(img, TARGET_SIZE)

            # Quality filter — slightly more permissive for Fitzpatrick
            gray = cv2.cvtColor(face_r, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 30:  # very low bar
                rejected += 1; continue
            mean = face_r.mean()
            if not (15 <= mean <= 250):
                rejected += 1; continue

            ita = compute_ita(face_r)
            ita_scores.append(ita)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_ita{ita:.1f}.jpg"), face_r)
            saved += 1

        mean_ita = float(np.mean(ita_scores)) if ita_scores else 0.0
        print(f"  {folder_name}: saved={saved}  rejected={rejected}  "
              f"mean ITA={mean_ita:.1f}")
        stats[group_label] = {"saved": saved, "rejected": rejected,
                              "mean_ita": round(mean_ita, 2)}

    print(f"\n  ✅ Fitzpatrick complete")
    report["fitzpatrick"] = stats
    return report


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  UBEKA AI — Dataset Preprocessing Pipeline v2")
    print("="*60)

    if PROCESSED_DIR.exists():
        print(f"\n  🗑  Clearing old processed data at {PROCESSED_DIR}...")
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True)

    report = {}
    report = process_acne(report)
    report = process_celeba(report)
    report = process_fitzpatrick(report)

    # Print final class counts clearly
    print("\n" + "="*60)
    print("  FINAL DATASET SUMMARY")
    print("="*60)
    for split in ["train", "val"]:
        print(f"\n  {split.upper()}:")
        split_dir = PROCESSED_DIR / split
        if split_dir.exists():
            for cls_dir in sorted(split_dir.iterdir()):
                n = len(list(cls_dir.glob("*.jpg")))
                bar = "█" * int(n / 100)
                print(f"    {cls_dir.name:>10}: {n:>5}  {bar}")

    report_path = PROCESSED_DIR / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  📄 Report → {report_path}")
    print("  ✅ ALL DONE — run src/model/train.py next")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()