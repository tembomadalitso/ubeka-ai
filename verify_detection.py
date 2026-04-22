"""
verify_detection.py
Run from ubeka-ai/ root:
    python verify_detection.py

What this does:
- Scans a sample of images from each acne level
- Runs face detection on each
- Reports detection rate per level
- Saves a visual grid of results to verify_output/
"""

import cv2
import numpy as np
from pathlib import Path
import os, random

# ── Config ────────────────────────────────────────────────────────────────────
ACNE_ROOT = Path("data/raw/acne/acne_1024")         # adjust if your path differs
OUTPUT_DIR = Path("verify_output")
SAMPLES_PER_CLASS = 5                       # how many images to test per level
TARGET_SIZE = (224, 224)

FOLDERS = {
    0: "acne0_1024",
    1: "acne1_1024",
    2: "acne2_1024",
    3: "acne3_1024",
}

SEVERITY_NAMES = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}
# ─────────────────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

OUTPUT_DIR.mkdir(exist_ok=True)


def try_detect(img_bgr):
    """Returns (face_crop_rgb, method_used, bbox_or_None)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.1 * min(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        y2 = min(img_bgr.shape[0], y + h + pad)
        crop = img_bgr[y1:y2, x1:x2]
        return cv2.cvtColor(cv2.resize(crop, TARGET_SIZE), cv2.COLOR_BGR2RGB), "cascade", (x1,y1,x2,y2)

    # Fallback: return full image
    fallback = cv2.cvtColor(cv2.resize(img_bgr, TARGET_SIZE), cv2.COLOR_BGR2RGB)
    return fallback, "fallback_full_image", None


def draw_result_tile(img_bgr, method, bbox, label_text):
    """Draw detection result on a copy of the image for visualization."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Draw bounding box if detected
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label banner at top
    color = (0, 200, 0) if method == "cascade" else (0, 165, 255)
    cv2.rectangle(vis, (0, 0), (w, 28), color, -1)
    cv2.putText(vis, label_text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def run_verification():
    print("\n" + "="*55)
    print("  UBEKA AI — Face Detection Verification")
    print("="*55)

    all_results = {}   # level -> list of {"path", "method", "detected"}
    grid_tiles = []    # for the output image grid

    for level, folder_name in FOLDERS.items():
        folder_path = ACNE_ROOT / folder_name
        if not folder_path.exists():
            print(f"\n⚠️  Folder not found: {folder_path}  — skipping level {level}")
            continue

        # Gather all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']:
            images.extend(folder_path.glob(ext))

        if not images:
            print(f"\n⚠️  No images in {folder_path}")
            continue

        # Sample randomly
        sample = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
        level_results = []

        print(f"\nLevel {level} — {SEVERITY_NAMES[level].upper()} ({folder_name})")
        print(f"  Total images in folder: {len(images)}")
        print(f"  Testing {len(sample)} samples:")

        for img_path in sample:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"    ✗ Could not read: {img_path.name}")
                continue

            face_rgb, method, bbox = try_detect(img_bgr)
            detected = method == "cascade"
            status = "✅ detected" if detected else "⚠️  fallback"
            print(f"    {status}  {img_path.name}  [{img_bgr.shape[1]}x{img_bgr.shape[0]}]")

            level_results.append({
                "path": str(img_path),
                "method": method,
                "detected": detected
            })

            # Build tile for grid
            label = f"L{level} {SEVERITY_NAMES[level]} | {status.strip()}"
            tile = draw_result_tile(img_bgr, method, bbox, label)
            tile_resized = cv2.resize(tile, (200, 200))
            grid_tiles.append(tile_resized)

        # Per-level summary
        n_detected = sum(1 for r in level_results if r["detected"])
        rate = n_detected / len(level_results) * 100 if level_results else 0
        print(f"  → Detection rate: {n_detected}/{len(level_results)} ({rate:.0f}%)")
        all_results[level] = level_results

    # ── Save visual grid ──────────────────────────────────────────────────────
    if grid_tiles:
        cols = SAMPLES_PER_CLASS
        rows = (len(grid_tiles) + cols - 1) // cols
        # Pad to fill grid
        while len(grid_tiles) < rows * cols:
            grid_tiles.append(np.zeros((200, 200, 3), dtype=np.uint8))

        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.hstack(grid_tiles[r*cols:(r+1)*cols]))
        grid = np.vstack(row_imgs)
        out_path = OUTPUT_DIR / "detection_results.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"\n📸 Visual grid saved → {out_path}")

    # ── Overall summary ───────────────────────────────────────────────────────
    all_detected = sum(r["detected"] for results in all_results.values() for r in results)
    all_total = sum(len(results) for results in all_results.values())

    print("\n" + "="*55)
    print(f"  OVERALL DETECTION RATE: {all_detected}/{all_total} "
          f"({all_detected/all_total*100:.0f}%)" if all_total else "  No images tested.")

    if all_total and (all_detected / all_total) < 0.5:
        print("\n  ⚠️  LOW DETECTION RATE — possible causes:")
        print("     • Images are close-up crops (face fills entire frame)")
        print("     • Low contrast / dark skin tone (Haar cascade bias)")
        print("     • Try: lower minNeighbors to 3, or minSize to (30,30)")
        print("     • Fallback (full image) will be used for missed detections")
        print("     • This is OKAY for Acne04 — images are pre-cropped faces")
    else:
        print("\n  ✅ Detection looks good. Ready to train!")

    print("="*55 + "\n")


if __name__ == "__main__":
    run_verification()