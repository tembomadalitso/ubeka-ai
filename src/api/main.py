"""
main.py — Ubeka AI API (Final)
================================
Supports:
  POST /analyze        — single image (backward compatible)
  POST /analyze-batch  — multiple images at once
  GET  /health         — health check with model info

Response always includes:
  label, severity, confidence, uncertain, all_probabilities, note
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import os
import sys
import asyncio

import cv2
import numpy as np

# ── Imports ────────────────────────────────────────────────────────────────────
api_dir     = Path(__file__).resolve().parent
src_dir     = api_dir.parent
project_root = src_dir.parent

sys.path.insert(0, str(src_dir))

from inference.predict import load_model, predict_with_tta, predict_image

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Ubeka AI — Skin Analysis API",
    description="Acne severity classification focused on African skin tones",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/ubeka_uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ── Model paths (prefer calibrated) ───────────────────────────────────────────
CALIBRATED_PATH = project_root / "models" / "model.pth"
MODEL_PATH      = project_root / "models" / "model.pth"

active_model_path = CALIBRATED_PATH if CALIBRATED_PATH.exists() else MODEL_PATH

if not active_model_path.exists():
    print("⚠️  No model found — API will return 503 on /analyze")
    model = None
    model_info = {"loaded": False, "path": str(active_model_path)}
else:
    print(f"🚀 Loading model: {active_model_path.name}")
    model = load_model(str(active_model_path))
    model_type = "calibrated" if CALIBRATED_PATH.exists() else "standard"
    print(f"✅ Model loaded ({model_type})")
    model_info = {
        "loaded": True,
        "type":   model_type,
        "path":   str(active_model_path),
    }

# ── Face / skin validation ─────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def validate_image(image_path: str):
    """
    Light validation:
      - File is a readable image
      - Contains some detectable skin tone (very permissive HSV range)
    Returns (is_valid: bool, note: str)
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, "Cannot read image file"

    h, w = img.shape[:2]
    if h < 32 or w < 32:
        return False, "Image too small (minimum 32×32 px)"

    # Skin detection — broad HSV range covering all skin tones
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 15, 40]), np.array([30, 255, 255]))
    skin_ratio = np.sum(mask > 0) / (h * w)

    if skin_ratio < 0.015:
        return False, "No visible skin detected in image"

    # Face detection (non-blocking — we proceed even without a face)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    if len(faces) > 0:
        return True, "Face detected"
    else:
        return True, "No face detected — using full image for analysis"


def normalise_result(raw: dict, note: str = "") -> dict:
    """
    Normalise predict.py output to consistent API response shape.
    Supports both old (label) and new (severity) field names.
    """
    label = raw.get("severity") or raw.get("label") or "unknown"
    return {
        "skin_condition":    "acne",
        "label":             label,          # UI reads this
        "severity":          label,          # brief JSON format
        "severity_level":    raw.get("severity_level", -1),
        "confidence":        round(raw.get("confidence", 0), 4),
        "uncertain":         raw.get("uncertain", False),
        "all_probabilities": raw.get("all_probabilities", {}),
        "tta_passes":        raw.get("tta_passes", 1),
        "note":              note,
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/#home")
def root():
    return {
        "message": "Ubeka AI API is running",
        "version": "2.0.0",
        "endpoints": ["/analyze", "/analyze-batch", "/health"],
        "model": model_info,
    }


@app.get("/health")
def health():
    return {
        "status":       "ok" if model else "degraded",
        "model_loaded": model is not None,
        "model_info":   model_info,
        "upload_dir":   str(UPLOAD_DIR),
    }


@app.post("/analyze")
async def analyze_single(file: UploadFile = File(...)):
    """
    Single image analysis (backward compatible with original API).
    Returns JSON with label, confidence, all_probabilities.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG/PNG/WEBP)")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")

    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}.jpg"
    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        is_valid, note = validate_image(str(temp_path))
        if not is_valid:
            raise HTTPException(status_code=400, detail=note)

        raw    = predict_with_tta(model, str(temp_path))
        result = normalise_result(raw, note)
        print(f"[analyze] {file.filename} → {result['label']} ({result['confidence']:.3f})")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] /analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)


@app.post("/analyze-batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """
    Batch image analysis — up to 10 images at once.
    Returns list of results in same order as uploaded files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")

    results = []
    temp_paths = []

    try:
        # Save all files first
        for f in files:
            if not f.content_type or not f.content_type.startswith("image/"):
                results.append({
                    "filename": f.filename,
                    "ok": False,
                    "error": "Not an image file"
                })
                temp_paths.append(None)
                continue

            temp = UPLOAD_DIR / f"{uuid.uuid4()}.jpg"
            with open(temp, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            temp_paths.append(temp)
            results.append({"filename": f.filename, "ok": True})

        # Analyse
        for i, (result, temp) in enumerate(zip(results, temp_paths)):
            if not result["ok"] or temp is None:
                continue
            try:
                is_valid, note = validate_image(str(temp))
                if not is_valid:
                    results[i] = {**result, "ok": False, "error": note}
                    continue

                raw = predict_with_tta(model, str(temp))
                norm = normalise_result(raw, note)
                results[i] = {**result, **norm}
                print(f"[batch] {result['filename']} → {norm['label']} ({norm['confidence']:.3f})")

            except Exception as e:
                results[i] = {**result, "ok": False, "error": str(e)}

    finally:
        for t in temp_paths:
            if t and t.exists():
                os.remove(t)

    return {
        "total":      len(results),
        "successful": sum(1 for r in results if r.get("ok", False)),
        "results":    results,
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────
frontend_dir = project_root / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        index = frontend_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"error": "Frontend not found"}