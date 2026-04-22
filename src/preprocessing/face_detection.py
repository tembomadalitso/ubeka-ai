import cv2
import numpy as np
from pathlib import Path

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_and_crop_face(image_path: str, target_size: tuple = (224, 224)):
    """
    Detect face in image, crop it, resize to target_size.
    Returns: numpy array (H, W, C) in RGB, or None if no face found.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        # Fallback: use full image (acne dataset images are already face-cropped)
        face_crop = img
    else:
        # Take the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add 10% padding around face
        pad = int(0.1 * min(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        face_crop = img[y1:y2, x1:x2]

    # Resize and convert BGR -> RGB
    face_resized = cv2.resize(face_crop, target_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    return face_rgb


def crop_with_celeba_bbox(image_path: str, bbox: dict, target_size=(224, 224)):
    """
    Use CelebA's pre-computed bounding box for precise face crop.
    bbox: dict with keys x_1, y_1, width, height
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    x, y, w, h = bbox['x_1'], bbox['y_1'], bbox['width'], bbox['height']
    face_crop = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, target_size)
    return cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)