from torchvision import transforms

# ── Standard training transforms ──────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Validation — no augmentation ──────────────────────────────────────────────
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Dark skin specific transforms ─────────────────────────────────────────────
# Applied to images with ITA <= -30 (group3_dark) during training.
# More aggressive brightness/contrast jitter to create variety within
# the narrow luminance range of dark skin images.
dark_skin_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.5,   # wider than standard 0.3
        contrast=0.5,
        saturation=0.4,
        hue=0.1
    ),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Label map ─────────────────────────────────────────────────────────────────
SEVERITY_LABELS = {
    0: "clear",
    1: "mild",
    2: "moderate",
    3: "severe"
}

CONFIDENCE_THRESHOLD = 0.60