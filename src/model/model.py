import torch
import torch.nn as nn
from torchvision import models

class AcneClassifier(nn.Module):
    """
    MobileNetV2 fine-tuned for 4-class acne severity classification.
    Pretrained on ImageNet — we freeze early layers and retrain the head.
    """

    def __init__(self, num_classes: int = 4, freeze_base: bool = True):
        super().__init__()
        # Load pretrained MobileNetV2
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        if freeze_base:
            # Freeze all feature extraction layers
            for param in self.base.features.parameters():
                param.requires_grad = False

        # Replace the classifier head
        in_features = self.base.last_channel  # 1280
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base(x)


def load_model(checkpoint_path: str, num_classes: int = 4, device: str = 'cpu'):
    """Load a saved model checkpoint."""
    model = AcneClassifier(num_classes=num_classes, freeze_base=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model