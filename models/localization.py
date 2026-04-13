import torch
import torch.nn as nn
from .vgg11 import VGG11BN
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    """
    VGG11-BN backbone with a 4-output regression head for bounding boxes.
    """
    def __init__(self, num_classes=37):
        super().__init__()
        backbone       = VGG11BN(num_classes=num_classes)
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),
            nn.Linear(1024, 4),
            nn.Sigmoid()  # Normalised [0, 1] coordinates
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.bbox_head(x)
