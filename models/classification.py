import torch.nn as nn
from .vgg11 import VGG11BN

class VGG11Classification(nn.Module):
    """
    VGG11-BN specifically configured for image classification.
    """
    def __init__(self, num_classes=37, drop_p=0.5):
        super().__init__()
        self.model = VGG11BN(num_classes=num_classes, drop_p=drop_p)

    def forward(self, x):
        return self.model(x)
