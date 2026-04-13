import torch
import torch.nn as nn
from .layers import conv_block, CustomDropout

class VGG11BN(nn.Module):
    """
    VGG11 built entirely from scratch using torch.nn primitives.
    """
    def __init__(self, num_classes: int = 37, drop_p: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3,   64),  nn.MaxPool2d(2, 2),
            conv_block(64,  128), nn.MaxPool2d(2, 2),
            conv_block(128, 256), conv_block(256, 256), nn.MaxPool2d(2, 2),
            conv_block(256, 512), conv_block(512, 512), nn.MaxPool2d(2, 2),
            conv_block(512, 512), conv_block(512, 512), nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(drop_p),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(drop_p),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))
