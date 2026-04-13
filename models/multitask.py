import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import conv_block, CustomDropout
from .segmentation import UpBlock

class MultiTaskPerceptionModel(nn.Module):
    """
    Integrates the components into a single multi-task learning architecture.
    Provides three outputs in a single forward pass:
      1. Breed Label (37-class logits)
      2. Bounding Box (4 continuous coords regression)
      3. Segmentation Mask (Dense, pixel-wise spatial map)
    """
    def __init__(self, classifier_path='saved_models/classifier.pth', 
                 localizer_path='saved_models/localizer.pth', 
                 unet_path='saved_models/unet.pth', 
                 num_classes=37, num_seg_classes=3):
        import gdown
        import os
        # Ensure directory exists
        os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
        # Download weights if not present (optional, but gdown handles overwrite/skip)
        gdown.download(id="1JpGQQj9k18zDJaWHUPqIqcuWWWoLVVC3", output=classifier_path, quiet=False)
        gdown.download(id="1cmDEkZT442HOWlWD140WwypoVpwh5APS", output=localizer_path, quiet=False)
        gdown.download(id="1Srbbhglqkude-ix2f56pL3LJeuLfm9sN", output=unet_path, quiet=False)
        
        super().__init__()
        # Backbone (Shared) VGG11 Features
        self.features = nn.Sequential(
            conv_block(3,   64),  nn.MaxPool2d(2, 2),    # block 1
            conv_block(64,  128), nn.MaxPool2d(2, 2),    # block 2
            conv_block(128, 256), conv_block(256, 256), nn.MaxPool2d(2, 2), # block 3
            conv_block(256, 512), conv_block(512, 512), nn.MaxPool2d(2, 2), # block 4
            conv_block(512, 512), conv_block(512, 512), nn.MaxPool2d(2, 2)  # block 5
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),
            nn.Linear(4096, num_classes),
        )

        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),
            nn.Linear(1024, 4),
            nn.Sigmoid(),
        )

        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64,  64,  32)
        self.seg_head = nn.Conv2d(32, num_seg_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = x
        skips = []
        for layer in self.features[0:2]: f = layer(f)  # block 1
        skips.append(f)
        for layer in self.features[2:4]: f = layer(f)  # block 2
        skips.append(f)
        for layer in self.features[4:7]: f = layer(f)  # block 3
        skips.append(f)
        for layer in self.features[7:10]: f = layer(f) # block 4
        skips.append(f)
        for layer in self.features[10:13]: f = layer(f)# block 5
        
        pool = self.avgpool(f)
        logits = self.classifier(pool)
        bbox   = self.bbox_head(pool)
        
        d = self.up4(f, skips[3])
        d = self.up3(d, skips[2])
        d = self.up2(d, skips[1])
        d = self.up1(d, skips[0])
        mask = self.seg_head(d)
        
        mask = F.interpolate(mask, size=(max(x.shape[2], mask.shape[2]), max(x.shape[3], mask.shape[3])), mode="bilinear", align_corners=False)
        mask = mask[:, :, :x.shape[2], :x.shape[3]] 
        return logits, bbox, mask
