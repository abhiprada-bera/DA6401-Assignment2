import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Task 1.1: Custom Dropout Module
# ─────────────────────────────────────────────────────────────────────────────
class CustomDropout(nn.Module):
    """
    Inverted dropout implemented from first principles.
    Does NOT use nn.Dropout or F.dropout.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0,1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        mask  = (torch.rand_like(x) > self.p).float()
        scale = 1.0 / (1.0 - self.p)
        return x * mask * scale

    def extra_repr(self):
        return f"p={self.p}"

# ─────────────────────────────────────────────────────────────────────────────
# Task 1.1: Standard VGG11 Architecture from Scratch
# ─────────────────────────────────────────────────────────────────────────────
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

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


# ─────────────────────────────────────────────────────────────────────────────
# Task 1.2: Custom Intersection over Union (IoU) Loss module
# ─────────────────────────────────────────────────────────────────────────────
class CustomIoULoss(nn.Module):
    """
    Computes Intersection over Union (IoU) Loss for bounding box regression.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        # pred, gt: (N, 4) normalised [x1,y1,x2,y2] tensors.
        xi1 = torch.max(pred[:,0], gt[:,0])
        yi1 = torch.max(pred[:,1], gt[:,1])
        xi2 = torch.min(pred[:,2], gt[:,2])
        yi2 = torch.min(pred[:,3], gt[:,3])
        
        inter = (xi2-xi1).clamp(min=0) * (yi2-yi1).clamp(min=0)
        area_p = (pred[:,2]-pred[:,0]).clamp(min=0) * (pred[:,3]-pred[:,1]).clamp(min=0)
        area_g = (gt[:,2]-gt[:,0]).clamp(min=0)   * (gt[:,3]-gt[:,1]).clamp(min=0)
        
        union  = area_p + area_g - inter
        iou = inter / (union + self.eps)
        
        # Loss is defined as 1 - IoU
        loss = 1.0 - iou
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1.4: Unified Multi-Task Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskPerceptionModel(nn.Module):
    """
    Integrates the components into a single multi-task learning architecture.
    Provides three outputs in a single forward pass:
      1. Breed Label (37-class logits)
      2. Bounding Box (4 continuous coords regression)
      3. Segmentation Mask (Dense, pixel-wise spatial map)
    """
    def __init__(self, classifier_path='saved_models/classifier.pth', localizer_path='saved_models/localizer.pth', unet_path='saved_models/unet.pth', num_classes=37, num_seg_classes=3):
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
            conv_block(3,   64),  nn.MaxPool2d(2, 2),    # block 1: pool idx 1
            conv_block(64,  128), nn.MaxPool2d(2, 2),    # block 2: pool idx 3
            conv_block(128, 256), conv_block(256, 256), nn.MaxPool2d(2, 2), # block 3: pool idx 6
            conv_block(256, 512), conv_block(512, 512), nn.MaxPool2d(2, 2), # block 4: pool idx 9
            conv_block(512, 512), conv_block(512, 512), nn.MaxPool2d(2, 2)  # block 5: pool idx 12
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # HEAD 1: Classification (Task 1) - directly from VGG11
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

        # HEAD 2: Bounding Box Regression (Task 2)
        # Attached to the encoder's pooled core
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),
            nn.Linear(1024, 4),
            nn.Sigmoid(),
        )

        # HEAD 3: Semantic Segmentation (Task 3)
        # Using a U-net expansive path
        class UpBlock(nn.Module):
            def __init__(self, in_c, skip_c, out_c):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2)
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
                )
            def forward(self, x, skip):
                x_up = self.up(x)
                diffY = skip.size()[2] - x_up.size()[2]
                diffX = skip.size()[3] - x_up.size()[3]
                x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
                return self.conv(torch.cat([x_up, skip], dim=1))

        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64,  64,  32)
        self.seg_head = nn.Conv2d(32, num_seg_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
        # Forward pass returning the three respective tasks
        
        # 1. ENCODER PASS with intermediate skips
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
        
        # 2. BOTTLENECK / CLASSIFICATION + BBOX
        pool = self.avgpool(f)
        logits = self.classifier(pool)
        bbox   = self.bbox_head(pool)
        
        # 3. DECODER PASS for Segmentation Features
        # (Learnable Upsampling + concatenation)
        d = self.up4(f, skips[3])
        d = self.up3(d, skips[2])
        d = self.up2(d, skips[1])
        d = self.up1(d, skips[0])
        
        # Final head to get to required mask dimensions
        mask = self.seg_head(d)
        
        # Depending on input size, there might be a final interpolation required to match the exact original image size
        # if the upsampling doesn't inherently put it exactly there due to pooling parities.
        mask = F.interpolate(mask, size=(max(x.shape[2], mask.shape[2]), max(x.shape[3], mask.shape[3])), mode="bilinear", align_corners=False)
        mask = mask[:, :, :x.shape[2], :x.shape[3]] # Crop exact size

        return logits, bbox, mask
