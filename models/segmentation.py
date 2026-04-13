import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11BN

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c+skip_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
            
    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.shape[2]-x.shape[2]; dx = skip.shape[3]-x.shape[3]
        x  = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x, skip], 1))

class UNet(nn.Module):
    def __init__(self, num_seg_classes=3):
        super().__init__()
        bb    = VGG11BN(num_classes=37)
        feats = list(bb.features.children())
        self.enc1 = nn.Sequential(*feats[0:2])
        self.enc2 = nn.Sequential(*feats[2:4])
        self.enc3 = nn.Sequential(*feats[4:7])
        self.enc4 = nn.Sequential(*feats[7:10])
        self.enc5 = nn.Sequential(*feats[10:])
        self.up4  = UpBlock(512, 512, 256)
        self.up3  = UpBlock(256, 256, 128)
        self.up2  = UpBlock(128, 128, 64)
        self.up1  = UpBlock(64,  64,  32)
        self.head = nn.Conv2d(32, num_seg_classes, 1)
        
    def forward(self, x):
        s1 = self.enc1(x);  s2 = self.enc2(s1)
        s3 = self.enc3(s2); s4 = self.enc4(s3); s5 = self.enc5(s4)
        d  = self.up4(s5, s4); d = self.up3(d, s3)
        d  = self.up2(d,  s2); d = self.up1(d, s1)
        return self.head(d)
