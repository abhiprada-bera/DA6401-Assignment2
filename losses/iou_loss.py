import torch
import torch.nn as nn

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
