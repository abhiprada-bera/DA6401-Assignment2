import torch
import torch.nn as nn

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

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
