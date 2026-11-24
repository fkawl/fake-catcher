import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def spectrum(rgb):
    """Per-image normalized log amplitude; DC is removed. RGB is in [0,1]."""
    gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    value = torch.log1p(torch.fft.fftshift(torch.fft.fft2(gray, norm='ortho'), dim=(-2, -1)).abs())
    h, w = value.shape[-2:]
    yy = torch.arange(h, device=rgb.device) - h // 2
    xx = torch.arange(w, device=rgb.device) - w // 2
    mask = (yy[:, None].square() + xx[None, :].square()) >= (min(h, w) * .03) ** 2
    value = value * mask
    return value / value.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class Detector(nn.Module):
    def __init__(self, variant='rgb_fft', pretrained=False):
        super().__init__()
        if variant not in {'rgb', 'rgb_fft'}:
            raise ValueError('variant must be rgb or rgb_fft')
        self.variant = variant
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.spatial = mobilenet_v3_small(weights=weights).features
        self.frequency = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        ) if variant == 'rgb_fft' else None
        self.head = nn.Sequential(nn.Linear(576 + (64 if self.frequency else 0), 128),
                                  nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, rgb, return_features=False):
        feature = self.spatial((rgb - self.mean) / self.std)
        pooled = feature.mean((-2, -1))
        if self.frequency is not None:
            pooled = torch.cat([pooled, self.frequency(spectrum(rgb)).flatten(1)], 1)
        logits = self.head(pooled).squeeze(1)
        return (logits, feature) if return_features else logits
