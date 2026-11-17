"""직접 작성한 Zero-DCE 재현과 고정 필터 기반 개선 모듈."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('none', 'bright', 'noise', 'both')

@dataclass
class EnhancementConfig:
    g_min: float = 0.2
    kappa: float = 1.0
    noise_strength: float = 5.0
    max_gain: float = 10.0
    edge_threshold: float = 0.1
    mix_max: float = 0.7
    brightness_sigma: float = 3.0
    residual_sigma: float = 1.0
    smooth_sigma: float = 0.8

    def __post_init__(self):
        if not 0 <= self.g_min <= 1 or not 0 <= self.mix_max <= 1:
            raise ValueError('g_min과 mix_max는 0~1이어야 합니다.')
        if min(self.kappa, self.edge_threshold, self.brightness_sigma,
               self.residual_sigma, self.smooth_sigma) <= 0:
            raise ValueError('지수, 에지 임계값, sigma는 양수여야 합니다.')
        if self.noise_strength < 0 or self.max_gain < 1:
            raise ValueError('noise_strength >= 0, max_gain >= 1이어야 합니다.')

def luminance(x):
    return (x * x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)

def gaussian(x, sigma):
    radius = max(1, int(3 * sigma + 0.5))
    axis = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    weights = torch.exp(-axis.square() / (2 * sigma**2))
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None]
    return F.conv2d(F.pad(x, (radius,) * 4, mode='replicate'),
                    kernel.expand(x.shape[1], 1, -1, -1), groups=x.shape[1])

def edge_magnitude(y):
    kernel = y.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernels = torch.stack((kernel, kernel.T))[:, None]
    return F.conv2d(F.pad(y, (1,) * 4, mode='replicate'), kernels).square().sum(1, keepdim=True).add(1e-12).sqrt()

class ZeroDCE(nn.Module):
    def __init__(self, mode='none', config=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'지원 mode: {MODES}')
        self.mode = mode
        self.config = config or EnhancementConfig()
        channels = [(3,32),(32,32),(32,32),(32,32),(64,32),(64,32),(64,24)]
        self.layers = nn.ModuleList(nn.Conv2d(a,b,3,padding=1) for a,b in channels)
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0, 0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        a = F.relu(self.layers[0](x))
        b = F.relu(self.layers[1](a))
        c = F.relu(self.layers[2](b))
        d = F.relu(self.layers[3](c))
        e = F.relu(self.layers[4](torch.cat((c,d),1)))
        f = F.relu(self.layers[5](torch.cat((b,e),1)))
        curves = torch.tanh(self.layers[6](torch.cat((a,f),1)))
        enhanced = x
        # 공식 구현의 부호 convention: I + r * (I² - I).
        for r in curves.chunk(8, dim=1):
            enhanced = enhanced + r * (enhanced.square() - enhanced)
        cfg = self.config
        y = luminance(x)
        maps = {}
        if self.mode in ('bright','both'):
            brightness = gaussian(y, cfg.brightness_sigma)
            strength = cfg.g_min + (1-cfg.g_min)*(1-brightness).clamp(0,1).pow(cfg.kappa)
            enhanced = x + strength*(enhanced-x)
            maps['brightness'] = brightness
        if self.mode in ('noise','both'):
            residual = y - gaussian(y, cfg.residual_sigma)
            noise = F.avg_pool2d(F.pad(residual.square(),(1,)*4,mode='replicate'),3,1).add(1e-12).sqrt()
            edges = (edge_magnitude(gaussian(y,cfg.brightness_sigma))/cfg.edge_threshold).clamp(0,1)
            noise = noise*(1-edges)
            gain = (luminance(enhanced)/(y+1e-6)).clamp(1,cfg.max_gain)
            mix = (1-torch.exp(-cfg.noise_strength*gain*noise)).clamp(max=cfg.mix_max)
            enhanced = enhanced*(1-mix)+gaussian(enhanced,cfg.smooth_sigma)*mix
            maps.update(noise=noise, mix=mix)
        return enhanced, curves, maps
