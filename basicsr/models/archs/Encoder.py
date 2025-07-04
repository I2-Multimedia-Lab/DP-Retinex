import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAttention(nn.Module):
    """
    Memory-efficient attention mechanism
    """

    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)

        dots = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class IlluminationAwareBlock(nn.Module):
    """
    Specialized block for illumination feature extraction
    """

    def __init__(self, dim, reduction_ratio=4):
        super().__init__()

        # Local feature extraction
        self.conv1 = nn.Conv2d(dim, dim // reduction_ratio, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim // reduction_ratio, dim // reduction_ratio, 3, padding=1, groups=dim // reduction_ratio),
            nn.BatchNorm2d(dim // reduction_ratio),
            nn.GELU()
        )

        # Illumination-aware attention
        self.attention = EfficientAttention(dim // reduction_ratio, heads=4)

        # Channel expansion
        self.conv3 = nn.Conv2d(dim // reduction_ratio, dim, 1)

        # Spatial pooling branch
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x

        # Local processing
        x = self.conv1(x)
        x = self.conv2(x)

        # Global attention
        x = self.attention(x)
        x = self.conv3(x)

        # Illumination-aware modulation
        x = x * self.pool_branch(residual)

        # Residual connection
        return residual + x * self.gamma


class EfficientIlluminationDecoder(nn.Module):
    """
    Efficient decoder for extracting illumination features
    Input: B×3×H×W
    Output: B×C (where C is hidden_dim)
    """

    def __init__(self, in_channels=3, hidden_dim=256, num_blocks=3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Progressive feature refinement
        self.blocks = nn.ModuleList([
            IlluminationAwareBlock(64) for _ in range(num_blocks)
        ])

        # Efficient spatial reduction
        self.reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(64, 128, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Final projection to hidden dimension
        self.final_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial feature extraction
        B, C, H, W = x.shape
        if (H*W) > (512*512):
             x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)  # B×C×H/2×W/2

        x = self.input_proj(x)  # B×64×H/4×W/4

        # Illumination feature refinement
        for block in self.blocks:
            x = block(x)

        # Spatial reduction and final projection
        x = self.reduction(x)  # B×128×1×1
        x = self.final_proj(x)  # B×hidden_dim

        return x

class denoise(nn.Module):
    def __init__(self, hidden_dim, feats=64, timesteps=4):
        super(denoise, self).__init__()
        self.max_period = timesteps * 10
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, hidden_dim),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x, t, c):
        t = t.float()
        t = t / self.max_period
        t = t.view(-1, 1)
        fea = self.mlp(torch.cat([x, t, c], dim=1))
        return fea


