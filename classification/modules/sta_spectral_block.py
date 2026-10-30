"""Pinned UDCNet CDFFN computation with a bottleneck residual STA adapter."""

import torch
from torch import nn
from torch.nn import functional

UPSTREAM_COMMIT = "ec76afee61fe4cdd4741ed2dde765137d782c4b7"
UPSTREAM_SOURCE_SHA256 = "276010d10b12be949085bf8ca3ea4d641b5a9e037f335bdf9c910265d4a49047"


class LocalEnhancementReference(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        for dilation in (3, 5, 7, 9, 12):
            self.add_module(
                f"Dconv{dilation}",
                nn.Sequential(
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm2d(channels),
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                ),
            )
        self.out = nn.Sequential(
            nn.Conv2d(channels * 5, channels * 3, 1),
            nn.BatchNorm2d(channels * 3),
            nn.Conv2d(channels * 3, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        previous = None
        for dilation in (3, 5, 7, 9, 12):
            branch_input = features if previous is None else features + previous
            previous = getattr(self, f"Dconv{dilation}")(branch_input)
            outputs.append(previous)
        return self.out(torch.cat(outputs, dim=1)) + features


class CDFFNReference(nn.Module):
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.dwconv0 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.project_out = nn.Conv2d(channels * 2, channels, 1, bias=bias)
        self.project_in = nn.Conv2d(channels, channels, 1, bias=bias)
        self.LE = LocalEnhancementReference(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.project_in(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            spectrum = torch.fft.fft2(projected.float(), dim=(-2, -1))
            amplitude = self.relu(self.norm(torch.abs(spectrum)))
            gated = functional.gelu(amplitude) * amplitude
            transformed = torch.fft.ifft2(torch.fft.fft2(gated.float(), dim=(-2, -1)), dim=(-2, -1))
            frequency_features = self.relu(self.norm(torch.abs(transformed)))
        spatial_features = self.dwconv0(projected)
        spatial_features = functional.gelu(spatial_features) * spatial_features
        spatial_features = self.LE(spatial_features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.project_out(torch.cat((frequency_features, spatial_features.float()), dim=1))


class STACDFFNReference(nn.Module):
    def __init__(self, channels: int = 128, bottleneck_dim: int = 64):
        super().__init__()
        if channels < 1 or bottleneck_dim < 1:
            raise ValueError("channels and bottleneck_dim must be positive")
        self.norm = nn.LayerNorm(channels, eps=1e-5)
        self.down = nn.Conv2d(channels, bottleneck_dim, 1, bias=False)
        self.ffn = CDFFNReference(bottleneck_dim, bias=False)
        self.up = nn.Conv2d(bottleneck_dim, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        enhanced = self.ffn(self.down(normalized))
        with torch.autocast(device_type=features.device.type, enabled=False):
            residual = self.up(enhanced.float())
        return features + residual.to(features.dtype)


class SpatialPoolingChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def channel_weights(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.project(features).flatten(2)
        with torch.autocast(device_type=features.device.type, enabled=False):
            attention = logits.float().softmax(dim=-1)
            pooled = (attention * features.float().flatten(2)).sum(dim=-1)
            weights = pooled.sigmoid().unsqueeze(-1).unsqueeze(-1)
        return weights.to(features.dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.channel_weights(features)


class BidirectionalSpatialExchange(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.to_spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.to_controlled = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, spatial: torch.Tensor, controlled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_new = spatial + self.alpha * self.to_spatial(controlled)
        controlled_new = controlled + self.beta * self.to_controlled(spatial)
        return spatial_new, controlled_new


class STASpatialControl(nn.Module):
    """Parameter-matched spatial control for the lightweight spectral adapter."""

    def __init__(
        self, channels: int = 128, bottleneck_dim: int = 64,
        channel_attention: bool = False, branch_exchange: bool = False,
    ):
        super().__init__()
        if channels < 1 or bottleneck_dim < 1:
            raise ValueError("channels and bottleneck_dim must be positive")
        if channel_attention and branch_exchange:
            raise ValueError("Channel attention and branch exchange require independent experiments")
        self.norm = nn.LayerNorm(channels, eps=1e-5)
        self.down = nn.Conv2d(channels, bottleneck_dim, 1, bias=False)
        self.spatial = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            3,
            padding=1,
            groups=bottleneck_dim,
            bias=False,
        )
        self.gate = nn.Conv2d(bottleneck_dim, bottleneck_dim, 1, bias=False)
        self.up = nn.Conv2d(bottleneck_dim * 2, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        with torch.random.fork_rng(devices=[]):
            self.channel_attention = (
                SpatialPoolingChannelAttention(bottleneck_dim) if channel_attention else nn.Identity()
            )
            self.exchange = BidirectionalSpatialExchange(bottleneck_dim) if branch_exchange else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        projected = self.down(normalized)
        spatial = self.spatial(projected)
        spatial = functional.gelu(spatial) * spatial
        spatial = self.channel_attention(spatial)
        controlled = projected * (1 + torch.tanh(self.gate(projected)))
        if self.exchange is not None:
            spatial, controlled = self.exchange(spatial, controlled)
        residual = self.up(torch.cat((spatial, controlled), dim=1))
        return features + residual
