"""Axial enhancement with optional spatial direction source and weak residual."""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def cell_histogram(theta: torch.Tensor, nbins: int = 36, cell_size: int = 8) -> torch.Tensor:
    """Count angle bins per cell, excluding padded pixels, then encode bin centers."""
    if theta.ndim != 4 or theta.shape[1] != 1 or nbins < 1 or cell_size < 1:
        raise ValueError("Expected B×1×H×W angles and positive bin/cell sizes")
    batch, _, height, width = theta.shape
    indices = (theta.float() * (nbins / torch.pi)).floor().long().clamp(0, nbins - 1)
    pad_h, pad_w = (-height) % cell_size, (-width) % cell_size
    valid = F.pad(torch.ones_like(theta, dtype=torch.float32), (0, pad_w, 0, pad_h))
    indices = F.pad(indices, (0, pad_w, 0, pad_h))
    grid_h, grid_w = (height + pad_h) // cell_size, (width + pad_w) // cell_size

    def arrange_cells(tensor):
        return tensor.reshape(batch, 1, grid_h, cell_size, grid_w, cell_size).permute(
            0, 2, 4, 1, 3, 5,
        ).reshape(batch, grid_h, grid_w, cell_size * cell_size)

    weights = arrange_cells(valid)
    counts = torch.zeros(batch, grid_h, grid_w, nbins, device=theta.device, dtype=torch.float32)
    counts.scatter_add_(-1, arrange_cells(indices), weights)
    hist = counts / weights.sum(-1, keepdim=True).clamp_min(1)
    centers = (torch.arange(nbins, device=theta.device).float() + 0.5) * (torch.pi / nbins)
    return (hist * centers).permute(0, 3, 1, 2).contiguous()


class EdgeConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("EdgeConv channels must be positive and even")
        half = channels // 2
        self.in_proj = nn.Conv2d(channels, half, 1, bias=False)
        self.horizontal = nn.Conv2d(half, half, (1, 3), padding=(0, 1), groups=half, bias=False)
        self.vertical = nn.Conv2d(half, half, (3, 1), padding=(1, 0), groups=half, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        projected = self.in_proj(feature)
        return self.out_proj(torch.cat((self.horizontal(projected), self.vertical(projected)), dim=1))


class DirectionEncoder(nn.Module):
    def __init__(self, hidden_channels: int = 64, nbins: int = 36, cell_size: int = 8):
        super().__init__()
        if hidden_channels < 8 or hidden_channels % 8 or nbins < 1 or cell_size < 1:
            raise ValueError("Hidden channels must be divisible by 8; bin/cell sizes must be positive")
        self.nbins, self.cell_size = nbins, cell_size
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2).contiguous())
        self.encode = nn.Sequential(
            nn.Conv2d(nbins, hidden_channels, 1, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def histogram(self, feature: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type=feature.device.type, enabled=False):
            gray = F.pad(feature.float().mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="replicate")
            dx = F.conv2d(gray, self.sobel_x.float())
            dy = F.conv2d(gray, self.sobel_y.float())
            theta = torch.atan2(dy, dx).abs()
            return cell_histogram(theta, self.nbins, self.cell_size)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.encode(self.histogram(feature).to(feature.dtype))


class DEGAxisAdapter(nn.Module):
    def __init__(
        self, channels: int = 224, hidden_channels: int = 64, nbins: int = 36,
        cell_size: int = 8, use_direction: bool = True, use_axis: bool = True, use_gate: bool = True,
        residual_scale: float = 1.0, max_ratio: float | None = None,
    ):
        super().__init__()
        if channels < 1 or hidden_channels < 8 or hidden_channels % 8 or nbins < 1 or cell_size < 1:
            raise ValueError("Positive dimensions and hidden_channels divisible by 8 are required")
        if not 0 < residual_scale <= 1 or (max_ratio is not None and not 0 < max_ratio <= 1):
            raise ValueError("Residual scale and optional norm ratio must be in (0, 1]")
        self.use_direction, self.use_axis, self.use_gate = use_direction, use_axis, use_gate
        self.residual_scale, self.max_ratio = residual_scale, max_ratio
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.value_edge = EdgeConv(hidden_channels)
        self.value_norm = nn.GroupNorm(8, hidden_channels)
        if use_direction:
            self.direction_encoder = DirectionEncoder(hidden_channels, nbins, cell_size)
        if use_axis:
            self.axis_down = nn.Conv1d(hidden_channels, hidden_channels // 2, 1)
            self.axis_up = nn.Conv1d(hidden_channels // 2, hidden_channels, 1)
        if use_gate:
            self.gate_edge = EdgeConv(hidden_channels)
            self.gate_norm = nn.GroupNorm(8, hidden_channels)
        self.up = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def axis_attention(self, feature: torch.Tensor) -> torch.Tensor:
        """Return per-channel FP32 weights normalized over H×W, not channels."""
        height, width = feature.shape[-2:]
        summaries = torch.cat((feature.mean(dim=3), feature.mean(dim=2)), dim=2)
        encoded = self.axis_up(F.gelu(self.axis_down(summaries)))
        height_encoded, width_encoded = encoded.split((height, width), dim=2)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            logits = height_encoded.float().unsqueeze(3) * width_encoded.float().unsqueeze(2)
            return logits.flatten(2).softmax(dim=2)

    def axis_weight(self, feature: torch.Tensor) -> torch.Tensor:
        attention = self.axis_attention(feature)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            summary = (attention * feature.float().flatten(2)).sum(dim=2)
            return (2 * summary.sigmoid()).unsqueeze(2).unsqueeze(3).to(feature.dtype)

    def forward(self, feature: torch.Tensor, direction_source: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        projected = self.down(normalized)
        if direction_source is not None and (
            direction_source.ndim != 4 or direction_source.shape[0] != feature.shape[0]
            or direction_source.shape[-2:] != feature.shape[-2:]
        ):
            raise ValueError("Direction source must match the feature batch and spatial size")
        direction_feature = projected if direction_source is None else direction_source
        epsilon = self.direction_encoder(direction_feature) if self.use_direction else torch.zeros_like(projected[:, :, :1, :1])
        conditioned = projected + epsilon
        if self.use_axis:
            axis_gate = self.axis_weight(conditioned)
        else:
            with torch.autocast(device_type=feature.device.type, enabled=False):
                axis_gate = (2 * conditioned.float().mean((2, 3), keepdim=True).sigmoid()).to(projected.dtype)
        if self.use_gate:
            logits = self.gate_norm(self.gate_edge(conditioned))
            with torch.autocast(device_type=feature.device.type, enabled=False):
                direction_gate = logits.float().sigmoid().to(projected.dtype)
        else:
            direction_gate = torch.ones_like(projected)
        value = self.value_norm(self.value_edge(projected))
        residual = self.up(value * direction_gate * axis_gate)
        if self.residual_scale != 1:
            residual = residual * self.residual_scale
        if self.max_ratio is not None:
            with torch.autocast(device_type=feature.device.type, enabled=False):
                source_norm = feature.float().flatten(1).norm(dim=1)
                residual_norm = residual.float().flatten(1).norm(dim=1).clamp_min(1e-12)
                factor = (self.max_ratio * source_norm / residual_norm).clamp(max=1)
            residual = residual * factor[:, None, None, None].to(residual.dtype)
        return feature + residual
