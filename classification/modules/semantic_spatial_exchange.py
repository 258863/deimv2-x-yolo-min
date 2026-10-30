import math

import torch
from torch import nn


class SemanticSpatialExchange(nn.Module):
    def __init__(
        self, semantic_channels: int = 384, spatial_channels: int = 128,
        hidden_channels: int = 64, direction: str = "bidirectional",
        guidance_max_ratio: float | None = None,
    ):
        super().__init__()
        if min(semantic_channels, spatial_channels, hidden_channels) < 1 or hidden_channels % 8:
            raise ValueError("Positive channel counts and hidden_channels divisible by 8 are required")
        if direction not in ("bidirectional", "semantic_to_spatial"):
            raise ValueError(f"Unknown interaction direction: {direction}")
        if guidance_max_ratio is not None:
            if direction != "semantic_to_spatial" or not math.isfinite(guidance_max_ratio) or guidance_max_ratio <= 0:
                raise ValueError("guidance_max_ratio must be positive, finite and used with semantic_to_spatial")
        self.direction = direction
        self.guidance_max_ratio = guidance_max_ratio
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(spatial_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.to_semantic = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, semantic_channels, 1, bias=False),
        )
        self.to_spatial = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, spatial_channels, 1, bias=False),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))
        if direction == "semantic_to_spatial":
            # Keep historical initialization order for identical retained-path weights.
            del self.spatial_projection
            del self.to_semantic
            del self.alpha

    def forward(self, semantic: torch.Tensor, spatial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_projected = self.semantic_projection(semantic)
        if self.direction == "semantic_to_spatial":
            correction = self.to_spatial(semantic_projected)
            if self.guidance_max_ratio is None:
                return semantic, spatial + self.beta * correction
            # Bound each sample's C×H×W residual norm; keep norms/addition in FP32 under AMP.
            residual = self.beta.float() * correction.float()
            spatial_fp32 = spatial.float()
            residual_norm = torch.linalg.vector_norm(residual.flatten(1), dim=1)
            spatial_norm = torch.linalg.vector_norm(spatial_fp32.flatten(1), dim=1)
            scale = (self.guidance_max_ratio * spatial_norm / residual_norm.clamp_min(1e-12)).clamp(max=1)
            return semantic, spatial_fp32 + residual * scale[:, None, None, None]
        spatial_projected = self.spatial_projection(spatial)
        semantic_new = semantic + self.alpha * self.to_semantic(spatial_projected)
        spatial_new = spatial + self.beta * self.to_spatial(semantic_projected)
        return semantic_new, spatial_new
