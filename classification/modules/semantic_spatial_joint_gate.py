import torch
from torch import nn


class SemanticSpatialJointGate(nn.Module):
    """Jointly predict a positional gate and reweight only the spatial feature."""

    def __init__(
        self, semantic_channels: int = 384, spatial_channels: int = 128,
        hidden_channels: int = 64,
    ):
        super().__init__()
        if min(semantic_channels, spatial_channels, hidden_channels) < 1 or hidden_channels % 8:
            raise ValueError("Positive channel counts and hidden_channels divisible by 8 are required")
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
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1, bias=False),
        )
        self.strength = nn.Parameter(torch.zeros(()))

    def spatial_weights(self, semantic: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        projected = torch.cat((self.semantic_projection(semantic), self.spatial_projection(spatial)), dim=1)
        logits = self.gate(projected)
        # FP32 preserves small learned adjustments that would round to one in FP16.
        return 1 + self.strength.float() * logits.float().tanh()

    def forward(self, semantic: torch.Tensor, spatial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.spatial_weights(semantic, spatial)
        return semantic, spatial.float() * weights
