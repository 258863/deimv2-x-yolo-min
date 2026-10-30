"""DEIMv2 feature pyramid with an image-classification head."""

from collections.abc import Sequence

import torch
from torch import nn

from engine.backbone import DINOv3STAs

from .modules import STACDFFNReference, STASpatialControl, SemanticSpatialExchange, SemanticSpatialJointGate, DEGAxisAdapter


class DEIMv2SpectrogramClassifier(nn.Module):
    """Pool DEIMv2's three feature levels and predict one image-level class."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "dinov3_vits16",
        weights_path: str | None = None,
        interaction_indexes: Sequence[int] = (5, 8, 11),
        hidden_dim: int = 224,
        conv_inplane: int = 32,
        use_sta: bool = True,
        freeze_dinov3: bool = False,
        dropout: float = 0.2,
        sta_variant: str = "baseline",
        sta_bottleneck_dim: int = 64,
        sta_adapter_stages: Sequence[str] = ("conv3",),
        cross_branch_interaction: bool = False,
        module_dim: int = 64,
        interaction_direction: str = "bidirectional",
        guidance_max_ratio: float | None = None,
        interaction_type: str = "residual",
        interaction_position: str = "fusion",
        fusion_deg: bool = False,
        fusion_deg_direction: bool = True,
        fusion_deg_axis: bool = True,
        fusion_deg_gate: bool = True,
        fusion_deg_stage: str = "c3",
        fusion_deg_position: str = "post_fusion",
        fusion_deg_residual_scale: float = 1.0,
        fusion_deg_max_ratio: float | None = None,
        fusion_deg_direction_source: str = "projected",
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if fusion_deg_stage not in ("c3", "c4"):
            raise ValueError(f"Unknown fusion DEG stage: {fusion_deg_stage}")
        if fusion_deg_position not in ("pre_fusion", "post_fusion"):
            raise ValueError(f"Unknown fusion DEG position: {fusion_deg_position}")
        if fusion_deg_direction_source not in ("projected", "spatial"):
            raise ValueError(f"Unknown DEG direction source: {fusion_deg_direction_source}")
        if fusion_deg and fusion_deg_direction_source == "spatial" and not fusion_deg_direction:
            raise ValueError("Spatial direction source requires the direction encoder")
        if interaction_type not in ("residual", "joint_gate"):
            raise ValueError(f"Unknown interaction type: {interaction_type}")
        if interaction_position not in ("fusion", "pre_s1_ca"):
            raise ValueError(f"Unknown interaction position: {interaction_position}")
        if interaction_position == "pre_s1_ca":
            if (not cross_branch_interaction or interaction_type != "residual"
                    or interaction_direction not in ("semantic_to_spatial", "bidirectional") or sta_variant != "s1_ca"
                    or tuple(sta_adapter_stages) != ("conv3",)):
                raise ValueError("pre_s1_ca requires residual interaction and S1-CA at conv3 only")
        if interaction_type == "joint_gate":
            if not cross_branch_interaction or interaction_direction != "semantic_to_spatial":
                raise ValueError("joint_gate requires cross_branch_interaction and semantic_to_spatial")
            if guidance_max_ratio is not None:
                raise ValueError("guidance_max_ratio applies only to residual guidance, not joint_gate")

        self.backbone = DINOv3STAs(
            name=backbone_name,
            weights_path=weights_path,
            interaction_indexes=list(interaction_indexes),
            finetune=not freeze_dinov3,
            use_sta=use_sta,
            conv_inplane=conv_inplane,
            hidden_dim=hidden_dim,
        )
        self.backbone.interaction_position = interaction_position
        feature_dim = hidden_dim * 3
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)
        if sta_variant not in ("baseline", "r1", "s1", "s1_ca", "s1_x"):
            raise ValueError(f"Unknown STA variant: {sta_variant}")
        stages = tuple(sta_adapter_stages)
        allowed_stages = {"conv2", "conv3", "conv4"}
        if not stages or len(stages) != len(set(stages)) or not set(stages) <= allowed_stages:
            raise ValueError("sta_adapter_stages must contain unique conv2/conv3/conv4 names")
        self.sta_adapter_stages = stages
        if sta_variant != "baseline":
            if not use_sta:
                raise ValueError(f"{sta_variant.upper()} requires use_sta=True")
            adapter_class = STACDFFNReference if sta_variant == "r1" else STASpatialControl
            module_name = "cdffn_r1" if sta_variant == "r1" else "spatial_s1"
            with torch.random.fork_rng(devices=[]):
                adapter_options = {"channel_attention": True} if sta_variant == "s1_ca" else {}
                if sta_variant == "s1_x":
                    adapter_options = {"branch_exchange": True}
                channels = {"conv2": conv_inplane * 2, "conv3": conv_inplane * 4, "conv4": conv_inplane * 4}
                for stage in stages:
                    getattr(self.backbone.sta, stage).add_module(
                        module_name, adapter_class(channels[stage], sta_bottleneck_dim, **adapter_options)
                    )
        if cross_branch_interaction:
            if not use_sta:
                raise ValueError("Semantic-spatial interaction requires use_sta=True")
            with torch.random.fork_rng(devices=[]):
                if interaction_type == "joint_gate":
                    self.backbone.semantic_spatial_exchange = SemanticSpatialJointGate(
                        semantic_channels=self.backbone.dinov3.embed_dim,
                        spatial_channels=conv_inplane * 4,
                        hidden_channels=module_dim,
                    )
                else:
                    self.backbone.semantic_spatial_exchange = SemanticSpatialExchange(
                        semantic_channels=self.backbone.dinov3.embed_dim,
                        spatial_channels=conv_inplane * 4,
                        hidden_channels=module_dim,
                        direction=interaction_direction,
                        guidance_max_ratio=guidance_max_ratio,
                    )
        if fusion_deg:
            if not use_sta:
                raise ValueError("Fusion DEG requires use_sta=True")
            self.backbone.fusion_deg_stage = fusion_deg_stage
            self.backbone.fusion_deg_position = fusion_deg_position
            self.backbone.fusion_deg_direction_source = fusion_deg_direction_source
            fusion_conv = self.backbone.convs[1 if fusion_deg_stage == "c3" else 2]
            channels = fusion_conv.in_channels if fusion_deg_position == "pre_fusion" else fusion_conv.out_channels
            with torch.random.fork_rng(devices=[]):
                self.backbone.deg_axis_adapter = DEGAxisAdapter(
                    channels=channels, hidden_channels=module_dim,
                    use_direction=fusion_deg_direction, use_axis=fusion_deg_axis, use_gate=fusion_deg_gate,
                    residual_scale=fusion_deg_residual_scale, max_ratio=fusion_deg_max_ratio,
                )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return torch.cat([feature.mean(dim=(-2, -1)) for feature in features], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))
