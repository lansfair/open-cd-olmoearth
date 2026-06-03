from __future__ import annotations

import torch
from mmengine.model import BaseModule
from mmseg.models.necks import MultiLevelNeck
from opencd.registry import MODELS


@MODELS.register_module()
class OLMoEarthFeatureFusionPyramid(BaseModule):
    """Fuse bi-temporal OLMoEarth features and build a feature pyramid.

    OLMoEarthBackbone returns one dense feature map. After bi-temporal fusion,
    MultiLevelNeck replicates that single map into the four scales expected by
    UPerHead.
    """

    def __init__(
        self,
        policy: str = "abs_diff",
        embed_dim: int = 768,
        out_channels: int | None = None,
        scales=(1, 0.5, 0.25, 0.125),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        out_indices=(0,),
    ) -> None:
        super().__init__()
        if policy not in ("concat", "sum", "diff", "abs_diff"):
            raise ValueError(f"Unsupported fusion policy: {policy}")
        self.policy = policy
        self.out_indices = out_indices
        fused_dim = embed_dim * 2 if policy == "concat" else embed_dim
        out_channels = out_channels or fused_dim
        self.pyramid = MultiLevelNeck(
            in_channels=[fused_dim],
            out_channels=out_channels,
            scales=list(scales),
            norm_cfg=norm_cfg,
        )

    def _fuse(self, x1, x2):
        if self.policy == "concat":
            return torch.cat([x1, x2], dim=1)
        if self.policy == "sum":
            return x1 + x2
        if self.policy == "diff":
            return x2 - x1
        return torch.abs(x1 - x2)

    def forward(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("Feature lists from both dates must match.")
        fused = tuple(self._fuse(x1[i], x2[i]) for i in self.out_indices)
        return self.pyramid(fused)
