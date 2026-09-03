from __future__ import annotations

import torch
from mmengine.model import BaseModule
from mmseg.models.necks import MultiLevelNeck
from opencd.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class UniverSatFeatureFusionPyramid(BaseModule):
    """Fuse two UniverSat dense maps and construct a UPerNet pyramid."""

    def __init__(
        self,
        policy: str = "abs_diff",
        embed_dim: int = 768,
        out_channels: int | None = None,
        scales=(4, 2, 1, 0.5),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        out_indices=(0,),
    ) -> None:
        super().__init__()
        if policy not in ("concat", "sum", "diff", "abs_diff"):
            raise ValueError(f"Unsupported fusion policy: {policy}")
        self.policy = policy
        self.out_indices = tuple(out_indices)
        fused_dim = embed_dim * 2 if policy == "concat" else embed_dim
        out_channels = out_channels or fused_dim
        self.pyramid = MultiLevelNeck(
            in_channels=[fused_dim],
            out_channels=out_channels,
            scales=list(scales),
            norm_cfg=norm_cfg,
        )

    def _fuse(self, from_feature: Tensor, to_feature: Tensor) -> Tensor:
        if self.policy == "concat":
            return torch.cat([from_feature, to_feature], dim=1)
        if self.policy == "sum":
            return from_feature + to_feature
        if self.policy == "diff":
            return to_feature - from_feature
        return torch.abs(from_feature - to_feature)

    def forward(self, from_features, to_features):
        if len(from_features) != len(to_features):
            raise ValueError("Feature lists from both observations must match.")
        fused = tuple(
            self._fuse(from_features[index], to_features[index])
            for index in self.out_indices
        )
        return self.pyramid(fused)
