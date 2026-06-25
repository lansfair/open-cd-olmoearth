from __future__ import annotations

import re
from typing import Any

import torch
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from opencd.registry import MODELS
from torch import Tensor

from ..utils import build_olmoearth_model, get_modality_bands, get_sample_field
from .olmoearth_backbone import _import_olmoearth_types, _normalize_band_name


@MODELS.register_module()
class OlmoEarthMultiModalBackbone(BaseModule):
    """Dense OLMoEarth encoder that fuses two modalities in one forward pass."""

    def __init__(
        self,
        model_config_path: str,
        from_modality: str = "sentinel2_l2a",
        to_modality: str = "sentinel1",
        patch_size: int = 16,
        num_timesteps: int = 1,
        out_channels: int = 768,
        fusion_policy: str = "abs_diff",
        fast_pass: bool | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if fusion_policy not in {"abs_diff", "diff", "sum", "concat"}:
            raise ValueError(f"Unsupported fusion_policy: {fusion_policy}")
        self.from_modality = from_modality
        self.to_modality = to_modality
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self.out_channels = out_channels
        self.fusion_policy = fusion_policy
        self.fast_pass = fast_pass
        self.from_band_names = list(get_modality_bands(from_modality))
        self.to_band_names = list(get_modality_bands(to_modality))
        self.from_sample_field = get_sample_field(from_modality)
        self.to_sample_field = get_sample_field(to_modality)
        self.model = build_olmoearth_model(model_config_path)
        self.encoder = self.model.encoder
        self.encoder.remove_masked_tokens = self._remove_masked_tokens_sort_compat
        self._freeze_unused_pretrain_parameters()
        self._from_metainfo: list[dict[str, Any]] | None = None
        self._to_metainfo: list[dict[str, Any]] | None = None

    @staticmethod
    def _remove_masked_tokens_sort_compat(
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sortable_mask = mask
        if mask.dtype == torch.bool:
            sortable_mask = mask.to(torch.uint8)
        sorted_mask, indices = torch.sort(
            sortable_mask,
            dim=1,
            descending=True,
            stable=True,
        )
        sorted_mask = sorted_mask.to(torch.bool)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)
        seq_lengths = sorted_mask.sum(-1)
        max_length = seq_lengths.max()
        x = x[:, :max_length]
        updated_mask = sorted_mask[:, :max_length]
        return x, indices, updated_mask, seq_lengths, max_length

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "OLMoEarth init_cfg checkpoint must be a state_dict or "
                "contain 'state_dict'/'model'."
            )
        cleaned = {}
        for key, value in checkpoint.items():
            key = re.sub(r"^(module\.)+", "", key)
            key = re.sub(r"^(model\.)+", "", key)
            cleaned[key] = value
        return cleaned

    def init_weights(self) -> None:
        if self.init_cfg is None:
            return
        if not isinstance(self.init_cfg, dict):
            raise TypeError("OlmoEarthMultiModalBackbone init_cfg must be a dict.")
        if self.init_cfg.get("type") != "Pretrained":
            super().init_weights()
            return
        checkpoint_path = self.init_cfg.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError(
                "OlmoEarthMultiModalBackbone init_cfg requires a checkpoint path."
            )
        checkpoint = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            map_location="cpu",
            logger=None,
        )
        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self._is_init = True

    def set_batch_metainfo(
        self,
        from_metainfo: list[dict[str, Any]] | None,
        to_metainfo: list[dict[str, Any]] | None,
    ) -> None:
        self._from_metainfo = from_metainfo
        self._to_metainfo = to_metainfo

    def _modality_enum(self, sample_field: str):
        _, _, Modality = _import_olmoearth_types()
        return getattr(Modality, sample_field.upper())

    def _get_bandsets(self, sample_field: str, band_names: list[str]) -> list[list[str]]:
        modality = self._modality_enum(sample_field)
        for attr in ("band_sets", "bandsets", "band_groups"):
            if not hasattr(modality, attr):
                continue
            value = getattr(modality, attr)
            if value is None:
                continue
            resolved = []
            for group in value:
                if hasattr(group, "bands"):
                    resolved.append([_normalize_band_name(x) for x in group.bands])
                else:
                    resolved.append([_normalize_band_name(x) for x in group])
            return resolved
        return [[_normalize_band_name(band)] for band in band_names]

    def _freeze_unused_pretrain_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

        sample_fields = {self.from_sample_field, self.to_sample_field}
        for name, param in self.encoder.named_parameters():
            if name.startswith("project_and_aggregate."):
                continue

            if name.startswith("patch_embeddings.per_modality_embeddings."):
                parts = name.split(".")
                if len(parts) < 4 or parts[2] not in sample_fields:
                    continue
                param.requires_grad = True
                continue

            if name.startswith(
                "composite_encodings.per_modality_channel_embeddings."
            ):
                parts = name.split(".")
                if len(parts) < 3 or parts[2] not in sample_fields:
                    continue
                param.requires_grad = True
                continue

            param.requires_grad = True

    def _default_timestamps(self, batch_size: int, device: torch.device) -> Tensor:
        timestamps = torch.tensor([1, 1, 2025], dtype=torch.long, device=device)
        return timestamps[None, None, :].repeat(batch_size, self.num_timesteps, 1)

    def _timestamps_from_metainfo(
        self,
        metainfo: list[dict[str, Any]] | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if not metainfo:
            return self._default_timestamps(batch_size, device)
        timestamps = []
        for meta in metainfo:
            value = meta.get("timestamps")
            if value is None:
                timestamps.append(self._default_timestamps(1, device).squeeze(0))
                continue
            tensor = torch.as_tensor(value, dtype=torch.long, device=device)
            if tensor.ndim != 2 or tensor.shape[-1] != 3:
                raise ValueError(
                    "timestamps must have shape (T, 3), "
                    f"got {tuple(tensor.shape)}"
                )
            timestamps.append(tensor)
        return torch.stack(timestamps, dim=0)

    def _present_bands_from_metainfo(
        self,
        metainfo: list[dict[str, Any]] | None,
        band_names: list[str],
        batch_size: int,
    ) -> list[set[str]]:
        if not metainfo:
            all_bands = {_normalize_band_name(band) for band in band_names}
            return [all_bands for _ in range(batch_size)]
        out = []
        for meta in metainfo:
            present = meta.get("present_bands") or band_names
            out.append({_normalize_band_name(band) for band in present})
        return out

    def _build_bandset_mask(
        self,
        sample_field: str,
        band_names: list[str],
        metainfo: list[dict[str, Any]] | None,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        _, MaskValue, _ = _import_olmoearth_types()
        bandsets = self._get_bandsets(sample_field, band_names)
        present_by_sample = self._present_bands_from_metainfo(
            metainfo, band_names, batch_size
        )
        mask = torch.full(
            (batch_size, height, width, self.num_timesteps, len(bandsets)),
            float(MaskValue.MISSING.value),
            dtype=torch.float32,
            device=device,
        )
        for sample_idx, present in enumerate(present_by_sample):
            for bandset_idx, bandset in enumerate(bandsets):
                if any(band in present for band in bandset):
                    mask[sample_idx, :, :, :, bandset_idx] = float(
                        MaskValue.ONLINE_ENCODER.value
                    )
        return mask

    def _reshape_image(self, inputs: Tensor, band_names: list[str]) -> Tensor:
        batch_size, channels, height, width = inputs.shape
        expected_channels = len(band_names) * self.num_timesteps
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels "
                f"({len(band_names)} bands x {self.num_timesteps} timesteps), "
                f"got {channels}"
            )
        image = inputs.reshape(
            batch_size, len(band_names), self.num_timesteps, height, width
        )
        return image.permute(0, 3, 4, 2, 1).contiguous()

    def _make_sample(self, inputs_from: Tensor, inputs_to: Tensor):
        MaskedOlmoEarthSample, _, _ = _import_olmoearth_types()
        batch_size, _, height, width = inputs_from.shape
        if inputs_to.shape[0] != batch_size or inputs_to.shape[2:] != (height, width):
            raise ValueError(
                "from/to inputs must share batch and spatial shape, got "
                f"{tuple(inputs_from.shape)} vs {tuple(inputs_to.shape)}"
            )
        from_timestamps = self._timestamps_from_metainfo(
            self._from_metainfo,
            batch_size,
            inputs_from.device,
        )
        kwargs = {
            self.from_sample_field: self._reshape_image(
                inputs_from, self.from_band_names
            ),
            f"{self.from_sample_field}_mask": self._build_bandset_mask(
                self.from_sample_field,
                self.from_band_names,
                self._from_metainfo,
                batch_size,
                height,
                width,
                inputs_from.device,
            ),
            self.to_sample_field: self._reshape_image(inputs_to, self.to_band_names),
            f"{self.to_sample_field}_mask": self._build_bandset_mask(
                self.to_sample_field,
                self.to_band_names,
                self._to_metainfo,
                batch_size,
                height,
                width,
                inputs_from.device,
            ),
            "timestamps": from_timestamps,
        }
        return MaskedOlmoEarthSample(**kwargs)

    @staticmethod
    def _has_missing_tokens(sample) -> bool:
        _, MaskValue, _ = _import_olmoearth_types()
        for name, value in sample.as_dict().items():
            if name.endswith("_mask") and value is not None:
                if (value == MaskValue.MISSING.value).any():
                    return True
        return False

    @staticmethod
    def _pool_tokens(tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            return tokens.mean(dim=(3, 4))
        keep = (mask == 0).to(dtype=tokens.dtype).unsqueeze(-1)
        summed = (tokens * keep).sum(dim=(3, 4))
        denom = keep.sum(dim=(3, 4)).clamp_min(1.0)
        return summed / denom

    def _fuse(self, x_from: Tensor, x_to: Tensor) -> Tensor:
        if self.fusion_policy == "concat":
            return torch.cat([x_from, x_to], dim=-1)
        if self.fusion_policy == "sum":
            return x_from + x_to
        if self.fusion_policy == "diff":
            return x_to - x_from
        return torch.abs(x_from - x_to)

    def forward(self, inputs_from: Tensor, inputs_to: Tensor) -> tuple[Tensor]:
        sample = self._make_sample(inputs_from, inputs_to)
        fast_pass = self.fast_pass
        if fast_pass is None:
            fast_pass = not self._has_missing_tokens(sample)
        encoder_out = self.encoder(
            sample,
            fast_pass=fast_pass,
            patch_size=self.patch_size,
        )
        tokens_and_masks = encoder_out["tokens_and_masks"]
        tokens_from = getattr(tokens_and_masks, self.from_sample_field)
        mask_from = getattr(tokens_and_masks, f"{self.from_sample_field}_mask")
        tokens_to = getattr(tokens_and_masks, self.to_sample_field)
        mask_to = getattr(tokens_and_masks, f"{self.to_sample_field}_mask")
        pooled_from = self._pool_tokens(tokens_from, mask_from)
        pooled_to = self._pool_tokens(tokens_to, mask_to)
        fused = self._fuse(pooled_from, pooled_to)
        return (fused.permute(0, 3, 1, 2).contiguous(),)
