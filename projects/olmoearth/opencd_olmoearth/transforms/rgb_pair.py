from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from opencd.registry import TRANSFORMS

from ..utils import get_modality_bands, rgb_to_pseudo_s2


def _load_computed_norm(modality: str) -> dict[str, dict[str, float]]:
    from olmoearth_pretrain.data.normalize import load_computed_config

    return load_computed_config()[modality]


@TRANSFORMS.register_module()
class RGBPairToOlmoEarth(BaseTransform):
    """Convert loaded pre/post RGB images for OLMoEarth change detection."""

    def __init__(
        self,
        modality: str,
        rgb_channel_order: str = "BGR",
        input_value_range: str = "0_255",
        std_multiplier: float = 2.0,
        pre_timestamp=(1, 1, 2025),
        post_timestamp=(2, 1, 2025),
    ) -> None:
        if modality not in {"sentinel2_l2a", "rgb"}:
            raise ValueError("modality must be sentinel2_l2a or rgb")
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"0_255", "0_1"}:
            raise ValueError("input_value_range must be 0_255 or 0_1")
        self.modality = modality
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.std_multiplier = std_multiplier
        self.pre_timestamp = tuple(int(x) for x in pre_timestamp)
        self.post_timestamp = tuple(int(x) for x in post_timestamp)
        self.band_names = list(get_modality_bands(modality))
        self.norm_config = (
            _load_computed_norm("sentinel2_l2a")
            if modality == "sentinel2_l2a"
            else None
        )

    def _to_unit_scale(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        if self.input_value_range == "0_255":
            return image / 255.0
        return image

    def _to_native_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")
        image = self._to_unit_scale(image)
        height, width = image.shape[:2]
        out = np.zeros((height, width, len(self.band_names)), dtype=np.float32)
        channel_to_index = {name: idx for idx, name in enumerate(self.rgb_channel_order)}
        for band_name in ("B", "G", "R"):
            out[..., self.band_names.index(band_name)] = image[
                ..., channel_to_index[band_name]
            ]
        return out

    def _convert_image(self, image: np.ndarray) -> np.ndarray:
        if self.modality == "sentinel2_l2a":
            return rgb_to_pseudo_s2(
                image=image,
                band_names=self.band_names,
                norm_config=self.norm_config,
                rgb_channel_order=self.rgb_channel_order,
                input_value_range=self.input_value_range,
                std_multiplier=self.std_multiplier,
            )
        return self._to_native_rgb(image)

    def _metainfo(self, timestamp) -> dict:
        present_bands = ["B04", "B03", "B02"] if self.modality == "sentinel2_l2a" else ["B", "G", "R"]
        return dict(
            olmoearth_modality=self.modality,
            olmoearth_num_timesteps=1,
            olmoearth_band_names=list(self.band_names),
            present_bands=present_bands,
            timestamps=np.asarray([timestamp], dtype=np.int64),
            source=f"rgb_pair_to_{self.modality}",
            olmoearth_rgb_adapter=dict(
                rgb_channel_order=self.rgb_channel_order,
                input_value_range=self.input_value_range,
            ),
        )

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        images = results.get("img")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError("RGBPairToOlmoEarth expects results['img'] with two images")
        img_from = self._convert_image(images[0])
        img_to = self._convert_image(images[1])
        if img_from.shape[:2] != img_to.shape[:2]:
            raise ValueError(
                "Image pair shapes do not match: "
                f"{img_from.shape} vs {img_to.shape}"
            )
        results["img"] = [img_from, img_to]
        results["img_shape"] = img_from.shape[:2]
        results["ori_shape"] = results.get("ori_shape", img_from.shape[:2])
        results["olmoearth_from_metainfo"] = self._metainfo(self.pre_timestamp)
        results["olmoearth_to_metainfo"] = self._metainfo(self.post_timestamp)
        return results
