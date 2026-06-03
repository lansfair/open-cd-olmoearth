from __future__ import annotations

from typing import Any

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from opencd.registry import TRANSFORMS

from ..utils import (
    RGB_TO_SENTINEL2_L2A,
    get_modality_bands,
    normalize_band,
    rgb_to_pseudo_s2,
)


def _load_computed_norm(modality: str) -> dict[str, dict[str, float]]:
    from olmoearth_pretrain.data.normalize import load_computed_config

    return load_computed_config()[modality]


@TRANSFORMS.register_module()
class LoadOLMoEarthBRIGHTPair(BaseTransform):
    """Load BRIGHT pre RGB and post SAR for OLMoEarth RGB-proxy training.

    BRIGHT's released post-event SAR rasters are single-channel uint8 images.
    Following the Copernicus-FM BRIGHT baseline, this transform repeats that
    SAR amplitude image to RGB before mapping both dates into OLMoEarth's
    Sentinel-2 L2A band layout. This is a fair comparison proxy, not a claim
    that BRIGHT SAR is native Sentinel-2 or dual-pol Sentinel-1.
    """

    def __init__(
        self,
        to_float32: bool = True,
        backend_args: dict | None = None,
        rgb_channel_order: str = "RGB",
        input_value_range: str = "0_255",
        std_multiplier: float = 2.0,
        post_sar_mode: str = "rgbproxy_to_s2",
        sar_db_range: tuple[float, float] = (-30.0, 5.0),
        pre_timestamp: tuple[int, int, int] = (1, 1, 2025),
        post_timestamp: tuple[int, int, int] = (2, 1, 2025),
    ) -> None:
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"0_255", "0_1", "s2"}:
            raise ValueError("input_value_range must be 0_255, 0_1, or s2")
        if post_sar_mode not in {"rgbproxy_to_s2", "s1_vv_zero_vh", "s1_dup2"}:
            raise ValueError(
                "post_sar_mode must be rgbproxy_to_s2, s1_vv_zero_vh, or s1_dup2"
            )
        if sar_db_range[0] >= sar_db_range[1]:
            raise ValueError("sar_db_range must be ordered as (min_db, max_db)")
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.std_multiplier = std_multiplier
        self.post_sar_mode = post_sar_mode
        self.sar_db_range = tuple(float(x) for x in sar_db_range)
        self.pre_timestamp = tuple(int(x) for x in pre_timestamp)
        self.post_timestamp = tuple(int(x) for x in post_timestamp)
        self.band_names = list(get_modality_bands("sentinel2_l2a"))
        self.norm_config = _load_computed_norm("sentinel2_l2a")
        self.s1_band_names = list(get_modality_bands("sentinel1"))
        self.s1_norm_config = _load_computed_norm("sentinel1")

    def _read_rgb(self, filename: str) -> np.ndarray:
        img_bytes = fileio.get(filename, backend_args=self.backend_args)
        image = mmcv.imfrombytes(img_bytes, flag="color", channel_order="rgb")
        if self.to_float32:
            image = image.astype(np.float32)
        return image

    def _read_sar_as_rgb(self, filename: str) -> np.ndarray:
        image = self._read_sar(filename)
        image = np.stack([image, image, image], axis=-1)
        if self.to_float32:
            image = image.astype(np.float32)
        return image

    def _read_sar(self, filename: str) -> np.ndarray:
        img_bytes = fileio.get(filename, backend_args=self.backend_args)
        image = mmcv.imfrombytes(img_bytes, flag="grayscale").squeeze()
        if self.to_float32:
            image = image.astype(np.float32)
        return image

    def _rgb_to_pseudo_s2(self, image: np.ndarray) -> np.ndarray:
        return rgb_to_pseudo_s2(
            image=image,
            band_names=self.band_names,
            norm_config=self.norm_config,
            rgb_channel_order=self.rgb_channel_order,
            input_value_range=self.input_value_range,
            std_multiplier=self.std_multiplier,
        )

    def _sar_to_db(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False) / 255.0
        min_db, max_db = self.sar_db_range
        return image * (max_db - min_db) + min_db

    def _sar_to_s1_proxy(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        sar_db = self._sar_to_db(image)
        h, w = sar_db.shape[:2]
        out = np.zeros((h, w, len(self.s1_band_names)), dtype=np.float32)
        vv_idx = self.s1_band_names.index("vv")
        vh_idx = self.s1_band_names.index("vh")
        out[..., vv_idx] = normalize_band(
            sar_db,
            "vv",
            self.s1_norm_config,
            self.std_multiplier,
        )
        present_bands = ["vv", "vh"]
        proxy_filled_bands = ["vh"]
        if self.post_sar_mode == "s1_dup2":
            out[..., vh_idx] = normalize_band(
                sar_db,
                "vh",
                self.s1_norm_config,
                self.std_multiplier,
            )
            proxy_filled_bands = ["vv", "vh"]
        return out, present_bands, proxy_filled_bands

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        pre_rgb = self._read_rgb(results["img_path"][0])
        post_sar = self._read_sar(results["img_path"][1])
        if pre_rgb.shape[:2] != post_sar.shape[:2]:
            raise ValueError(
                "BRIGHT pre/post image shapes do not match: "
                f"{pre_rgb.shape} vs {post_sar.shape}"
            )

        pre_image = self._rgb_to_pseudo_s2(pre_rgb)
        if self.post_sar_mode == "rgbproxy_to_s2":
            post_image = self._rgb_to_pseudo_s2(np.stack([post_sar] * 3, axis=-1))
            post_metainfo = dict(
                olmoearth_modality="sentinel2_l2a",
                olmoearth_num_timesteps=1,
                olmoearth_band_names=list(self.band_names),
                present_bands=list(RGB_TO_SENTINEL2_L2A.values()),
                timestamps=np.asarray([self.post_timestamp], dtype=np.int64),
                source="bright_post_sar_rgbproxy_to_s2",
            )
        else:
            post_image, present_bands, proxy_filled_bands = self._sar_to_s1_proxy(
                post_sar
            )
            post_metainfo = dict(
                olmoearth_modality="sentinel1",
                olmoearth_num_timesteps=1,
                olmoearth_band_names=list(self.s1_band_names),
                present_bands=present_bands,
                timestamps=np.asarray([self.post_timestamp], dtype=np.int64),
                source=f"bright_post_sar_{self.post_sar_mode}",
                sar_db_range=self.sar_db_range,
                proxy_filled_bands=proxy_filled_bands,
            )

        results["img"] = [pre_image, post_image]
        results["img_shape"] = pre_rgb.shape[:2]
        results["ori_shape"] = pre_rgb.shape[:2]
        mapped_bands = list(RGB_TO_SENTINEL2_L2A.values())
        results["olmoearth_from_metainfo"] = dict(
            olmoearth_modality="sentinel2_l2a",
            olmoearth_num_timesteps=1,
            olmoearth_band_names=list(self.band_names),
            present_bands=mapped_bands,
            timestamps=np.asarray([self.pre_timestamp], dtype=np.int64),
            source="bright_pre_rgb_to_s2",
        )
        results["olmoearth_to_metainfo"] = post_metainfo
        return results


@TRANSFORMS.register_module()
class LoadOLMoEarthBRIGHTAnnotations(BaseTransform):
    """Load BRIGHT four-class building damage labels."""

    def __init__(
        self,
        backend_args: dict | None = None,
        imdecode_backend: str = "pillow",
    ) -> None:
        self.backend_args = backend_args.copy() if backend_args else None
        self.imdecode_backend = imdecode_backend

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        img_bytes = fileio.get(results["seg_map_path"], backend_args=self.backend_args)
        seg_map = mmcv.imfrombytes(
            img_bytes,
            flag="grayscale",
            backend=self.imdecode_backend,
        ).squeeze().astype(np.uint8)
        seg_map[seg_map > 3] = 255

        if results.get("label_map", None) is not None:
            seg_map_copy = seg_map.copy()
            for old_id, new_id in results["label_map"].items():
                seg_map[seg_map_copy == old_id] = new_id

        results["gt_seg_map"] = seg_map
        results.setdefault("seg_fields", []).append("gt_seg_map")
        return results
