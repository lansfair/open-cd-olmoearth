from __future__ import annotations

from typing import Any

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadUniverSatBRIGHTPair(BaseTransform):
    """Load raw BRIGHT RGB/SAR observations before spatial augmentation.

    The single-channel post-event SAR image is repeated to three channels so
    Open-CD's reference BRIGHT photometric transform can run before modality
    proxy construction and normalization.
    """

    def __init__(
        self,
        to_float32: bool = True,
        backend_args: dict | None = None,
        rgb_channel_order: str = "RGB",
        input_value_range: str = "0_255",
        pre_date: int = 0,
        post_date: int = 1,
    ) -> None:
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"0_255", "0_1"}:
            raise ValueError("input_value_range must be 0_255 or 0_1")
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.pre_date = int(pre_date)
        self.post_date = int(post_date)

    def _read_rgb(self, filename: str) -> np.ndarray:
        img_bytes = fileio.get(filename, backend_args=self.backend_args)
        image = mmcv.imfrombytes(img_bytes, flag="color", channel_order="rgb")
        return image.astype(np.float32) if self.to_float32 else image

    def _read_sar(self, filename: str) -> np.ndarray:
        img_bytes = fileio.get(filename, backend_args=self.backend_args)
        image = mmcv.imfrombytes(img_bytes, flag="grayscale").squeeze()
        return image.astype(np.float32) if self.to_float32 else image

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        pre_rgb = self._read_rgb(results["img_path"][0])
        post_sar = self._read_sar(results["img_path"][1])
        if pre_rgb.shape[:2] != post_sar.shape[:2]:
            raise ValueError(
                "BRIGHT pre/post image shapes do not match: "
                f"{pre_rgb.shape} vs {post_sar.shape}."
            )

        results["img"] = [
            pre_rgb,
            np.repeat(post_sar[..., None], 3, axis=-1),
        ]
        results["img_shape"] = pre_rgb.shape[:2]
        results["ori_shape"] = pre_rgb.shape[:2]
        results["universat_rgb_channel_order"] = self.rgb_channel_order
        results["universat_input_value_range"] = self.input_value_range
        results["universat_from_date"] = self.pre_date
        results["universat_to_date"] = self.post_date
        return results


@TRANSFORMS.register_module()
class BuildUniverSatBRIGHTProxies(BaseTransform):
    """Construct and normalize explicit UniverSat modality proxies.

    Normalization is applied after random spatial and photometric augmentation.
    The defaults preserve the previous ``[-1, 1]`` mapping but are explicit
    and can be replaced by BRIGHT training-set statistics.
    """

    def __init__(
        self,
        rgb_mean: tuple[float, float, float] = (127.5, 127.5, 127.5),
        rgb_std: tuple[float, float, float] = (127.5, 127.5, 127.5),
        sar_mean: float = 127.5,
        sar_std: float = 127.5,
    ) -> None:
        self.rgb_mean = np.asarray(rgb_mean, dtype=np.float32)
        self.rgb_std = np.asarray(rgb_std, dtype=np.float32)
        self.sar_mean = float(sar_mean)
        self.sar_std = float(sar_std)
        if self.rgb_mean.shape != (3,) or self.rgb_std.shape != (3,):
            raise ValueError("rgb_mean and rgb_std must contain three values.")
        if np.any(self.rgb_std <= 0) or self.sar_std <= 0:
            raise ValueError("Normalization standard deviations must be positive.")

    def _rgb_to_s2_4band(
        self,
        image: np.ndarray,
        channel_order: str,
    ) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        channel_index = {
            name: index for index, name in enumerate(channel_order)
        }
        normalized = (image - self.rgb_mean) / self.rgb_std
        height, width = image.shape[:2]
        output = np.zeros((height, width, 4), dtype=np.float32)
        output[..., 0] = normalized[..., channel_index["B"]]
        output[..., 1] = normalized[..., channel_index["G"]]
        output[..., 2] = normalized[..., channel_index["R"]]
        return output

    def _sar_to_s1(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        amplitude = image.mean(axis=-1) if image.ndim == 3 else image
        amplitude = (amplitude - self.sar_mean) / self.sar_std
        height, width = amplitude.shape
        output = np.zeros((height, width, 3), dtype=np.float32)
        output[..., 0] = amplitude
        output[..., 1] = amplitude
        return output

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        if len(results.get("img", [])) != 2:
            raise ValueError("BRIGHT proxy construction requires two images.")
        channel_order = results.get("universat_rgb_channel_order", "RGB")
        results["img"] = [
            self._rgb_to_s2_4band(results["img"][0], channel_order),
            self._sar_to_s1(results["img"][1]),
        ]
        results["universat_from_modality"] = "s2_4band"
        results["universat_to_modality"] = "s1"
        results["universat_from_date"] = results.get(
            "universat_from_date",
            0,
        )
        results["universat_to_date"] = results.get(
            "universat_to_date",
            1,
        )
        results["universat_proxy_note"] = (
            "BRIGHT RGB->s2_4band(B08 missing); "
            "single-channel SAR->s1(VV/VH duplicated, ratio missing); "
            "normalization applied after augmentation"
        )
        return results


@TRANSFORMS.register_module()
class LoadUniverSatBRIGHTAnnotations(BaseTransform):
    """Load BRIGHT background/intact/damaged/destroyed labels."""

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

        if results.get("label_map") is not None:
            source = seg_map.copy()
            for old_id, new_id in results["label_map"].items():
                seg_map[source == old_id] = new_id

        results["gt_seg_map"] = seg_map
        results.setdefault("seg_fields", []).append("gt_seg_map")
        return results
