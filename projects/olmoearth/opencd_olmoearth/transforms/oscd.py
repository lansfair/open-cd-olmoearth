from __future__ import annotations

from typing import Any

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from opencd.registry import TRANSFORMS

try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class LoadOLMoEarthOSCDPair(BaseTransform):
    """Load OSCD band files and attach OLMoEarth metadata for both dates."""

    def __init__(
        self,
        band_names=(
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ),
        band_scales=None,
        mean=(
            1117.2,
            1041.8,
            946.5,
            2301.2,
            1199.1,
            2003.0,
            2374.0,
            2599.7,
            1820.6,
            1118.2,
            1353.7,
            732.1,
        ),
        std=(
            736.0,
            684.8,
            620.0,
            1545.5,
            791.9,
            1341.3,
            1595.4,
            1750.1,
            1216.5,
            736.7,
            897.3,
            475.1,
        ),
        pre_timestamp=(1, 1, 2025),
        post_timestamp=(2, 1, 2025),
        nan_to_num: bool = True,
        to_float32: bool = True,
    ) -> None:
        if gdal is None:
            raise RuntimeError("gdal is not installed")
        self.band_names = list(band_names)
        self.band_scales = None if band_scales is None else np.asarray(
            band_scales, dtype=np.float32
        )
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        self.pre_timestamp = tuple(int(x) for x in pre_timestamp)
        self.post_timestamp = tuple(int(x) for x in post_timestamp)
        self.nan_to_num = nan_to_num
        self.to_float32 = to_float32
        if len(self.band_names) != self.mean.shape[-1]:
            raise ValueError("mean length must match band_names.")
        if len(self.band_names) != self.std.shape[-1]:
            raise ValueError("std length must match band_names.")

    def _read_band(self, filename: str) -> np.ndarray:
        ds = gdal.Open(filename)
        if ds is None:
            raise FileNotFoundError(f"Unable to open file: {filename}")
        image = ds.ReadAsArray()
        if image.ndim == 3:
            if image.shape[0] != 1:
                raise ValueError(f"Expected one band in {filename}, got {image.shape}")
            image = image[0]
        if self.to_float32:
            image = image.astype(np.float32)
        if self.nan_to_num:
            image = np.nan_to_num(image)
        return image

    def _read_image(self, filenames: list[str]) -> np.ndarray:
        image = np.stack([self._read_band(filename) for filename in filenames], axis=-1)
        if self.band_scales is not None:
            if len(self.band_scales) != image.shape[-1]:
                raise ValueError("band_scales length must match image bands.")
            image = image * self.band_scales.reshape(1, 1, -1)
        return (image.astype(np.float32) - self.mean) / self.std

    def _metainfo(self, timestamp) -> dict:
        return dict(
            olmoearth_modality="sentinel2_l2a",
            olmoearth_num_timesteps=1,
            olmoearth_band_names=list(self.band_names),
            present_bands=list(self.band_names),
            timestamps=np.asarray([timestamp], dtype=np.int64),
            source="oscd_sentinel2_l2a",
        )

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        img_from = self._read_image(results["img_path"][0])
        img_to = self._read_image(results["img_path"][1])
        if img_from.shape[:2] != img_to.shape[:2]:
            raise ValueError(
                "OSCD image shapes do not match: "
                f"{img_from.shape} vs {img_to.shape}"
            )
        results["img"] = [img_from, img_to]
        results["img_shape"] = img_from.shape[:2]
        results["ori_shape"] = img_from.shape[:2]
        results["olmoearth_from_metainfo"] = self._metainfo(self.pre_timestamp)
        results["olmoearth_to_metainfo"] = self._metainfo(self.post_timestamp)
        return results


@TRANSFORMS.register_module()
class LoadOLMoEarthOSCDAnnotations(BaseTransform):
    """Load OSCD binary change masks."""

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
        ).squeeze()
        unique_values = set(int(v) for v in np.unique(seg_map))
        if unique_values.issubset({1, 2}):
            seg_map = (seg_map == 2).astype(np.uint8)
        else:
            seg_map = (seg_map >= 128).astype(np.uint8)

        if results.get("label_map", None) is not None:
            seg_map_copy = seg_map.copy()
            for old_id, new_id in results["label_map"].items():
                seg_map[seg_map_copy == old_id] = new_id

        results["gt_seg_map"] = seg_map
        results.setdefault("seg_fields", []).append("gt_seg_map")
        return results
