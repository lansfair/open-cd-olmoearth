from __future__ import annotations

import os.path as osp
from typing import Any, List

import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from opencd.datasets.basecddataset import _BaseCDDataset
from opencd.registry import DATASETS, TRANSFORMS

from ..utils import (
    RGB_TO_SENTINEL2_L2A,
    get_modality_bands,
    normalize_band,
)


LEVIR_CD_CLASSES = ("unchanged", "changed")
LEVIR_CD_PALETTE = [[0, 0, 0], [255, 255, 255]]


def _load_computed_norm(modality: str) -> dict[str, dict[str, float]]:
    from olmoearth_pretrain.data.normalize import load_computed_config

    return load_computed_config()[modality]


def _with_suffix(name: str, suffix: str) -> str:
    """Append suffix only when an annotation-list entry has no extension."""
    return name if name.endswith(suffix) else name + suffix


@DATASETS.register_module()
class OLMoEarthLEVIRCDDataset(_BaseCDDataset):
    """LEVIR-CD style binary change-detection dataset for OLMoEarth/Open-CD.

    Expected layout::

        data_root/
          train/A/*.png
          train/B/*.png
          train/label/*.png
          val/A/*.png
          val/B/*.png
          val/label/*.png
          test/A/*.png
          test/B/*.png
          test/label/*.png

    Labels are stored as 0 for unchanged/background and 255 for changed/
    foreground. ``LoadOLMoEarthLEVIRAnnotations`` maps them to class ids
    0 and 1; 255 remains available as ignore_index for padding.
    """

    METAINFO = dict(classes=LEVIR_CD_CLASSES, palette=LEVIR_CD_PALETTE)

    def __init__(
        self,
        img_suffix: str = ".png",
        seg_map_suffix: str = ".png",
        format_seg_map: str = "to_binary",
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs,
        )

    def _build_item(self, rel_img: str) -> dict:
        img_dir_from = self.data_prefix.get("img_path_from", None)
        img_dir_to = self.data_prefix.get("img_path_to", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)
        if img_dir_from is None or img_dir_to is None:
            raise ValueError("data_prefix must contain img_path_from and img_path_to")

        stem = rel_img[: -len(self.img_suffix)] if rel_img.endswith(self.img_suffix) else rel_img
        data_info = dict(
            img_path=[
                osp.join(img_dir_from, rel_img),
                osp.join(img_dir_to, rel_img),
            ],
            sample_id=stem.replace("\\", "/"),
            label_map=self.label_map,
            format_seg_map=self.format_seg_map,
            reduce_zero_label=self.reduce_zero_label,
            seg_fields=[],
        )
        if ann_dir is not None:
            data_info["seg_map_path"] = osp.join(ann_dir, stem + self.seg_map_suffix)
        return data_info

    def load_data_list(self) -> List[dict]:
        """Load image pairs either from ``ann_file`` or by scanning A/B dirs."""
        img_dir_from = self.data_prefix.get("img_path_from", None)
        img_dir_to = self.data_prefix.get("img_path_to", None)
        if img_dir_from is None or img_dir_to is None:
            raise ValueError("data_prefix must contain img_path_from and img_path_to")

        data_list: list[dict] = []
        ann_file = self.ann_file
        if ann_file and not osp.isabs(ann_file) and self.data_root is not None:
            ann_file = osp.join(self.data_root, ann_file)

        if ann_file and osp.isfile(ann_file):
            for line in mmengine.list_from_file(ann_file, backend_args=self.backend_args):
                item = line.strip()
                if not item:
                    continue
                rel_img = _with_suffix(item, self.img_suffix)
                data_list.append(self._build_item(rel_img))
            return data_list

        from_files = sorted(
            fileio.list_dir_or_file(
                dir_path=img_dir_from,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args,
            )
        )
        to_files = sorted(
            fileio.list_dir_or_file(
                dir_path=img_dir_to,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args,
            )
        )
        if from_files != to_files:
            missing_in_b = sorted(set(from_files) - set(to_files))[:5]
            missing_in_a = sorted(set(to_files) - set(from_files))[:5]
            raise AssertionError(
                "Images in A and B are not one-to-one correspondence. "
                f"Missing in B examples: {missing_in_b}; "
                f"missing in A examples: {missing_in_a}"
            )

        for rel_img in from_files:
            data_list.append(self._build_item(rel_img))
        return data_list


@TRANSFORMS.register_module()
class LoadOLMoEarthLEVIRPair(BaseTransform):
    """Load LEVIR-CD RGB pairs and map them to OLMoEarth Sentinel-2 L2A input.

    Only the RGB-related Sentinel-2 slots are filled:
    R -> B04, G -> B03, B -> B02. Other Sentinel-2 bands are zero-filled and
    marked as absent through ``present_bands`` metadata. Raw RGB values are
    first rescaled to the configured Sentinel-2 reflectance scale, then
    normalized by OLMoEarth's computed Sentinel-2 statistics.
    """

    def __init__(
        self,
        to_float32: bool = True,
        backend_args: dict | None = None,
        rgb_channel_order: str = "RGB",
        input_value_range: str = "0_255",
        input_max_value: float = 255.0,
        target_s2_scale: float = 10000.0,
        clip_to_s2_scale: bool = True,
        std_multiplier: float = 2.0,
        pre_timestamp: tuple[int, int, int] = (1, 1, 2025),
        post_timestamp: tuple[int, int, int] = (2, 1, 2025),
    ) -> None:
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"0_255", "0_1", "s2"}:
            raise ValueError("input_value_range must be '0_255', '0_1', or 's2'")
        if input_max_value <= 0:
            raise ValueError("input_max_value must be positive")
        if target_s2_scale <= 0:
            raise ValueError("target_s2_scale must be positive")
        if std_multiplier <= 0:
            raise ValueError("std_multiplier must be positive")

        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.input_max_value = float(input_max_value)
        self.target_s2_scale = float(target_s2_scale)
        self.clip_to_s2_scale = bool(clip_to_s2_scale)
        self.std_multiplier = float(std_multiplier)
        self.pre_timestamp = tuple(int(x) for x in pre_timestamp)
        self.post_timestamp = tuple(int(x) for x in post_timestamp)

        self.band_names = list(get_modality_bands("sentinel2_l2a"))
        self.norm_config = _load_computed_norm("sentinel2_l2a")

    def _read_rgb(self, filename: str) -> np.ndarray:
        img_bytes = fileio.get(filename, backend_args=self.backend_args)
        image = mmcv.imfrombytes(img_bytes, flag="color", channel_order="rgb")
        if self.to_float32:
            image = image.astype(np.float32)
        return image

    def _to_s2_scale(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        if self.input_value_range == "s2":
            scaled = image
        elif self.input_value_range == "0_1":
            scaled = image * self.target_s2_scale
        else:
            scaled = image * (self.target_s2_scale / self.input_max_value)
        if self.clip_to_s2_scale:
            scaled = np.clip(scaled, 0.0, self.target_s2_scale)
        return scaled

    def _rgb_to_pseudo_s2(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")

        image = self._to_s2_scale(image)
        h, w = image.shape[:2]
        out = np.zeros((h, w, len(self.band_names)), dtype=np.float32)
        channel_to_index = {name: idx for idx, name in enumerate(self.rgb_channel_order)}
        for rgb_name, s2_band in RGB_TO_SENTINEL2_L2A.items():
            rgb_idx = channel_to_index[rgb_name]
            band_idx = self.band_names.index(s2_band)
            out[..., band_idx] = normalize_band(
                image[..., rgb_idx],
                s2_band,
                self.norm_config,
                self.std_multiplier,
            )
        return out

    def _build_metainfo(self, timestamp: tuple[int, int, int], source: str) -> dict:
        return dict(
            olmoearth_modality="sentinel2_l2a",
            olmoearth_num_timesteps=1,
            olmoearth_band_names=list(self.band_names),
            present_bands=list(RGB_TO_SENTINEL2_L2A.values()),
            timestamps=np.asarray([timestamp], dtype=np.int64),
            source=source,
            input_value_range=self.input_value_range,
            input_max_value=self.input_max_value,
            target_s2_scale=self.target_s2_scale,
            proxy_filled_bands=list(RGB_TO_SENTINEL2_L2A.values()),
        )

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        pre_rgb = self._read_rgb(results["img_path"][0])
        post_rgb = self._read_rgb(results["img_path"][1])
        if pre_rgb.shape[:2] != post_rgb.shape[:2]:
            raise ValueError(
                "LEVIR-CD A/B image shapes do not match: "
                f"{pre_rgb.shape} vs {post_rgb.shape}"
            )

        results["img"] = [
            self._rgb_to_pseudo_s2(pre_rgb),
            self._rgb_to_pseudo_s2(post_rgb),
        ]
        results["img_shape"] = pre_rgb.shape[:2]
        results["ori_shape"] = pre_rgb.shape[:2]
        results["olmoearth_from_metainfo"] = self._build_metainfo(
            self.pre_timestamp, "levir_cd_A_rgb_to_s2"
        )
        results["olmoearth_to_metainfo"] = self._build_metainfo(
            self.post_timestamp, "levir_cd_B_rgb_to_s2"
        )
        return results


@TRANSFORMS.register_module()
class LoadOLMoEarthLEVIRAnnotations(BaseTransform):
    """Load LEVIR-CD 0/255 labels and convert them to class ids 0/1."""

    def __init__(
        self,
        backend_args: dict | None = None,
        imdecode_backend: str = "pillow",
        binary_threshold: int = 128,
    ) -> None:
        self.backend_args = backend_args.copy() if backend_args else None
        self.imdecode_backend = imdecode_backend
        self.binary_threshold = int(binary_threshold)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        img_bytes = fileio.get(results["seg_map_path"], backend_args=self.backend_args)
        seg_map = mmcv.imfrombytes(
            img_bytes,
            flag="grayscale",
            backend=self.imdecode_backend,
        ).squeeze()

        # Important: in this LEVIR-CD layout, 255 means foreground/change,
        # not ignore. Padding later still uses ignore_index=255.
        seg_map = (seg_map >= self.binary_threshold).astype(np.uint8)

        if results.get("label_map", None) is not None:
            seg_map_copy = seg_map.copy()
            for old_id, new_id in results["label_map"].items():
                seg_map[seg_map_copy == old_id] = new_id

        results["gt_seg_map"] = seg_map
        results.setdefault("seg_fields", []).append("gt_seg_map")
        return results
