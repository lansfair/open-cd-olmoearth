from __future__ import annotations

import glob
import os.path as osp
import re
from typing import List

import mmengine
from opencd.datasets.basecddataset import _BaseCDDataset
from opencd.registry import DATASETS

from ..utils import SENTINEL2_L2A_BANDS


@DATASETS.register_module()
class OLMoEarthOSCDDataset(_BaseCDDataset):
    """OSCD binary change detection dataset in the official folder layout."""

    METAINFO = dict(
        classes=("unchanged", "changed"),
        palette=[[0, 0, 0], [255, 255, 255]],
    )

    ALL_BANDS = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    )

    def __init__(
        self,
        img_suffix: str = ".tif",
        seg_map_suffix: str = ".png",
        format_seg_map: str = "to_binary",
        split: str = "train",
        bands=None,
        **kwargs,
    ) -> None:
        self.split = split
        self.bands = tuple(bands or SENTINEL2_L2A_BANDS)
        unsupported = set(self.bands) - set(self.ALL_BANDS)
        if unsupported:
            raise ValueError(f"Unsupported OSCD bands: {sorted(unsupported)}")
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs,
        )

    @staticmethod
    def _parse_band(path: str) -> str:
        match = re.search(r"B(?:0[1-9]|1[0-2]|8A)", osp.basename(path))
        if match is None:
            raise ValueError(f"Cannot parse Sentinel-2 band from {path}")
        return match.group(0)

    def _get_band_paths(
        self,
        images_root: str,
        region: str,
        temporal_index: int,
    ) -> list[str]:
        pattern = osp.join(images_root, region, f"imgs_{temporal_index}_rect", "*.tif")
        band_to_path = {self._parse_band(path): path for path in glob.glob(pattern)}
        missing = [band for band in self.bands if band not in band_to_path]
        if missing:
            raise FileNotFoundError(
                f"Missing bands for {region} time {temporal_index}: {missing}"
            )
        return [band_to_path[band] for band in self.bands]

    def load_data_list(self) -> List[dict]:
        split_name = self.split.capitalize()
        images_root = osp.join(
            self.data_root,
            "Onera Satellite Change Detection dataset - Images",
        )
        labels_root = osp.join(
            self.data_root,
            f"Onera Satellite Change Detection dataset - {split_name} Labels",
        )
        if not osp.isdir(images_root):
            raise FileNotFoundError(f"OSCD images folder not found: {images_root}")
        if not osp.isdir(labels_root):
            raise FileNotFoundError(f"OSCD labels folder not found: {labels_root}")

        regions = None
        if self.ann_file and osp.isfile(self.ann_file):
            regions = {
                line.strip()
                for line in mmengine.list_from_file(self.ann_file)
                if line.strip()
            }

        data_list = []
        for folder in sorted(glob.glob(osp.join(labels_root, "*"))):
            if not osp.isdir(folder):
                continue
            region = osp.basename(folder)
            if regions is not None and region not in regions:
                continue
            mask = osp.join(labels_root, region, "cm", f"{region}-cm.tif")
            if not osp.isfile(mask):
                mask = osp.join(labels_root, region, "cm", "cm.png")
            if not osp.isfile(mask):
                raise FileNotFoundError(f"OSCD mask not found: {mask}")
            data_list.append(
                dict(
                    img_path=[
                        self._get_band_paths(images_root, region, 1),
                        self._get_band_paths(images_root, region, 2),
                    ],
                    seg_map_path=mask,
                    sample_id=region,
                    label_map=self.label_map,
                    format_seg_map=self.format_seg_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                )
            )
        return data_list
