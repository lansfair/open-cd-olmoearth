from __future__ import annotations

import os.path as osp
from typing import List

import mmengine
from opencd.datasets.basecddataset import _BaseCDDataset
from opencd.registry import DATASETS

from ..utils import BRIGHT_CLASSES, BRIGHT_PALETTE


@DATASETS.register_module()
class OLMoEarthBRIGHTDataset(_BaseCDDataset):
    """BRIGHT building damage assessment dataset for OLMoEarth/Open-CD.

    Layout:
    ``pre-event/{id}_pre_disaster.tif``,
    ``post-event/{id}_post_disaster.tif`` and
    ``target/{id}_building_damage.tif``.
    """

    METAINFO = dict(classes=BRIGHT_CLASSES, palette=BRIGHT_PALETTE)

    def __init__(
        self,
        img_suffix: str = ".tif",
        seg_map_suffix: str = ".tif",
        format_seg_map: str = "unchanged",
        split: str = "train",
        **kwargs,
    ) -> None:
        self.split = split
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs,
        )

    def _load_ids(self) -> List[str]:
        if self.ann_file:
            ann_file = self.ann_file
            if not osp.isabs(ann_file):
                ann_file = osp.join(self.data_root, ann_file)
            return [
                item.strip()
                for item in mmengine.list_from_file(ann_file)
                if item.strip()
            ]

        split_file = osp.join(self.data_root, f"{self.split}_set.txt")
        if not osp.isfile(split_file):
            raise FileNotFoundError(f"BRIGHT split file not found: {split_file}")
        return [
            item.strip()
            for item in mmengine.list_from_file(split_file)
            if item.strip()
        ]

    def _resolve_pre_path(self, sample_id: str) -> str:
        candidates = [
            osp.join(
                self.data_root,
                "pre-event",
                f"{sample_id}_pre_disaster{self.img_suffix}",
            ),
            osp.join(
                self.data_root,
                "pre-event_wo_ukraine_myanmar_mexico",
                f"{sample_id}_pre_disaster{self.img_suffix}",
            ),
        ]
        for path in candidates:
            if osp.isfile(path):
                return path
        raise FileNotFoundError(f"BRIGHT pre-event image not found: {sample_id}")

    def _resolve_post_path(self, sample_id: str) -> str:
        path = osp.join(
            self.data_root,
            "post-event",
            f"{sample_id}_post_disaster{self.img_suffix}",
        )
        if not osp.isfile(path):
            raise FileNotFoundError(f"BRIGHT post-event image not found: {path}")
        return path

    def _resolve_label_path(self, sample_id: str) -> str:
        path = osp.join(
            self.data_root,
            "target",
            f"{sample_id}_building_damage{self.seg_map_suffix}",
        )
        if not osp.isfile(path):
            raise FileNotFoundError(f"BRIGHT target not found: {path}")
        return path

    def load_data_list(self) -> List[dict]:
        data_list = []
        for sample_id in self._load_ids():
            data_info = dict(
                img_path=[
                    self._resolve_pre_path(sample_id),
                    self._resolve_post_path(sample_id),
                ],
                seg_map_path=self._resolve_label_path(sample_id),
                sample_id=sample_id,
                label_map=self.label_map,
                format_seg_map=self.format_seg_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            data_list.append(data_info)
        return data_list

