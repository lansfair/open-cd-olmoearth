from __future__ import annotations

from typing import Any

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackUniverSatCDInputs(BaseTransform):
    """Pack heterogeneous observations into Open-CD's channel-pair tensor."""

    def __init__(
        self,
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "sample_id",
            "universat_from_modality",
            "universat_to_modality",
            "universat_from_date",
            "universat_to_date",
            "universat_proxy_note",
        ),
    ) -> None:
        self.meta_keys = tuple(meta_keys)

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim < 3:
            image = np.expand_dims(image, -1)
        chw = np.ascontiguousarray(image.transpose(2, 0, 1))
        return to_tensor(chw).contiguous()

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        packed: dict[str, Any] = {}
        if "img" in results:
            images = [self._image_to_tensor(image) for image in results["img"]]
            if len(images) != 2:
                raise ValueError("UniverSat change detection requires two images.")
            packed["inputs"] = torch.cat(images, dim=0)

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            gt_seg = results["gt_seg_map"]
            data_sample.gt_sem_seg = PixelData(
                data=to_tensor(gt_seg[None, ...].astype(np.int64))
            )
        data_sample.set_metainfo(
            {key: results[key] for key in self.meta_keys if key in results}
        )
        packed["data_samples"] = data_sample
        return packed
