from __future__ import annotations

from typing import Any

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackOLMoEarthCDInputs(BaseTransform):
    """Pack OLMoEarth change-detection image pairs for Open-CD.

    The two images are concatenated along channel dimension to stay compatible
    with Open-CD data preprocessors. Their separate OLMoEarth metadata is kept
    in ``olmoearth_from_metainfo`` and ``olmoearth_to_metainfo``.
    """

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
            "olmoearth_from_metainfo",
            "olmoearth_to_metainfo",
        ),
    ) -> None:
        self.meta_keys = meta_keys

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim < 3:
            image = np.expand_dims(image, -1)
        chw = np.ascontiguousarray(image.transpose(2, 0, 1))
        return to_tensor(chw).contiguous()

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        packed: dict[str, Any] = {}
        if "img" in results:
            imgs = [self._image_to_tensor(img) for img in results["img"]]
            packed["inputs"] = torch.cat(imgs, dim=0)

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            gt_seg = results["gt_seg_map"]
            data_sample.gt_sem_seg = PixelData(
                data=to_tensor(gt_seg[None, ...].astype(np.int64))
            )

        metainfo = {key: results[key] for key in self.meta_keys if key in results}
        data_sample.set_metainfo(metainfo)
        packed["data_samples"] = data_sample
        return packed

