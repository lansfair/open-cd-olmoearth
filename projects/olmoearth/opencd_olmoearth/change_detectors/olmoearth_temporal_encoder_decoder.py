from __future__ import annotations

from typing import List

import numpy as np
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.utils import OptSampleList, SampleList
from opencd.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class OLMoEarthTemporalEncoderDecoder(EncoderDecoder):
    """Joint temporal OLMoEarth encoder for change detection.

    The input tensor is still packed like Open-CD change detection data:
    ``[t1_bands, t2_bands]``. Before forwarding to OLMoEarth, it is reordered
    to band-major temporal layout: ``[band1_t1, band1_t2, band2_t1, ...]``.
    """

    def __init__(
        self,
        backbone: dict,
        decode_head: dict,
        neck: dict | None = None,
        auxiliary_head: dict | list[dict] | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        data_preprocessor: dict | None = None,
        pretrained: str | None = None,
        init_cfg: dict | list[dict] | None = None,
        backbone_inchannels: int = 12,
        num_timesteps: int = 2,
    ) -> None:
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.backbone_inchannels = backbone_inchannels
        self.num_timesteps = num_timesteps

    def _reorder_inputs(self, inputs: Tensor) -> Tensor:
        expected_channels = self.backbone_inchannels * self.num_timesteps
        if inputs.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels "
                f"({self.backbone_inchannels} bands x {self.num_timesteps} "
                f"timesteps), got {inputs.shape[1]}"
            )
        batch_size, _, height, width = inputs.shape
        inputs = inputs.reshape(
            batch_size,
            self.num_timesteps,
            self.backbone_inchannels,
            height,
            width,
        )
        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        return inputs.reshape(batch_size, expected_channels, height, width)

    def _temporal_metainfo(self, data_samples: OptSampleList) -> list[dict] | None:
        if data_samples is None:
            return None
        metas = []
        for data_sample in data_samples:
            from_meta = dict(data_sample.metainfo.get("olmoearth_from_metainfo", {}))
            to_meta = dict(data_sample.metainfo.get("olmoearth_to_metainfo", {}))
            meta = dict(from_meta)
            from_timestamps = from_meta.get("timestamps")
            to_timestamps = to_meta.get("timestamps")
            if from_timestamps is not None and to_timestamps is not None:
                meta["timestamps"] = np.concatenate(
                    [np.asarray(from_timestamps), np.asarray(to_timestamps)],
                    axis=0,
                )
            present_bands = from_meta.get("present_bands") or to_meta.get(
                "present_bands"
            )
            if present_bands is not None:
                meta["present_bands"] = list(present_bands)
            meta["olmoearth_num_timesteps"] = self.num_timesteps
            metas.append(meta)
        return metas

    def extract_feat(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> List[Tensor]:
        inputs = self._reorder_inputs(inputs)
        if hasattr(self.backbone, "set_batch_metainfo"):
            self.backbone.set_batch_metainfo(self._temporal_metainfo(data_samples))
        return super().extract_feat(inputs)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs, data_samples)
        losses = {}
        losses.update(self._decode_head_forward_train(x, data_samples))
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(x, data_samples))
        return losses

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        from mmseg.structures import SegDataSample

        data_samples = []
        for meta in batch_img_metas:
            sample = SegDataSample()
            sample.set_metainfo(meta)
            data_samples.append(sample)
        x = self.extract_feat(inputs, data_samples)
        return self.decode_head.predict(x, batch_img_metas, self.test_cfg)

    def predict(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> Tensor:
        x = self.extract_feat(inputs, data_samples)
        return self.decode_head.forward(x)

    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        return self.encode_decode(inputs, batch_img_metas)
