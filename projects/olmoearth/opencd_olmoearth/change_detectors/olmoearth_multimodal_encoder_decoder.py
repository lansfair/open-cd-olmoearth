from __future__ import annotations

from typing import List

from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.utils import OptSampleList, SampleList
from opencd.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class OLMoEarthMultiModalEncoderDecoder(EncoderDecoder):
    """Change detector that fuses two OLMoEarth modalities in one encoder."""

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
        backbone_from_inchannels: int = 12,
        backbone_to_inchannels: int = 2,
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
        self.backbone_from_inchannels = backbone_from_inchannels
        self.backbone_to_inchannels = backbone_to_inchannels

    def _split_inputs(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        expected_channels = self.backbone_from_inchannels + self.backbone_to_inchannels
        if inputs.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels "
                f"({self.backbone_from_inchannels} from + "
                f"{self.backbone_to_inchannels} to), got {inputs.shape[1]}"
            )
        return inputs.split(
            [self.backbone_from_inchannels, self.backbone_to_inchannels],
            dim=1,
        )

    @staticmethod
    def _extract_olmoearth_metas(
        data_samples: OptSampleList,
        key: str,
    ) -> list[dict] | None:
        if data_samples is None:
            return None
        metas = []
        for data_sample in data_samples:
            meta = data_sample.metainfo.get(key, {})
            if meta is None:
                meta = {}
            metas.append(dict(meta))
        return metas

    def extract_feat(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> List[Tensor]:
        img_from, img_to = self._split_inputs(inputs)

        if hasattr(self.backbone, "set_batch_metainfo"):
            self.backbone.set_batch_metainfo(
                self._extract_olmoearth_metas(
                    data_samples,
                    "olmoearth_from_metainfo",
                ),
                self._extract_olmoearth_metas(
                    data_samples,
                    "olmoearth_to_metainfo",
                ),
            )
        x = self.backbone(img_from, img_to)
        if self.with_neck:
            x = self.neck(x)
        return x

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
