from __future__ import annotations

from typing import List

from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
from opencd.registry import MODELS
from torch import Tensor

from mmseg.utils import OptSampleList, SampleList


@MODELS.register_module()
class OLMoEarthSiamEncoderDecoder(SiamEncoderDecoder):
    """Open-CD Siamese detector that passes OLMoEarth metadata per date."""

    def _split_inputs(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        expected_channels = self.backbone_inchannels * 2
        if inputs.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels "
                f"({self.backbone_inchannels} per date), got {inputs.shape[1]}"
            )
        img_from, img_to = inputs.split(self.backbone_inchannels, dim=1)
        return img_from, img_to

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
                )
            )
        feat_from = self.backbone(img_from)

        if hasattr(self.backbone, "set_batch_metainfo"):
            self.backbone.set_batch_metainfo(
                self._extract_olmoearth_metas(
                    data_samples,
                    "olmoearth_to_metainfo",
                )
            )
        feat_to = self.backbone(img_to)

        if self.with_neck:
            return self.neck(feat_from, feat_to)
        raise ValueError("`neck` is required for OLMoEarthSiamEncoderDecoder.")

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


@MODELS.register_module()
class OLMoEarthHeteroSiamEncoderDecoder(OLMoEarthSiamEncoderDecoder):
    """OLMoEarth change detector with separate pre/post modality backbones."""

    def __init__(
        self,
        backbone_from: dict,
        backbone_to: dict,
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
        if pretrained is not None:
            raise ValueError(
                "Use init_cfg inside backbone_from/backbone_to for OLMoEarth weights."
            )
        super(SiamEncoderDecoder, self).__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.backbone_from = MODELS.build(backbone_from)
        self.backbone_to = MODELS.build(backbone_to)
        self.backbone = self.backbone_from
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.backbone_from_inchannels = backbone_from_inchannels
        self.backbone_to_inchannels = backbone_to_inchannels
        self.backbone_inchannels = backbone_from_inchannels
        assert self.with_decode_head

    def _split_inputs(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        expected_channels = self.backbone_from_inchannels + self.backbone_to_inchannels
        if inputs.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels "
                f"({self.backbone_from_inchannels} pre + "
                f"{self.backbone_to_inchannels} post), got {inputs.shape[1]}"
            )
        return inputs.split(
            [self.backbone_from_inchannels, self.backbone_to_inchannels],
            dim=1,
        )

    def extract_feat(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> List[Tensor]:
        img_from, img_to = self._split_inputs(inputs)

        if hasattr(self.backbone_from, "set_batch_metainfo"):
            self.backbone_from.set_batch_metainfo(
                self._extract_olmoearth_metas(
                    data_samples,
                    "olmoearth_from_metainfo",
                )
            )
        feat_from = self.backbone_from(img_from)

        if hasattr(self.backbone_to, "set_batch_metainfo"):
            self.backbone_to.set_batch_metainfo(
                self._extract_olmoearth_metas(
                    data_samples,
                    "olmoearth_to_metainfo",
                )
            )
        feat_to = self.backbone_to(img_to)

        if self.with_neck:
            return self.neck(feat_from, feat_to)
        raise ValueError("`neck` is required for OLMoEarthHeteroSiamEncoderDecoder.")
