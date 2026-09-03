from __future__ import annotations

from typing import List

import torch
from mmseg.structures import SegDataSample
from mmseg.utils import OptSampleList, SampleList
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
from opencd.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class UniverSatSiamEncoderDecoder(SiamEncoderDecoder):
    """Open-CD Siamese detector for heterogeneous UniverSat observations.

    Open-CD keeps the before/after rasters in one channel-concatenated tensor.
    This detector splits them, restores the modality dictionaries and date
    tensors expected by UniverSat, applies one shared backbone to both
    observations, and delegates temporal feature fusion to the neck.

    The BRIGHT adapter uses ``s2_4band`` for the pre-disaster RGB proxy and
    ``s1`` for the post-disaster single-channel SAR proxy.
    """

    def __init__(
        self,
        *args,
        backbone_from_inchannels: int,
        backbone_to_inchannels: int,
        from_modality: str,
        to_modality: str,
        default_from_date: int = 0,
        default_to_date: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            backbone_inchannels=backbone_from_inchannels,
            **kwargs,
        )
        if backbone_from_inchannels <= 0 or backbone_to_inchannels <= 0:
            raise ValueError("Per-observation channel counts must be positive.")
        self.backbone_from_inchannels = int(backbone_from_inchannels)
        self.backbone_to_inchannels = int(backbone_to_inchannels)
        self.from_modality = from_modality
        self.to_modality = to_modality
        self.default_from_date = int(default_from_date)
        self.default_to_date = int(default_to_date)

    def _split_inputs(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        expected = self.backbone_from_inchannels + self.backbone_to_inchannels
        if inputs.ndim != 4 or inputs.shape[1] != expected:
            raise ValueError(
                "UniverSat change-detection inputs must have shape "
                f"(B, {expected}, H, W); got {tuple(inputs.shape)}."
            )
        return inputs.split(
            [self.backbone_from_inchannels, self.backbone_to_inchannels],
            dim=1,
        )

    @staticmethod
    def _extract_dates(
        data_samples: OptSampleList,
        key: str,
        batch_size: int,
        default: int,
        device: torch.device,
    ) -> Tensor:
        if data_samples is None:
            return torch.full(
                (batch_size, 1),
                default,
                dtype=torch.long,
                device=device,
            )
        if len(data_samples) != batch_size:
            raise ValueError(
                f"Expected {batch_size} data samples, got {len(data_samples)}."
            )
        values = []
        for data_sample in data_samples:
            value = data_sample.metainfo.get(key, default)
            date = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
            if date.numel() != 1:
                raise ValueError(
                    f"{key!r} must contain one date index per BRIGHT "
                    f"observation, got {date.tolist()}."
                )
            values.append(date)
        return torch.stack(values, dim=0)

    def _make_observation(
        self,
        image: Tensor,
        modality: str,
        date_key: str,
        default_date: int,
        data_samples: OptSampleList,
    ) -> dict[str, Tensor]:
        dates = self._extract_dates(
            data_samples,
            date_key,
            image.shape[0],
            default_date,
            image.device,
        )
        return {
            modality: image.unsqueeze(1),
            f"{modality}_dates": dates,
        }

    def extract_feat(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
    ) -> List[Tensor]:
        img_from, img_to = self._split_inputs(inputs)
        from_inputs = self._make_observation(
            img_from,
            self.from_modality,
            "universat_from_date",
            self.default_from_date,
            data_samples,
        )
        to_inputs = self._make_observation(
            img_to,
            self.to_modality,
            "universat_to_date",
            self.default_to_date,
            data_samples,
        )

        feat_from = self.backbone(from_inputs)
        feat_to = self.backbone(to_inputs)
        if self.with_neck:
            return self.neck(feat_from, feat_to)
        raise ValueError("`neck` is required for UniverSatSiamEncoderDecoder.")

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs, data_samples)
        losses = {}
        losses.update(self._decode_head_forward_train(x, data_samples))
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(x, data_samples))
        return losses

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
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
            batch_img_metas = [sample.metainfo for sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                    universat_from_date=self.default_from_date,
                    universat_to_date=self.default_to_date,
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
