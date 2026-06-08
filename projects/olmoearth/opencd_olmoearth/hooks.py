from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.structures import SegDataSample
from opencd.registry import HOOKS
from opencd.visualization import CDLocalVisualizer


@HOOKS.register_module()
class FreezeBackboneUntilEpochHook(Hook):
    """Freeze backbone parameters until a target epoch or iteration.

    ``unfreeze_epoch`` is kept for compatibility with existing OLMoEarth
    configs. In iter-based training, set it to the iteration count where the
    backbone should be unfrozen.
    """

    priority = "VERY_HIGH"

    def __init__(self, unfreeze_epoch: int | None = None) -> None:
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    @staticmethod
    def _set_backbone_requires_grad(model, requires_grad: bool) -> None:
        module = model.module if hasattr(model, "module") else model
        backbones = []
        for name in ("backbone", "backbone_from", "backbone_to"):
            backbone = getattr(module, name, None)
            if backbone is not None and backbone not in backbones:
                backbones.append(backbone)
        for backbone in backbones:
            for param in backbone.parameters():
                param.requires_grad = requires_grad

    def before_train(self, runner) -> None:
        if self.unfreeze_epoch is None:
            self._set_backbone_requires_grad(runner.model, False)
        else:
            self._set_backbone_requires_grad(runner.model, False)
            self._unfrozen = False

    def before_train_epoch(self, runner) -> None:
        if self.unfreeze_epoch is None:
            return
        if self._unfrozen:
            return
        if runner.epoch >= self.unfreeze_epoch:
            self._set_backbone_requires_grad(runner.model, True)
            self._unfrozen = True

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        if self.unfreeze_epoch is None:
            return
        if self._unfrozen:
            return
        if runner.iter >= self.unfreeze_epoch:
            self._set_backbone_requires_grad(runner.model, True)
            self._unfrozen = True


def _inputs_at(inputs, index: int):
    if isinstance(inputs, torch.Tensor):
        return inputs[index: index + 1]
    if isinstance(inputs, (list, tuple)):
        return [inputs[index]]
    raise TypeError(f"Unsupported visualization inputs: {type(inputs)}")


def _stretch_to_uint8(image: np.ndarray) -> np.ndarray:
    out = np.empty(image.shape, dtype=np.uint8)
    for channel_idx in range(image.shape[-1]):
        channel = image[..., channel_idx].astype(np.float32, copy=False)
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            out[..., channel_idx] = 0
            continue
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            out[..., channel_idx] = 0
            continue
        scaled = (channel - lo) / (hi - lo)
        out[..., channel_idx] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return out


@HOOKS.register_module()
class OLMoEarthOSCDVisualizationHook(SegVisualizationHook):
    """Visualize OSCD predictions from the batch tensor."""

    def __init__(
        self,
        draw: bool = False,
        interval: int = 50,
        show: bool = False,
        wait_time: float = 0.0,
    ) -> None:
        super().__init__(
            draw=draw,
            interval=interval,
            show=show,
            wait_time=wait_time,
            backend_args=None,
        )
        self._visualizer: CDLocalVisualizer = CDLocalVisualizer.get_current_instance()
        self._test_index = 0

    @staticmethod
    def _make_image(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.detach().cpu()
        image = tensor.numpy().transpose(1, 2, 0)
        return _stretch_to_uint8(image[..., [2, 1, 0]])

    @staticmethod
    def _sample_name(data_sample: SegDataSample, prefix: str) -> str:
        if "sample_id" in data_sample.metainfo:
            return f"{prefix}_{data_sample.metainfo['sample_id']}"
        return prefix

    def _draw_sample(
        self,
        inputs,
        data_sample: SegDataSample,
        output_idx: int,
        mode: str,
        step: int,
    ) -> None:
        img = self._make_image(_inputs_at(inputs, output_idx)[0])
        self._visualizer.add_datasample(
            self._sample_name(data_sample, mode),
            img,
            [],
            data_sample=data_sample,
            show=self.show,
            wait_time=self.wait_time,
            step=step,
            draw_gt=False,
        )

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
    ) -> None:
        if self.draw is False:
            return
        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.interval != 0:
            return
        inputs = data_batch["inputs"]
        for output_idx, data_sample in enumerate(outputs):
            self._draw_sample(
                inputs,
                data_sample,
                output_idx,
                "val",
                total_curr_iter,
            )

    def after_test_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
    ) -> None:
        if self.draw is False:
            return
        inputs = data_batch["inputs"]
        for output_idx, data_sample in enumerate(outputs):
            self._test_index += 1
            self._draw_sample(
                inputs,
                data_sample,
                output_idx,
                "test",
                self._test_index,
            )
