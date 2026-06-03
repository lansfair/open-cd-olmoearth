from __future__ import annotations

from mmengine.hooks import Hook
from opencd.registry import HOOKS


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
