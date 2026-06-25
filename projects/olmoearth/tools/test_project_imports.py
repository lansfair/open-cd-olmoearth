from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class _Registry:
    def __init__(self) -> None:
        self.modules = {}

    def register_module(self):
        def decorator(cls):
            self.modules[cls.__name__] = cls
            return cls

        return decorator


MODELS = _Registry()
DATASETS = _Registry()
TRANSFORMS = _Registry()
HOOKS = _Registry()


def _module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _install_stubs() -> None:
    class BaseModule(torch.nn.Module):
        def __init__(self, init_cfg=None) -> None:
            super().__init__()
            self.init_cfg = init_cfg

    class Hook:
        pass

    class BaseTransform:
        pass

    class PixelData:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class SegDataSample:
        def __init__(self) -> None:
            self.metainfo = {}

        def set_metainfo(self, metainfo) -> None:
            self.metainfo.update(metainfo)

    class CheckpointLoader:
        @staticmethod
        def load_checkpoint(*args, **kwargs):
            return {}

    class MultiLevelNeck(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.kwargs = kwargs

        def forward(self, inputs):
            return tuple(inputs)

    class SiamEncoderDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class EncoderDecoder(torch.nn.Module):
        def __init__(
            self,
            backbone=None,
            decode_head=None,
            neck=None,
            auxiliary_head=None,
            train_cfg=None,
            test_cfg=None,
            data_preprocessor=None,
            pretrained=None,
            init_cfg=None,
        ) -> None:
            super().__init__()
            self.backbone = backbone
            self.decode_head = decode_head
            self.neck = neck
            self.auxiliary_head = auxiliary_head
            self.train_cfg = train_cfg
            self.test_cfg = test_cfg
            self.data_preprocessor = data_preprocessor

    class _BaseCDDataset:
        pass

    _module("mmcv", imfrombytes=lambda *args, **kwargs: None)
    _module(
        "mmcv.transforms",
        BaseTransform=BaseTransform,
        to_tensor=torch.as_tensor,
    )
    _module("mmengine")
    _module("mmengine.model", BaseModule=BaseModule)
    _module("mmengine.hooks", Hook=Hook)
    _module("mmengine.runner")
    _module("mmengine.runner.checkpoint", CheckpointLoader=CheckpointLoader)
    _module("mmengine.structures", PixelData=PixelData)
    _module("mmengine.fileio", get=lambda *args, **kwargs: b"")

    _module("mmseg")
    _module("mmseg.registry", MODELS=MODELS, DATASETS=DATASETS, TRANSFORMS=TRANSFORMS)
    _module("mmseg.models")
    _module("mmseg.models.necks", MultiLevelNeck=MultiLevelNeck)
    _module("mmseg.models.segmentors")
    _module("mmseg.models.segmentors.encoder_decoder", EncoderDecoder=EncoderDecoder)
    _module("mmseg.structures", SegDataSample=SegDataSample)
    _module("mmseg.utils", OptSampleList=object, SampleList=list)

    _module("opencd")
    _module(
        "opencd.registry",
        MODELS=MODELS,
        DATASETS=DATASETS,
        TRANSFORMS=TRANSFORMS,
        HOOKS=HOOKS,
    )
    _module("opencd.datasets")
    _module("opencd.datasets.basecddataset", _BaseCDDataset=_BaseCDDataset)
    _module("opencd.models")
    _module("opencd.models.change_detectors")
    _module(
        "opencd.models.change_detectors.siamencoder_decoder",
        SiamEncoderDecoder=SiamEncoderDecoder,
    )


def main() -> None:
    _install_stubs()
    importlib.import_module("projects.olmoearth")

    expected = {
        "models": {
            "OlmoEarthBackbone",
            "OlmoEarthMultiModalBackbone",
            "OLMoEarthHeteroSiamEncoderDecoder",
            "OLMoEarthMultiModalEncoderDecoder",
            "OLMoEarthSiamEncoderDecoder",
            "OLMoEarthFeatureFusionPyramid",
        },
        "datasets": {"OLMoEarthBRIGHTDataset"},
        "transforms": {
            "LoadOLMoEarthBRIGHTPair",
            "LoadOLMoEarthBRIGHTAnnotations",
            "PackOLMoEarthCDInputs",
        },
        "hooks": {"FreezeBackboneUntilEpochHook"},
    }
    actual = {
        "models": set(MODELS.modules),
        "datasets": set(DATASETS.modules),
        "transforms": set(TRANSFORMS.modules),
        "hooks": set(HOOKS.modules),
    }
    for key, names in expected.items():
        missing = names - actual[key]
        if missing:
            raise AssertionError(f"Missing registered {key}: {sorted(missing)}")

    print("OLMoEarth Open-CD project import/register smoke test passed.")


if __name__ == "__main__":
    main()
