from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TRANSFORM_PATH = (
    ROOT
    / "projects"
    / "universat"
    / "opencd_universat"
    / "transforms"
    / "bright.py"
)


def _load_transform_module():
    class _Registry:
        @staticmethod
        def register_module():
            return lambda cls: cls

    class BaseTransform:
        pass

    mmcv = types.ModuleType("mmcv")
    mmcv.imfrombytes = lambda *args, **kwargs: None
    mmcv_transforms = types.ModuleType("mmcv.transforms")
    mmcv_transforms.BaseTransform = BaseTransform
    mmengine = types.ModuleType("mmengine")
    mmengine_fileio = types.ModuleType("mmengine.fileio")
    mmengine.fileio = mmengine_fileio
    opencd = types.ModuleType("opencd")
    opencd_registry = types.ModuleType("opencd.registry")
    opencd_registry.TRANSFORMS = _Registry()
    sys.modules.setdefault("mmcv", mmcv)
    sys.modules.setdefault("mmcv.transforms", mmcv_transforms)
    sys.modules.setdefault("mmengine", mmengine)
    sys.modules.setdefault("mmengine.fileio", mmengine_fileio)
    sys.modules.setdefault("opencd", opencd)
    sys.modules.setdefault("opencd.registry", opencd_registry)

    spec = importlib.util.spec_from_file_location(
        "universat_bright_transform",
        TRANSFORM_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_transform_module()
    transform = module.BuildUniverSatBRIGHTProxies()

    rgb = np.asarray([[[255.0, 127.5, 0.0]]], dtype=np.float32)
    s2 = transform._rgb_to_s2_4band(rgb, "RGB")
    assert s2.shape == (1, 1, 4)
    assert np.allclose(s2[0, 0], [-1.0, 0.0, 1.0, 0.0])

    sar = np.asarray([[[0.0] * 3, [255.0] * 3]], dtype=np.float32)
    s1 = transform._sar_to_s1(sar)
    assert s1.shape == (1, 2, 3)
    assert np.allclose(s1[0, :, 0], [-1.0, 1.0])
    assert np.allclose(s1[..., 0], s1[..., 1])
    assert np.allclose(s1[..., 2], 0.0)
    print("UniverSat BRIGHT proxy transform test passed.")


if __name__ == "__main__":
    main()
