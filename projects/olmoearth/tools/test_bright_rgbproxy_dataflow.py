from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
UTILS_DIR = ROOT / "projects" / "olmoearth" / "opencd_olmoearth" / "utils"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package_name in [
    "projects",
    "projects.olmoearth",
    "projects.olmoearth.opencd_olmoearth",
    "projects.olmoearth.opencd_olmoearth.utils",
]:
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules.setdefault(package_name, package)

modalities = _load_module(
    "projects.olmoearth.opencd_olmoearth.utils.modalities",
    UTILS_DIR / "modalities.py",
)
rgbproxy = _load_module(
    "projects.olmoearth.opencd_olmoearth.utils.rgbproxy",
    UTILS_DIR / "rgbproxy.py",
)


def _fake_norm_config() -> dict[str, dict[str, float]]:
    return {
        band: {"mean": 0.0, "std": 5000.0}
        for band in modalities.SENTINEL2_L2A_BANDS
    }


def _fake_s1_norm_config() -> dict[str, dict[str, float]]:
    return {
        "vv": {"mean": -10.0, "std": 5.0},
        "vh": {"mean": -20.0, "std": 10.0},
    }


def test_rgb_to_pseudo_s2_mapping() -> None:
    image = np.asarray([[[255, 0, 128]]], dtype=np.float32)
    pseudo = rgbproxy.rgb_to_pseudo_s2(
        image=image,
        band_names=list(modalities.SENTINEL2_L2A_BANDS),
        norm_config=_fake_norm_config(),
        input_value_range="0_255",
        std_multiplier=1.0,
    )
    band_names = list(modalities.SENTINEL2_L2A_BANDS)
    b02 = band_names.index("B02")
    b03 = band_names.index("B03")
    b04 = band_names.index("B04")

    assert pseudo.shape == (1, 1, 12)
    assert np.isclose(pseudo[0, 0, b04], 1.5)
    assert np.isclose(pseudo[0, 0, b03], 0.5)
    assert np.isclose(pseudo[0, 0, b02], 0.5 + 128 / 255)

    missing = np.delete(pseudo[0, 0], [b02, b03, b04])
    assert np.allclose(missing, 0.0)


def test_sar_repeat_to_rgb_proxy() -> None:
    sar = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    sar_rgb = np.stack([sar, sar, sar], axis=-1).astype(np.float32)
    pseudo = rgbproxy.rgb_to_pseudo_s2(
        image=sar_rgb,
        band_names=list(modalities.SENTINEL2_L2A_BANDS),
        norm_config=_fake_norm_config(),
        input_value_range="0_255",
        std_multiplier=1.0,
    )
    band_names = list(modalities.SENTINEL2_L2A_BANDS)
    b02 = band_names.index("B02")
    b03 = band_names.index("B03")
    b04 = band_names.index("B04")

    assert pseudo.shape == (2, 2, 12)
    assert np.allclose(pseudo[..., b02], pseudo[..., b03])
    assert np.allclose(pseudo[..., b03], pseudo[..., b04])
    assert np.isclose(pseudo[1, 1, b04], 0.5 + 4 / 255)


def test_s1_proxy_uses_olmoearth_computed_normalization() -> None:
    sar_db = np.asarray([[-30.0, 5.0]], dtype=np.float32)
    norm_config = _fake_s1_norm_config()

    vv = rgbproxy.normalize_band(
        sar_db,
        "vv",
        norm_config,
        std_multiplier=2.0,
    )
    vh = rgbproxy.normalize_band(
        sar_db,
        "vh",
        norm_config,
        std_multiplier=2.0,
    )

    expected_vv = (sar_db - (-10.0 - 2.0 * 5.0)) / (4.0 * 5.0)
    expected_vh = (sar_db - (-20.0 - 2.0 * 10.0)) / (4.0 * 10.0)
    assert np.allclose(vv, expected_vv)
    assert np.allclose(vh, expected_vh)


def main() -> None:
    test_rgb_to_pseudo_s2_mapping()
    test_sar_repeat_to_rgb_proxy()
    test_s1_proxy_uses_olmoearth_computed_normalization()
    print("BRIGHT rgbproxy/S1-proxy dataflow tests passed.")


if __name__ == "__main__":
    main()
