from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects" / "olmoearth"
CONFIG_DIR = PROJECT / "configs" / "bright"


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _assert_contains(path: Path, needles: list[str]) -> None:
    text = _read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise AssertionError(f"{path} missing expected strings: {missing}")


def _assert_python_syntax(paths: list[Path]) -> None:
    for path in paths:
        ast.parse(_read(path), filename=str(path))


def main() -> None:
    py_files = sorted(PROJECT.rglob("*.py"))
    _assert_python_syntax(py_files)

    _assert_contains(
        CONFIG_DIR / "olmoearth-base_upernet_1xb1-80k_bright-1024x1024-rgbproxy.py",
        [
            "olmoearth_bright_upernet_rgbproxy.py",
            "bright_1024_olmoearth_rgbproxy.py",
            "AmpOptimWrapper",
            "FreezeBackboneUntilEpochHook",
            "max_iters=80000",
        ],
    )
    _assert_contains(
        CONFIG_DIR
        / "olmoearth-base_upernet_1xb1-1iter_bright-1024x1024-rgbproxy-smoke.py",
        [
            "olmoearth-base_upernet_1xb1-80k_bright-1024x1024-rgbproxy.py",
            "max_iters=1",
            "val_interval=2",
            "num_workers=0",
            "FreezeBackboneUntilEpochHook",
        ],
    )
    _assert_contains(
        CONFIG_DIR / "olmoearth_bright_upernet_rgbproxy.py",
        [
            "custom_imports",
            "projects.olmoearth",
            "OLMoEarthSiamEncoderDecoder",
            "OlmoEarthBackbone",
            "OLMoEarthFeatureFusionPyramid",
            "backbone_inchannels=12",
            'modality="sentinel2_l2a"',
            "scales=[1, 0.5, 0.25, 0.125]",
            "test_cfg=dict(size_divisor=32)",
        ],
    )
    _assert_contains(
        CONFIG_DIR / "olmoearth_bright_upernet_s2_s1proxy.py",
        [
            "OLMoEarthHeteroSiamEncoderDecoder",
            "backbone_from_inchannels=12",
            "backbone_to_inchannels=2",
            'modality="sentinel2_l2a"',
            'modality="sentinel1"',
            "fast_pass=False",
        ],
    )
    _assert_contains(
        CONFIG_DIR
        / "olmoearth-base_upernet_1xb1-80k_bright-1024x1024-s1-vv-zero-vh.py",
        [
            "olmoearth_bright_upernet_s2_s1proxy.py",
            "bright_1024_olmoearth_s1_vv_zero_vh.py",
            "AmpOptimWrapper",
            "FreezeBackboneUntilEpochHook",
        ],
    )
    _assert_contains(
        CONFIG_DIR
        / "olmoearth-base_upernet_1xb1-80k_bright-1024x1024-s1-dup2.py",
        [
            "olmoearth_bright_upernet_s2_s1proxy.py",
            "bright_1024_olmoearth_s1_dup2.py",
            "AmpOptimWrapper",
            "FreezeBackboneUntilEpochHook",
        ],
    )
    _assert_contains(
        CONFIG_DIR / "bright_1024_olmoearth_rgbproxy.py",
        [
            "OLMoEarthBRIGHTDataset",
            "LoadOLMoEarthBRIGHTPair",
            "LoadOLMoEarthBRIGHTAnnotations",
            "PackOLMoEarthCDInputs",
            "MultiImgRandomCrop",
            "MultiImgRandomFlip",
        ],
    )
    _assert_contains(
        PROJECT / "opencd_olmoearth" / "transforms" / "bright.py",
        [
            "flag=\"grayscale\"",
            "np.stack([image, image, image], axis=-1)",
            "rgb_to_pseudo_s2",
            "bright_post_sar_rgbproxy_to_s2",
            "present_bands=mapped_bands",
            "post_sar_mode",
            "s1_vv_zero_vh",
            "s1_dup2",
            "sar_db_range",
            "proxy_filled_bands",
            "bright_post_sar_",
        ],
    )
    _assert_contains(
        PROJECT / "opencd_olmoearth" / "utils" / "rgbproxy.py",
        [
            "def rgb_to_pseudo_s2",
            "RGB_TO_SENTINEL2_L2A",
            "np.zeros",
            "normalize_band",
        ],
    )
    _assert_contains(
        PROJECT
        / "opencd_olmoearth"
        / "change_detectors"
        / "olmoearth_siamencoder_decoder.py",
        [
            "olmoearth_from_metainfo",
            "olmoearth_to_metainfo",
            "OLMoEarthHeteroSiamEncoderDecoder",
            "self.backbone.set_batch_metainfo",
            "self.backbone_from.set_batch_metainfo",
            "self.backbone_to.set_batch_metainfo",
            "self.encode_decode(inputs, batch_img_metas)",
        ],
    )
    _assert_contains(
        PROJECT / "opencd_olmoearth" / "necks" / "feature_fusion_pyramid.py",
        [
            "MultiLevelNeck",
            "in_channels=[fused_dim]",
            "out_channels=out_channels",
            "scales=list(scales)",
        ],
    )
    _assert_contains(
        PROJECT / "tools" / "smoke_build_runner.py",
        [
            "Config.fromfile",
            "Runner.from_cfg",
            "olmoearth-base_upernet_1xb1-1iter_bright-1024x1024-rgbproxy-smoke.py",
        ],
    )
    _assert_contains(
        PROJECT / "tools" / "check_server_env.py",
        [
            "olmoearth_pretrain",
            "Required package is not importable",
            "Config.fromfile",
            "cfg.olmoearth_config",
            "cfg.olmoearth_checkpoint",
            "BRIGHT data_root layout",
        ],
    )
    _assert_contains(
        PROJECT / "tools" / "test_bright_rgbproxy_dataflow.py",
        [
            "test_rgb_to_pseudo_s2_mapping",
            "test_sar_repeat_to_rgb_proxy",
            "test_s1_proxy_uses_olmoearth_computed_normalization",
            "BRIGHT rgbproxy/S1-proxy dataflow tests passed.",
        ],
    )
    _assert_contains(
        PROJECT / "tools" / "test_project_imports.py",
        [
            "importlib.import_module(\"projects.olmoearth\")",
            "OlmoEarthBackbone",
            "OLMoEarthHeteroSiamEncoderDecoder",
            "OLMoEarthSiamEncoderDecoder",
            "OLMoEarth Open-CD project import/register smoke test passed.",
        ],
    )
    print(f"Validated {len(py_files)} Python files and BRIGHT proxy configs.")


if __name__ == "__main__":
    main()
