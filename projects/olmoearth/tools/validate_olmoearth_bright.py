from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects" / "olmoearth"
CONFIG_DIR = PROJECT / "configs" / "bright"

EXPECTED_CONFIGS = {
    "olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py": [
        "default_runtime.py",
        'modality="sentinel2_l2a"',
        "backbone_inchannels=12",
        "unfreeze_epoch=None",
    ],
    "olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-finetune.py": [
        "s2proxy-frozen.py",
        "custom_hooks = []",
    ],
    "olmoearth-10m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-frozen.py": [
        "s2proxy-frozen.py",
        "OlmoEarth-10m",
        'pre_rgb_mode="native_rgb"',
        'post_sar_mode="native_sar"',
        'modality="rgb"',
        'modality="sar"',
        "unfreeze_epoch=None",
    ],
    "olmoearth-10m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-finetune.py": [
        "rgb-sar-frozen.py",
        "custom_hooks = []",
    ],
    "olmoearth-2m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-frozen.py": [
        "olmoearth-10m_upernet",
        "OlmoEarth-2m",
        'modality="rgb"',
        'modality="sar"',
        "unfreeze_epoch=None",
    ],
    "olmoearth-2m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-finetune.py": [
        "rgb-sar-frozen.py",
        "custom_hooks = []",
    ],
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    actual = {path.name for path in CONFIG_DIR.glob("*.py")}
    expected = set(EXPECTED_CONFIGS)
    if actual != expected:
        raise AssertionError(
            f"BRIGHT configs differ: missing={expected - actual}, extra={actual - expected}"
        )

    for filename, needles in EXPECTED_CONFIGS.items():
        path = CONFIG_DIR / filename
        text = _read(path)
        ast.parse(text, filename=str(path))
        missing = [needle for needle in needles if needle not in text]
        if missing:
            raise AssertionError(f"{path} missing expected strings: {missing}")

    for path in sorted((PROJECT / "opencd_olmoearth").rglob("*.py")):
        ast.parse(_read(path), filename=str(path))

    transform_text = _read(PROJECT / "opencd_olmoearth" / "transforms" / "bright.py")
    for needle in ("native_rgb", "native_sar", "rgbproxy_to_s2"):
        if needle not in transform_text:
            raise AssertionError(f"BRIGHT transform missing {needle}")

    print("Validated exactly 6 BRIGHT configs and project Python syntax.")


if __name__ == "__main__":
    main()
