from __future__ import annotations

import argparse
import importlib
from pathlib import Path


DEFAULT_CONFIG = (
    "projects/olmoearth/configs/bright/"
    "olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check server dependencies and paths for OLMoEarth BRIGHT."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-paths",
        action="store_true",
        help="Only check imports and config parsing.",
    )
    return parser.parse_args()


def _check_imports() -> None:
    packages = [
        "torch",
        "mmcv",
        "mmengine",
        "mmseg",
        "opencd",
        "olmoearth_pretrain",
    ]
    for package in packages:
        try:
            module = importlib.import_module(package)
        except ImportError as exc:
            raise ImportError(
                f"Required package is not importable: {package}. "
                "Activate the Open-CD/OLMoEarth training environment first."
            ) from exc
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] import {package}: {version}")


def _check_file(path: str | Path, label: str) -> None:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    print(f"[OK] {label}: {resolved}")


def _check_bright_layout(data_root: str | Path) -> None:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"BRIGHT data_root does not exist: {root}")

    pre_dirs = [
        root / "pre-event",
        root / "pre-event_wo_ukraine_myanmar_mexico",
    ]
    if not any(path.is_dir() for path in pre_dirs):
        raise FileNotFoundError(
            "BRIGHT pre-event directory not found; expected one of: "
            + ", ".join(str(path) for path in pre_dirs)
        )
    for name in ["post-event", "target"]:
        _check_file(root / name, f"BRIGHT {name} directory")
    for name in ["train_set.txt", "val_set.txt", "test_set.txt"]:
        _check_file(root / name, f"BRIGHT split {name}")
    print(f"[OK] BRIGHT data_root layout: {root}")


def main() -> None:
    args = parse_args()
    _check_imports()

    from mmengine.config import Config

    cfg = Config.fromfile(args.config)
    print(f"[OK] config parsed: {args.config}")
    print(f"[OK] model type: {cfg.model.type}")
    print(f"[OK] dataset type: {cfg.train_dataloader.dataset.type}")

    if args.skip_paths:
        return

    _check_file(cfg.olmoearth_config, "OLMoEarth config")
    _check_file(cfg.olmoearth_checkpoint, "OLMoEarth checkpoint")
    _check_bright_layout(cfg.data_root)


if __name__ == "__main__":
    main()
