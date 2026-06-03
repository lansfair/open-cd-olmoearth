from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Open-CD Runner for the OLMoEarth BRIGHT config."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=(
            "projects/olmoearth/configs/bright/"
            "olmoearth-base_upernet_1xb1-1iter_bright-1024x1024-rgbproxy-smoke.py"
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional temporary work directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    runner = Runner.from_cfg(cfg)
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    print(f"Runner: {type(runner).__name__}")
    print(f"Model: {type(model).__name__}")
    print(f"Backbone: {type(model.backbone).__name__}")
    print(f"Neck: {type(model.neck).__name__}")
    print(f"Work dir: {Path(cfg.work_dir)}")


if __name__ == "__main__":
    main()

