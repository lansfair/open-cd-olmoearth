from __future__ import annotations

import ast
import runpy
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def registered_classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "register_module"
            for decorator in node.decorator_list
        )
    }


def main() -> None:
    expected = {
        PROJECT
        / "opencd_universat"
        / "backbones"
        / "universat_backbone.py": {"UniverSatBackbone"},
        PROJECT
        / "opencd_universat"
        / "change_detectors"
        / "universat_siamencoder_decoder.py": {"UniverSatSiamEncoderDecoder"},
        PROJECT / "opencd_universat" / "datasets" / "bright.py": {
            "UniverSatBRIGHTDataset"
        },
        PROJECT
        / "opencd_universat"
        / "necks"
        / "feature_fusion_pyramid.py": {"UniverSatFeatureFusionPyramid"},
        PROJECT / "opencd_universat" / "transforms" / "bright.py": {
            "LoadUniverSatBRIGHTPair",
            "LoadUniverSatBRIGHTAnnotations",
            "BuildUniverSatBRIGHTProxies",
        },
        PROJECT / "opencd_universat" / "transforms" / "formatting.py": {
            "PackUniverSatCDInputs"
        },
    }
    for path, class_names in expected.items():
        missing = class_names - registered_classes(path)
        if missing:
            raise AssertionError(f"{path}: missing registrations {sorted(missing)}")

    config_dir = PROJECT / "configs" / "bright"
    frozen = runpy.run_path(
        str(
            config_dir
            / "universat-base_upernet_1xb1-40k_"
            "bright-256x256-rgb-sar-frozen.py"
        )
    )
    finetune = runpy.run_path(
        str(
            config_dir
            / "universat-base_upernet_1xb1-40k_"
            "bright-256x256-rgb-sar-finetune.py"
        )
    )
    model = frozen["model"]
    assert model["type"] == "UniverSatSiamEncoderDecoder"
    assert model["backbone_from_inchannels"] == 4
    assert model["backbone_to_inchannels"] == 3
    assert model["backbone"]["modalities"] == ["s2_4band", "s1"]
    assert model["test_cfg"]["mode"] == "slide"
    assert frozen["crop_size"] == (256, 256)
    assert frozen["train_pipeline"][2] == dict(
        type="MultiImgRandomCrop",
        crop_size=(256, 256),
        cat_max_ratio=0.75,
    )
    assert frozen["train_pipeline"][5]["type"] == (
        "MultiImgPhotoMetricDistortion"
    )
    assert frozen["train_pipeline"][6]["type"] == (
        "BuildUniverSatBRIGHTProxies"
    )
    assert model["test_cfg"]["crop_size"] == (256, 256)
    assert model["test_cfg"]["stride"] == (192, 192)
    assert frozen["train_cfg"]["max_iters"] == 40000
    assert frozen["train_cfg"]["val_interval"] == 4000
    assert frozen["param_scheduler"][0]["end"] == 3000
    assert frozen["param_scheduler"][1]["type"] == "CosineAnnealingLR"
    assert frozen["param_scheduler"][1]["end"] == 40000
    assert finetune["model"]["backbone"]["freeze_backbone"] is False
    print("UniverSat Open-CD registrations/config static test passed.")


if __name__ == "__main__":
    main()
