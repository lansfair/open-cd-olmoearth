from __future__ import annotations

import numpy as np

from .modalities import RGB_TO_SENTINEL2_L2A


def to_s2_scale(image: np.ndarray, input_value_range: str) -> np.ndarray:
    if input_value_range == "s2":
        return image
    if input_value_range == "0_1":
        return image * 10000.0
    if input_value_range == "0_255":
        return image * (10000.0 / 255.0)
    raise ValueError("input_value_range must be 0_255, 0_1, or s2")


def normalize_band(
    values: np.ndarray,
    band_name: str,
    norm_config: dict[str, dict[str, float]],
    std_multiplier: float,
) -> np.ndarray:
    stats = norm_config[band_name]
    min_val = stats["mean"] - std_multiplier * stats["std"]
    max_val = stats["mean"] + std_multiplier * stats["std"]
    return (values - min_val) / (max_val - min_val)


def rgb_to_pseudo_s2(
    image: np.ndarray,
    band_names: list[str],
    norm_config: dict[str, dict[str, float]],
    rgb_channel_order: str = "RGB",
    input_value_range: str = "0_255",
    std_multiplier: float = 2.0,
) -> np.ndarray:
    rgb_channel_order = rgb_channel_order.upper()
    if sorted(rgb_channel_order) != ["B", "G", "R"]:
        raise ValueError("rgb_channel_order must be a permutation of RGB")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")

    image = to_s2_scale(image.astype(np.float32, copy=False), input_value_range)
    h, w = image.shape[:2]
    out = np.zeros((h, w, len(band_names)), dtype=np.float32)
    channel_to_index = {name: idx for idx, name in enumerate(rgb_channel_order)}
    for rgb_name, s2_band in RGB_TO_SENTINEL2_L2A.items():
        rgb_idx = channel_to_index[rgb_name]
        band_idx = band_names.index(s2_band)
        out[..., band_idx] = normalize_band(
            image[..., rgb_idx],
            s2_band,
            norm_config,
            std_multiplier,
        )
    return out

