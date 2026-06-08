from __future__ import annotations

import numpy as np

from .modalities import RGB_TO_SENTINEL2_L2A


def to_s2_scale(
    image: np.ndarray,
    input_value_range: str,
    input_max_value: float = 255.0,
    target_s2_scale: float = 10000.0,
    clip_to_s2_scale: bool = True,
) -> np.ndarray:
    """Rescale RGB-like values to the Sentinel-2 reflectance scale.

    ``target_s2_scale`` and ``input_max_value`` are configurable so the
    0-255 -> 0-10000 conversion ratio is not hard-coded in dataset code.
    """
    if input_max_value <= 0:
        raise ValueError("input_max_value must be positive")
    if target_s2_scale <= 0:
        raise ValueError("target_s2_scale must be positive")

    image = image.astype(np.float32, copy=False)
    if input_value_range == "s2":
        scaled = image
    elif input_value_range == "0_1":
        scaled = image * target_s2_scale
    elif input_value_range == "0_255":
        scaled = image * (target_s2_scale / input_max_value)
    else:
        raise ValueError("input_value_range must be 0_255, 0_1, or s2")

    if clip_to_s2_scale:
        scaled = np.clip(scaled, 0.0, target_s2_scale)
    return scaled


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
    input_max_value: float = 255.0,
    target_s2_scale: float = 10000.0,
    clip_to_s2_scale: bool = True,
    std_multiplier: float = 2.0,
) -> np.ndarray:
    """Map RGB image values to OLMoEarth Sentinel-2 L2A band layout.

    Filled bands:
      R -> B04
      G -> B03
      B -> B02

    All other Sentinel-2 bands are left as zero. They should also be marked as
    missing through ``present_bands`` metadata so OLMoEarth masks them out
    instead of treating them as valid observations.
    """
    rgb_channel_order = rgb_channel_order.upper()
    if sorted(rgb_channel_order) != ["B", "G", "R"]:
        raise ValueError("rgb_channel_order must be a permutation of RGB")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")

    image = to_s2_scale(
        image=image,
        input_value_range=input_value_range,
        input_max_value=input_max_value,
        target_s2_scale=target_s2_scale,
        clip_to_s2_scale=clip_to_s2_scale,
    )
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
