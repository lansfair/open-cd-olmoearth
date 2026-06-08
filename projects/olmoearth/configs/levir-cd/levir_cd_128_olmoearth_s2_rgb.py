dataset_type = "OLMoEarthLEVIRCDDataset"
data_root = "D:/xxx/levir-cd"
crop_size = (128, 128)

# Configurable RGB -> Sentinel-2 reflectance-scale conversion.
# For normal 8-bit RGB PNG:
#   scaled_rgb = rgb * (target_s2_scale / input_max_value)
# If your PNG has already been scaled to [0, 10000], set input_value_range="s2".
input_value_range = "0_255"
input_max_value = 255.0
target_s2_scale = 10000.0

load_pair_cfg = dict(
    type="LoadOLMoEarthLEVIRPair",
    expected_shape=crop_size,
    rgb_channel_order="RGB",
    input_value_range=input_value_range,
    input_max_value=input_max_value,
    target_s2_scale=target_s2_scale,
    clip_to_s2_scale=True,
)

train_pipeline = [
    load_pair_cfg,
    dict(type="LoadOLMoEarthLEVIRAnnotations", binary_threshold=128),
    # Images are already 128x128, so no random crop/resize is needed.
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackOLMoEarthCDInputs"),
]

test_pipeline = [
    load_pair_cfg,
    dict(type="LoadOLMoEarthLEVIRAnnotations", binary_threshold=128),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="train/A",
            img_path_to="train/B",
            seg_map_path="train/label",
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="val/A",
            img_path_to="val/B",
            seg_map_path="val/label",
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="test/A",
            img_path_to="test/B",
            seg_map_path="test/label",
        ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type="mmseg.IoUMetric",
    iou_metrics=["mIoU", "mFscore"],
    ignore_index=255,
)
test_evaluator = val_evaluator
