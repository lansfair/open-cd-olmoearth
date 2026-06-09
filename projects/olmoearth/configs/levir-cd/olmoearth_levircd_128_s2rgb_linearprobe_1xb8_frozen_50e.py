# Copyright (c) Open-CD. All rights reserved.
# Single-file config for LEVIR-CD with a linear-probe decoder.
#
# What "linear probe decoder" means here:
#   - The decoder is mmseg.FCNHead(num_convs=0).
#   - num_convs=0 removes FCNHead's intermediate conv blocks.
#   - concat_input=False disables the extra concat conv.
#   - dropout_ratio=0.0 disables dropout.
#   - Therefore the decode head only keeps cls_seg, a 1x1 conv classifier:
#       768 -> 2 classes
#
# No custom LinearProbeHead or custom LinearProbeNeck is required.
#
# Data:
#   A RGB 128x128 png -> pseudo Sentinel-2 L2A
#   B RGB 128x128 png -> pseudo Sentinel-2 L2A
#   label png: 0 = unchanged/background, 255 = changed/foreground
#
# Run from repository root:
#   python tools/train.py projects/olmoearth/configs/levir-cd/olmoearth_levircd_128_s2rgb_fcnhead_numconvs0_1xb8_50e_single.py
#
# This config intentionally does NOT use _base_ inheritance.

# -------------------------------------------------------------------------
# 0. Registry / custom project import
# -------------------------------------------------------------------------
default_scope = "opencd"

custom_imports = dict(
    imports=["projects.olmoearth"],
    allow_failed_imports=False,
)

# -------------------------------------------------------------------------
# 1. Paths
# -------------------------------------------------------------------------
data_root = "D:/xxx/levir-cd"

# OLMoEarth-v1-Base directory. It should contain:
#   config.json
#   weights.pth
olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"

work_dir = "./work_dirs/olmoearth_levircd_128_s2rgb_fcnhead_numconvs0_1xb8_50e_single"

# -------------------------------------------------------------------------
# 2. Dataset / RGB -> Sentinel-2 L2A proxy
# -------------------------------------------------------------------------
dataset_type = "OLMoEarthLEVIRCDDataset"
crop_size = (128, 128)

# RGB scaling.
# The transform computes scale_factor as:
#   target_s2_scale / input_max_value
#
# uint8 PNG:             input_max_value = 255.0
# 0~1 float RGB:         input_max_value = 1.0
# uint16 RGB:            input_max_value = 65535.0
# already 0~10000 RGB:   input_max_value = 10000.0
input_max_value = 255.0
target_s2_scale = 10000.0
rgb_channel_order = "RGB"

# RGB mapping to OLMoEarth Sentinel-2 L2A:
#   R -> B04
#   G -> B03
#   B -> B02
#
# Other Sentinel-2 bands are zero-filled. Only B04/B03/B02 are marked as
# present, so the zero-filled bands are treated as missing by OLMoEarth.
present_bands = ["B04", "B03", "B02"]

pre_timestamp = (1, 1, 2025)
post_timestamp = (2, 1, 2025)

train_pipeline = [
    dict(
        type="LoadOLMoEarthLEVIRPair",
        to_float32=True,
        rgb_channel_order=rgb_channel_order,
        input_max_value=input_max_value,
        target_s2_scale=target_s2_scale,
        expected_shape=crop_size,
        pre_timestamp=pre_timestamp,
        post_timestamp=post_timestamp,
    ),
    dict(type="LoadOLMoEarthLEVIRAnnotations"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackOLMoEarthCDInputs"),
]

test_pipeline = [
    dict(
        type="LoadOLMoEarthLEVIRPair",
        to_float32=True,
        rgb_channel_order=rgb_channel_order,
        input_max_value=input_max_value,
        target_s2_scale=target_s2_scale,
        expected_shape=crop_size,
        pre_timestamp=pre_timestamp,
        post_timestamp=post_timestamp,
    ),
    dict(type="LoadOLMoEarthLEVIRAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="train",
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="val",
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="mmseg.IoUMetric",
    iou_metrics=["mIoU", "mFscore"],
    ignore_index=255,
)
test_evaluator = val_evaluator

# -------------------------------------------------------------------------
# 3. Model: OLMoEarth backbone + existing fusion neck + FCNHead(num_convs=0)
# -------------------------------------------------------------------------
norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=16),
)

patch_size = 16
embed_dim = 768
num_classes = 2

model = dict(
    type="OLMoEarthSiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,

    # Two dates are packed as 12 + 12 channels.
    # The detector splits them into two 12-channel OLMoEarth S2 inputs.
    backbone_inchannels=12,

    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=1,
        out_channels=embed_dim,

        # Keep False because only B02/B03/B04 are observed.
        # Other Sentinel-2 bands are zero-filled and masked as missing.
        fast_pass=False,

        init_cfg=dict(type="Pretrained", checkpoint=olmoearth_checkpoint),
    ),

    # Keep the original project neck. The current OLMoEarthSiamEncoderDecoder
    # requires a neck to fuse the two temporal features.
    #
    # For the FCNHead linear decoder, only one fused feature level is needed.
    # scale=4 upsamples OLMoEarth's stride-16 feature to about stride-4,
    # which is consistent with dense change segmentation.
    neck=dict(
        type="OLMoEarthFeatureFusionPyramid",
        policy="abs_diff",
        embed_dim=embed_dim,
        out_channels=embed_dim,
        scales=[4],
        norm_cfg=norm_cfg,
    ),

    # Linear-probe decoder.
    #
    # In mmseg.FCNHead:
    #   num_convs=0      -> no intermediate conv blocks
    #   concat_input=False -> no concat conv
    #   dropout_ratio=0.0  -> no dropout
    #
    # The only learnable layer inside this decode head is cls_seg:
    #   Conv2d(768, 2, kernel_size=1)
    decode_head=dict(
        type="mmseg.FCNHead",
        in_channels=embed_dim,
        in_index=0,
        channels=embed_dim,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.0,
        num_classes=num_classes,
        ignore_index=255,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss",
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0,
        ),
    ),

    # No auxiliary head. The goal is to use only the linear FCNHead classifier.
    auxiliary_head=None,

    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# -------------------------------------------------------------------------
# 4. Training: epoch-based, 50 epochs
# -------------------------------------------------------------------------
max_epochs = 50

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Freeze OLMoEarth backbone during the whole run.
# This does not change the backbone architecture; it only keeps its pretrained
# parameters fixed while training the neck and FCNHead decoder.
linear_probe_freeze_until_epoch = 1000000000

custom_hooks = [
    dict(
        type="FreezeBackboneUntilEpochHook",
        unfreeze_epoch=linear_probe_freeze_until_epoch,
    ),
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=5e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0, decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=5,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# -------------------------------------------------------------------------
# 5. Runtime / hooks
# -------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_best="mIoU",
        rule="greater",
        save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(
        type="CDVisualizationHook",
        interval=1,
        img_shape=(crop_size[0], crop_size[1], 3),
    ),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="CDLocalVisBackend")]
visualizer = dict(
    type="CDLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    alpha=1.0,
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"

load_from = None
resume = False
randomness = dict(seed=0)
auto_scale_lr = dict(enable=False, base_batch_size=8)

tta_model = dict(type="mmseg.SegTTAModel")
