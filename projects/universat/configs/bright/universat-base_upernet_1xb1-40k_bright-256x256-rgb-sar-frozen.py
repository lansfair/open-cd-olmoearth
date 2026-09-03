_base_ = "../../../../configs/_base_/default_runtime.py"

custom_imports = dict(imports=["projects.universat"], allow_failed_imports=False)

data_root = __import__("os").environ.get(
    "BRIGHT_DATA_ROOT",
    "F:/data/DFC2025 BRIGHT",
)
crop_size = (256, 256)
norm_cfg = dict(type="SyncBN", requires_grad=True)

train_pipeline = [
    dict(type="LoadUniverSatBRIGHTPair"),
    dict(type="LoadUniverSatBRIGHTAnnotations"),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(
        type="MultiImgPhotoMetricDistortion",
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(type="BuildUniverSatBRIGHTProxies"),
    dict(type="PackUniverSatCDInputs"),
]
test_pipeline = [
    dict(type="LoadUniverSatBRIGHTPair"),
    dict(type="LoadUniverSatBRIGHTAnnotations"),
    dict(type="BuildUniverSatBRIGHTProxies"),
    dict(type="PackUniverSatCDInputs"),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="UniverSatBRIGHTDataset",
        data_root=data_root,
        ann_file="train_set.txt",
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatBRIGHTDataset",
        data_root=data_root,
        ann_file="val_set.txt",
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatBRIGHTDataset",
        data_root=data_root,
        ann_file="test_set.txt",
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type="mmseg.IoUMetric",
    iou_metrics=["mIoU", "mFscore"],
    ignore_index=255,
)
test_evaluator = val_evaluator

universat_checkpoint = (
    __import__("os").environ.get(
        "UNIVERSAT_CHECKPOINT",
        "checkpoints/universat_base.safetensors",
    )
)
embed_dim = 768
pyramid_channels = 256

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=4),
)

model = dict(
    type="UniverSatSiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_from_inchannels=4,
    backbone_to_inchannels=3,
    from_modality="s2_4band",
    to_modality="s1",
    default_from_date=0,
    default_to_date=1,
    backbone=dict(
        type="UniverSatBackbone",
        modalities=["s2_4band", "s1"],
        embed_dim=embed_dim,
        num_heads=12,
        patch_size=40,
        output_grid=None,
        block_type=("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"),
        n_registers=4,
        gating=True,
        compile_encoder=False,
        freeze_backbone=True,
        frozen_stages=-1,
        init_cfg=dict(type="Pretrained", checkpoint=universat_checkpoint),
    ),
    neck=dict(
        type="UniverSatFeatureFusionPyramid",
        policy="abs_diff",
        embed_dim=embed_dim,
        out_channels=pyramid_channels,
        scales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type="mmseg.UPerHead",
        in_channels=[pyramid_channels] * 4,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=4,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type="mmseg.CrossEntropyLoss",
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_weight=1.0,
            ),
            dict(
                type="mmseg.DiceLoss",
                use_sigmoid=False,
                loss_weight=0.5,
                ignore_index=255,
            ),
        ],
    ),
    auxiliary_head=dict(
        type="mmseg.FCNHead",
        in_channels=pyramid_channels,
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss",
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=0.4,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(192, 192)),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=3000),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        T_max=37000,
        begin=3000,
        end=40000,
        by_epoch=False,
    ),
]
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=3,
        save_best="mIoU",
        rule="greater",
        save_last=True,
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

log_processor = dict(by_epoch=False)
randomness = dict(seed=0)
auto_scale_lr = dict(enable=False, base_batch_size=1)
work_dir = (
    "./work_dirs/"
    "universat-base_upernet_1xb1-40k_bright-256x256-rgb-sar-frozen"
)
