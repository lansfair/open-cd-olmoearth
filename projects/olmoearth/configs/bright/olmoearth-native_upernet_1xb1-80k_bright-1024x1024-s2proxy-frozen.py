_base_ = "../../../../configs/_base_/default_runtime.py"

custom_imports = dict(imports=["projects.olmoearth"], allow_failed_imports=False)

data_root = "F:/data/DFC2025 BRIGHT"
crop_size = (1024, 1024)
norm_cfg = dict(type="SyncBN", requires_grad=True)

train_pipeline = [
    dict(type="LoadOLMoEarthBRIGHTPair"),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackOLMoEarthCDInputs"),
]
test_pipeline = [
    dict(type="LoadOLMoEarthBRIGHTPair"),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="OLMoEarthBRIGHTDataset",
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
        type="OLMoEarthBRIGHTDataset",
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
        type="OLMoEarthBRIGHTDataset",
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

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
embed_dim = 768

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=16),
)

model = dict(
    type="OLMoEarthSiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_inchannels=12,
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="sentinel2_l2a",
        patch_size=16,
        num_timesteps=1,
        out_channels=embed_dim,
        fast_pass=False,
        init_cfg=dict(type="Pretrained", checkpoint=olmoearth_checkpoint),
    ),
    neck=dict(
        type="OLMoEarthFeatureFusionPyramid",
        policy="abs_diff",
        embed_dim=embed_dim,
        out_channels=embed_dim,
        scales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type="mmseg.UPerHead",
        in_channels=[embed_dim] * 4,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
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
        in_channels=embed_dim,
        in_index=2,
        channels=256,
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
    test_cfg=dict(mode="whole"),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=1000,
        end=80000,
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
work_dir = "./work_dirs/olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen"
