# 作用：使用 OLMoEarth-10m 原生 rgb 模态在 LEVIR-CD 上训练变化检测模型。
import os

_base_ = "../../../../configs/_base_/default_runtime.py"

custom_imports = dict(imports=["projects.olmoearth"], allow_failed_imports=False)

dataset_type = "LEVIR_CD_Dataset"
data_root = os.path.join(os.environ.get("MM_ARCHIVE_DATA_HOME", "data"), "LEVIR-CD")

crop_size = (512, 512)
patch_size = 16
embed_dim = 768
num_classes = 2
norm_cfg = dict(type="SyncBN", requires_grad=True)

train_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgLoadAnnotations"),
    dict(type="MultiImgRandomRotate", prob=0.5, degree=180),
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
    dict(
        type="RGBPairToOlmoEarth",
        modality="rgb",
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOLMoEarthCDInputs"),
]

test_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgResize", scale=(1024, 1024), keep_ratio=True),
    dict(type="MultiImgLoadAnnotations"),
    dict(
        type="RGBPairToOlmoEarth",
        modality="rgb",
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="train/A",
            img_path_to="train/B",
            seg_map_path="train/label",
        ),
        test_mode=False,
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
        data_prefix=dict(
            img_path_from="val/A",
            img_path_to="val/B",
            seg_map_path="val/label",
        ),
        test_mode=True,
        pipeline=test_pipeline,
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
        data_prefix=dict(
            img_path_from="test/A",
            img_path_to="test/B",
            seg_map_path="test/label",
        ),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type="mmseg.IoUMetric", iou_metrics=["mFscore", "mIoU"])
test_evaluator = val_evaluator

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-10m"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="OLMoEarthSiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_inchannels=4,
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="rgb",
        patch_size=patch_size,
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
        num_classes=num_classes,
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
        num_classes=num_classes,
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
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(256, 256)),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
log_processor = dict(by_epoch=False)
randomness = dict(seed=0)

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
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    visualization=dict(type="CDVisualizationHook", interval=1, img_shape=(1024, 1024, 3)),
)

auto_scale_lr = dict(enable=False, base_batch_size=8)
work_dir = "./work_dirs/olmoearth-10m_upernet_4xb2-40k_levircd-rgb-512x512"
