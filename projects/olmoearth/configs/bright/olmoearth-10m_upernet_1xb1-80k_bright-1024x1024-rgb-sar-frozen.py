_base_ = "./olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py"

crop_size = (1024, 1024)
norm_cfg = dict(type="SyncBN", requires_grad=True)
embed_dim = 768

train_pipeline = [
    dict(
        type="LoadOLMoEarthBRIGHTPair",
        pre_rgb_mode="native_rgb",
        post_sar_mode="native_sar",
    ),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackOLMoEarthCDInputs"),
]
test_pipeline = [
    dict(
        type="LoadOLMoEarthBRIGHTPair",
        pre_rgb_mode="native_rgb",
        post_sar_mode="native_sar",
    ),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-10m"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
olmoearth_init = dict(type="Pretrained", checkpoint=olmoearth_checkpoint)

model = dict(
    _delete_=True,
    type="OLMoEarthHeteroSiamEncoderDecoder",
    data_preprocessor=dict(
        type="DualInputSegDataPreProcessor",
        size=crop_size,
        pad_val=0,
        seg_pad_val=255,
        test_cfg=dict(size_divisor=16),
    ),
    pretrained=None,
    backbone_from_inchannels=4,
    backbone_to_inchannels=1,
    backbone_from=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="rgb",
        patch_size=16,
        num_timesteps=1,
        out_channels=embed_dim,
        fast_pass=False,
        init_cfg=olmoearth_init,
    ),
    backbone_to=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="sar",
        patch_size=16,
        num_timesteps=1,
        out_channels=embed_dim,
        fast_pass=False,
        init_cfg=olmoearth_init,
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

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]
work_dir = "./work_dirs/olmoearth-10m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-frozen"
