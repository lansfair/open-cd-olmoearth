_base_ = "./olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py"

crop_size = (1024, 1024)
norm_cfg = dict(type="SyncBN", requires_grad=True)
embed_dim = 768

train_pipeline = [
    dict(
        type="LoadOLMoEarthBRIGHTPair",
        pre_rgb_mode="s2_proxy",
        post_sar_mode="s1_dup2",
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
        pre_rgb_mode="s2_proxy",
        post_sar_mode="s1_dup2",
    ),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

model = dict(
    _delete_=True,
    type="OLMoEarthMultiModalEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_from_inchannels=12,
    backbone_to_inchannels=2,
    backbone=dict(
        type="OlmoEarthMultiModalBackbone",
        model_config_path=olmoearth_config,
        from_modality="sentinel2_l2a",
        to_modality="sentinel1",
        patch_size=16,
        num_timesteps=1,
        out_channels=embed_dim,
        fusion_policy="abs_diff",
        fast_pass=False,
        init_cfg=dict(type="Pretrained", checkpoint=olmoearth_checkpoint),
    ),
    neck=dict(
        type="mmseg.MultiLevelNeck",
        in_channels=[embed_dim],
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
work_dir = "./work_dirs/olmoearth-native_multimodal-upernet_1xb1-80k_bright-1024x1024-s2-s1dup2-frozen"
