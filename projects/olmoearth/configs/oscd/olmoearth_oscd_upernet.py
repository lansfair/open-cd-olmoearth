custom_imports = dict(
    imports=["projects.olmoearth"],
    allow_failed_imports=False,
)

norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    size=(256, 256),
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32),
)

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
patch_size = 4
embed_dim = 768

model = dict(
    type="OLMoEarthSiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_inchannels=12,
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=1,
        out_channels=embed_dim,
        pooling_type="mean",
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
        in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss",
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type="mmseg.FCNHead",
        in_channels=embed_dim,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
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
    test_cfg=dict(mode="slide", crop_size=(256, 256), stride=(128, 128)),
)
