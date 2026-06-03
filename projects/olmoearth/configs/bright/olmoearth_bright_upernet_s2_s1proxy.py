custom_imports = dict(
    imports=["projects.olmoearth"],
    allow_failed_imports=False,
)

norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    size=(1024, 1024),
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=16),
)

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
patch_size = 16
embed_dim = 768

olmoearth_init = dict(type="Pretrained", checkpoint=olmoearth_checkpoint)

model = dict(
    type="OLMoEarthHeteroSiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_from_inchannels=12,
    backbone_to_inchannels=2,
    backbone_from=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=1,
        out_channels=embed_dim,
        fast_pass=False,
        init_cfg=olmoearth_init,
    ),
    backbone_to=dict(
        type="OlmoEarthBackbone",
        model_config_path=olmoearth_config,
        modality="sentinel1",
        patch_size=patch_size,
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
        in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
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
