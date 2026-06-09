_base_ = "../../../../configs/common/standard_512x512_40k_levircd.py"

custom_imports = dict(
    imports=["projects.dinov3.opencd_dinov3"],
    allow_failed_imports=False,
)

work_dir = "./work_dirs/dinov3-vitl16_upernet_4xb2-40k_levircd-512x512"

dinov3_root = "/mnt/ht2-nas2/EO_test/dataset/dinov3_pretrained"
dinov3_repo_dir = "projects/dinov3/dinov3-main"
dinov3_weights_path = (
    f"{dinov3_root}/DINOv3 ViT SAT-493M/dinov3_vitl16_pretrain_sat493m.pth"
)

crop_size = (512, 512)
patch_size = 16
embed_dim = 1024
num_classes = 2
norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=patch_size,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="SiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_inchannels=3,
    backbone=dict(
        type="DINOv3ViTBackbone",
        repo_dir=dinov3_repo_dir,
        model_name="dinov3_vitl16",
        weights_path=dinov3_weights_path,
        patch_size=patch_size,
        out_channels=embed_dim,
        freeze=True,
    ),
    neck=dict(
        type="DINOv3FeatureFusionPyramid",
        policy="abs_diff",
        embed_dim=embed_dim,
        out_channels=embed_dim,
        scales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
        num_inputs=1,
    ),
    decode_head=dict(
        type="mmseg.UPerHead",
        in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
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

train_dataloader = dict(batch_size=2)

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
)

auto_scale_lr = dict(enable=False, base_batch_size=8)
