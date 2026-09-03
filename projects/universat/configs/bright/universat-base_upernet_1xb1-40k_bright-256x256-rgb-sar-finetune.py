_base_ = (
    "./universat-base_upernet_1xb1-40k_"
    "bright-256x256-rgb-sar-frozen.py"
)

model = dict(
    backbone=dict(
        freeze_backbone=False,
        frozen_stages=-1,
    ),
)

optim_wrapper = dict(
    optimizer=dict(lr=2e-5),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "decode_head": dict(lr_mult=5.0),
            "auxiliary_head": dict(lr_mult=5.0),
        }
    ),
)

work_dir = (
    "./work_dirs/"
    "universat-base_upernet_1xb1-40k_bright-256x256-rgb-sar-finetune"
)
