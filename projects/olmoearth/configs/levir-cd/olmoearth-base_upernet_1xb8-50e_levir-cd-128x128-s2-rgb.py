_base_ = [
    "./olmoearth_levir_cd_upernet_s2_rgb.py",
    "./levir_cd_128_olmoearth_s2_rgb.py",
    "../../../../configs/_base_/default_runtime.py",
]

work_dir = "./work_dirs/olmoearth-base_upernet_1xb8-50e_levir-cd-128x128-s2-rgb"

max_epochs = 50
warmup_epochs = 2
val_interval = 1
checkpoint_interval = 5

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

log_processor = dict(by_epoch=True)
randomness = dict(seed=0)

# Freeze OLMoEarth for the first 5 epochs, then fine-tune it.
custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=5)]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
    ),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=warmup_epochs,
        end=max_epochs,
        by_epoch=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=checkpoint_interval,
        max_keep_ckpts=3,
        save_best="mIoU",
        rule="greater",
        save_last=True,
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
)

auto_scale_lr = dict(enable=False, base_batch_size=8)
