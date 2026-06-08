_base_ = [
    "./olmoearth_levir_cd_upernet_rgbproxy.py",
    "./levir_cd_512_olmoearth_rgbproxy.py",
    "../../../../configs/_base_/default_runtime.py",
]

work_dir = "./work_dirs/olmoearth-base_upernet_1xb1-40k_levir-cd-512x512-rgbproxy"

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=40000,
    val_interval=4000,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

log_processor = dict(by_epoch=False)
randomness = dict(seed=0)

# Same idea as BRIGHT: freeze OLMoEarth at the beginning, then fine-tune.
custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=10000)]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
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
        save_last=True,
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

auto_scale_lr = dict(enable=False, base_batch_size=1)
