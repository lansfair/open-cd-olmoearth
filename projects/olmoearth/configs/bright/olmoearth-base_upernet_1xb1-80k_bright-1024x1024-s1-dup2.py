_base_ = [
    "./olmoearth_bright_upernet_s2_s1proxy.py",
    "./bright_1024_olmoearth_s1_dup2.py",
    "../../../../configs/_base_/default_runtime.py",
]

work_dir = "./work_dirs/olmoearth-base_upernet_1xb1-80k_bright-1024x1024-s1-dup2"

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=80000,
    val_interval=4000,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

log_processor = dict(by_epoch=False)
randomness = dict(seed=0)

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
olmoearth_init = dict(type="Pretrained", checkpoint=olmoearth_checkpoint)

model = dict(
    backbone_from=dict(
        model_config_path=olmoearth_config,
        init_cfg=olmoearth_init,
        modality="sentinel2_l2a",
    ),
    backbone_to=dict(
        model_config_path=olmoearth_config,
        init_cfg=olmoearth_init,
        modality="sentinel1",
    ),
)

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=10000)]

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

auto_scale_lr = dict(enable=False, base_batch_size=1)
