_base_ = [
    "./olmoearth_temporal_oscd_upernet.py",
    "./oscd_s2_256.py",
    "../../../../configs/_base_/default_runtime.py",
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
log_processor = dict(by_epoch=True)
randomness = dict(seed=0)

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"

model = dict(
    backbone=dict(
        model_config_path=olmoearth_config,
        init_cfg=dict(type="Pretrained", checkpoint=olmoearth_checkpoint),
        modality="sentinel2_l2a",
    )
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type="OneCycleLR",
        eta_max=1e-3,
        pct_start=0.0,
        anneal_strategy="cos",
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best="mIoU",
        rule="greater",
        save_last=True,
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    visualization=dict(
        type="OLMoEarthOSCDVisualizationHook",
        interval=1,
        draw=True,
        show=False,
    ),
)
