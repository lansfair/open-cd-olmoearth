_base_ = "./olmoearth-base_upernet_1xb1-80k_bright-1024x1024-rgbproxy.py"

work_dir = "./work_dirs/olmoearth-base_upernet_1xb1-1iter_bright-1024x1024-rgbproxy-smoke"

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

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=1,
    val_interval=2,
)

train_dataloader = dict(num_workers=0, persistent_workers=False)
val_dataloader = dict(num_workers=0, persistent_workers=False)
test_dataloader = dict(num_workers=0, persistent_workers=False)

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2,
        max_keep_ckpts=1,
        save_last=False,
    ),
    logger=dict(type="LoggerHook", interval=1, log_metric_by_epoch=False),
)
