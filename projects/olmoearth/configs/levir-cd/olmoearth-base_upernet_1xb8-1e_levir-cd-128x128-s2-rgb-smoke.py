_base_ = "./olmoearth-base_upernet_1xb8-50e_levir-cd-128x128-s2-rgb.py"

work_dir = "./work_dirs/olmoearth-base_upernet_1xb8-1e_levir-cd-128x128-s2-rgb-smoke"

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=1,
    val_interval=2,
)

train_dataloader = dict(num_workers=0, persistent_workers=False)
val_dataloader = dict(num_workers=0, persistent_workers=False)
test_dataloader = dict(num_workers=0, persistent_workers=False)

# Keep the backbone frozen during the one-epoch smoke run.
custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0,
        by_epoch=True,
        begin=0,
        end=1,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_last=False,
    ),
    logger=dict(type="LoggerHook", interval=1, log_metric_by_epoch=True),
)
