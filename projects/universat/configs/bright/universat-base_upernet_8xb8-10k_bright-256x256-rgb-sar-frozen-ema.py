_base_ = "./universat-base_upernet_1xb1-40k_bright-256x256-rgb-sar-frozen.py"

# Multi-GPU training recipe adapted from the single-GPU BRIGHT configuration
# referenced by the comparison project's README. The reference uses batch 8
# for 80k iterations. Eight GPUs at batch 8/GPU use 1/8 as many iterations so
# that both recipes process the same 640k samples.
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
)

optim_wrapper = dict(
    optimizer=dict(
        # Linear scaling from reference global batch 8 to global batch 64.
        lr=4e-3,
        weight_decay=5e-4,
    ),
)

train_cfg = dict(
    max_iters=10000,
    val_interval=1000,
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=625,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=8e-6,
        T_max=9375,
        begin=625,
        end=10000,
        by_epoch=False,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        interval=1000,
        max_keep_ckpts=5,
    ),
    logger=dict(interval=6),
)

custom_hooks = [
    dict(
        type="EMAHook",
        # Match the reference EMA decay per processed sample across 8x fewer
        # optimizer steps: 1 - (1 - 2e-4) ** 8 ~= 1.6e-3.
        momentum=1.6e-3,
        update_buffers=True,
        priority="LOWEST",
    ),
]

auto_scale_lr = dict(enable=False, base_batch_size=64)
work_dir = (
    "./work_dirs/"
    "universat-base_upernet_8xb8-10k_bright-256x256-rgb-sar-frozen-ema"
)
