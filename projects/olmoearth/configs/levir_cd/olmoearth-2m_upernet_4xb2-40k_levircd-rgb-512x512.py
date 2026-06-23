_base_ = "./olmoearth-rgb_upernet_4xb2-40k_levircd-512x512_base.py"

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-2m"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"

model = dict(
    data_preprocessor=dict(
        test_cfg=dict(size_divisor=16),
    ),
    backbone=dict(
        model_config_path=olmoearth_config,
        patch_size=16,
        init_cfg=dict(type="Pretrained", checkpoint=olmoearth_checkpoint),
    ),
)

work_dir = "./work_dirs/olmoearth-2m_upernet_4xb2-40k_levircd-rgb-512x512"
