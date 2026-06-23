_base_ = "./olmoearth-rgb_upernet_4xb2-40k_levircd-512x512_base.py"

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-10m"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"

model = dict(
    backbone=dict(
        model_config_path=olmoearth_config,
        init_cfg=dict(type="Pretrained", checkpoint=olmoearth_checkpoint),
    ),
)

work_dir = "./work_dirs/olmoearth-10m_upernet_4xb2-40k_levircd-rgb-512x512"
