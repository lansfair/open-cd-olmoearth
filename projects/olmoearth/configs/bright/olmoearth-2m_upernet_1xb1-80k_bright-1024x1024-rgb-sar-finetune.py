_base_ = "./olmoearth-base_upernet_1xb1-80k_bright-1024x1024-rgb-sar.py"

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-2m"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
olmoearth_init = dict(type="Pretrained", checkpoint=olmoearth_checkpoint)

model = dict(
    backbone_from=dict(
        model_config_path=olmoearth_config,
        init_cfg=olmoearth_init,
        modality="rgb",
    ),
    backbone_to=dict(
        model_config_path=olmoearth_config,
        init_cfg=olmoearth_init,
        modality="sar",
    ),
)

custom_hooks = []
work_dir = "./work_dirs/olmoearth-2m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-finetune"
