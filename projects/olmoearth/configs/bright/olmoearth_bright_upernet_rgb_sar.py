_base_ = "./olmoearth_bright_upernet_s2_s1proxy.py"

# This model config must come from an OLMoEarth pretraining run whose encoder
# contains the generic `rgb` and `sar` tokenization branches.
olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-RGB-SAR"
olmoearth_checkpoint = f"{olmoearth_model_dir}/weights.pth"
olmoearth_config = f"{olmoearth_model_dir}/config.json"
olmoearth_init = dict(type="Pretrained", checkpoint=olmoearth_checkpoint)

model = dict(
    backbone_from_inchannels=4,
    backbone_to_inchannels=1,
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
