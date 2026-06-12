_base_ = ["./olmoearth-base_upernet_4xb2-40k_levircd-rgb-s2proxy-p16-512x512.py"]

train_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgLoadAnnotations"),
    dict(type="MultiImgRandomRotate", prob=0.5, degree=180),
    dict(type="MultiImgRandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(
        type="MultiImgPhotoMetricDistortion",
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(
        type="RGBPairToOlmoEarth",
        modality="rgb",
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOLMoEarthCDInputs"),
]

test_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgResize", scale=(1024, 1024), keep_ratio=True),
    dict(type="MultiImgLoadAnnotations"),
    dict(
        type="RGBPairToOlmoEarth",
        modality="rgb",
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

model = dict(
    backbone_inchannels=4,
    backbone=dict(modality="rgb"),
)

work_dir = "./work_dirs/olmoearth-base_upernet_4xb2-40k_levircd-rgb-native-p16-512x512"
