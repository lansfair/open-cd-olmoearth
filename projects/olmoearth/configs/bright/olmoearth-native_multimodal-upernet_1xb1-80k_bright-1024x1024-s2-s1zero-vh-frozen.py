_base_ = "./olmoearth-native_multimodal-upernet_1xb1-80k_bright-1024x1024-s2-s1dup2-frozen.py"

train_pipeline = [
    dict(
        type="LoadOLMoEarthBRIGHTPair",
        pre_rgb_mode="s2_proxy",
        post_sar_mode="s1_vv_zero_vh",
    ),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackOLMoEarthCDInputs"),
]
test_pipeline = [
    dict(
        type="LoadOLMoEarthBRIGHTPair",
        pre_rgb_mode="s2_proxy",
        post_sar_mode="s1_vv_zero_vh",
    ),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

work_dir = "./work_dirs/olmoearth-native_multimodal-upernet_1xb1-80k_bright-1024x1024-s2-s1zero-vh-frozen"
