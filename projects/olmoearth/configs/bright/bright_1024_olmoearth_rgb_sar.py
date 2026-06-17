_base_ = "./bright_1024_olmoearth_rgbproxy.py"

# BRIGHT provides a three-channel pre-event optical image and a single-channel
# post-event commercial SAR image. Keep both in the native generic modalities
# used by the RGB+SAR OLMoEarth pretraining checkpoint. NIR is zero-filled but
# marked absent, so its independent RGB bandset is masked by the backbone.
pre_rgb_mode = "native_rgb"
post_sar_mode = "native_sar"

train_pipeline = [
    dict(
        type="LoadOLMoEarthBRIGHTPair",
        pre_rgb_mode=pre_rgb_mode,
        post_sar_mode=post_sar_mode,
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
        pre_rgb_mode=pre_rgb_mode,
        post_sar_mode=post_sar_mode,
    ),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
