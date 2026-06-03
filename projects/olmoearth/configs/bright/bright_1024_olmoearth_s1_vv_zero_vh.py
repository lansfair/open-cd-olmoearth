_base_ = "./bright_1024_olmoearth_rgbproxy.py"

# BRIGHT post-event SAR is released as single-channel uint8 commercial VHR SAR.
# This proxy maps uint8 values to an approximate Sentinel-1 dB range, then uses
# OLMoEarth's computed Sentinel-1 normalization: (x - (mean - 2*std)) / (4*std).
# OLMoEarth masks Sentinel-1 at bandset level, so the VV/VH bandset stays
# online; VH is explicitly zero-filled and recorded as a proxy-filled band.
post_sar_mode = "s1_vv_zero_vh"
sar_db_range = (-30.0, 5.0)

train_pipeline = [
    dict(type="LoadOLMoEarthBRIGHTPair", post_sar_mode=post_sar_mode, sar_db_range=sar_db_range),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackOLMoEarthCDInputs"),
]

test_pipeline = [
    dict(type="LoadOLMoEarthBRIGHTPair", post_sar_mode=post_sar_mode, sar_db_range=sar_db_range),
    dict(type="LoadOLMoEarthBRIGHTAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
