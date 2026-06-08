dataset_type = "OLMoEarthOSCDDataset"
data_root = "/mnt/ht2-nas2/EO_test/wry/Copernicus/Data/OSCD/"
crop_size = (256, 256)

train_pipeline = [
    dict(type="LoadOLMoEarthOSCDPair"),
    dict(type="LoadOLMoEarthOSCDAnnotations"),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="horizontal"),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackOLMoEarthCDInputs"),
]

test_pipeline = [
    dict(type="LoadOLMoEarthOSCDPair"),
    dict(type="LoadOLMoEarthOSCDAnnotations"),
    dict(type="PackOLMoEarthCDInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split="train",
            pipeline=train_pipeline,
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type="mmseg.IoUMetric", iou_metrics=["mFscore", "mIoU"])
test_evaluator = val_evaluator
