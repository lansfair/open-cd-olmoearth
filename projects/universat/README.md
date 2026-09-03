# UniverSat for Open-CD

This project ports the UniverSat encoder into Open-CD and provides a BRIGHT
four-class building-damage configuration. It follows the integration pattern
used by `projects/olmoearth` while keeping the UniverSat code and registrations
independent.

## BRIGHT input contract

The public BRIGHT rasters do not contain native Sentinel-2 and dual-pol
Sentinel-1 observations:

- pre-disaster RGB is mapped to UniverSat `s2_4band` (B02, B03, B04, B08);
- B08 is unavailable and is filled with normalized zero;
- post-disaster single-channel SAR is mapped to `s1` (VV, VH, VV/VH ratio);
- the SAR amplitude is duplicated into VV/VH and the unavailable ratio is zero.

This is an explicit proxy experiment. It must not be reported as native
S1/S2 performance.

The transform uses date indices 0 and 1 by default because the standard BRIGHT
split files do not provide per-raster acquisition dates. Override `pre_date`
and `post_date` in `LoadUniverSatBRIGHTPair` when authoritative dates are
available.

## Setup

```powershell
pip install -r projects/universat/requirements.txt
$env:BRIGHT_DATA_ROOT = "F:/data/DFC2025 BRIGHT"
$env:UNIVERSAT_CHECKPOINT = "D:/models/universat_base.safetensors"
```

The expected BRIGHT layout is:

```text
BRIGHT/
├── train_set.txt
├── val_set.txt
├── test_set.txt
├── pre-event/
├── post-event/
└── target/
```

## Train

Frozen-backbone linear probing:

```powershell
python tools/train.py projects/universat/configs/bright/universat-base_upernet_1xb1-40k_bright-256x256-rgb-sar-frozen.py
```

Full fine-tuning with a lower backbone learning rate:

```powershell
python tools/train.py projects/universat/configs/bright/universat-base_upernet_1xb1-40k_bright-256x256-rgb-sar-finetune.py
```

Training synchronously random-crops the pre-event image, post-event image, and
label to 256x256. Validation and testing use overlapping 256x256 sliding
windows with a 192-pixel stride, avoiding the prohibitive token count of a
whole 1024x1024 UniverSat forward pass with the paper's 40 m patch size.

The 40k iteration schedule, 3k linear warmup, cosine decay, class-ratio-aware
random crop, and two-axis random flips follow the server BRIGHT reference
configuration. Photometric distortion is applied to the raw `[0, 255]`
observations before `BuildUniverSatBRIGHTProxies` constructs and normalizes the
S2/S1 proxy channels. The normalization mean and standard deviation are
explicit transform parameters and can be replaced by BRIGHT training-set
statistics.

The shared frozen configuration uses BF16 autocast to avoid FP16 gradient-scale
overflows observed on A100 GPUs. The fine-tuning configuration inherits this
setting. BF16 training requires an Ampere-or-newer NVIDIA GPU; on older hardware,
override `optim_wrapper.dtype=float16` or use a regular `OptimWrapper`.

## Architecture

The before and after images remain channel-concatenated through Open-CD's data
pipeline. `UniverSatSiamEncoderDecoder` splits the 4+3 channels, restores
one-step modality/date dictionaries, applies one shared UniverSat backbone,
and fuses the two dense feature maps with absolute difference. A
`MultiLevelNeck` then builds the four scales consumed by UPerHead.
