# OLMoEarth on Open-CD BRIGHT

This project adapts OLMoEarth encoders to BRIGHT four-class building-damage
segmentation. BRIGHT provides pre-event RGB and post-event single-channel SAR.

## Configurations

The BRIGHT directory contains the core comparison configs plus an optional
token-level early-fusion variant:

| Pretrained model | Input adaptation | Backbone frozen | Full fine-tune |
| --- | --- | --- | --- |
| Official OLMoEarth-v1 | S2 proxy + S2 proxy | `olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py` | `olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-finetune.py` |
| Official OLMoEarth-v1 | S2 proxy + S2 proxy, 2-timestep token fusion | `olmoearth-native_temporal-upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py` | `olmoearth-native_temporal-upernet_1xb1-80k_bright-1024x1024-s2proxy-finetune.py` |
| Official OLMoEarth-v1 | S2 proxy + S1 proxy, SAR duplicated to VV/VH | `olmoearth-native_multimodal-upernet_1xb1-80k_bright-1024x1024-s2-s1dup2-frozen.py` | `olmoearth-native_multimodal-upernet_1xb1-80k_bright-1024x1024-s2-s1dup2-finetune.py` |
| Official OLMoEarth-v1 | S2 proxy + S1 proxy, SAR as VV and VH zero-filled | `olmoearth-native_multimodal-upernet_1xb1-80k_bright-1024x1024-s2-s1zero-vh-frozen.py` | `olmoearth-native_multimodal-upernet_1xb1-80k_bright-1024x1024-s2-s1zero-vh-finetune.py` |
| OLMoEarth-10m | native RGB + native SAR | `olmoearth-10m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-frozen.py` | `olmoearth-10m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-finetune.py` |
| OLMoEarth-2m | native RGB + native SAR | `olmoearth-2m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-frozen.py` | `olmoearth-2m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-finetune.py` |

LEVIR-CD RGB configs live in the same project:

```text
projects/olmoearth/configs/levir_cd/olmoearth-10m_upernet_4xb2-40k_levircd-rgb-512x512.py
projects/olmoearth/configs/levir_cd/olmoearth-2m_upernet_4xb2-40k_levircd-rgb-512x512.py
projects/olmoearth/configs/levir_cd/olmoearth-base_upernet_4xb2-40k_levircd-rgb-s2proxy-p16-512x512.py
```

They expect the standard Open-CD LEVIR-CD layout:

```text
${MM_ARCHIVE_DATA_HOME:-data}/LEVIR-CD/train/{A,B,label}
${MM_ARCHIVE_DATA_HOME:-data}/LEVIR-CD/val/{A,B,label}
${MM_ARCHIVE_DATA_HOME:-data}/LEVIR-CD/test/{A,B,label}
```

The official model maps both dates to its Sentinel-2 L2A interface. For the
post-event image, the single SAR channel is repeated as an RGB proxy before
mapping into the visible Sentinel-2 bands. The 10m and 2m models use their
pretrained generic `rgb` and single-channel `sar` branches directly. BRIGHT has
no NIR channel, so NIR is zero-filled and marked missing.

The temporal S2-proxy variant keeps the same data adaptation but replaces the
Siamese abs-diff encoder path with `OLMoEarthTemporalEncoderDecoder`: the two
dates are packed as two timesteps and fused inside the OLMoEarth encoder before
the decoder.

The S2+S1 variants use `OlmoEarthMultiModalBackbone`, which creates one
`MaskedOlmoEarthSample` containing both `sentinel2_l2a` and `sentinel1`, so
their tokens attend to each other inside the same OLMoEarth encoder pass.

Frozen configs set `unfreeze_epoch=None`; fine-tune configs remove the freeze
hook and train the selected encoder parameters from iteration zero. Derived
entries inherit from another BRIGHT config. The native frozen file is the sole complete base and inherits Open-CD's shared
`default_runtime.py`.

Update `data_root` and each model's `olmoearth_model_dir` for the training
server. Every model directory must contain an exported `config.json` and
`weights.pth`.

## Validation

```bash
python projects/olmoearth/tools/validate_olmoearth_bright.py
python projects/olmoearth/tools/test_bright_rgbproxy_dataflow.py
python projects/olmoearth/tools/test_project_imports.py
```

Launch training with the selected config, for example:

```bash
python tools/train.py \
  projects/olmoearth/configs/bright/olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py
```

For LEVIR-CD RGB training:

```bash
python tools/train.py \
  projects/olmoearth/configs/levir_cd/olmoearth-10m_upernet_4xb2-40k_levircd-rgb-512x512.py

python tools/train.py \
  projects/olmoearth/configs/levir_cd/olmoearth-2m_upernet_4xb2-40k_levircd-rgb-512x512.py

python tools/train.py \
  projects/olmoearth/configs/levir_cd/olmoearth-base_upernet_4xb2-40k_levircd-rgb-s2proxy-p16-512x512.py
```
