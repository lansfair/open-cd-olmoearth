# OLMoEarth on Open-CD BRIGHT

This project adapts OLMoEarth encoders to BRIGHT four-class building-damage
segmentation. BRIGHT provides pre-event RGB and post-event single-channel SAR.

## Configurations

The BRIGHT directory intentionally contains only the six experiment entry
configs required for the comparison:

| Pretrained model | Input adaptation | Backbone frozen | Full fine-tune |
| --- | --- | --- | --- |
| Official OLMoEarth-v1 | S2 proxy + S2 proxy | `olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py` | `olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-finetune.py` |
| OLMoEarth-10m | native RGB + native SAR | `olmoearth-10m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-frozen.py` | `olmoearth-10m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-finetune.py` |
| OLMoEarth-2m | native RGB + native SAR | `olmoearth-2m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-frozen.py` | `olmoearth-2m_upernet_1xb1-80k_bright-1024x1024-rgb-sar-finetune.py` |

The official model maps both dates to its Sentinel-2 L2A interface. For the
post-event image, the single SAR channel is repeated as an RGB proxy before
mapping into the visible Sentinel-2 bands. The 10m and 2m models use their
pretrained generic `rgb` and single-channel `sar` branches directly. BRIGHT has
no NIR channel, so NIR is zero-filled and marked missing.

Frozen configs set `unfreeze_epoch=None`; fine-tune configs remove the freeze
hook and train the selected encoder parameters from iteration zero. The five
derived entries inherit only from another one of these six files. The native
frozen file is the sole complete base and inherits Open-CD's shared
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
