# OLMoEarth on Open-CD BRIGHT

This project ports OLMoEarth into Open-CD for the BRIGHT / DFC2025 Track II
building damage assessment task.

## SAR Handling

The local `copernicusfm_bright` baseline treats BRIGHT post-event SAR as a
single-channel uint8 raster, repeats it to three channels, and feeds it through
the same RGB-compatible Copernicus-FM wrapper as the pre-event optical image.
This matches TorchGeo's BRIGHT dataset description: pre-disaster images are
three-channel optical images and post-disaster SAR images are single-channel
images repeated to three channels.

The provided OLMoEarth config follows that comparison protocol:

```text
pre-event RGB        -> RGB proxy -> OLMoEarth Sentinel-2 L2A interface
post-event SAR uint8 -> RGB proxy -> OLMoEarth Sentinel-2 L2A interface
```

This is intentionally named `rgbproxy`: it is a fair baseline against
Copernicus-FM BRIGHT, not a claim that BRIGHT SAR is native Sentinel-2 or
dual-polarization Sentinel-1.

Two Sentinel-1 proxy ablations are also provided. They keep the pre-event RGB
image on the OLMoEarth Sentinel-2 L2A interface, but route the post-event SAR
image through a separate OLMoEarth Sentinel-1 backbone:

```text
pre-event RGB        -> RGB proxy -> OLMoEarth Sentinel-2 L2A interface
post-event SAR uint8 -> S1 proxy  -> OLMoEarth Sentinel-1 interface
```

BRIGHT SAR is single-polarization commercial VHR SAR, not standard Sentinel-1
VV/VH. For these ablations, the released uint8 SAR image is first mapped to an
approximate dB range, `[-30, 5]`, then normalized exactly like OLMoEarth
`Strategy.COMPUTED`: `(x - (mean - 2*std)) / (4*std)` using the computed
Sentinel-1 band statistics. OLMoEarth masks Sentinel-1 at the VV/VH bandset
level, so both proxy configs fill the complete two-band bandset:

- `s1-vv-zero-vh`: SAR fills the VV slot and VH is zero-filled.
- `s1-dup2`: the same SAR image fills both VV and VH slots.

The config does not enable OLMoEarth `fast_pass=True`. Only RGB-derived
Sentinel-2 bands are present in this proxy input; the remaining Sentinel-2 bands
are treated as missing. Enabling `fast_pass` would incorrectly treat those
filled channels as observed bands.

References:

- https://docs.torchgeo.org/en/stable/api/datasets/bright.html
- https://essd.copernicus.org/articles/17/6217/2025/index.html

## Train

Make sure `olmoearth_pretrain` is importable in the training environment. For
example:

```bash
export PYTHONPATH=/path/to/olmoearth_pretrain:$PYTHONPATH
```

Update the model path in:

```text
projects/olmoearth/configs/bright/olmoearth_bright_upernet_rgbproxy.py
```

Then run:

```bash
python tools/train.py \
  projects/olmoearth/configs/bright/olmoearth-base_upernet_1xb1-80k_bright-1024x1024-rgbproxy.py
```

Available BRIGHT configs:

```text
projects/olmoearth/configs/bright/olmoearth-base_upernet_1xb1-80k_bright-1024x1024-rgbproxy.py
projects/olmoearth/configs/bright/olmoearth-base_upernet_1xb1-80k_bright-1024x1024-s1-vv-zero-vh.py
projects/olmoearth/configs/bright/olmoearth-base_upernet_1xb1-80k_bright-1024x1024-s1-dup2.py
```

The config uses Open-CD's iter-based BRIGHT schedule, OLMoEarth dense features,
absolute feature difference fusion, MultiLevelNeck to produce stride
16/32/64/128 features, and UPerNet for four-class damage segmentation.

## Validate

This repository includes a dependency-light validator for the project files:

```bash
python projects/olmoearth/tools/validate_olmoearth_bright.py
python projects/olmoearth/tools/test_bright_rgbproxy_dataflow.py
python projects/olmoearth/tools/test_project_imports.py
```

The first command checks project/config invariants. The second command performs
a small numeric test for the RGB/SAR-to-pseudo-Sentinel-2 data path without
requiring Open-CD runtime dependencies. The third command imports the project
with lightweight stubs and verifies that Open-CD registry decorators expose the
expected model, dataset, transform, neck, and hook classes.

In a full Open-CD environment, additionally run:

```bash
python projects/olmoearth/tools/check_server_env.py
python projects/olmoearth/tools/smoke_build_runner.py
```

The environment check verifies imports, parses the config, and checks that the
OLMoEarth checkpoint/config plus BRIGHT data layout exist before building the
runner.

Then run a one-iteration smoke train on the real BRIGHT root before launching
the full 80k schedule. The smoke config skips validation/checkpoint work and
only checks one training forward/backward step.

For that smoke train, use the included config:

```bash
python tools/train.py \
  projects/olmoearth/configs/bright/olmoearth-base_upernet_1xb1-1iter_bright-1024x1024-rgbproxy-smoke.py
```

## Optimization Notes

- AMP is enabled by default through `AmpOptimWrapper`.
- The full config freezes the OLMoEarth backbone until iteration 10000, then
  fine-tunes end-to-end.
- The one-iteration smoke config keeps the backbone frozen to minimize the
  first environment check.
- For a faster linear-probe style comparison, extract OLMoEarth embeddings for
  both dates and train only the fusion/decode head. That is a separate offline
  workflow and should not be mixed with this online full-finetune config.
