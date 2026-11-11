---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - earth-observation
  - remote-sensing
  - land-cover
  - semantic-segmentation
  - tessera
  - sweden
---

# IMINT TESSERA b28: land cover over Sweden with barrblandskog

A 29-class land-cover segmentation model for Sweden at 10 m, with a second
head that estimates crown cover per tree species. It runs on pre-computed
[TESSERA](https://github.com/ucam-eo/tessera) annual embeddings plus eleven
auxiliary rasters. Class 28, *barrblandskog* (mixed pine/spruce conifer
forest, NMD 2023 code 113), is new in this model.

Made by RISE Research Institutes of Sweden in the IMINT Engine project
([github.com/TobiasEdman/imintengine](https://github.com/TobiasEdman/imintengine)).
This is a pre-release shared for review; the evaluation is not yet
published.

## What is in this repository

| File | Content |
|---|---|
| `model.safetensors` | The weights: 395 042 learnable parameters plus BatchNorm statistics, keys and tensors exactly as in the training checkpoint. |
| `config.json` | The checkpoint's model config and the input/output contract: class names, fraction species, aux channel order and normalization, embedding source. |
| `MANIFEST.json` | sha256 and size of every file, the source checkpoint's sha256, and the commits that trained and exported the model. |
| `README.md`, `LICENSE`, `NOTICE` | This card, Apache-2.0, and credits for the data the model was trained on. |

The release contains no TESSERA encoder weights. The model is a small
convolutional head; the TESSERA embeddings are its input.

## Model

- **Architecture:** 1×1 and 3×3 convolutions over the 128-dimensional
  embedding, gated fusion with the projected aux stack, then two parallel
  1×1 heads: 29 class logits and 4 crown-cover fraction logits
  (`imint/fm/tessera_seg.py`).
- **Training:** IM-023 barrblandskog retrain, run `im023-tessera-b28`,
  2026-11-06, on one H100. The weights are from the best epoch, 27 of 30
  (`epoch` in `config.json`). Code commit
  `a6d36ff08acfee129f9c48457d8f75390aa6f2f9`, recipe
  `scripts/im023_retrain.py --variant main`
  (`k8s/im023/train-tessera-b28-job.yaml`). Focal + Dice loss with a Lovász
  term (weight 0.3), square-root class weighting capped at 5, label
  smoothing 0.05, and an L1 fraction loss with weight 1.0; batch 8, learning
  rate 1e-4.
- **Data:** 7 874 labelled tiles of 512 × 512 px at 10 m on the NMD grid
  (EPSG:3006). The trainer splits them by the MD5 hash of the tile name
  (`imint/training/unified_dataset.py`, 10 % to validation): 7 052 for
  training and 822 for validation. The split is random over tiles, not a
  spatial holdout, so neighbouring tiles can fall on both sides.
- **Labels:** the NMD 2023 base layer with year-matched LPIS crop parcels and
  Skogsstyrelsen clear-cuts laid over it. The forest types come from a small
  head trained on Swedish NFI field plots (temporary plots) on top of an
  earlier TESSERA model's features, and applied to every tile
  (`scripts/distill_forest_labels.py`). Pixels with NMD 2023 code 113 then
  form class 28.
- **Fraction targets:** the NMD 2023 Trädslag layer (crown cover 0–100 %) for
  four species: tall (pine), gran (spruce), trivial (birch, aspen, alder
  etc.) and adel (oak, beech, ash etc.). The fraction loss is computed only
  where at least one species has crown cover and NMD does not flag the
  pixel as uncertain, so the head never learned to output zero on open
  land or water.

## Classes

| Index | Class | Index | Class |
|---|---|---|---|
| 0 | bakgrund (ignored in training) | 15 | slåttervall |
| 1 | tallskog | 16 | bete |
| 2 | granskog | 17 | potatis |
| 3 | lövskog | 18 | sockerbetor |
| 4 | blandskog | 19 | trindsäd |
| 5 | sumpskog | 20 | råg |
| 6 | tillfälligt ej skog | 21 | majs |
| 7 | våtmark | 22 | hygge |
| 8 | öppen mark | 23 | torvtäkt |
| 9 | bebyggelse | 24 | buskdominerad mark |
| 10 | vatten | 25 | risdominerad mark |
| 11 | vete | 26 | gräsdominerad mark |
| 12 | korn | 27 | öppen mark utan vegetation |
| 13 | havre | 28 | barrblandskog |
| 14 | oljeväxter | | |

The names are the repository's schema (`imint/schema/unified_schema.py`);
`config.json` lists them in index order.

## Inputs

All inputs are 504 × 504 px: the centre crop of a 512 × 512 tile on the
NMD 10 m grid. The model is fully convolutional, so other sizes also run.

1. **Embedding**, shape `(1, 128, H, W)`, float32: the TESSERA annual
   embedding for the map year, fetched with geotessera 0.10.2 and
   dequantized. It is used as delivered, with no further normalization.
2. **Aux stack**, shape `(1, 11, H, W)`, float32, in this order. It is
   required: the model also runs with `aux=None`, but then skips the aux
   fusion and its output is meaningless. Per channel: handle missing values
   as the table says, apply the transform, then `z = (x - mean) / std`.

   | # | Channel | Source | Missing values | Transform | mean | std |
   |---|---|---|---|---|---|---|
   | 0 | height | Trädhöjd, in metres | NaN → mean | — | 7.36 | 6.55 |
   | 1 | volume | Skogliga grunddata | 0 is a value | log1p(max(x, 0)) | 3.56 | 1.14 |
   | 2 | basal_area | Skogliga grunddata | 0 is a value | log1p(max(x, 0)) | 2.42 | 0.71 |
   | 3 | diameter | Skogliga grunddata | 0 is a value | — | 16.33 | 7.84 |
   | 4 | dem | Copernicus DEM GLO-30 | 0 is a value | — | 264.03 | 215.37 |
   | 5 | vpp_sosd | HR-VPP | ≤ 0 → mean | YYDDD → x % 1000 (day of year) | 129.40 | 31.11 |
   | 6 | vpp_eosd | HR-VPP | ≤ 0 → mean | YYDDD → x % 1000 (day of year) | 263.40 | 38.32 |
   | 7 | vpp_length | HR-VPP | 0 is a value | — | 141.61 | 41.39 |
   | 8 | vpp_maxv | HR-VPP | 0 is a value | — | 0.88 | 0.57 |
   | 9 | vpp_minv | HR-VPP | 0 is a value | — | 0.04 | 0.05 |
   | 10 | markfukt | SLU Markfuktighetskarta | NaN → mean | — | 0.50 | 0.25 |

   `normalize_aux_channel` in `imint/training/unified_dataset.py` is the
   reference implementation; `config.json` repeats the table in
   machine-readable form (`input.aux`, `input.aux_normalization`). HR-VPP
   must be for the same year as the embedding.

No temporal or location coordinates are used.

## Outputs

- **Class logits**, `(1, 29, H, W)`. Take the argmax over the class axis.
  Class 0 was the ignore index during training.
- **Fraction logits**, `(1, 4, H, W)`, returned with `return_fractions=True`
  in the order tall, gran, trivial, adel. The sigmoid estimates each
  species' crown cover as a fraction in [0, 1]. It is only meaningful on
  pixels with trees: the head was never trained where NMD shows no tree
  cover, so read it together with the forest classes.

## Usage

The repository is private: log in first (`hf auth login`, or set
`HF_TOKEN`). Download a pinned revision, then check out the code at the
commit named `exporter_git_sha` in `MANIFEST.json`:

```bash
git clone https://github.com/TobiasEdman/imintengine.git
cd imintengine
pip install -e '.[inference]'
python -c "from huggingface_hub import snapshot_download; snapshot_download('RISE-Sweden/imint-tessera-b28', revision='<commit>', local_dir='imint-tessera-b28')"
git checkout <exporter_git_sha from imint-tessera-b28/MANIFEST.json>
```

```python
import torch
from scripts.hf_release import load_release

# Checks every file against MANIFEST.json, and MANIFEST.json against the
# sha256 the release was announced with, before any weight is read.
model, config = load_release("imint-tessera-b28",
                             expected_manifest_sha256="<announced sha256>")
with torch.no_grad():
    logits, fractions = model(embedding, aux=aux, temporal_coords=None,
                              location_coords=None, return_fractions=True)
classes = logits.argmax(1)
crown_cover = torch.sigmoid(fractions)
```

For a tile in the repository's training format (`.npz`), the training
code builds the inputs:

```python
from scripts.inference_comparison import _build_inference_inputs

aux_names = [a["name"] for a in config["input"]["aux"]]
inputs = _build_inference_inputs("tile.npz", "cpu", config["input"]["img_size"],
                                 aux_names, family="tessera")
embedding, aux = inputs["img5d"], inputs["aux"]   # (1, 128, 504, 504), (1, 11, 504, 504)
```

## Limitations

- Sweden only, on the NMD 10 m grid; the labels come from Swedish national
  maps and registers.
- Needs TESSERA embeddings and HR-VPP for the map year. HR-VPP 2025 is not
  yet published.
- The labels are map products, not field truth. Class 28 follows NMD 2023's
  stand-level code 113, so its boundaries are stand boundaries rather than
  per-pixel species.
- The evaluation against independent field data is not published yet.

## License

The weights are released under the Apache License 2.0 (`LICENSE`),
Copyright 2026 RISE Research Institutes of Sweden AB. `NOTICE` lists the
data the model was trained on. The TESSERA embeddings are CC0 1.0; their
authors ask to be cited:

> Feng et al., *TESSERA: Temporal Embeddings of Surface Spectra for Earth
> Representation and Analysis*, arXiv:2506.20380.
