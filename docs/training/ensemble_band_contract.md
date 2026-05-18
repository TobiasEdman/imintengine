# Ensemble band contract

Which models the LULC ensemble trains/runs, what bands each consumes, and
the on-disk band order every tile `.npz` must satisfy.

Source of truth for the model list: `MODEL_REGISTRY` in
`imint/fm/registry.py`. This doc exists because the band order was
previously implicit in code only — and a hardcoded band list in
`scripts/sen2cor_pipeline/run_sen2cor_per_scene.py` drifted to the wrong
NIR band (B08 instead of B8A) precisely because nothing documented the
contract.

## The ensemble — 7 registered models

| Model | Family | Temporal | Input dataset key(s) |
|-------|--------|----------|----------------------|
| `prithvi_300m` | prithvi | yes (1–4 frames) | `spectral` |
| `prithvi_600m` | prithvi | yes (1–4 frames) | `spectral` |
| `terramind_v1_base` | terramind | no (1 frame) | `spectral`, `s1_vv_vh` |
| `thor_v1_base` | thor | no (1 frame) | `spectral`, `s1_vv_vh` |
| `clay_v1_5` | clay | no (1 frame) | `s2_clay` |
| `croma_base` | croma | no (1 frame) | `s2_croma`, `s1_vv_vh` |
| `tessera_v1` | tessera | no (1 frame) | `tessera` |

## On-disk tensors — the dataset contract

Every tile `.npz` stores spectral data once; per-model loaders re-stack
from these tensors. No model gets its own private copy of a band.

| Tensor | Shape | Band order |
|--------|-------|------------|
| `spectral` | `(n_frames, 6, H, W)` | **B02, B03, B04, B8A, B11, B12** |
| `b08` (enrichment) | `(H, W)` | B08 — 842 nm broad NIR |
| `rededge` (enrichment) | `(3, H, W)` | B05, B06, B07 |
| `s1_vv_vh` (enrichment) | `(2, H, W)` | VV, VH |
| `tessera` (enrichment) | `(128, H, W)` | pre-computed embedding |

The 6-band `spectral` order is the Prithvi/HLS order — the canonical
constant is `PRITHVI_BANDS` in `imint/training/tile_fetch.py`. Any code
that writes a spectral frame **must** import and use that constant, never
a local literal.

### The NIR rule

`spectral` slot 3 is **B8A** (865 nm narrow NIR) — for **every frame**,
including the autumn frame `frame_2016`. B08 (842 nm broad NIR) is a
*different physical band* and lives only in the separate `b08`
enrichment. The two are not interchangeable.

A frame written with B08 in slot 3 is silently wrong: within a single
4-frame `spectral` tensor, frame 0 would carry B08 while frames 1–3
carry B8A.

## Per-model band consumption

### prithvi_300m / prithvi_600m — `spectral` directly
6 bands, up to 4 temporal frames. Expect slot 3 = B8A (HLS narrow NIR;
the band Prithvi-EO-2.0 was pretrained on).

### terramind_v1_base — `spectral` directly
6 bands, single frame. Registry declares `bands.S2L2A = [BLUE, GREEN,
RED, NIR_NARROW, SWIR_1, SWIR_2]` — `NIR_NARROW` = B8A. Matches the
`spectral` order exactly.

### thor_v1_base — `spectral` directly  ⚠ OPEN QUESTION
6 bands, single frame. Registry `loader_kwargs.model_bands` declares
`[BLUE, GREEN, RED, NIR_BROAD, SWIR_1, SWIR_2, VV, VH]` — `NIR_BROAD`
= **B08**, which contradicts the B8A `spectral` tensor it is fed.
Either the registry spec is wrong or the THOR loader remaps. **Must be
resolved before stage-5 THOR training.** This discrepancy predates the
sen2cor pipeline and is not caused by it.

### clay_v1_5 — `s2_clay` (built by loader)
`imint.fm.loaders.clay.build_s2_clay_tensor` stacks 10 bands:
`spectral` (B02,B03,B04,B8A,B11,B12) + `b08` + `rededge`
(B05,B06,B07) → blue, green, red, rededge1-3, **B08 nir**, **B8A
nir08**, swir16, swir22. Clay uses *both* NIR bands. Requires the
`rededge` enrichment; missing rededge degrades the input.

### croma_base — `s2_croma` (built by loader)
`imint.fm.loaders.croma.build_s2_croma_tensor` stacks 12 bands:
B01, B02–B07, **B08**, **B8A**, B09, B11, B12. Uses both NIR bands.
B01/B09 (60 m atmospheric) are zero-padded when absent.

### tessera_v1 — `tessera` embedding
No raw S2 bands. Consumes the 128-D annual embedding baked in by
`scripts/enrich_tiles_tessera.py`. Band order is not applicable.

## Summary — what a tile must carry to serve all 7 models

| Need | Bands | Used by |
|------|-------|---------|
| `spectral` (B8A in slot 3) | B02,B03,B04,B8A,B11,B12 | prithvi×2, terramind, thor, clay, croma |
| `b08` enrichment | B08 | clay, croma |
| `rededge` enrichment | B05,B06,B07 | clay, croma |
| B01/B09 | — | croma (zero-padded if absent) |
| `s1_vv_vh` enrichment | VV,VH | terramind, thor, croma |
| `tessera` enrichment | embedding | tessera |

## Suggested test

Add `tests/test_ensemble_band_contract.py` asserting:
- `PRITHVI_BANDS[3] == "B8A"`
- every spectral-writing script imports `PRITHVI_BANDS` rather than a
  local literal
- `_FRAME_BANDS` in the sen2cor runner equals `PRITHVI_BANDS`
