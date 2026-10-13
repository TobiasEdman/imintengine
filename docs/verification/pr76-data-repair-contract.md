# Label reference and autumn repair contract

`build_labels.py` fails when the LPIS reference for the resolved tile year is
missing. Other LPIS years are never substituted. An operator may explicitly
allow NMD-only labeling for a particular tile year with the repeatable option
`--allow-nmd-only-year YEAR`. There are no default allowed years, and this
option does not assert that any live dataset/reference inventory was inspected.

An allowance only applies when that exact year's LPIS reference is missing.
An available same-year LPIS file is still queried; a corrupt file still fails.
The allowance does not relax either SKS reference requirement or alter the
existing year-filtered harvest overlay. Here “NMD-only” refers to absence of
LPIS crop labels; normal eligible SKS overlays remain in place.

The tile must have no crop evidence: a `crop_` filename, nonzero `n_parcels` or
`label_mask`, positive `parcel_area_ha`, or a stored crop class (11–21) blocks
this exception. In addition, the canonical NMD merge must show that no pixel
is eligible for an LPIS crop overlay. This deliberately rejects ambiguous open
land as well as cropland; an explicit year flag cannot silently remove crop
supervision. Stale crop evidence also causes rejection rather than guessing.
Both NMD2018 and NMD2023 use their existing merge rules for that decision.

Each successful output (in-place or label sidecar) records `label_year` and
`label_lpis_policy`: `same_year_lpis` or `nmd_only_missing_lpis`. Per-tile results
include the policy/year and the CLI prints the explicit year allowlist. Source
tiles remain byte-identical in sidecar mode. Failures preserve existing source
and sidecar files. There is no crop-year fallback and no live inventory action
is necessary to understand or test this contract.

## Autumn repair scripts

`patch_autumn_frame.py` reads the tile without pickle and requires four-frame
raster/metadata shapes. It derives the tile year from the production resolver;
the requested fetch window must be in year−1. The returned acquisition must
lie within that window. A successful update replaces frame zero and updates
`dates[0]`, `doy[0]`, and `temporal_mask[0]` together. It runs production
`validate_smoke_temporal_metadata` before writing and on the temporary NPZ
before atomic replacement. Thus wrong-year/out-of-season dates, inconsistent
DOY or invalid remaining frames fail without replacing the original. A prior
partial repair is not silently counted as already repaired: only a tile that
passes validation is skipped. Otherwise the requested scene is fetched again.

If `spectral` and `image` both exist, their dtype and values must agree before
repair. Both aliases receive the same edit. Conflicting aliases are rejected;
the script does not guess which source is authoritative. A lone alias retains
its original key. The blanking script applies the same alias rule.

`blank_failed_crop_tiles.py` retains its specific 2018 failed-autumn scope and
non-crop keep behavior. For crop tiles it zeros the first six bands and sets
date/DOY/mask to empty/0/0 together, preserving later frames. Blanking does NOT
make a tile eligible for the sealed smoke cohort: its production validator
correctly rejects the unavailable autumn frame. No placeholder acquisition is
invented. Already-blank states are skipped only when frame-zero data and
metadata agree. Both scripts clean failed temporary writes and retain the
original tile on validation/write failure.

Both scripts require `dates`, `doy` and `temporal_mask` of length four; missing
metadata is rejected rather than inferred for unrelated frames. Object arrays
are rejected without unpickling. Autumn CLI acquisition still needs explicit
operator authorization and credentials; `--dry-run` still fetches but does not
write. Merely importing that module no longer loads environment credentials.
These changes do not authorize running either repair against a live dataset.

## Offline verification

Native tests use small synthetic NPZ/GeoTIFF/GeoParquet files and real
rasterization, production temporal validation, compressed write/reload and
atomic replacement. The acquisition boundary returns a synthetic scene; socket
connections are denied. Tests cover in-place and sidecar preservation, exact
year opt-in, crop guards, available/corrupt/missing reference controls, both
aliases, partial metadata repair, and disk/post-write validation failures.

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  -p no:cacheprovider tests/test_pr76_data_followup.py \
  tests/test_repair_script_io.py tests/test_build_labels_rebuild.py \
  tests/test_build_labels_pickle_safety.py
```

No live references, cache regeneration, training, GPU, deployment, historical
metric edits or commit-author policy changes are part of this work.
