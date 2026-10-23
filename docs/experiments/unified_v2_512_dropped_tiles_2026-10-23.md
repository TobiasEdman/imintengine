# unified_v2_512: ten tiles dropped, the corrupted directory retired (task 0115)

**Decision (owner, 2026-10-23):** the ten tiles below are dropped from
`/data/unified_v2_512` for good instead of being fetched again, and
`/data/unified_v2_512_corrupted` is retired. The owner deletes the directory
with `k8s/retire-corrupted-0115-job.yaml`. The names are in
`unified_v2_512_dropped_tiles_20261023.json`.

## What the check found

The read-only check of 2026-10-22
(`~/.agents/continuity/evidence/ImintEngine/corrupted-check-0115-20261022.json`)
compared the 231 files in the corrupted directory with the live directory:

| | files |
|---|---|
| a readable copy in `/data/unified_v2_512` | 220 |
| a copy in the live directory that cannot be read | 0 |
| no copy in the live directory | 11 |

Of the 11, one is a leftover temporary file (`tile_368631_6143498.tmp.npz`),
not a tile. The other ten are the dropped tiles:

`tile_555366_6801136`, `tile_616126_6925422`, `tile_617120_6514080`,
`tile_618702_6929973`, `tile_618702_6932473`, `tile_621126_6917922`,
`urban_413223_6256470`, `urban_431494_6216721`, `urban_436888_6247191`,
`urban_442961_6228264`.

## Why they are dropped, not fetched again

- **They are already out of training.** `k8s/quarantine-corrupt-tiles-job.yaml`
  rebuilt `split_train.txt` and `split_val.txt` from the files left in the
  live directory, and these ten are not there. Dropping them changes nothing
  that a model sees.
- **A fetch alone would not give a tile like its neighbours.** The fetch writes
  spectral data, labels and aux layers. A complete tile also needs the label
  sidecars (one made on a GPU), the VPP stamp, the 2016 frame from sen2cor
  (special image, GPU), Tessera and S1 v4 last, and the splits rebuilt. Most
  of these steps cannot be limited to named tiles.
- **Full parity is not possible.** The cohort carries `tessera_v1`, whose
  source answers 410 (`k8s/tessera-promote-v2-job.yaml`), so a fetched tile
  would always lack it.

Ten tiles are about 0.13 % of the 7 882 tiles counted on 2026-10-22.

## What still refers to the directory

- **Jobs in this repository:** `k8s/quarantine-corrupt-tiles-job.yaml`, which
  made it, and `k8s/refill-512-watcher-job.yaml`, whose Klass A reads its names.
  The watcher's glob of a missing directory is empty, so after the deletion
  Klass A is 0 and the watcher stops at its "Nothing to do for Klass A" gate.
  Nothing else in the repository reads it.
- **Files on the PVC** that list tile names, not paths into the directory:
  `tile_locations.json`, `tile_locations_full.json`, `patch_list.json`,
  `patch_autumn_done.json`, `patch_autumn_failed.json`,
  `blank_crop_result.json`, `refetch_list.json`, `unified_v2_512/vpp_known_empty.json`.
  `unified_v2_512/quarantined_tiles.txt` keeps the quarantine's own record.
