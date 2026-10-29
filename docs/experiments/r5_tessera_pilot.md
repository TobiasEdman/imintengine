# TESSERA v2 R5 pilot: preparation and fixed acceptance

This prepares the 2026-10-19 decision: R5 adds LUCAS crop type to R4,
using TESSERA only, after the NFI freeze. No production dataset, checkpoint,
cluster job or GPU budget is changed by this PR. The experiment declaration is
[`config/r5_tessera_pilot.json`](../../config/r5_tessera_pilot.json).

## Fixed criterion

Both conditions must hold before calling the pilot successful:

1. Held-out LUCAS crop accuracy is strictly higher for R5 than R4.
2. LULC validation mIoU under `miou-ignore-target-v2` falls by at most 0.005.

Crop accuracy samples the **28-class segmentation prediction** at the
protected crop points. Its denominator includes every protected crop row;
non-crop predictions count as errors. There is no renormalization among crop
logits and no substitution of the auxiliary MLP's accuracy. Both arms use the
same point order, truth, vocabulary, tile bytes, centre crop, pixel masks,
validation labels and evaluator/runtime. The comparator recalculates both
metrics from confusion matrices and rejects missing or unequal identity hashes
or target supports. Its 1e-12 mIoU boundary tolerance covers float arithmetic.

The historical crop OOF table used retired v1 embeddings. The historical R4
mIoU v2 rescore changed the metric, not the embedding dataset or validation
membership. Neither is a reproducible promoted-v2 baseline. Keep those results
as historical context; do not supply them as acceptance records.

## Training exposure and the matched control

The frozen crop split used `(tile_name, point_id)`, so one physical point can
appear on both sides under different tiles or years. The previously observed
404 shared IDs are a finding to recompute, not a constant to subtract.

The preparation reads the hash-bound frozen distill and protected parquets,
and joins them to a hash-bound full LUCAS source index. Frozen label and pixel
coordinates must match that source exactly. The source restores `Year`, which
the six-column frozen parquet does not carry. Canonical integer and decimal
string point IDs normalize to the same identity; aliases, nulls, floats and
booleans are refused.

All protected point IDs are excluded from head fitting across **all source
index years and tiles**. Any tile containing such an ID is also excluded from
student training. The clean head-training partition is intersected with the
retained student train split. Both student arms must therefore use the same
filtered cohort: a fresh R4 control and R5, with the same cold-start seed and
R4 training regime. The extra control's GPU cost needs approval; this code
neither schedules it nor silently treats historical R4 as a matched control.

This protects LUCAS truth exposure and known indexed tile co-location. It does
not establish unseen imagery for the historical teacher: R4 already learned
from map labels and imagery. Nor does the index prove absence of exposure in
unindexed overlapping footprints. Report those limits with results. The NFI
freeze is a sequencing prerequisite; this preparation does not rerun or change
its experiment.

## Concrete producer

[`scripts/r5_tessera_pilot.py`](../../scripts/r5_tessera_pilot.py) provides:

- `prepare`: verify frozen inputs, recompute exclusion, require all 11 crop
  classes after exclusion, check promoted embeddings, exact annual matching and
  training `frame_2016`, then create a private bundle with byte inventories,
  clean/protected indices and matched train/validation membership. No input is
  modified and an existing output directory is refused.
- `produce`: authenticate that bundle and the sealed image, load the bound R4
  teacher with the existing strict checkpoint loader, use the existing
  pre-classifier extraction path, fit the standardized 128-hidden MLP, and use
  existing dense inference and numpy head forwarding to produce crop sidecars.
  Keys and truth must exactly match the clean training index. The head uses a
  fixed 500-iteration limit with no random point-wise early-stopping partition;
  the protected holdout never selects a head or epoch.
- `compare`: consume two matched evaluation records and recompute the fixed
  acceptance decision. Failure is exit 2; incomparable records raise an error.

Only already-crop pixels (classes 11..21) in the 504×504 centre window of each
512×512 R4 training label are replaced. `label_mask` contains raw SJV crop
codes, not Boolean validity; it remains byte-identical metadata. Forest,
background, crop/non-crop boundaries, border pixels and all other sidecar
arrays remain identical. Fraction sidecars stay separately
bound to their original bytes; finite four-species percentages in [0,100],
exact tile geometry and a binary uncertainty mask are checked before production
and again by the explicit-cohort loader. Malformed fractions fail closed. No holdout or validation pseudo-labels are
produced; both arms must validate against the original R4 labels.

`backbone=tessera_v1` is the registry architecture name. The data contract is
**`tessera_source=geotessera-0.10.2`**, with valid nonempty `(128,512,512)` arrays,
positive presence flags and exact embedding/spectral/label year agreement.
The preparation also rejects positive-area footprint overlap between training
and validation or protected holdout tiles, independently of names and years.
Geometry uses the shared bbox resolver; no missing geometry is silently accepted.
The old all-backbone crop split is not rebuilt: its immutable point identities
are reused, while a new inventory records the promoted embedding bytes.

## Private input manifest

`prepare --inputs <private-json> --out <new-private-directory>` requires these
JSON fields. Every artifact record contains `path` and a reviewed full
`sha256`; there are no automatic "accept current bytes" defaults:

| Field | Required artifact |
|---|---|
| `crop_freeze` | Existing crop split MANIFEST.json; binds both parquets |
| `source_index` | Full LUCAS index, including integer `Year` |
| `nfi_freeze` | Completed NFI `manifest.json`; verifies its holdout bytes |
| `teacher_checkpoint` | Historical TESSERA R4 feature teacher |
| `cohort` | JSON with explicit original `train` and `val` tile stems |
| `data_dir` | Flat promoted input tile root |
| `label_dir` | R4 forest-distilled label root |
| `frac_dir` | Original Trädslag fraction root |

The cohort must come from the R4 loader's resolved membership, not an assumed
reconstruction of historical filename hashing. The preparation's matched
student comparison is a new controlled experiment even if the historical
membership cannot be recovered. All indices, inventories and per-tile outputs
remain on private storage. Commit no real IDs, coordinates, tile names or
results to the public repository.

The output `manifest.json` is the commit marker for a preparation bundle.
A failed or interrupted run with no marker is incomplete. Existing directories
are never automatically discarded or restarted. `produce` additionally takes
`--bundle-sha256`, `--runtime-manifest`, `--source-git-sha`, `--image`,
`--device` and a fresh `--out` directory. It reuses the crop-distill image's
source seal and requires its model interpreter. Rebuild that existing runtime
from a reviewed source commit before pinning a real digest; the old image does
not contain this producer.

## Remaining execution boundary

This PR implements and locally verifies preparation, the crop producer,
student runner and paired evaluator. It does **not authorize a training launch**. Before any GPU work, review the private
preparation counts, pin the new image/source, and authorize one measured cell.
Record GPU model/count, extraction and dense-pass seconds, peak VRAM, CPU head
fit seconds, tile/point counts, image/source and input-bundle hashes. Use that
measurement plus a student epoch probe to propose the complete two-arm cost;
GPU-hour cost is a user decision. A single forward pass is not a reliable
estimate of 30-epoch student training cost.

[`scripts/run_r5_tessera.py`](../../scripts/run_r5_tessera.py) supplies the
actual `probe`, `train` and `evaluate` entrypoints. All require the verified
bundle, a hash-bound producer/training artifact, and the sealed runtime.
The student uses existing `LULCTrainer` with exact ordered membership through
`UnifiedDataset(tile_names=...)`. Missing, filtered or unreadable tiles fail;
they never substitute another sample or discover a nested tile directory. R5 reads its crop sidecars only for
training; both arms validate against the original R4 sidecars and retain the
fraction head. The shared R4 regime uses seed 42, 30 maximum epochs, batch 8,
504 grid, all eleven ordered R4 aux channels, gated fusion, focal/dice/Lovasz and the original optimizer settings.
Determinism is enabled for both arms. Sqrt class weights are computed once
per run from the same original R4 labels on the clean cohort; tile sampling is
explicitly uniform with replacement in both arms. Those settings and every
resolved config field are persisted and compared, rather than inheriting a
mutable global class_stats.json or environment variable. Each training report
retains the canonical producer manifest and exact selected label inventory;
evaluation carries the complete training manifest identity.

`probe` changes epochs to one and saves its epoch checkpoint, measures elapsed
time and peak GPU memory, then marks the result `probe_complete`. It uses the
complete fixed cohort. `evaluate` writes actual ordered key/truth hashes,
LULC sample/mask/label hashes, per-point predictions, cached rasters and both
confusion matrices. The LULC mask is `target != 0`, exactly the shared v2
metric. The comparator rejects probe results even if their numeric metrics
happen to meet the criterion. Checkpoint selection uses the same LULC
validation policy in both arms; LUCAS holdout accuracy never selects an epoch.

The existing `docker/ladder-crop-distill/build.sh` and Dockerfile build this
code from a committed source SHA. Their archive selection includes all
`config/`, `imint/` and `scripts/` files and the runtime seal hashes the full
payload. CI now selects that image when either R5 entrypoint, its declaration
or the dataset changes; push selection covers the full event commit range.
The image's model smoke imports both new entrypoints.
A real published digest and its CI smoke remain a bootstrap requirement.

Only after review, image publication, the one-cell measurement and budget
approval may an
explicit suspended Kubernetes Job be rendered with real image and input
identities. There is no placeholder Job, no change to `RUNGS`, and no R5 queue
entry. A preparation report or successful synthetic test is not a pilot result.

## Local verification

`python -m pytest tests/test_r5_tessera_pilot.py -q` runs synthetic CPU tests,
including an actual registry TESSERA checkpoint through the existing strict
loader, feature extractor, MLP, dense inference, sidecar write and readback,
then one complete CPU training epoch for each student arm and paired
evaluation with verified identical evaluation hashes.
The integration fixture uses a smaller grid for CPU testing; it does not claim
production resolution performance, production-data readiness or GPU timing.
Tests assert cross-year point and spatial exclusion, source truth integrity, rejected v1 /
empty / wrong-year inputs, immutable bundle bytes, unchanged forest/metadata,
and failed comparisons when evaluation identities differ.
