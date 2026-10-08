# Task 0103: held-out NFI model and NMD evaluation

Owner: Codex, branch `agent/te/codex/eval-nfi-fieldtruth`.
Reviewer: Claude. This PR replaces #65, #62 and #63 after review; those PRs
remain open until the replacement is reviewed. No changes are pushed to
their author branches.

## Execution boundary approved by Tobias, 2026-10-08

Preparation and CPU verification are authorized. Input freezing must wait
for terminal successful `tessera-promote-v2` and a clean
`/data/manifests/tessera_promotion.json`. Inference and the real joint NMD
scoring require a separate go/no-go after the frozen set and verification
have been presented. This document does not grant that go-ahead.

The 475 tiles in `unified_v2_512_nfi2024_staging` are evaluation data.
They remain in their own directory and must never join training. This work
changes evaluation consumers; it does not change any running training code.

## Prespecified comparison for go/no-go

The primary table contains the seven ladder backbones at rungs 1–4.
Every cell uses its class head, with the native crop/backbone recorded by
the ladder configuration and the checkpoint's own auxiliary configuration.
Forest class predictions collapse to 0 (nonforest) and 1–4 (forest types).
The NFI truth dominance threshold is 0.7; treeless observations are class 0.
Rung-4 fraction-head evaluation would be a separately specified secondary
analysis. No threshold is tuned on these held-out observations.

A model winner means the highest observed overall accuracy on the common
population. Statistical support is reported separately. The joint report
uses the existing spatial-block bootstrap (50 km EPSG:3006 grid, 10,000
draws, seed 20260818) and a paired sign-flip randomization over whole blocks.
Holm correction covers the complete model-model/model-NMD comparison family.
The block test assumes model-label exchangeability within independent blocks;
small block sets use exact enumeration, larger sets use 10,000 Monte Carlo
draws with the plus-one correction. Plot-level McNemar is diagnostic only. Pair confidence intervals and
equivalence checks are marginal, not simultaneous. A nonsignificant
difference does not establish equivalence. SESOI is 0.02. Class supports,
confusion matrices, paired differences and NMD coverage exclusions remain
visible.

These historical checkpoints do not constitute runtime-identical controlled
ablations. In particular Tessera evaluation uses the promoted v2 source;
older models were trained with earlier inputs. The result answers which
available checkpoint performs best under this recorded evaluation protocol,
not whether a single architectural change caused a difference.

## Observation independence and selection

Identity is `(TractID, PlotID, Year)`. A test tile is insufficient:
the same observation can occur in a teacher's training tile.

1. Read each of the seven actual teacher feature tables and split records.
   Verify tile membership and the recorded train/test row counts. Recover
   missing feature years only through an unambiguous observation/tile join
   to the teacher's source index; never infer a year from a tile name.
2. Form the union of all teacher-training plot-years. Exclude that union
   from the candidate NFI index across every tile. Different inventory
   years remain different observations.
3. Reject any campaign tile in a teacher's training list. Keep campaign and
   training roots disjoint; require exactly 475 campaign tile files.
4. Require the promoted Tessera source, valid flags/arrays and current
   SAR prerequisites. Check every campaign tile, including unindexed tiles.
   Record modality gaps. Require NFI year to equal spectral year.
5. Intersect native model crop support, choose the lexically first eligible
   tile per observation, and retain exactly one row per plot-year. Reject
   inconsistent field truth.
6. Save `holdout.parquet`, `training.parquet` and `manifest.json` in a
   new run directory on the data volume. The manifest binds code, runtime
   image digest, checkpoints, teacher artifacts, indices, selected tiles,
   promotion evidence and NMD rasters by content hash.

The promotion producer uses a Counter and omits zero-valued
`FLAG_ON_EMPTY`. The gate accepts an omitted value only together with
successful completion, the expected source, complete cohort counts for
`has_v1` and `stamped`, and no remaining v2/unreadable records.
The per-tile audit independently rejects positive flags on invalid arrays.

Raw NFI records and coordinates stay in the data environment. Return only
aggregate support/exclusion counts, hashes and execution evidence for review.

## Preparation after the promotion gate

Capture the actual completed Kubernetes Job JSON using explicit context
`icekube` and namespace `prithvi-training-default`. Read the promotion
report through the data mount. Run `scripts/prepare_nfi_holdout.py --help`
from the committed, reviewed source in the pinned evaluation environment.
Supply the v4 candidate index, the teacher source index, seven-teacher root,
checkpoint root, separate cohort/staging roots, job/report evidence,
NMD rasters, a digest-pinned runtime image and a new output directory.

The command performs no inference or NMD sampling. It refuses a dirty
source tree, incomplete promotion, ambiguous years, staging leakage and
empty support. Changed files during preparation abort before the manifest
is written. Do not call historical predictions reproducible solely because
the filenames match: historical dumps lack the required provenance.

Present for go/no-go:

- Terminal job/report evidence and zero invalid positive Tessera flags.
- Plot-year overlap = 0 against the union of the seven teachers; campaign
  tiles in training = 0.
- Unique held-out count, class/year support, modality and border exclusions.
- Manifest digest, checkpoint identities, runtime image and code identity.
- CPU regression results, in-environment preparation verification, Claude's
  SHA-bound review and the bounded execution plan.

## Execution only after go/no-go

Each cell runs `validate_against_nfi.py --holdout-manifest MANIFEST --cell CELL`
with its recorded checkpoint and a per-plot output. Native crop geometry is
per tile. Authenticated input readers check the exact checkpoint/tile bytes
consumed. The sidecar `.parquet.meta.json` binds the prediction dump to its
cell, checkpoint and freeze digest. A changed input is a failure.

After all cells finish, `score_nfi_holdout.py` verifies every expected dump,
samples the frozen NMD raster(s), and applies one common coverage mask to
every model and baseline. No source gets its own easier denominator.
`ladder_fieldtruth_standings.py` remains a diagnostic convenience unless
an NFI freeze is supplied; LUCAS is always marked diagnostic there.

The separate 28-class tile benchmark is not the NFI field-truth answer.
Its resume fingerprint binds tile/label bytes, scoring source and model
configuration, and its reports persist that fingerprint. Missing cells or
zero scored foreground support cannot pass as a completed evaluation.

## Current validation scope

CPU tests exercise real NPZ/parquet/JSON I/O, actual preprocessing and
authenticated reader failure paths with synthetic model forwards. They do
not establish the runtime or scientific result of a real GPU evaluation.
The frozen population, in-cluster preparation report and GPU result are
pending the promotion gate and subsequent go/no-go.
