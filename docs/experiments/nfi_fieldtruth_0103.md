# Task 0103: held-out NFI model and NMD evaluation

Owner: Codex, branch `agent/te/codex/eval-nfi-fieldtruth`.
Reviewer: Claude. This PR replaces #65, #62 and #63 after review; those PRs
were closed after Claude review. No changes were pushed to their author
branches.

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
Assign each tract-year to the grid cell of its centroid in the complete
frozen population. Bootstrap and sign-flip use that same assignment; NMD
sampling retains each observation's original coordinates.
Holm correction covers the complete model-model/model-NMD comparison family.
The block test assumes model-label exchangeability within independent blocks;
the complete signed-sum distribution is computed by dynamic programming
on integer block sums, with no Monte Carlo p-value floor. Plot-level McNemar
is diagnostic only. Pair confidence intervals and
equivalence checks are marginal, not simultaneous. A nonsignificant
difference does not establish equivalence. SESOI is 0.02. Class supports,
confusion matrices, paired differences and NMD coverage exclusions remain
visible.

These historical checkpoints do not constitute runtime-identical controlled
ablations. In particular Tessera evaluation uses the promoted v2 source;
older models were trained with earlier inputs. The result answers which
available checkpoint performs best under this recorded evaluation protocol,
not whether a single architectural change caused a difference.

Tobias withdrew the 2024-only scope on 2026-10-08 and requested evenly
distributed inventory years. The current candidate audit is dominated by
2024 (2,212 of 2,354 independent plot-years before modality/crop checks).
It is not the final study population. Target years and per-year counts need
to be set before a balanced selection and any additional fetch campaign;
no new fetch is authorized by this protocol. Do not freeze the current
unbalanced candidate population as the final evaluation.

The current selector can include all eligible held-out observations
(`--population all`). Every row records `tile_role`; report descriptive
accuracy and confusion matrices separately for campaign and cohort tiles.
Cohort tiles may carry the exact NMD2023 targets seen during segmentation
training. Their NFI measurements remain held out, but comparison with NMD
there measures agreement with independent field truth on familiar imagery
and label locations. This dependence must be stated alongside the result.
`--population campaign` selects only eligible campaign observations before
freezing as a diagnostic option. Neither existing selector by itself
establishes the requested year balance. Final year selection and equal
counts must be prespecified and recorded in the approved manifest.

## Observation independence and selection

Identity is `(TractID, PlotID, Year)`. A test tile is insufficient:
the same observation can occur in a teacher's training tile.

1. Read each of the seven actual teacher feature tables and split records.
   Verify tile membership and the recorded train/test row counts. Recover
   missing feature years only through an unambiguous observation/tile join
   to the teacher's source index; never infer a year from a tile name.
2. Check each head's seed, raw training-row count and feature width against
   its split/table. Require all eligible dense-label sidecars, and verify
   every file in each model directory carries that head's SHA256 prefix.
   Historical heads lack a split/feature digest and downstream checkpoints
   lack a consumed-sidecar inventory. These consistency checks cannot
   retrospectively prove the exact training lineage. Conservatively exclude
   the union of **all seven recorded feature pools**, including old test
   observations, from every candidate tile. Save this exclusion population
   in `training.parquet`; report the actual recorded train-union separately.
   Different inventory years remain different observations.
3. Reject any campaign tile in a teacher's training list. Keep campaign and
   training roots disjoint; require exactly 475 campaign tile files.
4. Require the promoted Tessera source, valid flags/arrays and current
   SAR prerequisites. Check every campaign tile, including unindexed tiles.
   Check spectral/Tessera/SAR/B08/rededge grid and frame shapes, then the
   union of auxiliary channels declared by all 28 authenticated checkpoints.
   Missing/malformed stored aux excludes the tile. Legitimate physical zeros
   remain valid; partial markfukt NaNs are recorded. Native neutral fill for
   absent 2016 SAR baselines and optional CROMA B01/B09 padding is recorded
   for go/no-go; unsupported ERA5 checkpoint channels fail preparation.
   Require NFI year to equal spectral year using the training year resolver;
   unknown or contradictory years are excluded. Both explicit year fields
   must be finite integer scalars when present; otherwise dates resolve the
   growing-season year. Require four valid DOYs and finite location metadata.
   Prithvi inference reuses training's coordinate builder: prior autumn has
   year-1, growing frames have year, and single-frame input has DOY zero.
   Campaign tiles with dates and no explicit year therefore use their actual
   year, with no 2022 substitute and no changes to stored input data.
5. Intersect native model crop support, prefer an eligible campaign tile,
   then choose the lexically first eligible tile within the role per observation, and retain exactly one row per plot-year. Reject
   inconsistent field truth.
6. Save `holdout.parquet`, `training.parquet` and `manifest.json` in a
   new run directory on the data volume. The manifest binds code, runtime
   image digest, checkpoints, teacher artifacts, indices, selected tiles,
   promotion evidence and NMD rasters by content hash.

The promotion producer uses a Counter and omits zero-valued
`FLAG_ON_EMPTY`. The gate accepts an omitted value only together with
successful completion, the expected source, complete cohort counts for
`has_v1` and `stamped`, explicit `has_v2_left=0`, and
`promoted + already = cohort_count`. Missing error-only Counter keys mean
zero; required producer counters cannot be absent.
The per-tile audit independently rejects positive flags on invalid arrays.

Raw NFI records and coordinates stay in the data environment. Return only
aggregate support/exclusion counts, hashes and execution evidence for review.

## Preparation after the promotion gate

Capture the actual completed Kubernetes Job JSON using explicit context
`icekube` and namespace `prithvi-training-default`. Read the promotion
report through the data mount. Run `scripts/prepare_nfi_holdout.py --help`
from the committed, reviewed source in the pinned evaluation environment.
Supply the v4 candidate index, the teacher source index, seven-teacher root,
checkpoint root, `--distill-root /cephfs/distill`, separate cohort/staging
roots, job/report evidence,
NMD rasters, a digest-pinned runtime image and a new output directory.

Use the existing `docker/ladder-crop-distill` image rebuilt at the reviewed
source SHA, and `/opt/venvs/scoring/bin/python` for preparation. Supply
`--runtime-manifest /opt/provenance/runtime.json` and `--source-git-sha SHA`.
The baked runtime manifest verifies the complete source tree, dependency
identities and interpreter without requiring Git in the container. An image
built before this change cannot be substituted. The image build exercises
the preparation imports and the sealed-source verifier in the actual CPU
environment without PyTorch or field data. Preparation invokes a CPU-only
metadata child under `/opt/venvs/model/bin/python`: `weights_only=True`,
`map_location="meta"`, authenticated private checkpoint copies, no model
construction or forward. Mount a writable Pod-private `TMPDIR` large enough
for one checkpoint; keep source/runtime and input data mounts read-only.
The model-image smoke exercises Rasterio with a two-pixel synthetic raster
and runs the checkpoint metadata child with a synthetic safe checkpoint and
private TMPDIR. The scoring-image smoke also imports standings. Neither smoke samples real NMD or performs inference.
Capture the actual Pod `imageID` alongside the requested digest; the runtime
manifest checks content, but cannot independently discover its OCI identity.

The command performs no inference or NMD sampling. It refuses altered
source or the wrong interpreter, incomplete promotion, ambiguous years, staging leakage and
empty support. Changed files during preparation abort before the manifest
is written. Do not call historical predictions reproducible solely because
the filenames match: historical dumps lack the required provenance.

Present for go/no-go:

- Terminal job/report evidence and zero invalid positive Tessera flags.
- Plot-year overlap = 0 against all seven teacher feature pools; campaign
  tiles in training = 0.
- Unique held-out count, class/year support, modality and border exclusions,
  native neutral-fill gaps and historical provenance limitations.
- Manifest digest, checkpoint identities, runtime image and code identity.
- CPU regression results, in-environment preparation verification, Claude's
  SHA-bound review and the bounded execution plan.

## Execution only after go/no-go

Each cell runs `validate_against_nfi.py --holdout-manifest MANIFEST --cell CELL`
with `--expected-manifest-sha256 APPROVED_SHA256`, its recorded checkpoint
and a per-plot output. Supply the externally approved digest to every frozen
consumer (validate, score, compare and standings); never derive approval from
the mutable manifest being consumed. Incomplete source/28-cell/protocol
rosters fail closed. Every frozen consumer verifies the full sealed source,
dependencies and its interpreter against the preparation runtime.
Native crop geometry is
per tile. Authenticated input readers check the exact checkpoint/tile bytes
consumed. The sidecar `.parquet.meta.json` binds the prediction dump to its
cell, checkpoint, freeze digest, source payload, source commit and runtime
identity. A changed input is a failure.

After all cells finish, run `score_nfi_holdout.py` with
`/opt/venvs/model/bin/python`, which contains the pinned Rasterio sampler.
This remains CPU scoring; the separate preparation/scoring interpreter
does not include Rasterio. The command verifies every expected dump,
samples the frozen NMD raster(s), and applies one common coverage mask to
every model and baseline. No source gets its own easier denominator.
`ladder_fieldtruth_standings.py` remains a diagnostic convenience unless
an NFI freeze is supplied; it uses the verified scoring interpreter.
LUCAS is always marked diagnostic there.

Observation-level independence is established against the recorded teacher
feature pools. It is not a claim of geographic independence from every
historical training tile. The candidate support is dominated by the 2024
campaign; a result on that population does not establish performance across
inventory years. Report the surviving year and tile-role counts after the
all-feature exclusion and input checks. For 2024 observations, NMD2023 is
one year earlier (NMD2018 is six years earlier), so temporal mismatch remains
a limitation. Promoted
Tessera v2 is not proven prediction-equivalent to the historical v1 inputs;
the comparison ranks these stored checkpoints under the new input protocol.

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
