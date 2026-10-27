# Multi-year NFI held-out evaluation, step 6: inference on the frozen set

Task 0118, run 2026-10-27. This follows [the probe and run plan](nfi_inference_probe_2026-10-27.md)
and applies its Jobs to [the freeze](nfi_multiyear_freeze_2026-10-27.md) (manifest
sha256 `daaa4287…ff2c`). Tobias confirmed the plan on 2026-10-27 (~19:55Z). No held-out
score is in this record, in the JSON companion or in any log: the Jobs and the
verification print counts, timelines and digests only. Scoring against NMD needs
Tobias's own go.

## Verdict

**All 28 cells (7 models × 4 rungs) are inferred, published and verified.** The seven
Jobs (`k8s/nfi-inference-job.template.yaml`, applied from main `eb39d3a` at 16:33Z)
each exited 0 with no failed cell. A read-only verification Job then ran the scorer's
own checks over the 28 files: every file's provenance matches the frozen manifest,
the frozen checkpoint and the freeze's runtime; every file covers the 5,194 frozen
plot-years exactly once on their frozen tiles; and every file's truth equals the
frozen truth. The 28 prediction digests are distinct.

| Model | Rung 1 | Rung 2 | Rung 3 | Rung 4 | Pod finished (UTC) |
|---|---|---|---|---|---|
| clay | 47 min | 48 min | 48 min | 48 min | 19:46 |
| croma | 95 min | 95 min | 95 min | 96 min | 23:02 |
| prithvi300m | 33 min | 30 min | 33 min | 33 min | 18:49 |
| prithvi300m4f | 40 min | 42 min | 40 min | 40 min | 19:21 |
| prithvi600m | 60 min | 63 min | 60 min | 61 min | 20:44 |
| terramind | 30 min | 30 min | 30 min | 30 min | 18:40 |
| tessera | 40 min | 42 min | 40 min | 39 min | 19:21 |

The run used 23.4 GPU-hours on 2080 Ti nodes; with the probe, 24.5. Each cell's first
8 minutes are the identity checks (manifest, sealed source, checkpoint, 1,269 tile
hashes); seven Jobs reading at once slowed them little.

`prithvi600m_r1` produced the same prediction sha256 (`d0c95f23…`) as the probe did
three hours earlier on another node: frozen-mode inference is reproducible.

## Where things are

- Prediction files and metadata: `nfi/nfi_inference_2026-10-27/` on the PVC, 28
  parquet + 28 `.meta.json`, 0644. They carry NFI plot identities and stay on the
  PVC; the record names none.
- Job logs: the namespace's `gpu-reaper` deletes a finished GPU Job after 60 minutes,
  archiving its logs, spec and events to `/cephfs/ops/reaper_archive/` first. Six of
  the seven Jobs were collected from there.
- Verification: `k8s/nfi-inference-verify-job.yaml` (model interpreter, UID 2000,
  PVC read-only). An earlier pass over the first 25 cells found only the three cells
  then still running.

## Next

Scoring: `scripts/score_nfi_holdout.py --holdout-manifest … --dump-dir
nfi/nfi_inference_2026-10-27 --expected-manifest-sha256 daaa4287…`, which needs
Tobias's go. Nothing in this step computed a score.
