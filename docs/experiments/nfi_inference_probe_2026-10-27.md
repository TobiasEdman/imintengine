# Multi-year NFI held-out evaluation, step 6: inference probe and run plan

Task 0118, 2026-10-27. This follows [the freeze](nfi_multiyear_freeze_2026-10-27.md)
(manifest sha256 `daaa4287…ff2c`). Tobias gave the go for "build + one-cell
probe"; after the probe he chose seven parallel Jobs on 2080 Ti GPUs, to run
once Codex has reviewed them. No held-out score is in this record or in any
log: the Jobs print a timeline and provenance only.

## Verdict

**One cell ran end to end in frozen mode on one 2080 Ti in 60 minutes, and
its prediction file carries the provenance the scorer verifies.** The cell
was `prithvi600m_r1`, the largest backbone, so the other 27 should take at
most as long: about 28 GPU-hours for all of them.

| Phase | Time |
|---|---|
| Identity checks (manifest, sealed source, checkpoint, 1,269 tile hashes) | 7 min 41 s |
| Model load | 27 s |
| Inference over 1,269 tiles, 5,194 plot-years | 51 min 55 s |
| **Total** | **60 min 6 s** |

GPU memory peaked at 4.2 of 11 GB, and the GPU was busy about half the time.
The cell is I/O-bound: the validator hashes each tile again before reading it.
An H100 would not shorten it much.

The prediction file's metadata recorded: status `success`, all 5,194
plot-years in the model's crop, the frozen manifest digest, the checkpoint's
sha256 as frozen, and the runtime image, source sha, source payload and
runtime manifest digests of the image the freeze ran in. Those are the fields
`verify_prediction_dump` compares before scoring.

## The probe

`k8s/nfi-inference-probe-job.yaml` runs `scripts/validate_against_nfi.py`
with `--holdout-manifest`, `--cell` and `--expected-manifest-sha256`, in the
sealed image at `4cb93b4`, as UID 2000 with the PVC read-only. Its outputs
went to the Pod's emptyDir and are gone, so this cell runs again in the full
run.

Try 1 passed every identity check and then failed while importing torch:
UID 2000 has no passwd entry in the image, and torch's compile cache derives
its directory from the user name. Setting the cache directories explicitly
fixed it. The image was not rebuilt, since its digest is bound in the
manifest.

| Try | Pod node | Start–end (UTC) | Exit | Log sha256 |
|---|---|---|---|---|
| 1 | p02r02srv05 | 13:43:48–13:51:46 | 1 | `1aed2ef0…1f00` |
| 2 | p02r02srv05 | 13:53:04–14:53:10 | 0 | `1a26c73e…e1de` |

## The run

`k8s/nfi-inference-job.template.yaml`, rendered once per model (clay, croma,
prithvi300m, prithvi300m4f, prithvi600m, terramind, tessera), gives seven
Jobs that each run their model's four rungs in sequence: about 4 hours
wall-clock if shared-storage reads keep up. Seven Jobs, not one Indexed Job,
because the cluster runs Kubernetes 1.28: there, one index's failure would
stop the other six.

Each Job writes only `nfi/nfi_inference_2026-10-27/` on the PVC; the frozen
set and everything else stays read-only. A cell is computed into the Pod's
emptyDir and then copied by exclusive create, parquet before metadata, with
the copy re-hashed against the metadata's `prediction_sha256`. An existing
output is refused, a failed rung does not stop the next, and the Job fails at
the end if any cell failed. `tests/test_nfi_inference_job.py` runs the
template's own script against a fake validator for each of these paths;
six mutations of the script are each caught.

## Next

Codex reviews the Jobs; then the seven Jobs run. Scoring against NMD
(`scripts/score_nfi_holdout.py`) needs all 28 prediction files and Tobias's
own go.
