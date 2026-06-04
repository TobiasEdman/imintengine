# scripts/diagnostics/

Read-only diagnostic scripts written during the 2026-06 audit-tile
refetch debugging arc. Each script answers one specific operational
question against the live cluster's `/cephfs/` mount. None of them
modify state; safe to run alongside production fetch jobs.

All three were used in production-pod kubectl-exec sessions (the npz
files + audit JSON live in `/cephfs/`, not on the developer machine).

## Scripts

### `audit_strategy.py`

**Question:** *Of the 6786 audit-flagged tiles, which slots are actually
broken, and how many slot-fetches does the work really require?*

Per-tile per-slot validity check against the on-disk npz:
- frame all-zero? (no spectral data in that slot)
- date string empty? (slot was never fetched)
- DOY > `cap_doy` (244)? (the pre-PR#15 late-autumn bug — the original
  audit criterion)
- `temporal_mask[slot]==0`? (marked missing)

Outputs a histogram of broken-slots-per-tile, per-slot frequency
breakdown, and a rough PU/fetch economics estimate. Was the basis for
deciding to swap from SH Process (~109 k PU needed, monthly cap 30 k)
to DES-only + Sen2Cor 2017 backfill.

```bash
kubectl cp scripts/diagnostics/audit_strategy.py POD:/tmp/
kubectl exec POD -- python3 -u /tmp/audit_strategy.py > /cephfs/strategy_$(date +%s).txt
```

### `audit_funnel.py`

**Question:** *For a sample of failed tiles, how many candidates does the
ranker actually provide per slot, and how does ERA5 distribute across
them?* (i.e. *"are we trying enough options before giving up?"*)

Samples `SAMPLE_N` (default 25) tiles from the audit list, runs
`rank_stac_era5_candidates()` for each slot, reports raw STAC count vs
post-rank count, plus ERA5 best-of-slot distribution.

Was the diagnostic that proved the unified flow iterates all candidates
(not just `top_dates[0]` as some earlier code paths did).

```bash
SAMPLE_N=25 kubectl exec POD -- python3 -u /tmp/audit_funnel.py
```

### `probe_scl.py`

**Question:** *For top ranked candidates, what does SH Process's SCL
actually report, and does it match what the ERA5-adaptive gate would
accept?*

Calls `cdse_s2._prescreen_scl()` directly (bypassing
`fetch_s2_scene`'s built-in two-stage so the cloud fraction is visible
even when it would otherwise be silently rejected). Reports actual SCL
cloud_fraction vs the era5_to_scl_gate ceiling per candidate.

Runs **locally** (uses your `.env` credentials, hits public CDSE
endpoints). Used to confirm the SH Process PU exhaustion via the live
403 `ACCESS_INSUFFICIENT_PROCESSING_UNITS` response — the first probe
v1 wrongly used `crs="EPSG:3006"` (short form) instead of `_CRS_3006`
(OGC URI) and got the silent `(None, 1.0)` sentinel for everything; v2
fixed the CRS and surfaced the real 403.

```bash
python3 -u scripts/diagnostics/probe_scl.py
```

## Why these are not unit tests

They depend on live external services (CDSE STAC, Open-Meteo ERA5, SH
Process Process API) and on a mounted `/cephfs/` with the production
audit JSON and tile npz files. They are operational tools, not
regression tests. Test code lives under `tests/`.

## Known hardcoded paths

- `/cephfs/audits/frame_audit_512_*.json` — audit-job output
- `/cephfs/unified_v2_512/<tile>.npz` — per-tile data

Both are conventions of the production pod. Edit the path constants at
the top of each script if running against a different layout.
