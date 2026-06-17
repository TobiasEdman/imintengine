# SPEC — Dataset completion (promote-prep for the re-coreg campaign)

**Created:** 2026-06-17
**Status:** DECIDED 2026-06-17 — runs as one combined campaign **when Phase-1
(the des refetch) completes** (~ETA from the dashboard). Build can land ahead.
**Runs:** on `/data/unified_v2_512_recoreg`, AFTER Phase-1/2 finish, BEFORE `promote_recoreg.py`.
**Why:** the re-coreg campaign surfaced three gaps that would ship an incomplete
dataset if promoted as-is. All three reuse existing repo pipelines.

## Decisions (locked 2026-06-17, via buttons)
- **#1 VPP source:** WEkEO (`cdse_vpp src="wekeo"`) — PU-free, sidesteps the CDSE 403.
- **#2 Label:** **carry-forward** from the original `unified_v2_512` (not rebuild).
- **#3a Cloud test:** **ERA5 + actual-pixel** (SCL/COT) — not ERA5-only.
- **#3b Replacement:** **replace** failing frames with a cleaner **same-window**
  date (re-select within same year+slot window; never-worse-guarded).
- **#3c Threshold:** replace when cloud fraction **> 20%** (aggressive).
- **Trigger:** kick off when the `campaign-phase1-des-recoreg` job reaches completion.

## Sequencing
POST-Phase-1/2, PRE-promote. Order: **#3 (frame replace) → #2 (label) + #1 (VPP) → verify → promote**.
#3 changes spectral (re-fetches frames); #1/#2 are date-independent field
additions and can be one combined promote-prep pass. Labels are land-cover
(date-independent), so #3 replacing a cloudy frame never invalidates #2's label.
#1 and #2 both edit `_recoreg` npz fields → run sequentially (no write races).
Cross-cutting: atomic writes (tmp+`os.replace`); cache to disk; idempotent/
resumable; `--workers 2` for des; pin images by digest; `Verified-by:` trailers.

---

## 1. VPP backfill — ~2,500 tiles (~36%) with empty/missing VPP
**Problem:** ~36% of tiles carry all-zero `vpp_{sosd,eosd,length,maxv,minv}`,
inherited from the original (refetch preserves aux, never fetches VPP). Root
cause: VPP via CDSE SH-Process hit PU exhaustion (HTTP 403); flagged in the
2026-05-08 audit, only partially backfilled.
**Reuse:** `imint.training.cdse_vpp` — has a **WEkEO** path (`src="wekeo"`,
`$VPP_WEKEO_DIR=/data/vpp_wekeo`): free, local, PU-free; CDSE is the fallback.
Plus `imint.training.vpp_windows`, `wekeo_vpp`. Template: `k8s/prefetch-vpp-wekeo-job.yaml`.
**Plan:**
1. Enumerate `_recoreg` tiles where `vpp_sosd` is absent or all-zero.
2. Per tile: fetch the 5 VPP bands for its EPSG:3006 bbox + `tessera_year` via
   `cdse_vpp(src="wekeo")` (PU-free); CDSE only on WEkEO miss.
3. Write `vpp_*` into the npz atomically; preserve everything else.
4. Cache COGs to disk; idempotent re-runs.
5. **Verify:** re-count empty-VPP → ~0 except genuinely no-phenology tiles
   (water/urban) — record those as a known-empty set, don't retry.
**k8s:** `campaign-vpp-backfill-job.yaml`.
**⚠ Open:** WEkEO coverage/credentials for the gap tiles; the genuine-no-VPP set.

## 2. Label restore — all `_recoreg` tiles (refetch drops the label)
**Problem:** `refetch_tile` drops `label`/`label_mask`/`label_year` (in
`_REFETCH_DROP`) — `_recoreg` tiles have no training target. Must restore pre-promote.
**Two approaches:**
- **(A) Carry-forward [recommended]:** copy `label`/`label_mask`/`label_year`
  from the same-named original `unified_v2_512` tile. Valid because re-coreg keeps
  the canonical bbox/grid (M1 snaps to the NMD 10 m lattice, M2 sub-pixel-shifts
  frames to the M1 anchor, M3 absolute is OFF). The dashboard's `build_label`
  already renders the original label over `_recoreg` and it aligns. Cheap, no
  reference data.
- **(B) Rebuild [fallback]:** `unified_schema.merge_all()` (NMD+LPIS+SKS) per tile
  — needs `/data/{nmd,lpis,sks}` on PVC + compute. Authoritative; robust to any
  grid shift.
**Plan (A):** `scripts/restore_recoreg_labels.py` — per tile, read label fields
cross-dir from `{orig}/{name}.npz`, write atomically; skip if original lacks label.
**Verify:** sample N — overlay original label on the `_recoreg` anchor B04, confirm
class edges track land-cover edges; cross-check a few vs a fresh `merge_all`
(carry-forward == rebuild). Promote-block on misalignment.
**⚠ Decision:** carry-forward (A) vs rebuild (B). Recommend A + alignment check;
refetch dropped the label conservatively, but A is provably valid for M1+M2-only.

## 3. ERA5 per-frame QC + replace cloudy frames
**Goal:** test every frame (slots 0-3 + `frame_2016`) against cloud criteria;
replace failures with a cleaner **same-window** date.
**The test (reuse):** `imint.training.optimal_fetch.era5_prefilter_dates` — is the
frame's stored date in the ERA5-clear set for its window?
- **⚠ Caveat:** ERA5 is a *predictor*, not ground truth — the frame-0 example
  (~30% cloud) almost certainly passed ERA5 yet was visibly cloudy. **Recommend
  pairing ERA5 with an actual-pixel cloud measure** (SCL cloud-fraction if stored,
  or the `cot_l1c` analyzer / a B02>0.2 brightness proxy). ERA5 gates the *date*;
  the pixel measure gates the *actual* frame.
**The replace:**
- Re-select a cleaner date in the **same slot window** via
  `optimal_fetch_dates(mode="era5_then_scl")` (repo-mandated clean-scene selector),
  re-fetch that single slot via `fetch_tile_spectral` (M1+M2 with the rest) — des
  for ≥2018, sen2cor pipeline for pre-2018 (`frame_2016`/slot 0).
- **Temporal-matching guard:** replacement stays in the same year + slot window, so
  crop frames keep the label-year signal.
- **Never-worse guard:** replace only if the new frame's cloud fraction is lower;
  else keep the original.
**⚠ Deviation:** re-coreg deliberately *reuses stored dates* (never re-selects).
#3 *introduces* re-selection for failed frames — a policy change (same window/year,
cleaner day). Needs explicit sign-off.
**Plan:**
- **Phase A (measure, cheap):** ERA5(+pixel) test on all frames → report per-slot
  fail-rate + the (tile, slot) replace list. Size the burden first.
- **Phase B (replace):** re-fetch the failing subset (des/sen2cor) + M2 + write
  back; year-matched; never-worse-guarded.
**k8s:** a cheap measure job + a heavier replace job.
**⚠ Open decisions:** (i) ERA5-only vs ERA5+pixel-cloud (recommend +pixel);
(ii) accept same-window date re-selection; (iii) cloud-% replace threshold.

---

## Reuse map (don't rebuild)
- VPP: `imint.training.cdse_vpp` (WEkEO path), `vpp_windows`, `wekeo_vpp`.
- Label: `imint.training.unified_schema.merge_all`; original tiles for carry-forward.
- Frame QC/replace: `imint.training.optimal_fetch.{era5_prefilter_dates,optimal_fetch_dates}`,
  `imint.training.fetch_spectral.fetch_tile_spectral`, the `scripts/sen2cor_pipeline/`.
