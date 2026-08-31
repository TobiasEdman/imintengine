# Plan — season-composite S1 (SAR) enrichment, incl. the 2016 clearcut anchor

**Status:** IMPLEMENTED v3 (PC-RTC) 2026-08-17 · **Created:** 2026-08-14

## PIVOT — 2026-08-17: source switched to Planetary Computer sentinel-1-rtc
The CDSE-GRD implementation below (v2) is superseded. User decision: since S1
is being introduced to the models fresh, adopt the better product now —
**PC `sentinel-1-rtc` (γ⁰, terrain-corrected)** with **windowed COG reads**
(no product downloads). Same composite semantics (season + 2016 anchor,
single orbit, median ≤3 scenes), keys now `s1_enrich_v=3`,
`s1_source="pc-rtc-gamma0"`, **linear γ⁰** stored (fixes a latent CDSE-v2
double-log: dB was stored AND log10'd again by the model normalizers).
Measured: ~567 tiles/h (12× CDSE), georegistration verified against NMD
water labels, 2016 coverage confirmed (collection starts 2014-10).
Implementation: `imint/training/pc_s1_rtc.py`, `--s1-backend pc-rtc`
(default); CDSE path kept as fallback. Full run launched 2026-08-17.

**Motivation (user, 2026-08-14):** the current ±3-day S1↔S2 co-dating is the
wrong constraint for slow-changing land cover. It matches an S1 GRD scene to
**each S2 frame date within ±3 days**, so only **6,011/7,882** tiles get any
SAR, and even those are per-frame masked (`s1_temporal_mask`). Land cover is
stable over weeks; the winning backbone (**Tessera, 0.589**) uses *annual*
S1+S2. So: fetch S1 over **the same month as each S2 frame**, per-orbit
median-composited — full coverage, less speckle — **and give the 2016
clearcut-anchor frame (`frame_2016`) the same SAR treatment** so the
change-detection signal isn't SAR-blind.

## What changes (scope revised 2026-08-14 — user: one season composite suffices)
Replace "one scene within ±3 days per frame" with **one growing-season
composite per tile-year**. Rationale: the model glue consumes exactly ONE S1
frame anyway (`forward_router.py` reads `s1_vv_vh` as (B, 2, H, W);
`unified_dataset.py:1118` selects a single frame from the stored stack) —
per-frame composites were over-provisioned. Verified 2026-08-14.

- Window = the tile's **growing season** in its label year (VPP SOSD→EOSD via
  `tile_fetch._get_vpp_doy_windows`; fallback May–Sep at the tile's latitude).
- Query CDSE STAC for all IW-GRDH S1 scenes in that window, **filter to one
  orbit direction** (the tile's dominant / most-complete orbit — never mix
  ASC+DESC, their geometry differs), σ⁰-calibrate each (reuse
  `cdse_s1_stac`), reject >10% nodata scenes.
- **Per-pixel median** VV and VH across the surviving same-orbit scenes → one
  clean (2, H, W) SAR composite. Median suppresses speckle; a season at
  ~12-day S1A revisit yields ~6–10 passes/orbit → cap scenes used (below).

## Composites to build (2 per tile)
| Composite | Window | Purpose |
|---|---|---|
| **Season (label year)** | VPP growing season of the tile's label year | what CROMA/TerraMind train + infer on |
| **Season 2016** | same DOY window, year 2016 | SAR analogue of the `frame_2016` **clearcut change-detection anchor** (`unified_dataset.py:1194` — frame 0 "2016 summer background"): backscatter-then vs backscatter-now marks harvest (volume-scattering loss), weather-independent unlike the optical anchor. Only where `has_frame_2016==1` (~90% of 512 tiles, verified 2026-08-14) |

Note on the 2016 composite: the current SAR models (CROMA/TerraMind) consume a
single (2, H, W) S1 input and cannot yet see a before/after pair — wiring the
2016 SAR anchor in (e.g. a ΔVV/ΔVH change channel, or a 4-channel
[now, 2016] SAR input) is a separate, later model change. Fetching it now
rides the same enrichment pass at ~half the marginal cost and gives the
clearcut head a SAR change signal to grow into.

Keys: `s1_vv_vh` (2, H, W) season composite (replaces the old (T*2, H, W)
stack), `s1_vv_vh_2016` (2, H, W), `s1_dates`/`s1_dates_2016` (contributing
scene dates), `s1_orbit`, and a version marker `s1_enrich_v` = 2 so the old
±3-day data is overwritten, not skipped. The dataset's frame-selection code
(`unified_dataset.py:1118-`) simplifies to a direct read of the composite —
update it in the same pass (keep backward-compat read of v1 stacks OFF; v2
is a clean break, all tiles re-enriched).

## Orbit consistency (the correctness risk)
Median across mixed ASC/DESC orbits corrupts backscatter (different look
geometry). Pick **one orbit per tile** — the one with the most valid passes
across the windows — and use it for all frames of that tile. Record the
chosen orbit in a new `s1_orbit` key for provenance.

## Budget (CDSE — real constraint, plan around it)
S1 STAC bills **OData bandwidth (~12 TB/mo)** + **COG requests (~50k/mo)** —
separate from the spectral PU pool (`feedback_cdse_pu_budget_priorities`).
Rough cost (revised scope): 2 composites × 7,882 tiles × **≤3 scenes each**
≈ **~30–45k COG requests** — fits the 50k/mo ceiling in ONE pass, no
multi-month throttling needed. Safety margins:
1. **Cap scenes/composite at 3** (median of 3 is the speckle sweet spot;
   scenes are spread across the season, not clustered).
2. **Cache** per (orbit, product_id) in `$S1_CACHE_DIR` — the STAC backend
   already skips cached products; adjacent tiles share scenes heavily, so
   real request count is far below the naive estimate.
3. Fallback to **single best-scene per season** if budget tightens
   (still fixes coverage; drops the speckle-median bonus).
NEVER route this through the PU pool; STAC/OData only.

## Implementation
1. `imint/training/cdse_s1_stac.py` — add `fetch_s1_season_composite(bbox,
   doy_window, year, orbit, max_scenes=3)` returning per-orbit median VV/VH +
   the contributing dates; reuse existing σ⁰ calibration + nodata reject.
2. `scripts/enrich_tiles_s1.py` — swap the per-frame ±3-day loop for the two
   season composites (label-year + 2016 where `has_frame_2016`); write
   `s1_orbit` + `s1_enrich_v=2` so old ±3-day S1 is **overwritten** (not
   skipped by `has_s1`), atomically (`.npz.tmp` rename, as today).
3. `imint/training/unified_dataset.py` — replace the frame-selection read
   (~L1118) with a direct read of the (2, H, W) composite; require
   `s1_enrich_v==2` for CROMA/TerraMind tiles (fail-loud on v1 leftovers).
4. `k8s/enrich-s1-season-job.yaml` — CDSE STAC re-enrich over
   `/cephfs/unified_v2_512`, `envFrom` CDSE creds, resource-guarded.

## Follow-up (after composites land) — ΔSAR-as-aux for the clearcut head
The pretrained SAR encoders are locked to 2-channel input, so the 2016 anchor
enters through the **aux path** (same fusion the 11 aux channels ride):
add **ΔVV/ΔVH** (label-year − 2016, dB) as 2 extra aux channels
(`config.enabled_aux_names`), 11→13. No encoder surgery; the decoder's aux
fusion learns the change signal — harvest = sharp backscatter drop, which is
exactly ΔSAR. Needs a retrain to take effect; do together with the CROMA/
TerraMind re-train so it's one training round, not two.

## Verification (before the numbers count)
- Coverage: `has_s1`==1 with `s1_enrich_v==2` on ~100% of tiles;
  `s1_vv_vh_2016` present wherever `has_frame_2016==1`.
- Speckle: median composite has lower local variance than a single scene on
  a spot tile (quantify on 3–5 tiles).
- Orbit: single `s1_orbit` per tile; no ASC/DESC mixing.
- Then **re-train CROMA + TerraMind** (they consume `s1_vv_vh`; Clay is
  optical-only, unaffected) on the corrected S1 — those are the numbers that
  enter the paper.

## Out of scope
Not touching S2/optical, the label pipeline, or Clay. Model forwards
(`forward_router.py`) unchanged — they already consume a single (B, 2, H, W)
SAR input; only the dataset-side read simplifies (Implementation §3).
