# Training Tile Fetch — Cloud Filtering and Pre-2018 S2 Data

*Lesson learned: April 2026*

---

## Problem: 3,850 tiles with all-2018 autumn frames

After fetching 5,609 training tiles we found that 3,850 tiles had all four
temporal frames from 2018. Frame 0 should be autumn of `year - 1` (September–October
2017 for 2018-year tiles) but instead contained a 2018 scene.

**Root cause — two compounding bugs:**

1. **DES STAC has no pre-2018 index.** `_stac_available_dates()` queries
   `https://explorer.digitalearth.se/stac/search`. This archive only indexes
   Sentinel-2 L2A data from 2018 onwards. For any `date_range < 2018-01-01`
   it returns zero candidates. The code then fell through to synthetic
   date generation.

2. **Synthetic candidates were too sparse and cloud threshold was too strict.**
   The fallback generated one candidate every `(d1-d0).days // 6` days
   (≈ every 10 days for a 60-day window) and passed `cloud_threshold=0.15`
   to `fetch_s2_scene`. Swedish September–October has narrow clear windows;
   the few clear 2017 dates (`2017-09-10`, `2017-09-25`) were missed by the
   sparse grid, and when hit, rejected by the strict cloud test.

**Confirmed:** CDSE Sentinel Hub has Sentinel-2 data from 2016 onwards
(see https://collections.sentinel-hub.com/sentinel-2-l2a/). The problem
was purely the DES STAC index, not data availability.

---

## Fix in `imint/training/tile_fetch.py`

### 1. Skip DES STAC for pre-2018 date ranges

```python
# DES STAC only indexes S2 from 2018 onwards — skip for pre-2018 ranges
if date_end >= "2018-01-01":
    try:
        dates = _stac_available_dates(
            coords_wgs84, date_start, date_end,
            scene_cloud_max=scene_cloud_max,
        )
        candidates.extend(dates)
    except Exception:
        pass
```

### 2. Denser synthetic candidates for pre-2018

```python
# Pre-2018: probe every 3 days (S2A revisit ~5 days, single satellite era)
# Post-2018 (S2A+S2B): STAC already found good candidates
step = 3 if date_end < "2018-01-01" else max(1, (d1 - d0).days // 6)
```

### 3. Two-pass cloud filtering via `cloud_threshold` parameter

`_fetch_single_scene` now accepts `cloud_threshold` and `haze_threshold`
as parameters (previously hardcoded to 0.15 / 0.08).

---

## Two-pass cloud filtering — the pattern

Matching the showcase pipeline design, training fetches use two independent
cloud checks:

| Pass | Parameter | Scale | Source | Autumn value | Growing season value |
|------|-----------|-------|--------|--------------|----------------------|
| 1 — full swath | `scene_cloud_max` | 0–100 % | STAC `eo:cloud_cover` metadata | 60.0 | 30.0 |
| 2 — tile cutout | `cloud_threshold` | 0–1 fraction | `fetch_s2_scene` SCL check | 0.30 | 0.15 |

**Why different values per frame type:**
- A full 290 km Sentinel-2 swath can be 60% cloudy globally yet still have
  a clear 2.56 km tile window — `scene_cloud_max` is a coarse pre-filter only.
- `cloud_threshold` is the authoritative check on the actual spectral cutout.
  Growing season frames need ≤ 15% for accurate NDVI / LAI phenology.
  Autumn frames (stubble, bare soil, early frost) tolerate up to 30% because:
  (a) October cloud cover in Sweden averages 70–80%,
  (b) the autumn frame is a context/background frame, not phenologically critical.

---

## `patch_autumn_frame.py` — bulk fix script

Used to patch frame 0 in-place for the 3,850 same-year tiles:

```
python scripts/patch_autumn_frame.py \
    --patch-list /data/patch_list.json \
    --data-dir   /data/unified_v2 \
    --workers    6
```

Key constants (after fix):
```python
SCENE_CLOUD_MAX  = 60.0   # STAC swath filter (0-100 %)
TILE_CLOUD_MAX   = 0.30   # tile spectral acceptance (0-1 fraction)
TILE_HAZE_MAX    = 0.12
MAX_CANDIDATES   = 16     # 3-day step over 75-day window ≈ 25 tries
```

Fetch window: `{prev_year}-09-01` to `{prev_year}-10-31` (from `analyze_date_issues.py`).

Tiles that fail (no clear scene found in Sep–Oct y-1) are logged to
`/data/patch_autumn_failed.json` for manual review.

---

## K8s job

`k8s/patch-autumn-frames-job.yaml` — launched after Phase B
(`fix-tile-dates-job.yaml`, 524 scrambled tiles) releases the PVC.

```bash
kubectl apply -f k8s/patch-autumn-frames-job.yaml -n prithvi-training-default
```

---

## Data pipeline order

```
analyze_date_issues.py  → patch_list.json (3850) + refetch_list.json (524)
                                │                          │
                     patch-autumn-frames          fix-tile-dates
                     (patch frame 0 in-place)     (delete + re-fetch)
                                │                          │
                                └──────────┬───────────────┘
                                      build_labels.py
                            (rebuild all labels with merge_all())
```
