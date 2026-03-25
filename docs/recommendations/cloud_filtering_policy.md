# Cloud Filtering Policy — ImintEngine Showcase Pipelines

*Date: 2025-03-25*

---

## Standard: Two-stage cloud filtering

All showcase generators and analysis pipelines should use this two-stage approach:

### Stage 1: STAC scene-level pre-filter
- **Threshold: 50%** scene cloud cover
- Purpose: wide net — keeps many candidate dates for date selection
- Source: DES STAC metadata (`eo:cloud_cover`)
- Fast (metadata only, no imagery downloaded)

### Stage 2: SCL tile-level filter (per-pixel)
- **Primary threshold: 5%** cloud fraction within AOI
- **Fallback threshold: 10%** — used only if no dates pass 5% for a given year
- Source: Sentinel-2 Scene Classification Layer (SCL)
- SCL classes masked: 3 (cloud shadow), 8 (cloud medium), 9 (cloud high), 10 (cirrus)
- Computed after band fetch, on the actual tile pixels

### Fallback logic (pseudocode)

```python
# Try strict 5% first
result = fetch_sentinel2_data(..., cloud_threshold=0.05)
if result.cloud_fraction <= 0.05:
    # Accept — clean tile
    use(result)
elif result.cloud_fraction <= 0.10:
    # Fallback — acceptable if no better date exists for this year
    warn(f"Using fallback 10% threshold: {result.cloud_fraction:.1%}")
    use(result)
else:
    # Reject — too cloudy
    skip(result)
```

### Why not just use scene-level cloud?

Scene-level cloud cover from STAC metadata describes the **entire Sentinel-2 tile** (~100x100 km). Our AOI is typically much smaller (~10x10 km). A scene with 30% cloud overall may have 0% cloud over our AOI — or 100%. The SCL-based tile-level check is the only reliable way to ensure cloud-free data for the specific area being analyzed.

### Why the spectral change maps are sensitive to clouds

The multispectral change detection (L2 norm across B02, B03, B04, B08, B11, B12) computes euclidean distance between a baseline and current image. Even thin cloud or haze causes large spectral shifts across all bands simultaneously, producing hot spots in the stability/max-change maps that are cloud artifacts, not real landscape change. The 5% SCL threshold eliminates this.

---

## Applies to

- `scripts/generate_vegetationskant_showcase.py` — Kust / Vegetationskant
- `scripts/generate_kustlinje_showcase.py` — Kust / Kustlinje
- `imint/fetch.py` — `fetch_grazing_timeseries()`, `fetch_sentinel2_data()`
- Any future showcase generators
