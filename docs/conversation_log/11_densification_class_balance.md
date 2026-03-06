# Densification & Class Balance

> SCB tatort, sea cells, sumpskog densification strategies for addressing class imbalance in the LULC training dataset.

---

## Class Distribution Problem

At 2,092 tiles (~142.8M pixels), 8 classes were under 2%:

| Class | % of pixels | Strategy |
|-------|-------------|----------|
| developed_infrastructure | 0.22% | SCB tatort densification |
| forest_wetland_deciduous | 0.25% | Sumpskog densification + class weights |
| developed_buildings | 0.27% | SCB tatort densification |
| forest_wetland_spruce | 0.37% | Sumpskog densification |
| water_sea | 0.45% | Sea cell densification |
| open_land_bare | 0.46% | Class weights |
| forest_wetland_mixed | 0.92% | Sumpskog densification |
| forest_wetland_temp | 1.03% | Class weights |

Three approaches combined:
1. **Grid densification** — extra sampling in targeted regions
2. **Inverse-frequency class weights** — up to 10x weight (in `class_schema.py`)
3. **Weighted sampler** — tiles with rare classes sampled more often

---

## SCB Tatort (Urban Locality) Densification

Source: [SCB Statistiska tatorter](https://www.scb.se/vara-tjanster/oppna-data/oppna-geodata/statistiska-tatorter/)

SCB WFS endpoint returns GeoJSON in EPSG:3006, 2,011 localities with population data.

### Strategy
| Population | Count | Method | Extra cells |
|------------|-------|--------|-------------|
| >10k | 126 | polygon-bbox, 5km spacing | ~300-500 |
| 2k-10k | 332 | centroid, 1 extra point | ~200-300 |
| <2k | 1,252 | skip (too small) | 0 |

### Implementation

**File:** `imint/training/scb_tatort.py`

- `generate_scb_densification_regions()` — downloads and caches SCB Tatorter 2018 GeoJSON from WFS
- Parses tatort polygons, filters by population, pads small bboxes to at least `patch_size_m` (2,240m)
- Returns densification region dicts compatible with `densify_grid()`

WFS URL:
```
https://geodata.scb.se/geoserver/stat/wfs?service=WFS&REQUEST=GetFeature&version=1.1.0&TYPENAMES=stat:Tatorter_2018&outputFormat=application/json
```

SCB cache (168MB) loads from disk in ~1 second on subsequent runs.

---

## Sea Cell Densification (Territorial Waters)

### Problem
`filter_land_cells()` rejected all ocean cells, preventing water_sea class training.

### Data Sources

1. **HaV havsplaneomraden** — only covered from 1 NM offshore, missed inner territorial waters
2. **Sjofartsverket territorial boundary** — 2,322 turning points in two segments ("Fast land och Oland" + "Gotland")
   - URL: `https://www.sjofartsverket.se/globalassets/tjanster/sjokort/sjoterritoriets_gransterritorialgrans.zip`

### Building the Territorial Waters Polygon
1. Convert points to EPSG:3006
2. Group by segment, sort by `Lopnr_dels`
3. Close into Polygon geometries
4. Union polygons, subtract Swedish land mask
5. Result: 10/16 test cells pass (correctly excludes Finnish-side Gulf of Bothnia)

### Two-Step Sea Cell Filter
1. Distance from Swedish land <= `max_distance_m` (5km)
2. Inside Swedish territorial waters polygon

### Implementation

**File:** `imint/training/sampler.py` (modified)

- `filter_land_cells()` — optionally returns sea cells (`return_sea_cells=True`)
- `_build_territorial_waters()` — downloads Sjofartsverket zip, builds polygon, caches as GeoJSON
- `filter_sea_cells_swedish_waters()` — two-step filter

Config:
```python
enable_sea_densification: bool = False
max_sea_distance_m: int = 5_000
```

### Encoding Issues
- macOS `unzip` couldn't handle Swedish characters; used `ditto -x -k` / Python `zipfile`
- pyshp `UnicodeDecodeError` on `.dbf`; fixed with `encoding='latin-1'`

---

## Sumpskog (Swamp Forest) Densification

Source: [Skogsstyrelsen Sumpskog](https://geodpags.skogsstyrelsen.se/arcgis/rest/services/Geodataportal/GeodataportalVisaSumpskog/MapServer)

297,517 sumpskog polygons in EPSG:3006 with tree type mappings to LULC classes.

### Approach
- Scan Sweden with 25km grid, query ArcGIS REST endpoint per cell (0.27s/query, ~3 min total)
- Cells with >= density threshold become densification regions
- Cache results in `skg_sumpskog_scan.json`

### Dilution Analysis
Adding tiles with ~3% sumpskog doesn't help individual wetland subclasses reach 2%. `forest_wetland_deciduous` is only 3.1% of all wetland pixels. Geographic densification alone insufficient — class weights + tile oversampling more effective for extremely rare subclasses.

### Final Defaults
- 5% density threshold / 10km spacing -> 772 extra cells (127 regions)

### Implementation

**File:** `imint/training/skg_sumpskog.py`

```python
_ARCGIS_URL = "https://geodpags.skogsstyrelsen.se/arcgis/rest/services/Geodataportal/GeodataportalVisaSumpskog/MapServer/0/query"
_SCAN_SPACING_M = 25_000
```

Config:
```python
enable_sumpskog_densification: bool = False
sumpskog_min_density_pct: float = 5.0
sumpskog_densify_spacing_m: int = 10_000
```

---

## NMD Land-Fraction Filter Fix

### Problem
NMD pre-filter rejected intentional sea cells: 124/227 coastal cells (55%) incorrectly marked "failed" because `land_frac < 0.05`.

### Fix Evolution
1. Added `skip_land_filter` field to GridCell for coastal cells
2. User: ALL cells should skip the land filter
3. Removed `land_frac` filter entirely
4. User: still need nodata filter -> added `valid_frac < 0.01` check

Final state: only tiles with <1% valid pixels (entirely outside NMD coverage) are skipped.

---

## Production Grid Numbers

10km grid with all densification enabled:

| Stage | Total Cells | Added |
|-------|-------------|-------|
| Base grid (10km, land) | 4,381 | — |
| + Coastal water cells | 4,608 | +227 |
| + SCB tatort (pop >= 2k) | 5,279 | +671 |
| + Sumpskog (>= 5%) | 6,051 | +772 |

### Run Command
```bash
python scripts/prepare_lulc_data.py --data-dir data/lulc_full \
  --enable-scb-densification --enable-sea-densification --enable-sumpskog-densification
```

---

## Densification Results (Last 100 Tiles vs Total)

| Class | Last 100 | Total | Change |
|-------|----------|-------|--------|
| developed_buildings | 3.86% | 0.31% | +3.55% |
| developed_infrastructure | 3.37% | 0.25% | +3.11% |
| developed_roads | 6.75% | 2.30% | +4.45% |
| water_sea | 6.15% | 2.94% | +3.22% |
| forest_pine | 10.68% | 17.98% | -7.30% |

SCB and sea densification working as intended.

---

## Final Dataset

- **5,801 tiles completed**, 250 failed
- Split: 3,840 train / 949 val / 1,012 test
- Year distribution: 67% from 2018, 33% from 2019

---

## SLU GET (Geodata Extraction Tool)

URL: `https://maps.slu.se/get/`
Auth: Shibboleth (university SSO)

API endpoints for ordering data downloads (bbox + CRS -> email link). Relevant datasets: Topografi 10, Fastighetskartan, Sjokort S-57, Hojddata, Skogskarta.

Size limits by workload class (red: 1.6 km^2, orange: 2.5 km^2, gray: 62.5 km^2, yellow: 1,200 km^2).

Created `docs/slu_get_sjokort.md` with step-by-step download instructions for nautical charts.

---

## Infrastructure Class Discussion

NMD `developed_infrastructure` (code 52, class 16): non-building, non-road built-up at 0.36% of pixels.

Decision: Run 19-class as baseline, use `use_grouped_classes=True` (10-class) as primary schema. In the 10-class schema, buildings + infrastructure + roads merge to "developed" (~3.2% combined).

---

## Key Files Modified

| File | Changes |
|------|---------|
| `imint/training/scb_tatort.py` | **New** — SCB WFS download, tatort regions |
| `imint/training/skg_sumpskog.py` | **New** — ArcGIS sumpskog scan |
| `imint/training/sampler.py` | Sea cells, territorial waters polygon |
| `imint/training/config.py` | Densification flags, year config |
| `imint/training/prepare_data.py` | 3-stage grid generation, nodata filter |
| `scripts/prepare_lulc_data.py` | 6 new CLI arguments |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `fcb37e7` | SCB tatort, coastal, sumpskog densification; simplified DES auth; years fix |
