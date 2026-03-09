# Marine Commercial Showcase & Dashboard Improvements

> Commercial shipping tab (Kalmarsund), AI2 vessel detection, coordinate convention fix, grazing statistics with NMD cross-reference.

---

## Marine Commercial Tab

### Area & Date

| Parameter | Value |
|-----------|-------|
| **Location** | Kalmarsund (Kalmar Strait), east coast Sweden |
| **Center** | lon 18.04, lat 56.89 |
| **Bbox** | 17.980-18.040 E, 56.845-56.920 N (~30 km2) |
| **Primary date** | 2025-07-15 (peak Baltic shipping) |
| **Heatmap range** | 2025-07-01 to 2025-07-31 |

### Panels (9 total)

| Panel | Key | Analysis |
|-------|-----|----------|
| Sentinel-2 RGB | mc-rgb | True color with sjokort background toggle |
| YOLO fartyg | mc-vessels | YOLO vessel detections (GeoJSON vector overlay) |
| AI2 fartyg | mc-ai2-vessels | AI2 vessel detections with attributes |
| YOLO varmekarta | mc-vessel-heatmap | Multi-date YOLO heatmap (Jul 2025) |
| AI2 varmekarta | mc-ai2-heatmap | Multi-date AI2 heatmap (Jul 2025) |
| NMD marktacke | mc-nmd | Land cover classification |
| NDVI | mc-ndvi | Vegetation index |
| NDWI | mc-ndwi | Water index |
| COT | mc-cot | Cloud optical thickness |

### AI2 Vessel Detection

The AI2VesselAnalyzer (`imint/analyzers/ai2_vessels.py`) uses the xView3 model from Allen AI / Google (Nature 2023). When `predict_attributes=True`, it predicts:
- **vessel_type**: fishing, tanker, cargo, tug, etc.
- **speed_knots**: estimated speed
- **heading_deg**: heading in degrees
- **length_m**: estimated vessel length

AI2 detections are stored in `mc_ai2_vessels.geojson` with attributes merged into GeoJSON properties.

### Generation Script

```bash
.venv/bin/python scripts/generate_marine_commercial_showcase.py
```

Fetches Sentinel-2 via DES, runs SpectralAnalyzer, NMDAnalyzer, COTAnalyzer, MarineVesselAnalyzer (YOLO), AI2VesselAnalyzer, and multi-date heatmaps for both models.

---

## Marine Tab Restructuring

The original single "Marin" tab was split into two sub-tabs:
- **Marin -- Fritid** (leisure): Hunnebostrand, Bohuslan (existing)
- **Marin -- Kommersiell** (commercial): Kalmarsund (new)

Both tabs share the sjokort background toggle pattern.

---

## GeoJSON Coordinate Convention Fix

### Problem

GeoJSON files had inconsistent y-coordinate conventions:
- **YOLO vessel exporters** (`save_vessel_overlay`, `save_ai2_vessel_overlay`): Raw pixel y-down (row 0 = top)
- **LPIS/shoreline exporters** (`save_regions_geojson`): Pre-flipped to y-up (`H - row`)

The universal `coordsToLatLng` flip in `makeGeoJSON()` (`imgH - y`) worked for y-down files but double-flipped y-up files.

### Solution

Standardized all GeoJSON files to **y-down pixel coordinates** (row 0 = top). Applied `imgH - y` transformation to un-flip the pre-flipped files:
- `lpis.geojson` (imgH=344)
- `erosion.geojson` (imgH=508)
- `segformer-shorelines.geojson` (imgH=508)
- `coastline-shorelines.geojson` (imgH=508)

The universal flip in `makeGeoJSON()` now correctly handles all files.

---

## Grazing Statistics (Betesmark Tab)

### Charts Added

Four Chart.js charts in the Betesmark tab:

1. **Klassificeringsresultat** -- Horizontal bar: 68 active grazing, 8 no activity, 4 errors
2. **Areal per klass** -- Horizontal bar: area (ha) per classification
3. **Konfidensfordelning** -- Vertical bar: confidence score distribution
4. **NMD marktacke inom betesblock** -- Horizontal bar: NMD land cover within LPIS blocks (92.4% grassland, 6.6% wetland, 1.0% cropland)

Data sources:
- Classification/area/confidence: Computed from `lpis.geojson` properties
- NMD within LPIS: From `grazing_meta.json` (`nmd_within_lpis`)

### Vessel Confidence Fix

The popup for vessel detections showed "Konfidens: 0%" because YOLO GeoJSON uses `score` property while AI2 uses `confidence`. Fixed to read both: `p.confidence || p.score || 0`.

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/generate_marine_commercial_showcase.py` | Generate commercial shipping showcase images + GeoJSON |
| `docs/js/tab-data.js` | Tab configurations, legends, GeoJSON paths |
| `docs/js/app.js` | Map rendering, chart init, GeoJSON styling/popups |
| `docs/index.html` | Dashboard HTML shell with tab structure |
| `docs/data/mc_vessels.geojson` | YOLO vessel detections (Kalmarsund) |
| `docs/data/mc_ai2_vessels.geojson` | AI2 vessel detections with attributes |
| `docs/data/chart-data.json` | Chart data for all tabs |
| `docs/showcase/marine_commercial/` | 10 PNGs + sjokort placeholder |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `5c44f44` | feat: add marine commercial tab, fix vector coordinate convention, add grazing statistics |
