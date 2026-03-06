# Auxiliary Channels & Late Fusion

> Height, volume, basal area, diameter from Skogsstyrelsen, DEM from Copernicus, AuxEncoder late fusion architecture, normalization statistics.

---

## Auxiliary Data Sources

### Skogsstyrelsen Tree Height

**Endpoint:** Public kartportal proxy (no auth required). URL configured via `.skg_endpoints` file (gitignored).

Discovery: Opened kartportal in Chrome DevTools, found OpenLayers map layers, traced proxy URL that forwards to authenticated ImageServer without requiring authentication.

**File:** `imint/training/skg_height.py`
- `fetch_height_tile(west, south, east, north, *, size_px, cache_dir)` -> (H, W) float32 meters
- Converts decimeters -> meters, clamps negatives to 0, caches as `.npy`
- Supports non-square tiles via `size_px: int | tuple[int, int]`
- `_MOSAIC_RULE = '{"mosaicMethod":"esriMosaicNone","where":"ProductName = \'THF\'"}'`

Test: Ljungby forest 256x256, 0-30.3m, 89.9% forest, 0.7s. Cache: 70x faster (0.51s -> 0.007s).

### Skogliga Grunddata (Volume, Basal Area, Diameter)

**Endpoint:** Skogsstyrelsen kartportal proxy (URL in `.skg_endpoints`)

10-band TIFF via `renderingRule: {"rasterFunction":"none"}`:

| Band | Variable | Unit |
|------|----------|------|
| 0 | Volym (virkesforrad) | m3sk/ha |
| 1 | Medelhojd (Hgv) | dm |
| 2 | Grundyta (Gy) | m2/ha |
| 3 | Medeldiameter (Dgv) | cm |
| 4 | Biomassa | ton/ha |
| 5 | Tradhojd (laser) | dm |
| 6 | Krontackning | — |
| 7 | Scanningsdatum | code |
| 8 | NMD produktivitet | class |
| 9 | Omdrev | version |

**File:** `imint/training/skg_grunddata.py`
- `fetch_volume_tile()`, `fetch_basal_area_tile()`, `fetch_diameter_tile()`
- All delegate to `fetch_grunddata_tile(band=N, cache_prefix="...")`

### Copernicus DEM GLO-30

**Source:** AWS S3 (public, no auth): `https://copernicus-dem-30m.s3.amazonaws.com/`

**File:** `imint/training/copernicus_dem.py`
- `fetch_dem_tile()` -> (H, W) float32 meters above sea level
- Uses rasterio `WarpedVRT` for on-the-fly reprojection EPSG:4326 -> EPSG:3006 (bilinear)
- Handles multi-tile mosaic when bbox spans multiple 1-degree COG tiles

30m native resolution, resampled to 10m via bilinear interpolation. The AuxEncoder's 3x3 conv (50m receptive field at 10m) covers ~2 native DEM pixels — sufficient for slope/aspect learning.

---

## Why These Aux Channels Help

Direct physical discrimination examples:
- Height=25m, volume=300 m3sk/ha, basal_area=30 m2/ha -> near 100% mature conifer
- Tall (15m) vs bushes (2m): similar NIR but different height
- Forest (20m) vs cropland (0.5m): both green vegetation
- Buildings (8m) vs asphalt (0m): similar reflectance

---

## Late Fusion Architecture

### Why Late Fusion

1. **Full resolution** — aux data at 10m (224x224 px), backbone internal at 14x14 tokens. Late fusion preserves 16x more spatial detail.
2. **Different data types** — Prithvi trained on optical reflectance; aux from laser scanning (ALS). Different statistical properties.
3. **Zero risk to backbone** — Prithvi's 300M weights completely untouched.
4. **Regional flexibility** — can train different CNN encoders for areas with different data availability.

### Implementation

```
Input (B, 6, 1, H, W) -> PrithviViT backbone (frozen/partial unfreeze)
    -> 4 feature levels [256, 512, 1024, 1024]
    -> UPerNet decoder -> (B, 256, H, W)

Aux (B, N, H, W) -> AuxEncoder (2x Conv3x3+BN+ReLU) -> (B, 64, H, W)

Cat [decoded, aux_feat] -> (B, 320, H, W)
    -> Fusion Conv 1x1 -> (B, 256, H, W)
    -> SegmentationHead -> logits
```

**File:** `imint/fm/upernet.py` (modified)
- `AuxEncoder`: 2x `ConvBnRelu` layers, N -> 64 channels
- `PrithviSegmentationModel.__init__` takes `n_aux_channels: int = 0`
- `forward(x, aux=None)` — fuses when aux provided, passes through when not

Fusion uses 1x1 ConvBnRelu (320 -> 256) so the segmentation head weights are compatible with old checkpoints.

---

## Dataset Integration

**File:** `imint/training/dataset.py`

Aux channels stacked into `(N, H, W)` array, z-score normalized:

```python
aux_channels = []  # built from config flags
for ch_name in ['height', 'volume', 'basal_area', 'diameter', 'dem']:
    if getattr(config, f'enable_{ch_name}_channel', False):
        aux_channels.append(ch_name)
```

Z-score normalization:
```python
for i, ch_name in enumerate(aux_names):
    mean, std = self.config.aux_norm[ch_name]
    aux_stack[i] = (aux_stack[i] - mean) / max(std, 1e-6)
```

Missing aux channels zero-filled (35 tiles missing DEM — sea/border tiles):
```python
if ch_name in data:
    aux_arrays.append(data[ch_name].astype(np.float32))
else:
    aux_arrays.append(np.zeros((h, w), dtype=np.float32))
```

Augmentation functions handle both `(C, H, W)` and `(N, H, W)` aux via stacked format (flip/rotate axes adjusted from (0,1) to (1,2)).

---

## Normalization Statistics

Computed from all tiles using `scripts/compute_aux_stats.py`:

| Channel | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| height | 7.31 m | 6.49 | 0.1 | 49.7 |
| volume | 109.69 m3/ha | 106.77 | 1 | 3,507 |
| basal_area | 15.07 m2/ha | 9.95 | 1 | 137 |
| diameter | 15.28 cm | 7.88 | 1 | 100 |
| dem | 261.56 m | 214.59 | 0 | 1,647 |

---

## Config

```python
enable_height_channel: bool = False
enable_volume_channel: bool = False
enable_basal_area_channel: bool = False
enable_diameter_channel: bool = False
enable_dem_channel: bool = False
aux_cache_enabled: bool = True

aux_norm: dict = {
    "height":     (7.31, 6.49),
    "volume":     (109.69, 106.77),
    "basal_area": (15.07, 9.95),
    "diameter":   (15.28, 7.88),
    "dem":        (261.56, 214.59),
}
```

Channel order (fixed, for AuxEncoder): height -> volume -> basal_area -> diameter -> dem

---

## Prefetch Scripts

### `scripts/prefetch_aux.py`
- Generalizes over all aux channels
- Checks per-tile which channels are missing, fetches only missing ones
- Atomic rewrite with progress tracking
- Registry: `_CHANNEL_FETCHERS = {"height": ..., "volume": ..., ...}`

### Prefetch Results

| Channel | Tiles | Time | Rate | Failures |
|---------|-------|------|------|----------|
| Height | 5,801 | 12.3 min | 7.9/s | 0 |
| Volume+basal+diameter | 5,730 | ~1.5h | ~1.1/s | 0 |
| DEM | 5,801 | ~24 min | ~4/s | 0 |

Final: 5,766/5,801 (99.4%) have all 5 aux channels. 29 missing DEM (sea/border), 6 missing volume/basal/diameter.

---

## Slope/Gradient Discussion

AuxEncoder can learn slope directly from raw DEM. First 3x3 conv has same receptive field as Sobel operator — can learn `dz/dx`, `dz/dy`, slope, aspect, curvature. With 64 output channels and ReLU, also learns non-linear combinations and elevation thresholds (e.g., "alpine > 800m").

Decision: Start with raw DEM. Slope/aspect can be added as optional extra channels later.

---

## DEM Resampling (30m -> 10m)

Bilinear interpolation during reprojection. Elevation correct, slope slightly smoothed, curvature weaker. Still effective because:
- AuxEncoder receptive field = 50m (covers ~2 native DEM pixels)
- Elevation itself is highly informative
- All aux have similar resampling (Skogsstyrelsen native 12.5m -> 10m)
- Spatial alignment most important

---

## Aux Training Results

Training with `--enable-all-aux` (5 channels):

| Metric | Baseline (ep 5) | Aux (ep 5) |
|--------|-----------------|------------|
| mIoU | 0.3530 | 0.3870 |
| Worst class | pine 0.098 | deciduous 0.101 |

Aux already +3.4 percentage points at same epoch. Epoch times highly variable on MPS (1,680s-7,101s vs ~230s baseline) due to throttling.

---

## Multitemporal Capabilities

Prithvi supports up to 4 timesteps (`num_frames=4`). Strategy: best single image per time period (not composite).

### 4 Seasonal Windows
| Frame | Period | Purpose |
|-------|--------|---------|
| 0 | Apr-May | Conifer vs leafless deciduous, snowmelt |
| 1 | Jun-Jul | Full vegetation, max NDVI separation |
| 2 | Aug-Sep | Leaf color change, deciduous/mixed separation |
| 3 | Jan-Feb | Snow cover, urban/water/open land separation |

### Implementation
- Config: `enable_multitemporal`, `num_temporal_frames`, `seasonal_windows`
- `fetch_seasonal_dates()` — STAC discovery per seasonal window across years
- `fetch_seasonal_image()` — wrapper around `fetch_des_data()` for single-frame fetch
- `_fetch_worker_multitemporal()` — full pipeline per tile
- Tile format: `image (T*6, H, W)`, `label (H, W)`, `dates (T,)`, `doy (T,)`, `temporal_mask (T,)`
- Zero-padding for missing seasons when `seasonal_require_all=False`

---

## Key Files

| File | Changes |
|------|---------|
| `imint/training/skg_height.py` | **New** — tree height from kartportal proxy |
| `imint/training/skg_grunddata.py` | **New** — volume, basal area, diameter |
| `imint/training/copernicus_dem.py` | **New** — DEM from AWS S3 |
| `imint/fm/upernet.py` | AuxEncoder, fusion in forward() |
| `imint/training/dataset.py` | Stacked aux, z-score norm, zero-fill |
| `imint/training/trainer.py` | `_count_aux_channels()`, `_collect_aux()` |
| `imint/training/evaluate.py` | Aux passed through to model |
| `imint/training/prepare_data.py` | Aux fetch in both workers |
| `imint/fetch.py` | `fetch_seasonal_dates()`, `fetch_seasonal_image()` |
| `scripts/prefetch_aux.py` | **New** — batch aux prefetch |
| `scripts/compute_aux_stats.py` | **New** — normalization statistics |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `bf81a0f` | Evaluation section + dashboard + dataset fixes |
