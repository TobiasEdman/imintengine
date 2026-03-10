# Phenology & Seasonal Data Strategy

> Impact analysis of adding HR-VPP phenology and multi-temporal Sentinel-2 data on forest class separation. SOSD/EOSD directly addresses deciduous/conifer confusion. Three architectural options evaluated. Literature shows +5-15pp overall accuracy from temporal information.

---

## Current vs Proposed Input

| Data | Channels | What It Captures |
|------|----------|------------------|
| **Current**: Single summer S2 frame | 6 (B02-B04, B8A, B11, B12) | One snapshot, Jun-Aug |
| **HR-VPP Phenology** | 5 (SOSD, EOSD, LENGTH, MAXV, MINV) | When vegetation wakes up, peaks, and shuts down |
| **Seasonal S2 Frames** | 4 x 6 = 24 | Spring (Apr-May), early summer (Jun-Jul), late summer (Aug-Sep), winter (Jan-Feb) |
| **Current aux** | 5 (height, volume, basal_area, diameter, DEM) | Forestry structure |

Currently: 6 spectral + 5 aux = 11 channels. Potential: 24 spectral + 10 aux = 34 channels.

---

## Per-Class Impact Analysis

### forest_deciduous (9.9% -> potentially 25-40%)

The largest payoff. Currently trying to distinguish deciduous from coniferous using one summer image when both are fully green and spectrally near-identical.

Phenology and seasons attack this directly:
- **SOSD** (Start of Season): Deciduous leafs out 2-4 weeks later than coniferous greens up. Birch in northern Sweden: late May. Spruce: already green.
- **EOSD** (End of Season): Deciduous drops leaves in September. Conifers stay green into October+.
- **Spring frame** (Apr-May): Deciduous = bare branches (low NDVI), coniferous = green canopy. This is the **single most discriminating signal** for forest type, per the literature.
- **Winter frame** (Jan-Feb): Same effect plus snow-on-branches differences.

Literature: going from single-date to multi-temporal boosted overall accuracy from ~75% to 84-88%, with the biggest gains in forest type separation.

### forest_pine vs forest_spruce (26.8% / 36.3%)

Harder to separate, but phenology helps:
- Pine has slightly earlier green-up and grows on drier, sunnier sites (different DEM/aspect correlation)
- MAXV differs: spruce has denser canopy -> higher peak NDVI
- Winter frame: snow sits differently on pine (open crowns) vs spruce (dense pyramidal canopy)

### forest_mixed (33.5%)

Mixed forest becomes tractable if we can identify the deciduous vs coniferous signal within a pixel:
- Phenology shows intermediate SOSD/EOSD values -- between pure deciduous and pure coniferous
- Seasonal amplitude (MAXV - MINV) is higher in mixed stands (deciduous component swings more)
- This is exactly the "blandskogsforvirring" problem SLU documented -- phenology is their recommended solution

### forest_wetland (20.3%)

- LENGTH (season length) is shorter in wetland forests -- waterlogged roots delay growth
- SOSD is later on wet soils (cold, slow to warm)
- Combined with DEM from aux channels, this becomes much more separable

### cropland, open_wetland, open_land

Already decent, but seasonal data helps distinguish:
- Cropland has a sharp spring green-up (planting) and autumn drop (harvest)
- Open wetland stays green longer than upland open land
- Clearcuts (currently in open_land) have a distinctive phenology recovery pattern

---

## Architectural Options

### Option A: VPP via AuxEncoder (easy, low risk)

Add the 5 VPP channels to the existing AuxEncoder late fusion. Goes from 5 -> 10 aux channels. Minimal code change -- just enable `--enable-vpp`.

**Pro**: Simple, proven architecture. Already prepared in the codebase.
**Con**: Doesn't use multi-temporal S2 spectral data.

### Option B: Multi-temporal Backbone (high impact, more work)

Feed 4 seasonal frames through Prithvi (which supports multi-temporal input via `num_temporal_frames`). The backbone sees the full seasonal cycle.

**Pro**: Most powerful -- the backbone learns temporal features directly.
**Con**: 4x memory and compute. Need all 4 frames per tile (76 tiles already failed for cloud cover).

### Option C: Hybrid (recommended)

Multi-temporal S2 frames through backbone (spring + summer, 2 frames -- achievable with existing data) plus VPP + forestry via AuxEncoder (10 aux channels).

This gives the backbone spectral seasonality while VPP provides clean, cloud-interpolated phenology as a fallback.

---

## Risks and Considerations

1. **Data availability**: 76 far-north tiles (1.7%) have no clear spring/winter frames. VPP fills this gap -- it's derived from a full year of observations and handles cloud gaps.

2. **Label ceiling still applies**: Even with perfect temporal discrimination, NMD labels cap accuracy. But forest subtype labels are probably 60-70% accurate (not 50%), and a model that learns the correct 70% of deciduous pixels would score much higher than 9.9% IoU.

3. **Training time**: More channels = more parameters in AuxEncoder. But late fusion keeps it manageable -- adding 5 VPP channels adds only ~0.1% parameters.

4. **VPP fetch not yet run**: Code is ready (`imint/training/cdse_vpp.py`, `executors/vpp_fetch.py`) but data hasn't been fetched. That's the next step before training with it.

---

## Recommended Next Steps

1. **Fetch VPP data** and run AuxEncoder with 10 channels (Option A) first -- minimal risk, biggest bang for the buck
2. Explore multi-temporal backbone (Option B/C) if the AuxEncoder run finishes well
3. Combine with label cleaning (see [21_nmd_accuracy_and_label_noise.md](21_nmd_accuracy_and_label_noise.md)) for maximum improvement

---

## Key Files

| File | Purpose |
|------|---------|
| `imint/training/cdse_vpp.py` | HR-VPP fetch via Sentinel Hub Process API |
| `executors/vpp_fetch.py` | ColonyOS VPP fetch executor |
| `config/vpp_fetch_job.json` | VPP job specification |
| `imint/fm/upernet.py` | AuxEncoder (currently 5ch, extensible to 10ch) |
| `imint/training/config.py` | TrainingConfig with VPP enable flags |
