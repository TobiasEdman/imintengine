# NMD Label Accuracy & Label Noise Analysis

> Research into NMD (Nationella Marktackedata) classification accuracy from quality reports and academic literature. Identifies label noise as the primary performance ceiling. Includes 5 label cleaning approaches with phased implementation recommendation.

---

## NMD Classification Accuracy

From NMD quality evaluation reports and regional comparison studies:

- **Overall agreement**: 70-80% at the most detailed class level (code level 3)
- **Best accuracy** in Gotaland (southern Sweden), **decreasing northward**
- Our validation region (lat 64-66 N, Norrland) is the **worst area** for NMD accuracy
- Largest differences in **mixed forest** (blandskog) and **deciduous forest** (lovskog) -- explicitly called out as worst classes across all regions
- Pine and spruce forests: mapped with "good accuracy"
- Deciduous (triviallövskog): mapped with only "acceptable accuracy"
- "Other open land" also has poor agreement

### Independent Tree Species Research (Sentinel-2)

- Best overall accuracy: 84-88% for pure stands only
- Pine vs spruce confusion is a known, persistent problem -- spectrally very similar
- Multi-temporal data (spring + summer + autumn + winter) helps most, but our model only uses Jun-Aug
- SLU researchers coined "blandskogsforvirring" ("mixed forest confusion") as a documented problem

---

## Per-Class IoU Explained by NMD Quality

| Class | Our IoU | NMD Quality | Explanation |
|-------|---------|-------------|-------------|
| forest_deciduous | 9.9% | Worst class | NMD labels are noisy. Deciduous patches are small, mixed, spectrally variable (birch, aspen, alder, beech all lumped). Our ceiling IS the label quality |
| forest_mixed | 33.5% | Second worst | NMD can't reliably distinguish mixed from pure stands. Inherently ambiguous class |
| forest_wetland | 20.3% | Moderate | Confused with both regular forest and open wetland. Transitional class |
| forest_pine | 26.8% | "Good" | Pine/spruce spectral similarity causes mutual confusion. Only ~3.7 month growing season in our data |
| forest_spruce | 36.3% | "Good" | Better than pine because spruce is denser/darker, more distinct |
| open_wetland | 62.4% | Good | Distinct spectral signature (wet, low vegetation) |
| cropland | 71.3% | Very good | Clear pattern (plowed/vegetated cycles), stable NMD accuracy |
| water | 93.4% | Excellent | Trivially separable, NMD near-perfect |
| open_land | 47.2% | Moderate | Heterogeneous class (bare rock, grassland, clearcuts) |
| developed | 31.5% | Moderate | Small patches mixed with vegetation |

---

## Key Insights

1. **Label noise is the bottleneck, not the model.** NMD's own accuracy for deciduous and mixed forest is probably 50-60%. Our model cannot exceed the accuracy of its training labels -- 9.9% IoU for deciduous may actually reflect learning a ~60% accurate label at 10m resolution.

2. **Geographic bias.** NMD accuracy decreases northward. Our validation split (lat 64-66 N) is in the worst region. The model probably performs better in southern Sweden than these numbers show.

3. **Temporal limitation.** We use only summer imagery (Jun-Aug). The literature shows that adding spring (leaf-out timing separates deciduous from coniferous) and autumn (leaf-off) scenes dramatically improves forest type separation.

4. **Aux channels help most where NMD is most reliable.** Cropland (+8.3pp) and open_wetland (+7.5pp) got the biggest boost from forestry auxiliary data because: (a) height=0 clearly separates non-forest, (b) DEM helps distinguish wetland. But aux data can't fix noisy forest subtype labels.

---

## Label Cleaning Approaches

### Approach 1: Cross-Reference with Forestry Aux Data (already available)

We already have Skogsstyrelsen's forestry rasters (height, volume, basal_area, diameter) loaded as auxiliary channels. These can directly flag bad NMD labels:

| Rule | Catches |
|------|---------|
| NMD = any forest, but height < 2m and volume = 0 | Clearcuts/young regeneration mislabeled as mature forest |
| NMD = open_land/cropland, but height > 10m and volume > 50 | Forest mislabeled as open land |
| NMD = deciduous, but winter NDVI > 0.4 | Evergreen conifer mislabeled as deciduous |
| NMD = pine/spruce, but spring NDVI < 0.2 | Deciduous mislabeled as conifer (bare in spring) |
| NMD = forest_wetland, but DEM > 400m and slope > 15 deg | Upland forest mislabeled as wetland |

No new data fetches needed.

### Approach 2: Skogsstyrelsen Tree Species Map (new data, high value)

Skogsstyrelsen publishes a dominant tree species raster ("tradslag") derived from airborne lidar + NFI field plots -- completely independent from NMD's satellite-based classification.

- SKS says pine-dominated but NMD says deciduous -> flag as suspect
- SKS says deciduous-dominated but NMD says spruce -> flag as suspect
- Where both agree -> high confidence label, upweight in training

This is the single most powerful cleaning source because it's independently derived from lidar, not satellite imagery. Requires one new raster fetch per tile from Skogsstyrelsen's WCS.

### Approach 3: Multi-Source Voting

| Product | Resolution | Strengths |
|---------|-----------|-----------|
| NMD 2018 | 10m | Our current labels -- satellite + lidar based |
| ESA WorldCover | 10m | Global, Sentinel-1 + S2, independent classification |
| CORINE Land Cover | 100m | Coarser but human-validated |
| Skogsstyrelsen tradslag | 12.5m | Lidar-based forest type |
| Our own model predictions | 10m | After training, use model disagreement |

Where 3+ sources agree -> high confidence. Where NMD disagrees with multiple others -> suspect label.

### Approach 4: Loss-Weighted Training (no data change needed)

Instead of removing bad labels, downweight uncertain pixels during training:

```python
confidence = compute_label_confidence(nmd_label, height, volume, dem)
loss = focal_loss(pred, target) * confidence
```

The confidence function checks: does forest height agree with forest/non-forest label? Is the pixel in a class transition zone (edge pixels are noisy)? Does the tile's NMD class distribution match regional statistics?

### Approach 5: Model-Based Self-Training (after initial training)

Use the current 43% mIoU model to identify suspect labels:
1. Run inference on all training tiles
2. Find pixels where model prediction strongly disagrees with NMD label
3. For forest subtypes where NMD is known noisy, trust the model in high-confidence cases
4. Relabel or downweight those pixels
5. Retrain on cleaned labels

This is iterative pseudo-label refinement.

---

## Recommended Phased Implementation

### Phase 1 -- Immediate (zero new data)

Add confidence weighting to the trainer. Flag pixels where forestry aux data contradicts NMD:
- Height/volume says "not forest" but NMD says forest -> weight = 0.3
- Height/volume says "forest" but NMD says open -> weight = 0.3
- Edge pixels (within 2px of class boundary) -> weight = 0.5
- Everything else -> weight = 1.0

Implement in `dataset.py` `__getitem__` and pass as weight map to loss function.

### Phase 2 -- After VPP fetch

Add phenology-based cleaning:
- NMD = deciduous but SOSD < day 100 (very early) -> suspect, likely conifer
- NMD = conifer but LENGTH < 120 days -> suspect, likely deciduous

### Phase 3 -- If needed

Fetch Skogsstyrelsen tree species raster and do full multi-source voting. This would probably bump deciduous IoU from ~10% to 20-30% just from cleaner labels.

---

## Key Files

| File | Relevance |
|------|-----------|
| `imint/training/dataset.py` | Where confidence weights would be applied |
| `imint/training/losses.py` | Loss function with sample weighting |
| `imint/training/skg_height.py` | Height data for cross-reference |
| `imint/training/skg_grunddata.py` | Volume/basal area for cross-reference |
| `imint/training/cdse_vpp.py` | VPP phenology for Phase 2 |

---

## Sources

- NMD Regional Comparisons (Naturvardsverket PDF)
- NMD 2023 Quality Evaluation (Naturvardsverket PDF)
- NMD 2018 Product Description
- Tree Species Classification with Sentinel-2 (SLU/academic literature)
- Mixed Forest Confusion (SLU research)
