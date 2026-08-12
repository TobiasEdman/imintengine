# Finding — LUCAS × LPIS year-matched crop crosscheck

**Date:** 2026-08-12 · **Script:** `scripts/crosscheck_lucas_lpis.py`
**What:** independent crop-truth cross-check — LUCAS field survey vs LPIS
(jordbruksskiften) subsidy declarations, **strictly year-matched** (2018 LUCAS →
2018 LPIS, 2022 → 2022; crops rotate annually). No model in the loop — this
validates the crop *labels* (both the LUCAS→unified mapping and the LPIS
supervision the model trains on).

## Coverage
4,382 LUCAS crop points → **3,496 (79.8%) fall on a year-matched LPIS parcel**.
The 20% off-parcel are LUCAS crops on undeclared/non-subsidy land or slight
geolocation offset — not LPIS-validatable. Per year: 2018 584/825, 2022 2912/3557.

## Agreement (on-parcel, year-matched)
| LUCAS crop | n | agree | note |
|---|--:|--:|---|
| sockerbetor | 22 | 0.909 | |
| potatis | 20 | 0.900 | |
| slåttervall | 649 | 0.861 | |
| korn | 274 | 0.847 | |
| vete | 446 | 0.782 | mostly clean; some →korn/havre |
| havre | 142 | 0.768 | |
| trindsäd | 50 | 0.760 | |
| majs | 30 | 0.733 | |
| oljeväxter | 98 | 0.694 | |
| råg | 32 | 0.500 | thin |
| **bete** | **1733** | **0.251** | **systematic — see below** |

**Overall 0.534, but that's dragged down entirely by `bete`. Excluding bete,
agreement is 0.812** across the cereal/root/oil/ley classes.

## The bete mismatch is a grass-category boundary, not an error
LUCAS `bete` maps to LPIS `slåttervall` (1078) far more than LPIS `bete` (435),
but **87.3% of LUCAS bete points land on LPIS grass parcels {slåttervall ∪ bete}**.
So LUCAS bete *is* grass — it just straddles LPIS's split between slåttervall
(SJV 49,50 — mowing ley on arable) and bete (SJV 52-56 — registered pasture).
Two causes: (1) L0's LUCAS→bete mapping lumps agricultural grassland into bete,
coarser than LPIS's ley/pasture split; (2) the two survey systems genuinely
draw the ley/pasture line differently. The 15/16 (slåttervall/bete) distinction
is **not reliably separable across sources**.

## Implications
- **Crop supervision (classes 11-14, 17-21) is well-founded** — independent LUCAS
  field truth agrees with the LPIS training labels at 0.69-0.91. The crop half of
  the model's supervision is sound where it's spectrally distinct.
- **Grass classes 15/16 are fuzzy across truth sources** — model accuracy on
  bete (via LUCAS *or* LPIS) should be read as grass-vs-non-grass, not
  ley-vs-pasture. Reporting bete as a hard class overstates the achievable ceiling.
- **Year-matching is load-bearing and worked** — using the wrong year's LPIS
  would scramble every rotating-crop comparison.

## Gotcha recorded
Local LPIS parquets (`data/lpis/jordbruksskiften_*.parquet`) carry the SJV
**(Y,X) axis swap** (EPSG:3006 declares [Northing,Easting]); they predate the
PVC rewrite `ae5fe17`. Detected via `minx > miny` on total_bounds and fixed with
an affine flip `[0,1,1,0,0,0]`. Without it the spatial join returns 0 matches.
See `[[reference_sjv_wfs_axis_order]]`.
