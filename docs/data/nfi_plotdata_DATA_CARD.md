# SWE NFI plot data — SLU Riksskogstaxeringen · reference dataset

> ℹ️ **Reference / validation grade, not training labels.** Staged
> 2026-06-11. These are field-measured *sample* plots (r=7 m circles ≈ 1.5
> S2 pixels), suitable for **validating / calibrating** model outputs by
> year + location — **not** wall-to-wall segmentation labels.

## Files

| file | size | notes |
|---|---|---|
| `data/nfi/swe_nfi_plotdata2.xlsx` | 14 MB | source workbook (gitignored under `data/`); sheets `ReadMe` + `2007-2025` |
| `data/nfi/nfi_plots.parquet` | 4.1 MB | parquet cache, built on first `load_nfi_plots()` call |
| `data/nfi/info-data-for-calibration.pdf` | 133 KB | SLU's official info sheet for this dataset (license, variables, caveats) |

Loader: [`imint/training/slu_nfi.py`](../../imint/training/slu_nfi.py).

## What it is

Swedish National Forest Inventory (**Riksskogstaxeringen**, run by SLU) —
the published **calibration subset**: *temporary* field plots on forest
land, inventoried nationwide each year.

- **43,892 plots**, inventory years **2007–2025**. Temporary plots are a
  fresh sample each year (single visit) — there is **no per-plot time
  series**; don't expect to track the same plot across years.
- Circular plots, **r = 7 m → 153.94 m²** (un-divided plots only).
- **Boundary plots are excluded** — plots straddling stand or land-use
  boundaries were dropped, so each plot is interior to one stand (less
  mixed-pixel ambiguity for co-location).
- Coordinates: **Easting / Northing in SWEREF99 TM (EPSG:3006)** — the
  repo's CRS.
- **Forestland** per FAO definition; 5 land-use classes present.
- A **sample** (plots clustered in *tracts*), not a wall-to-wall raster.
- `Maturityclass`, `Stand_density`, `Stand_gaps` (and `SISpecies` /
  `SiteIndex`) are populated **only on productive forest**
  (`LandUseClass == 1`, 39,212 plots); the other 4,680 plots are NaN there.

## Coverage & fitness for the S2 tiles

**Spatial extent (EPSG:3006):** Easting 271,305–917,946 · Northing
6,142,436–7,659,756 — i.e. **whole of Sweden**.

**Per-year plot counts:**

| year | n | | year | n | | year | n |
|---|---|---|---|---|---|---|---|
| 2007 | 2018 | | 2014 | 2528 | | 2021 | 2238 |
| 2008 | 2002 | | 2015 | 2674 | | 2022 | 2318 |
| 2009 | 1873 | | 2016 | 2492 | | 2023 | 2329 |
| 2010 | 1904 | | 2017 | 2531 | | 2024 | 2389 |
| 2011 | 2168 | | 2018 | 2348 | | 2025 | 2395 |
| 2012 | 2415 | | 2019 | 2319 | | | |
| 2013 | 2626 | | 2020 | 2325 | | | |

> ⚠️ **18,661 plots (43%) are ≥ 2018** — the Sentinel-2 era for this repo's
> tiles. Pre-2018 plots have no matching clean S2 in `unified_v2/`. Per the
> repo's temporal-matching rule, co-locate a plot's `Year` to the same S2
> tile-year (autumn frame from year-1).
>
> ⚠️ **Plot footprint ≈ 1.5 S2 pixels.** This is *point / sub-pixel* truth.
> Co-locate by sampling the model output at the plot centre (or a small
> window), not by area-averaging — and expect mixed-pixel noise.
>
> ⚠️ **Position accuracy is year-dependent.** Plots were navigated with a
> handheld Garmin GPS **through 2023** (metre-level error — a real fraction
> of a 10 m S2 pixel) and with an Emlid RS3 RTK GNSS (**2–5 cm**) **from
> 2024**. For tight pixel-level validation, prefer 2024–2025 plots; treat
> ≤2023 plot positions as having metre-scale uncertainty.
>
> ⚠️ **Not for area statistics.** Per SLU, this subset "cannot be used to
> derive estimates of totals or mean values for a geographic area of any
> size." It is point reference/validation data only.

## Columns (47)

Names below are the `2007-2025` data sheet's own headers (source of truth;
the `ReadMe` sheet has a few cosmetic spelling variants). Units/notes from
`ReadMe`. The per-species columns (`Vol*`, `Ba*`, `MeanDbh*`, `NoStems*`)
break down into five species: **Pine = Scots pine, Contorta = Lodgepole
pine, Spruce = Norway spruce, Birch, OtherDec = other deciduous**.

| column | description (unit) |
|---|---|
| `Year` | Year of inventory (2007–2025) |
| `TractID` | Tract ID (a tract is a cluster of plots) |
| `PlotID` | Plot ID |
| `Inventory_Date` | Date of visit (YYYY-MM-DD) |
| `Easting` / `Northing` | Plot coordinates, metres, **SWEREF99 TM** |
| `PlotArea` | m² (circular r=7 → 153.94) |
| `LandUseClass` | National land-use class (see lookup) — forestland only |
| `Treelayers` | Number of tree layers (0–3) |
| `Fully_layered` | 1 if fully layered, else 0; NaN if `Treelayers==0` |
| `MeanHeight` | Mean stand height (m); BA-weighted if >7 m else arithmetic |
| `SiteIndex` | Site index (m), Hägglund |
| `SISpecies` | Species used for site index (10=Pine, 20=Spruce) |
| `StandAge` | Stand age (yrs); BA-weighted if MeanHeight≥7 m; top-class 101 = >100 yr |
| `Maturityclass` | Maturity / felling class (2-digit, see note) |
| `Stand_density` | Stand density (0–11) |
| `Stand_gaps` | Degree of gaps in the stand |
| `DwStem` / `DwBranch` / `DwRootStump` | Dry weight stem / branch+needles / stump+roots (kg/ha; Marklund / Petersson & Ståhl) |
| `VolPine` / `VolContorta` / `VolSpruce` / `VolBirch` / `VolOtherDec` | Volume by species (m³/ha) |
| `BaPine` / `BaContorta` / `BaSpruce` / `BaBirch` / `BaOtherDec` | Basal area by species (m²/ha) |
| `MeanDbhAll` / `MeanDbhPine` / `…Contorta` / `…Spruce` / `…Birch` / `…OtherDec` | Mean DBH, BA-weighted (mm) |
| `NoStemsPine` / `…Contorta` / `…Spruce` / `…Birch` / `…OtherDec` | Stems per species (no/ha) |
| `Soilmoisture_code` | Soil moisture class (see lookup) |
| `Soiltype_code` / `Soiltexture_code` / `Soildepth_code` / `Bottomlayertype_code` / `Fieldlayertype_code` | Site/soil codes — see SLU field instructions for code↔text |

## Code lookups

Decoded by `slu_nfi.decode_codes()` (verified against `ReadMe` + value
counts):

- **`LandUseClass`** — 1 Productive forest (39,212) · 4 Mire (2,392) ·
  5 Rockland (818) · 6 Sub-alpine spruce forest (1,008) · 7 Alpine (462).
  *(ReadMe also lists 8 = Other forest impediment; absent from the data.)*
- **`SISpecies`** — 10 Pine · 20 Spruce *(NaN on non-productive plots)*.
- **`Soilmoisture_code`** — 1 Dry (2,401) · 2 Fresh (28,721) ·
  3 Fresh-moist (10,313) · 4 Moist (2,314) · 5 Wet (143).

**`Maturityclass`** is a 2-digit code; the tens digit rises with stand
development. Observed: 11 (1,128) · 21/22/23 (2,444 / 1,826 / 4,658) ·
31/32/33/34 (8,236 / 2,713 / 4,189 / 260) · 41/42 (4,491 / 9,257) · 51 (10).
The **41 / 42 / 51 bucket (13,758 plots) is final-felling-age / overmature**
— the harvest-relevant tail. Exact class thresholds: SLU field instructions
p. 6:12 (`Soiltype/texture/depth/bottom/field-layer` code tables: same PDF):
<https://www.slu.se/globalassets/ew/org/centrb/rt/dokument/faltinst/nfi_fieldwork_instructions_eng.pdf>.
These remaining coded columns are **left as raw codes** by the loader (not
decoded) to avoid asserting an unverified mapping.

## Relevance to ImintEngine — *potential* uses (none implemented yet)

Field ground-truth that lines up with most forest outputs of the model.
**Use for validation / calibration**, co-located by `Year` + bbox:

| NFI signal | model output it can check |
|---|---|
| `Maturityclass` 41/42/51, `StandAge` | **Head 2 (avverkningsmogen)** — *independent field maturity*. Today Head 2 trains on SKS felling-notifications (intent); NFI is biological-maturity truth. **Validation/calibration only — never an input** (the synthetic `harvest_probability` aux was dropped for leaking this very target). |
| `Vol*` / `Ba*` / `NoStems*` by species | **Forest classes 1–4** (tall/gran/löv/blandskog) — dominant-species check on the LULC head |
| `MeanHeight` | **height aux channel** (grunddata band 5) — field-height check |
| `Soilmoisture_code` | **markfukt aux channel** — field soil-moisture check |
| `DwStem/Branch/RootStump`, `Vol*` | biomass / volume — not in the schema today; candidate regression target |

Deriving dominant-species → class or maturity → harvest-ready boolean is a
**labelling decision left to the consumer** — the loader stays neutral
(ingest ⊥ labelling, mirroring the fetch/label separation in this repo).

## Loading

```python
from imint.training.slu_nfi import load_nfi_plots, plots_in_bbox

# All plots (builds the parquet cache on first call):
df = load_nfi_plots()

# S2-era productive-forest plots, decoded, inside a tile bbox (EPSG:3006):
df = load_nfi_plots(years=range(2018, 2026), land_use=1, decode=True)
tile = plots_in_bbox(df, (west, south, east, north))
```

`bbox` is `(west, south, east, north)` in EPSG:3006 — the same convention as
`imint.training.spatial_parquet.SpatialParquet.query`.

## Provenance

- Received 2026-06-11: `swe_nfi_plotdata2.xlsx` + `info-data-for-calibration.pdf`
  (SLU's official info sheet), staged to `data/nfi/`.
- **License: CC0 1.0 Universal (Public Domain Dedication)** — free to use,
  modify and redistribute, no attribution required (citing SLU is still
  courteous). <https://creativecommons.org/publicdomain/zero/1.0/>
- Source: **SLU Riksskogstaxeringen** (Swedish NFI) ·
  <https://www.slu.se/en/Collaborative-Centres-and-Projects/the-swedish-national-forest-inventory/>
  · contacts Jonas Fridman, Mats Nilsson.
- Field methodology / code definitions: SLU NFI field-work instructions
  (English), linked above; dataset info sheet at
  `data/nfi/info-data-for-calibration.pdf`.
