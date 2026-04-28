# SPEC — WaterQualityAnalyzer (Sentinel-2, Bohuslän coastal waters)

**Created:** 2026-04-27
**Target repo:** `/Users/tobiasedman/Developer/ImintEngine`
**Status:** draft, pending fresh-session execution

## Context

A new `BaseAnalyzer` subclass that retrieves chlorophyll-a, total suspended solids, and CDOM from
Sentinel-2 L2A reflectance. Inspired by a 2026-04-08 Sentinel-2 true-color image of the Kattegatt
spring bloom (digitalearth.se). Validation focus is **Stigfjorden + waters off Mollösund**
(Bohuslän archipelago, between Tjörn and Orust) — Case-2 coastal water with strong
CDOM/sediment/phytoplankton mixing visible as plume streaks streaming SW from fjord mouths.

The analyzer combines **two AI methods** (Pahlevan MDN, ACOLITE C2RCC) with **two classical
indices** (NDCI, MCI) and writes them as four parallel outputs without fusion — research/diagnostic
mode, user picks the trusted retrieval downstream.

## In scope

- New analyzer module: `imint/analyzers/water_quality/` (package, not single file — 4 retrieval
  backends + water masking + AOI clipping)
- Registration in `ANALYZER_REGISTRY` in `imint/engine.py` as `"water_quality"`
- Config section in `config/analyzers.yaml`
- Hardcoded `imint/analyzers/water_quality/aoi/stigfjorden_skagerrak.geojson` (curated polygon,
  EPSG:4326, covers Stigfjorden between Tjörn and Orust + waters off Mollösund/Käringön +
  ~50 km arc into open Skagerrak — created 2026-04-27, v1-draft. Refinement, if needed, happens
  programmatically via Shapely + OSM coastline — no GIS desktop tool dependency)
- **Public Showcase integration** — surfaces results in the existing ImintEngine dashboard
  (`docs/index.html`) alongside kustlinje, vegetationskant, grazing, lulc, marine_commercial,
  marine_leisure showcases. See "Showcase integration" section below.
- Four retrieval backends, each producing one raster:
  - **MDN** (Pahlevan et al. 2020/2022) — pre-trained Mixture Density Network. Returns Chl-a (mg/m³),
    TSS (g/m³), aCDOM (m⁻¹), each with per-pixel σ. Weights downloaded on first run, cached to
    `~/.cache/imint/mdn/`. Source: `STREAM-RS/Mixture-Density-Networks` GitHub repo.
  - **C2RCC** — ACOLITE Python port of ESA's C2RCC neural net. Returns Chl-a, TSM, CDOM. No SNAP/JVM
    dependency. Source: ACOLITE PyPI package.
  - **NDCI** — `(B05 − B04) / (B05 + B04)`, classical, deterministic. Mishra & Mishra 2012.
  - **MCI** — `B05 − B04 − 0.389·(B06 − B04)`, classical, deterministic. Gower et al. 2005.
- Water masking via SCL class 6 (existing pipeline) with MNDWI fallback when SCL is unavailable
- AOI clipping: polygon-clipped from one or more S2 tiles at the analyzer level (new pattern —
  unlike other ImintEngine analyzers which receive a single pre-clipped tile)
- Output coordinate system: **EPSG:3006 (SWEREF99 TM)**, COG-formatted GeoTIFF, matching engine
  convention
- Per-tile output directory contains:
  - `chlorophyll_mdn.tif`, `chlorophyll_mdn_uncertainty.tif`
  - `tss_mdn.tif`, `tss_mdn_uncertainty.tif`
  - `acdom_mdn.tif`, `acdom_mdn_uncertainty.tif`
  - `chlorophyll_c2rcc.tif`, `tsm_c2rcc.tif`, `cdom_c2rcc.tif`
  - `ndci.tif`, `mci.tif`
  - `chlorophyll_spread.tif` — std across 4 Chl-a methods (disagreement map)
  - `water_mask.tif` — the mask applied
  - `summary.json` — per-method mean/p50/p95/max + AOI-clipped statistics
- Skip-and-warn failure handling (BaseAnalyzer.run wraps exceptions; per-method failures log a
  warning and omit that raster)
- Verification: visual sanity-check on 2026-04-08 Bohuslän tile — Chl-a rasters rendered as colormap
  must show structured gradients along the plume axes visible in the true-color image (mid-tile
  SW-flowing plume + lower-tile southerly plume)

## Showcase integration

This analyzer ships with a public-facing showcase, following the established repo pattern (see
`scripts/generate_kustlinje_showcase.py` and friends, `docs/js/tab-data.js` tab config,
`docs/index.html` tab surface).

**Showcase script:** `scripts/generate_water_quality_showcase.py`

- Fetches Sentinel-2 L2A timeseries for the Stigfjorden/Mollösund AOI using the same
  `fetch_grazing_timeseries()` pipeline already used by other showcases (cloud filtering via SCL,
  co-registration, NMD 10 m grid snapping)
- Selects best date per year — for water quality the season preference is **April–June**
  (spring bloom window in Kattegatt/Skagerrak), not the Jun–Aug summer window used by the land-
  cover showcases
- Verification fixed date: **2026-04-08** (the inspiring image) must be present and rendered
  even if a less-cloudy date exists for that year
- Runs `WaterQualityAnalyzer` on each selected date
- Renders PNGs via existing exporters (`imint/exporters/export.py`); add new exporters as needed
  for physical-unit rasters (Chl-a colormap, TSS colormap, CDOM colormap, spread map). Reuse
  `save_spectral_index_clean_png` for NDCI/MCI.
- Saves to `outputs/showcase/water_quality/<year>/*.png` and mirrors to
  `docs/showcase/water_quality/<year>/*.png` so the dashboard can serve them directly

**Dashboard tab (modify):** `docs/js/tab-data.js`

Add a new tab definition `f-water-quality` alongside the existing tabs. Layer toggles to expose:

- `wq-rgb` — true-color RGB baseline (always visible)
- `wq-chl-mdn` — MDN chlorophyll-a (mg/m³, viridis colormap, 0–25 mg/m³ stretch)
- `wq-chl-mdn-uncertainty` — MDN σ (greyscale)
- `wq-tss-mdn` — MDN TSS (g/m³, ocean colormap)
- `wq-acdom-mdn` — MDN aCDOM (m⁻¹, brown colormap)
- `wq-chl-c2rcc` — C2RCC chlorophyll-a (mg/m³, viridis stretch matching MDN)
- `wq-tsm-c2rcc` — C2RCC TSM (g/m³)
- `wq-cdom-c2rcc` — C2RCC CDOM (m⁻¹)
- `wq-ndci` — NDCI index ([-1, 1] diverging colormap)
- `wq-mci` — MCI index (cmocean phase or similar)
- `wq-spread` — Chl-a inter-method disagreement (red where methods disagree)
- `wq-water-mask` — binary water mask (transparency overlay)

**Dashboard tab (modify):** `docs/index.html`

Add tab button + content stub for "Vattenkvalitet" (Swedish — repo convention). Place
between the existing kustlinje and vegetationskant tabs in the navigation.

**Tab metadata for `tab-data.js`:**

```javascript
'water-quality': {
    title: 'Vattenkvalitet (Stigfjorden)',
    bbox: [/* AOI bbox in display CRS */],
    layers: [
        {id:'wq-rgb',                key:'rgb',               title:'RGB (true color)',                 legend:'rgb'},
        {id:'wq-chl-mdn',            key:'chl_mdn',           title:'Klorofyll-a (MDN, mg/m³)',         legend:'chl'},
        {id:'wq-chl-mdn-uncertainty', key:'chl_mdn_sigma',     title:'MDN osäkerhet (σ)',                legend:'sigma'},
        {id:'wq-tss-mdn',            key:'tss_mdn',           title:'TSS (MDN, g/m³)',                  legend:'tss'},
        {id:'wq-acdom-mdn',          key:'acdom_mdn',         title:'aCDOM (MDN, m⁻¹)',                legend:'cdom'},
        {id:'wq-chl-c2rcc',          key:'chl_c2rcc',         title:'Klorofyll-a (C2RCC, mg/m³)',       legend:'chl'},
        {id:'wq-tsm-c2rcc',          key:'tsm_c2rcc',         title:'TSM (C2RCC, g/m³)',                legend:'tss'},
        {id:'wq-cdom-c2rcc',         key:'cdom_c2rcc',        title:'CDOM (C2RCC, m⁻¹)',               legend:'cdom'},
        {id:'wq-ndci',               key:'ndci',              title:'NDCI (klorofyllindex)',            legend:'ndci'},
        {id:'wq-mci',                key:'mci',               title:'MCI (klorofyllindex)',             legend:'mci'},
        {id:'wq-spread',             key:'chl_spread',        title:'Metoders oenighet (Chl-a)',        legend:'spread'},
        {id:'wq-water-mask',         key:'water_mask',        title:'Vattenmask',                      legend:'mask'}
    ],
    images: {
        'wq-rgb':                    'showcase/water_quality/rgb.png',
        'wq-chl-mdn':                'showcase/water_quality/chl_mdn.png',
        'wq-chl-mdn-uncertainty':    'showcase/water_quality/chl_mdn_sigma.png',
        'wq-tss-mdn':                'showcase/water_quality/tss_mdn.png',
        'wq-acdom-mdn':              'showcase/water_quality/acdom_mdn.png',
        'wq-chl-c2rcc':              'showcase/water_quality/chl_c2rcc.png',
        'wq-tsm-c2rcc':              'showcase/water_quality/tsm_c2rcc.png',
        'wq-cdom-c2rcc':             'showcase/water_quality/cdom_c2rcc.png',
        'wq-ndci':                   'showcase/water_quality/ndci.png',
        'wq-mci':                    'showcase/water_quality/mci.png',
        'wq-spread':                 'showcase/water_quality/chl_spread.png',
        'wq-water-mask':             'showcase/water_quality/water_mask.png'
    },
    imgH: /* set after first render */,
    imgW: /* set after first render */,
    hasBgToggle: false
}
```

**Year navigation:** the showcase mirrors the existing per-year stepper UX (kustlinje does this).
Output paths include the year: `showcase/water_quality/<year>/<layer>.png`. The 2026 entry uses
2026-04-08 specifically (verification target).

**Frame proportions:** the AOI bounding box (`stigfjorden_skagerrak.geojson`) is sized to a
**1.5:1 landscape ratio in projected EPSG:3006 metres** at 58°N — matching the existing
`marine_commercial` showcase frame (893×588). All rendered PNGs inherit this aspect from the AOI
clip. Set `imgW`/`imgH` in `tab-data.js` to the actual pixel dimensions of the rendered RGB
PNG (recommend `imgW: 893, imgH: 588` to align with marine_commercial; or whatever the renderer
emits — measure once after first render and pin).

**Out of showcase scope:**

- No new colormap/legend infrastructure beyond what `imint/exporters/export.py` already provides
- No modifications to non-water-quality dashboard tabs
- No WordPress/digitalearth.se deployment — only the in-repo `docs/` dashboard
- No timeseries animation; static per-year PNGs only

## Out of scope

- **No FAI** (Floating Algae Index). Diatom spring bloom in Kattegatt does not form surface scum.
- **No bloom polygon vectorization** in v1. Output is rasters only. Threshold-based polygon export
  deferred to v2 when in-situ calibration is available.
- **No fusion / weighted ensemble.** Four methods, four rasters, plus a spread map.
- **No multi-date / temporal logic.** One `IMINTJob` = one date = one analysis. `previous_results`
  hook is not used.
- **No SMHI SHARK in-situ comparison.** Visual sanity-check only. SHARK integration deferred to v2.
- **No L1C atmospheric correction inside the analyzer.** Assumes input is L2A surface reflectance
  (existing pipeline). Note in docstring: absolute Chl-a values may be biased without C2RCC-AC; the
  in-band C2RCC retrieval here uses L2A SR as input.
- **No EPSG:4326 reprojection.** Native EPSG:3006 only. Web tiles can reproject downstream.
- **No web UI / dashboard.** Outputs are raster files; visualisation is the user's problem.
- **No training of MDN or C2RCC** — pre-trained weights only.

## Interface

Standard `BaseAnalyzer` contract — no API change to `IMINTJob` / `IMINTResult`:

```python
# imint/analyzers/water_quality/analyzer.py
from imint.analyzers.base import BaseAnalyzer, AnalysisResult

class WaterQualityAnalyzer(BaseAnalyzer):
    name = "water_quality"

    def analyze(
        self,
        rgb: np.ndarray,
        bands: dict[str, np.ndarray] | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
        scl: np.ndarray | None = None,
        geo: GeoContext | None = None,
    ) -> AnalysisResult: ...
```

Required bands: **B02, B03, B04, B05, B06, B07, B8A** (all already fetched by the engine).

`config/analyzers.yaml` entry:

```yaml
water_quality:
  enabled: true
  methods:
    mdn: { enabled: true, weights_url: "https://github.com/STREAM-RS/..." }
    c2rcc: { enabled: true }
    ndci: { enabled: true }
    mci: { enabled: true }
  aoi_geojson: "imint/analyzers/water_quality/aoi/stigfjorden_mollosund.geojson"
  bloom_threshold_chl: 5.0  # mg/m³, used in summary.json classification only
  output_format: "cog"      # GeoTIFF Cloud-Optimized
```

## Data & state

**Read:**
- `bands["B02"]` … `bands["B8A"]` — L2A surface reflectance, float32, scaled [0, 1]
- `scl` — Scene Classification Layer, uint8, used for water mask (class 6)
- `geo: GeoContext` — CRS + transform, used for AOI clipping
- `imint/analyzers/water_quality/aoi/stigfjorden_mollosund.geojson` — curated polygon
- `~/.cache/imint/mdn/*.pth` (downloaded on first run) or PVC mirror at `/data/models/mdn/` if
  present (not in v1 scope but path should not collide)

**Write:**
- `<output_dir>/water_quality/<date>/*.tif` — 13 GeoTIFFs listed in In Scope
- `<output_dir>/water_quality/<date>/summary.json` — statistics
- `~/.cache/imint/mdn/*.pth` — MDN weights cache (first run only)

**Mutate:** none. Read-only on inputs.

## Dependencies

**New deps (acceptable):**
- `acolite` (PyPI) — for C2RCC port. Pin to a tested version; check License (likely GPL-3, verify
  compatibility with ImintEngine licence before merging).
- `torch` — already in repo (Prithvi). MDN uses PyTorch.
- `requests` — already in repo. Used for MDN weight download.
- `shapely`, `rasterio`, `pyproj` — already in repo. Used for AOI clipping + reprojection.

**Must not break:**
- Existing analyzers: `spectral`, `shoreline`, `vegetation_edge`, `prithvi`, `nmd`, `cot`,
  `marine_vessels`, `change_detection`, `object_detection`, `samgeo`, `insar`, `grazing`
- `BaseAnalyzer.run()` exception-wrap contract
- `config/analyzers.yaml` schema for existing analyzers
- `IMINTJob` / `IMINTResult` dataclasses

**Off-limits:**
- SNAP / JVM dependency (rejected)
- Per-tile model training — pre-trained only
- Modifying any other repo

## Failure modes & verification

| Scenario | Expected behavior | Verification method |
|---|---|---|
| MDN weights download fails (network, 404) | Log warning, skip MDN rasters, continue with C2RCC + NDCI + MCI | Pytest with mocked HTTP 404 returns success=True with `mdn_failed: true` in summary |
| ACOLITE not installed | Log warning, skip C2RCC, continue with MDN + NDCI + MCI | Pytest with `sys.modules["acolite"] = None`; assert no crash |
| Required band missing (e.g. B05 absent) | Skip methods that need it, write only methods that can run; if no method can run, return `success=False` with explanatory error | Pytest with `bands` dict missing B05; assert NDCI/MCI skipped, MDN/C2RCC may still run depending on signature |
| `scl` is `None` | Fall back to MNDWI water mask `(B03 − B11)/(B03 + B11) > 0` | Pytest passing `scl=None`; assert mask non-empty over water |
| AOI polygon doesn't intersect tile bounds | Log warning, skip analyzer, return `success=True` with `out_of_aoi: true` | Pytest with a Stockholm-area tile; assert no rasters written |
| MDN inference produces NaN over land/cloud pixels | Mask NaN to nodata in output GeoTIFF | Inspect output raster; nodata mask aligns with water_mask inverse |
| C2RCC produces negative Chl-a | Clip to ≥ 0, log count of clipped pixels in summary | Inspect summary.json — `c2rcc_negative_clipped` field |
| Output dir not writable | Raise, let `BaseAnalyzer.run()` catch and return `success=False` | Pytest with read-only tmpdir |
| Verification: 2026-04-08 Bohuslän tile | All four Chl-a rasters show structured gradients along the two visible plume axes; spread map highlights inter-method disagreement near plume edges (high CDOM zones) | Manual review of rendered colormap PNGs alongside the inspiring true-color image |

## Constraints

**Hard limits:**
- Inference runtime per tile < 5 minutes on CPU (no GPU dependency for v1; MDN + C2RCC are small NNs)
- Memory < 8 GB peak — coastal AOI clip keeps tensors small
- No modifications to other ImintEngine analyzers
- Output bit depth: float32 for physical-unit rasters (Chl-a, TSS, CDOM), int16 for indices (NDCI,
  MCI), uint8 for water_mask
- No secrets, tokens, or credentials in the repo or weights cache

**Soft preferences:**
- Match existing analyzer code style (look at `imint/analyzers/shoreline.py` as the primary reference)
- Use `rasterio` for I/O, not GDAL CLI
- Type hints on all public functions
- Logging via Python `logging`, not `print`
- No new top-level imports in `imint/__init__.py` — keep the analyzer self-contained

## Tradeoffs accepted

- **No fusion / ensemble Chl-a output.** User chose four separate rasters over a fused product —
  more transparent for research/comparison, harder for a downstream automated consumer. Spread map
  is the compromise.
- **Polygon AOI clipping inside the analyzer**, breaking the "one tile in, one result out" pattern
  the rest of the engine follows. Justified because Stigfjorden spans tile boundaries and the AOI
  is the natural unit of analysis here. Documented as an exception in the analyzer's docstring.
- **Hardcoded AOI GeoJSON** instead of config-driven. Rigid (changing AOI = code edit + commit) but
  reproducible and version-controlled — better for a research artefact than a flexible production
  tool.
- **Visual-only validation** for v1. SMHI SHARK comparison would be more rigorous but blocks on
  data availability and adds scope. Visual match against a known plume signature is enough for the
  first cut; v2 adds in-situ.
- **L2A SR as input to C2RCC**, not L1C TOA. ACOLITE C2RCC normally expects TOA + atmospheric
  correction inside the same pipeline. Using L2A SR means atmospheric correction is already done by
  Sen2Cor (suboptimal for Case-2). Trade-off: no L1C fetch path needed in v1; absolute values
  biased but relative patterns intact. Documented as a known limitation.
- **Download-on-first-run for MDN weights** instead of vendored. Network dependency at first
  invocation; subsequent runs cached. Matches the "cache to disk" rule in CLAUDE.md.

## Execution hints

**Files likely to change / add:**
- `imint/analyzers/water_quality/__init__.py` (new, exports `WaterQualityAnalyzer`)
- `imint/analyzers/water_quality/analyzer.py` (new, ~150 LOC)
- `imint/analyzers/water_quality/mdn_inference.py` (new, ~120 LOC, includes weight download)
- `imint/analyzers/water_quality/c2rcc_wrapper.py` (new, ~80 LOC, wraps ACOLITE)
- `imint/analyzers/water_quality/classical_indices.py` (new, ~40 LOC, NDCI + MCI)
- `imint/analyzers/water_quality/water_mask.py` (new, ~30 LOC, SCL + MNDWI)
- `imint/analyzers/water_quality/aoi/stigfjorden_skagerrak.geojson` (**already created
  2026-04-27**, v1-draft polygon)
- `imint/engine.py` (modify, register in `ANALYZER_REGISTRY`)
- `config/analyzers.yaml` (modify, add `water_quality` section)
- `requirements.txt` / `pyproject.toml` (modify, add `acolite` dep)
- `tests/test_water_quality.py` (new, unit + integration tests)
- `scripts/generate_water_quality_showcase.py` (new, follows `generate_kustlinje_showcase.py` pattern)
- `imint/exporters/export.py` (modify — add `save_chlorophyll_png`, `save_tss_png`,
  `save_cdom_png`, `save_spread_png` if not derivable from existing exporters)
- `docs/js/tab-data.js` (modify — add `water-quality` tab definition)
- `docs/index.html` (modify — add "Vattenkvalitet" tab button + content stub)
- `docs/showcase/water_quality/<year>/*.png` (generated, gitignore'd or committed depending on
  repo policy — check existing showcase output handling)
- `outputs/showcase/water_quality/<year>/*.png` (generated, mirror)

**Tests to add:**
- Unit: `classical_indices.py` formulas with hand-computed values
- Unit: `water_mask.py` with synthetic SCL + bands
- Unit: `mdn_inference.py` with mocked weight download (HTTP 404 → graceful skip)
- Unit: `c2rcc_wrapper.py` with mocked ACOLITE absent → graceful skip
- Integration: full `analyze()` call on a fixture tile clipped to Stigfjorden polygon, assert all
  13 outputs exist + dimensions correct + summary.json schema
- Verification (manual, not in CI): run on real 2026-04-08 Bohuslän S2 tile, render Chl-a colormap
  PNGs, side-by-side compare to inspiring true-color image

**Rollback strategy:**
- Disable via `enabled: false` in `config/analyzers.yaml` — analyzer is skipped at registry-load
  time.
- Hard rollback: revert the registration commit in `imint/engine.py` and the config addition. The
  analyzer package can stay on disk; it's only invoked via the registry.
- MDN weights cache is in `~/.cache/imint/mdn/` — safe to delete, will re-download.

## Open questions

1. **Exact MGRS tile(s) covering Stigfjorden/Mollösund.** Likely `32VPL` or `33VVE` — needs to be
   verified against the AOI polygon centroid before the verification run. The executing session
   should resolve this with a one-shot `pyproj` lookup on the polygon.
2. **ACOLITE licence compatibility.** ACOLITE is GPL-3. Verify ImintEngine's licence allows
   linking. If not, the C2RCC backend may need to live in a separate optional module that the
   analyzer dynamically imports.
3. **MDN weight URL stability.** The `STREAM-RS/Mixture-Density-Networks` repo may have moved or
   re-released weights since 2026-04. Confirm the canonical download URL and pin a specific release
   tag.
4. **ACOLITE C2RCC API.** Whether ACOLITE's Python API exposes a clean `compute_c2rcc(bands, …)`
   entry point or whether it requires writing a temporary L2A scene to disk first. If the latter,
   the wrapper needs a tempdir lifecycle.
5. **Spread-map normalization.** When MDN returns mg/m³ and NDCI is unitless, computing a "spread"
   raster requires normalizing to a common scale. Options: (a) normalize each method to its own
   p50–p95 range and compute std of normalized values, (b) compute spread only across the two
   physical-unit methods (MDN + C2RCC) and ignore the indices. v1 should default to (b) and document
   it.
6. **Trophic-state classification.** Spec says `bloom_threshold_chl: 5.0` is for `summary.json`
   classification only. Should `summary.json` include Carlson TSI bins (oligo/meso/eu/hyper) or just
   binary above/below threshold? Defer decision to executing session; binary is enough for v1.
7. **AOI polygon construction.** ✅ Resolved 2026-04-27: hand-drawn v1-draft polygon committed
   at `imint/analyzers/water_quality/aoi/stigfjorden_skagerrak.geojson`. Bounding box sized to
   1.5:1 landscape aspect ratio in projected meters at 58°N to match the dashboard showcase
   frame proportions (precedent: `marine_commercial` tab). Refinement, if needed in v2, happens
   programmatically via Shapely + OSM coastline — no GIS desktop tool dependency.

---

**Reference paths cited in this spec:**
- Existing analyzer pattern: [imint/analyzers/shoreline.py](imint/analyzers/shoreline.py)
- Registry: [imint/engine.py](imint/engine.py)
- BaseAnalyzer: [imint/analyzers/base.py](imint/analyzers/base.py)
- Job dataclass: [imint/job.py](imint/job.py)
- Config: [config/analyzers.yaml](config/analyzers.yaml)
- Repo conventions: [CLAUDE.md](CLAUDE.md)
