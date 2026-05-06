# SPEC — Lilla Karlsö C2RCC tidsserie · Sillgrissle-säsong 2025

**Created:** 2026-05-06
**Target repo:** `/Users/tobiasedman/Developer/ImintEngine`
**Status:** draft, klar för fresh-session execution

## Mål

Köra ESA C2RCC (via SNAP Docker, samma pipeline som Vattenkvalitet-tabben)
på alla molnfria Sentinel-2-scener över en utvidgad AOI runt Lilla Karlsö
under sillgrisslornas häckningssäsong 2025. Resultat publiceras som ny
showcase med per-scen kartor + tidsserie-chart över medel-klorofyll.

## Bakgrund

Lilla Karlsö (Gotland, fågelreservat) har 2 000–3 000 häckande sillgrissle-par.
Häckningssäsong:
- Början maj — ägg läggs på klipphyllor (ett ägg per par)
- Mitten juni — ägg kläcks
- Slutet juni / början juli — ungar lämnar kolonin

Foderbasen är pelagisk småfisk (skarpsill, sill) som följer klorofyll-
dynamiken i Östersjön. C2RCC ger kvantitativa retrievals (chl-a, TSM, CDOM)
för optiskt komplexa Case-2-vatten — relevant för Östersjöns kustnära.

## AOI

Nuvarande Lilla Karlsö-demo (`demos/lilla_karlso/fetch_and_render.py`):
- 7 × 5 km, `west=18.02, south=57.28, east=18.10, north=57.34`

**Utvidgad AOI för C2RCC-jobbet:**

```python
BBOX_WGS84 = {
    "west":  17.91,
    "south": 57.21,
    "east":  18.21,
    "north": 57.41,
}
# ~22 × 22 km i WGS84, ~16 × 22 km efter projektion till EPSG:3006
```

Större AOI behövs eftersom:
1. Lilla Karlsö är ~2.5 × 1 km — i 7×5 km AOI dominerar kalksten-platån
2. Foderhabitat söks 5–15 km från kolonin (sillgrisslan dyker till 100 m djup)
3. C2RCC-stats kräver tillräckligt vatten-pixlar

Single-pass tile: T33VWE (UTM 33N) täcker hela AOI:n.

## Period

```python
PERIOD_START = "2025-04-15"
PERIOD_END   = "2025-07-31"
```

3.5 månader × Sentinel-2 5-dagars revisit = ~21 nominella scener.
Efter `era5_then_scl(max_aoi_cloud=0.05)` förväntas 8–15 rena.

## Pipeline

### Steg 1 — Sceneval (DES creds krävs)

```python
from imint.training.optimal_fetch import optimal_fetch_dates
plan = optimal_fetch_dates(
    bbox_wgs84=BBOX_WGS84,
    date_start=PERIOD_START, date_end=PERIOD_END,
    mode="era5_then_scl", max_aoi_cloud=0.05,
)
# plan.dates → ISO-datum för L1C SAFE-fetch
```

### Steg 2 — L1C SAFE från Google Cloud

```python
from imint.fetch import fetch_l1c_safe_from_gcp
SAFE_CACHE = Path("demos/lilla_karlso_birds/cache_l1c")
for date in plan.dates:
    safe = fetch_l1c_safe_from_gcp(
        date=date, coords=BBOX_WGS84,
        dest_dir=SAFE_CACHE,
        cloud_max=20.0,  # backup-filter; era5_then_scl är primärt
    )
```

~700 MB–1 GB per scen × ~12 = 10 GB lokal disk.

### Steg 3 — C2RCC via SNAP Docker

**KRITISKT:** Det finns en lokal Docker-image som heter
`imint-snap-c2rcc:latest` (1.39 GB) — repot-grep hittar **ingen**
referens till hur den invocades. Antagligen kördes den manuellt med
ad-hoc `docker run`-kommandon i en tidigare session.

**Fresh-session måste rekonstruera kör-protokollet.** Steg-för-steg:

1. Inspektera imagen:
   ```bash
   docker inspect imint-snap-c2rcc:latest | jq '.[0].Config'
   docker run --rm imint-snap-c2rcc:latest /usr/local/snap/bin/gpt -h | head -20
   ```

2. Sentinel-2 → C2RCC SNAP graph (rekonstruerat från commit 52d19ae):
   ```
   Read(SAFE)
     → Resample(referenceBand=B2)
     → Subset(geoRegion=AOI WGS84 polygon, copyMetadata=true)
     → c2rcc.msi(netSet=C2X-Nets, outputRtosa=false)
     → Write(format=BEAM-DIMAP)
   ```

3. Skriv graph som `docker/c2rcc-snap/lilla_karlso_graph.xml`
   (se `outputs/c2rcc_runs/c2rcc_2026_04_08.dim` för befintlig referens-
   output struktur — `iop_apig`, `iop_agelb`, `iop_bpart`, `iop_bwit`,
   `kd489`, `kdmin`, `rhow_B*`, `c2rcc_flags`).

4. Run-skript:
   ```bash
   docker run --rm \
     -v "$PWD/demos/lilla_karlso_birds/cache_l1c:/in:ro" \
     -v "$PWD/outputs/c2rcc_runs_lilla_karlso:/out" \
     -v "$PWD/docker/c2rcc-snap:/graph:ro" \
     imint-snap-c2rcc:latest \
     /usr/local/snap/bin/gpt /graph/lilla_karlso_graph.xml \
       -PinputSafe=/in/<DATE>.SAFE \
       -PoutputDim=/out/<DATE>.dim \
       -e
   ```

**Per-scen runtime:**
- amd64 native: 3–10 min
- Mac (x86-emulering på Apple Silicon): 9–30 min
- × 12 scener = 2–6 h på Mac, 1–2 h på Linux x86

### Steg 4 — Klorofyll-beräkning

C2RCC ger inte direkt chl-a — bara IOPs. Brockmanns standardrelation
(samma som befintlig analyzer):

```python
# imint/analyzers/water_quality/analyzer.py:save_water_quality_png
chl  = (iop_apig ** 1.04) * 21.0   # mg/m³
tsm  = 1.72 * iop_bpart + 3.1 * iop_bwit   # g/m³
cdom = iop_agelb   # m⁻¹

# Filtrera bort NN-floor-rester
chl[iop_apig < 0.001] = np.nan
```

### Steg 5 — Render PNG per datum

För varje datum, generera 4 PNG (1000×1000 px, transparent overlay):

| Fil | Innehåll | Cmap | vmin/vmax |
|---|---|---|---|
| `rgb.png` | B04/B03/B02, p2/p98 stretch | (RGB) | — |
| `chl.png` | chl-a (mg/m³) | DES (1a4338→fde725) | log10(0.5+1)..log10(25+1) |
| `tsm.png` | TSM (g/m³) | DES | 0..20 |
| `cdom.png` | CDOM (m⁻¹) | DES | 0..2 |

DES-palett (per CLAUDE.md regel):
```python
COT_CMAP = LinearSegmentedColormap.from_list("des", [
    (0.00, "#1a4338"), (0.25, "#cff8e4"),
    (0.50, "#fdd5c2"), (0.80, "#ff826c"),
    (1.00, "#ffffff"),
])
```

Output: `docs/showcase/lilla_karlso_birds/<date>/{rgb,chl,tsm,cdom}.png`

### Steg 6 — Tidsserie-aggregering

```python
records = []
for date in plan.dates:
    chl, water = read_dimap(f"outputs/c2rcc_runs_lilla_karlso/{date}.dim")
    valid = chl[water & np.isfinite(chl) & (chl > 0)]
    records.append({
        "date": date,
        "chl_p50": float(np.percentile(valid, 50)),
        "chl_p90": float(np.percentile(valid, 90)),
        "chl_mean": float(valid.mean()),
        "tsm_p50": ...,
        "cdom_p50": ...,
        "n_water_pixels": int(water.sum()),
        "n_valid_chl": int(valid.size),
    })
import json
Path("docs/showcase/lilla_karlso_birds/timeseries.json").write_text(
    json.dumps({"records": records, "aoi": BBOX_WGS84,
                "period": [PERIOD_START, PERIOD_END]}, indent=2))
```

### Steg 7 — Showcase

**Tab-placering — beslutspunkt:**
- A) Sub-tab under befintlig **Marin** (parallellt med Fritid/Sjöfart)
- B) Egen ny **🐦 Sillgrissla** parent
- C) Sub-tab under **Vattenkvalitet** (passar tematiskt)

Default-rekommendation: **C** (Vattenkvalitet → Bohuslän + Lilla Karlsö
som sub-tabs) — matchar tematiken och Vattenkvalitet är just nu single-tab.

**Struktur** (`renderTabDynamic`-mall):
- Summary cards: Antal scener, säsongs-mean chl, peak-bloom-datum, AOI-area
- Panels: 4 lager × N datum — använder `era5-set-tab`-mönstret för datumval
- Time-series chart längst ner: chl_p50 + chl_p90 över datum (Chart.js line)

`docs/js/tab-data.js`:
```javascript
lilla_karlso_birds: {
    title: 'Lilla Karlsö — Sillgrissle-häckning 2025 · klorofyll',
    summary: [...],
    intro: '...',
    panels: [
        {id:'lk-rgb-<date>', key:'rgb', ...},
        {id:'lk-chl-<date>', key:'chl', legend:'chl', ...},
        // alternativt: en panel per produkt med datum-toggle
    ],
    images: { ... },
    imgH: 1000, imgW: 1000,
    hasBgToggle: true,
}
```

## Filer att skapa

| Fil | Vad |
|---|---|
| `demos/lilla_karlso_birds/__init__.py` | (tom) |
| `demos/lilla_karlso_birds/fetch_safes.py` | optimal_fetch + L1C-loop |
| `demos/lilla_karlso_birds/run_c2rcc.py` | docker-loop över SAFE → DIMAP |
| `demos/lilla_karlso_birds/render.py` | DIMAP → 4 PNG/datum + manifest |
| `demos/lilla_karlso_birds/aggregate.py` | DIMAP → timeseries.json |
| `docker/c2rcc-snap/lilla_karlso_graph.xml` | SNAP graph (om inte återfinns) |
| `docker/c2rcc-snap/run.sh` | Wrapper-skript för docker run |
| `docs/js/tab-data.js` | Lägg till `lilla_karlso_birds` config |
| `docs/index.html` | Ny sub-tab + ev. Chart.js-canvas |

## Var körs det

**Rekommendation: Lokal Mac.**
- Docker daemon redan igång, image (`imint-snap-c2rcc:latest`) finns
- 10 GB SAFE-cache OK på laptop SSD
- 2–6 h totalt — kör nattjobb eller backgrund
- Direkt write till `docs/showcase/` utan rsync-steg
- Ingen ny PVC-konflikt

K8s-alternativ: kräver push av imagen till intern registry +
nytt job-yaml. För v1 — onödig komplexitet.

## Beroenden

- `imint.training.optimal_fetch.optimal_fetch_dates` ✅ finns
- `imint.fetch.fetch_l1c_safe_from_gcp` ✅ finns (`imint/fetch.py:1519`)
- `imint.analyzers.water_quality.analyzer` ✅ finns (chl/tsm/cdom-formler)
- DES openEO creds (.env DES_USER + DES_PASSWORD) ✅
- Docker daemon ✅
- `imint-snap-c2rcc:latest` ✅ lokalt (1.39 GB)
- SNAP graph-XML ❌ **MÅSTE REKONSTRUERAS** (inte i repot, inte i ~/Developer)

## Kritiska osäkerheter

1. **SNAP graph-XML saknas.** Den måste rekonstrueras från:
   - commit 52d19ae beskrivning ("Read → Resample(B2) → Subset → c2rcc.msi(netSet=C2X-Nets) → Write(BEAM-DIMAP)")
   - befintlig output-struktur i `outputs/c2rcc_runs/c2rcc_2026_04_08.dim`
   - SNAP CLI `gpt -h c2rcc.msi` för parameter-defaults

2. **GCP L1C SAFE-fetch behöver verifieras** för 2025-datum innan
   batch-kör — DES L1C-bug (docs/des_l1c_bug_report.md) gjorde det
   nödvändigt att gå GCP-vägen.

3. **AOI sträcker sig över UTM-zon-gräns?** 18°E är gränsen
   33N/34N. Lilla Karlsö (18.06°E) ligger i 33N. AOI 17.91–18.21°E
   ligger helt i 33N → enkelt single-tile. **Verifierat OK.**

## Verifiering

| Steg | Mätbar verifiering |
|---|---|
| Sceneval | `len(plan.dates) >= 5` |
| L1C-fetch | Varje SAFE har `MTD_MSIL1C.xml` + ≥10 IMG_DATA filer |
| C2RCC-output | `iop_apig.img` finns; max ≥ 0.001 (ej all-NN-floor) |
| Tidsserie | chl_p50 sanity: vårblom maj > juni-juli |
| Showcase | Ny tab/sub-tab renderar i preview, inga 404 |
| DES-palett | Alla matplotlib-cmaps använder DES-färger |
| Docker | `docker images | grep imint-snap-c2rcc` ger 1.39 GB image |

## Beslut som behöver tas i fresh session

| # | Fråga | Default om obesvarad |
|---|---|---|
| 1 | AOI-storlek? | 22×22 km enligt SPEC |
| 2 | Period? | 2025-04-15..07-31 |
| 3 | Tab-placering? | Sub-tab under Vattenkvalitet (alt: Marin) |
| 4 | Var kör? | Lokal Mac |
| 5 | Vilka produkter rendera? | Alla 4 (RGB+chl+TSM+CDOM) |
| 6 | Chart-data? | chl_p50, chl_p90 per datum |

## Briefing-prompt för fresh session

```
Läs /Users/tobiasedman/Developer/ImintEngine/SPEC_lilla_karlso_birds.md
och börja exekvera Lilla Karlsö C2RCC-pipelinen för sillgrisslesäsongen 2025.

Bekräfta först de 6 besluten i "Beslut som behöver tas". Default-svar
om jag inte säger annat:
- 22×22 km AOI runt Lilla Karlsö
- Period 2025-04-15..2025-07-31
- Sub-tab under Vattenkvalitet med namn "🐦 Lilla Karlsö — Sillgrissle 2025"
- Köra lokalt på Mac (image imint-snap-c2rcc:latest finns redan)
- Alla 4 produkter (RGB+chl+TSM+CDOM)
- Time-series chart med chl_p50 + chl_p90

Använd alltid:
- optimal_fetch_dates(mode="era5_then_scl") för Sentinel-2-urval
- DES varumärkespalett (#1a4338 #cff8e4 #fdd5c2 #ff826c)
- renderTabDynamic-mallen för showcase
- Co-Authored-By: Claude Opus 4.6 trailer i alla commits

Status idag (2026-05-06):
- Pirinen-stack komplett (alla 8 lager renderade) i Marktäcke-tab
- Atmosfär DES-paletted i 0c6fa80
- Senaste commit: da415d8 (Marktäcke-omstrukturering)

Första action: läs SPEC, kör docker images för att bekräfta att
imint-snap-c2rcc:latest finns, sedan optimal_fetch på utvidgad AOI
för att få en datumlista. Rapportera datumlistan till mig innan
du börjar L1C-fetch (det är 10 GB nedladdning).
```

## Källhänvisningar

- Sillgrissla häckningscykel: WWF Sverige, Naturskyddsföreningen Gotland
- C2RCC NN-floor + C2X-Nets: Brockmann et al. 2016, ESA LPS
- ESA SNAP: https://step.esa.int/main/download/snap-download/
- Sentinel-2 L1C på GCP: gs://gcp-public-data-sentinel-2 (anonymous, free)
- Befintlig water_quality-analyzer: commit 52d19ae (Bohuslän, 2026-04-08)
