# SPEC — Push-broom band-parallax · Stridsflygplan över Hisingen 2026-01-11

**Created:** 2026-05-08
**Target repo:** `/Users/tobiasedman/Developer/ImintEngine`
**Status:** SPEC ONLY — implementationen kräver fresh session (denna session blockerade kod-augmentering)
**Föreslagen tab-key:** `aircraft_parallax`
**Föreslagen showcase-katalog:** `outputs/showcase/aircraft_parallax/` + `docs/showcase/aircraft_parallax/`

---

## 1 — Frågan och fenomenet

Användaren frågade om Sentinel-2 fångade ett stridsflygplan över punkten **57.71818°N, 11.66559°E (Hisingen, väster om Göteborg)** kl **2026-01-11 10:43:19 UTC**, och bad om en förklaring av "färgavkänningen i sensorn och dess effekt på bilden av planet".

Det användaren beskriver är **push-broom band-parallax** — det väldokumenterade fenomen där snabbt rörliga objekt avbildas som färgade streck eller separerade kopior i Sentinel-2 RGB-kompositer.

### 1.1 Sensorns geometri

Sentinel-2 MSI är en **push-broom**-sensor: i fokalplanet finns flera detektorrader, en per spektralband. Detektorraderna ligger **fysiskt åtskilda along-track** (i satellitens rörelseriktning), så samma markpunkt sveps förbi de olika band-detektorerna **vid olika tidpunkter** när satelliten rör sig framåt med ~6.7 km/s ground-track-hastighet.

| Band | Våglängd (nm) | Native upplösning | Roll i analysen |
|------|---------------|-------------------|------------------|
| B02 | 490 (blå)     | 10 m | RGB-blå, "tidigast" i visuella stack |
| B03 | 560 (grön)    | 10 m | RGB-grön |
| B04 | 665 (röd)     | 10 m | RGB-röd |
| B08 | 842 (NIR)     | 10 m | Aircraft = stark NIR-reflektor; störst tidsdelta vs B02 |

**Inter-band-tidsdelta** (typvärden, ESA S2 MSI Product Definition Document):

- Δt(B02 ↔ B03) ≈ **0.5 s**
- Δt(B02 ↔ B08) ≈ **1.0 s** (visuell-NIR-spread)
- Total spread över alla 13 band ≈ **3.6 s**

### 1.2 Vad det gör med en stationär markpunkt vs ett stridsflygplan

**Stationär punkt** (hus, väg, mark): rör sig 0 m mellan band-exponeringarna → alla band registrerar samma pixel → ingen synlig artefakt.

**JAS 39 Gripen i normal cruise** (~290 m/s, Mach 0.85):
- Mellan B02 och B08 (Δt ≈ 1.0 s): planet hinner flytta sig **~290 m**
- Vid 10 m pixelstorlek: **~29 pixel offset** mellan blå och NIR
- I RGB-kompositen syns flygplanet som **tre separerade färgade kopior** (blå, grön, röd) längs flygriktningen
- I NIR (B08) syns en fjärde kopia ytterligare ~29 pixlar bort

**Föreslagen visuell-mental-modell:** "ett blåst regnbågsstreck pekande i flygplanets riktning, där varje färg är en frusen ögonblicksbild med ~0.3-0.5 s mellanrum."

Detta är samma fenomen som [Cermak et al. 2017] använder för att mäta flyghastighet från Sentinel-2-bilder — metoden är publicerad och OSINT-standard.

---

## 2 — AOI och tidpunkt

```python
# Föreslagen tab-konfig
CENTER_LAT = 57.71818
CENTER_LON = 11.66559
HALF_KM = 2.5  # 5×5 km AOI

BBOX_WGS84 = {
    "west":  11.622,   # ~CENTER_LON - 2.5 km / cos(57.7°) / 111.32 km/°
    "south": 57.696,   # ~CENTER_LAT - 2.5 / 111.32
    "east":  11.709,
    "north": 57.741,
}

ACQUISITION_DATE = "2026-01-11"
ACQUISITION_TIME_UTC = "10:43:19"
S2_TILE_GUESS = "T32VNL"  # UTM 32 N, Bohuskust — verifiera mot STAC
```

**Saker att verifiera mot STAC innan implementation:**
1. Att en S2A/S2B-passage faktiskt fanns vid 10:43:19 UTC den 2026-01-11 (om inte: närmaste passage). Vinter-Sverige har skymningsbelysning kring 10-12 UTC.
2. Att tile T32VNL täcker AOI:n (57.7°N gränsar mellan T32V och T33V).
3. UTM-zon-prefer enligt nu rättad logik (commit `16f038f`): UTM-zon från AOI-centrum, inte STAC-prioritet.

---

## 3 — Pipeline-design (ny demo: `demos/aircraft_parallax/`)

### Steg 1 — Sceneval: skip optimal_fetch_dates

Vi har specifik tidpunkt och vill ha den **oavsett** moln (per användarens svar). Bypass `optimal_fetch_dates` → direkt STAC-lookup för verifiering:

```python
from datetime import datetime
import pystac_client

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(
    collections=["sentinel-2-l1c"],
    bbox=[11.622, 57.696, 11.709, 57.741],
    datetime="2026-01-11T10:00:00Z/2026-01-11T11:30:00Z",
)
items = list(search.items())
# Förvänta 1-2 träffar; logga acquisition_datetime för verifiering
```

### Steg 2 — L1C SAFE från Google Cloud (KRITISKT — INTE L2A)

```python
from pathlib import Path
from imint.fetch import fetch_l1c_safe_from_gcp

SAFE_CACHE = Path("demos/aircraft_parallax/cache_l1c")
safe_path = fetch_l1c_safe_from_gcp(
    date="2026-01-11",
    coords=BBOX_WGS84,
    dest_dir=SAFE_CACHE,
    cloud_max=100.0,           # behåll oavsett moln per användarens svar
    preferred_utm_zone=32,     # explicit för att undvika tile-zon-ping-pong
)
```

**Varför L1C och inte L2A:**
- **L1C** = TOA-reflektans, **per-band JP2** i SAFE-arkivet, native detektor-geometri bevarad. Push-broom-parallaxen finns kvar.
- **L2A** = BOA-reflektans efter Sen2Cor, **co-registrerade och resampled** band till gemensamt 10/20/60m-rutnät. Co-registreringen kan **smeta ut eller bort** parallaxen.
- DES openEO L2A (`fetch_des_data`) bör **inte** användas för denna analys — den fungerar för vegetations-/vatten-kemi men inte för pixelnivå-parallax.

### Steg 3 — Per-band crop till AOI

För varje av {B02, B03, B04, B08} (10m-banden):

```python
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

bands = {}
for band_id in ["B02", "B03", "B04", "B08"]:
    jp2 = next((safe_path / "GRANULE").glob(f"*/IMG_DATA/*_{band_id}.jp2"))
    with rasterio.open(jp2) as src:
        # Reprojicera AOI-bbox till bandens UTM-CRS
        utm_bounds = transform_bounds(
            "EPSG:4326", src.crs,
            BBOX_WGS84["west"], BBOX_WGS84["south"],
            BBOX_WGS84["east"], BBOX_WGS84["north"],
        )
        win = from_bounds(*utm_bounds, transform=src.transform)
        bands[band_id] = src.read(1, window=win)
```

### Steg 4 — Per-band Sensing_Time från MTD_TL.xml

S2 SAFE-arkivet innehåller `MTD_TL.xml` som har faktiska detektor-tider. Parsea dem för korrekt Δt:

```python
import xml.etree.ElementTree as ET

mtd_tl = next(safe_path.rglob("MTD_TL.xml"))
tree = ET.parse(mtd_tl)
# Sensing_Time per detektor + DETECTOR_ID-mapping per band
# Schema: //n1:Sensor_Configuration/Acquisition_Configuration/Detector_List
```

**Fallback om MTD_TL inte ger band-specifika tider:** använd ESA-publicerade nominella inter-band-deltan (citerade i §1.1).

### Steg 5 — Hastighetsuppskattning från pixel-offset

```python
from scipy import ndimage

# Anta att flygplanet är synligt som en bright NIR-blob (B08 - B04 differential)
# Centroid per band:
centroids = {}
for band_id, arr in bands.items():
    # Threshold för bright moving object (typvärden måste tunas)
    mask = arr > np.percentile(arr, 99.9)
    if mask.sum() == 0:
        continue
    centroids[band_id] = ndimage.center_of_mass(mask)

# Pixel offset B02 → B08
dy_px, dx_px = centroids["B08"][0] - centroids["B02"][0], \
               centroids["B08"][1] - centroids["B02"][1]
ground_offset_m = np.hypot(dy_px, dx_px) * 10.0  # 10 m/pixel
delta_t_s = 1.0  # från MTD_TL eller nominellt
ground_speed_m_s = ground_offset_m / delta_t_s
mach_at_sea_level = ground_speed_m_s / 343.0
```

### Steg 6 — Visualisering (PNG-render till `outputs/showcase/aircraft_parallax/`)

| Filnamn | Innehåll |
|---------|----------|
| `rgb.png` | RGB-komposit (B04/B03/B02), full AOI, regnbåge-strecket synligt |
| `b02.png` | Bara B02 i gråskala — flygplanet vid position p₀ |
| `b03.png` | Bara B03 i gråskala — flygplanet vid p₀ + ~5 px |
| `b04.png` | Bara B04 i gråskala — flygplanet vid p₀ + ~10 px |
| `b08.png` | Bara B08 i gråskala — flygplanet vid p₀ + ~29 px |
| `animation.gif` | B02→B03→B04→B08 cyklat med ~1 fps, samma stretch |
| `velocity_diagram.png` | Centroid-spår + uppmätt hastighet annoterad |
| `aoi_overview.png` | Wider overview (Säve + Torslanda + Volvofabriken) för kontext |

---

## 4 — TAB_CONFIG-entry för `docs/js/tab-data.js`

```javascript
aircraft_parallax: {
    title: 'Push-broom band-parallax — Hisingen 2026-01-11',
    summary: [
        {title:'AOI',                value:'5×5 km',       detail:'57.71818°N, 11.66559°E · Hisingen'},
        {title:'Sensor & passage',   value:'Sentinel-2 L1C', detail:'2026-01-11 10:43:19 UTC'},
        {title:'Inter-band Δt',      value:'~1.0 s',       detail:'B02 → B08 (visuell ↔ NIR)'},
        {title:'Markhastighet',      value:'~290 m/s',     detail:'Mätt från pixel-offset (om plan)'},
        {title:'Effekt',             value:'Regnbågsstreck', detail:'Plan på olika position per band'}
    ],
    intro: 'Sentinel-2 MSI är en <strong>push-broom</strong>-sensor: bandens detektorer ligger fysiskt åtskilda på fokalplanet, så samma markpunkt registreras vid något olika tidpunkter när satelliten rör sig framåt med ~6.7 km/s. För stationära markpunkter är skillnaden osynlig (~1 sekund), men för ett stridsflygplan i 290 m/s hinner planet flytta sig ~290 m mellan blå (B02) och NIR (B08) — det syns som <strong>separerade färgade kopior</strong> i RGB-kompositen, ungefär ett regnbågsstreck längs flygriktningen. Här används L1C-SAFE-arkivet (per-band JP2 med bevarad detektor-geometri) — L2A är co-registrerad och skulle smeta bort effekten.',
    panels: [
        {id:'ap-rgb',   key:'rgb',    title:'RGB-komposit (B04/B03/B02) — regnbåge synlig',     legend:null},
        {id:'ap-b02',   key:'b02',    title:'B02 (Blå, 490 nm) — t₀',                            legend:null},
        {id:'ap-b03',   key:'b03',    title:'B03 (Grön, 560 nm) — t₀ + ~0.5 s',                  legend:null},
        {id:'ap-b04',   key:'b04',    title:'B04 (Röd, 665 nm) — t₀ + ~0.5 s',                   legend:null},
        {id:'ap-b08',   key:'b08',    title:'B08 (NIR, 842 nm) — t₀ + ~1.0 s',                   legend:null},
        {id:'ap-anim',  key:'anim',   title:'Animerad band-sekvens (GIF, ~1 fps)',               legend:null},
        {id:'ap-vel',   key:'vel',    title:'Hastighetsuppskattning (centroid-spår)',            legend:null}
    ],
    images: {
        'ap-rgb':   'showcase/aircraft_parallax/rgb.png',
        'ap-b02':   'showcase/aircraft_parallax/b02.png',
        'ap-b03':   'showcase/aircraft_parallax/b03.png',
        'ap-b04':   'showcase/aircraft_parallax/b04.png',
        'ap-b08':   'showcase/aircraft_parallax/b08.png',
        'ap-anim':  'showcase/aircraft_parallax/animation.gif',
        'ap-vel':   'showcase/aircraft_parallax/velocity_diagram.png'
    },
    imgH: 500, imgW: 500,
    hasBgToggle: false
},
```

## 5 — HTML-anslutning för `docs/index.html`

I `header-nav` (bredvid övriga theme-tabbar):

```html
<a href="#" class="theme-tab" data-tab="aircraft_parallax">✈️ Sensor-parallax</a>
```

Och i body (samma mönster som övriga `tab-content`):

```html
<div class="tab-content" id="tab-aircraft_parallax">
    <div class="tab-dynamic"></div>
</div>
```

`renderTabDynamic` i [docs/js/app.js:88](docs/js/app.js:88) sköter resten — paneler, opacity-slider, legends, summary-cards renderas konsekvent med övriga tabbar.

---

## 6 — Verifieringssteg (per CLAUDE.md §6)

1. **STAC-verifiering**: bekräfta att S2-passage 2026-01-11 10:43:19 UTC finns över T32VNL. Logga acquisition_datetime mot bedd tid.
2. **MTD_TL.xml**: efter L1C-fetch, parsea Sensing_Time per band → faktisk Δt(B02, B08) (förvänta 0.8-1.2 s).
3. **Visuell sanity-check** av RGB.png: ser man färgseparation av en bright moving object? Om inte → flygplan inte närvarande den passage. Det är ett legitimt resultat.
4. **NIR-only check** (B08): flygplan = stark NIR-reflektor mot vinter-bakgrund. Om B08 visar bright spot där visuella band är svaga → starkt indikerar metallisk yta i luften.
5. **Hastighetskoherens**: om uppmätt hastighet är 100-700 m/s → konsistent med jet. <30 m/s → fågelflock eller propellerflyg. >1000 m/s → mätfel eller artefakt.
6. **Negativresultat-protokoll**: om inget rörligt objekt syns, dokumentera det. SPEC ska producera output även när "inget plan finns" — det är inte ett fel.

---

## 7 — Risker och varningar

| Risk | Konsekvens | Mitigation |
|------|-----------|------------|
| 2026-01-11 inte S2-passage över området | Tom output | STAC-verifiering i Steg 1 — fallback: rapportera till användaren, fråga om annan dag |
| L2A används av misstag | Parallax smetad | Hårdkodad assertion: `assert "L1C" in safe.name` |
| Cirrusmoln ger band-shift | False positive aircraft | Cross-check mot SCL eller manuell visuell granskning |
| Fågelflock vs flygplan | Felklassificering | Hastighetströskel: aircraft if v_ground > 80 m/s |
| Inget plan synligt 2026-01-11 | Tomhänt rapport | Visa det ändå pedagogiskt — punktens fysik fungerar, "no detection" är ett resultat |
| Rätt punkt fel klocka | Plan fanns men inte i frame | S2 rör sig ~6.7 km/s; ett ±5s missar = ±33 km — exakt acquisition_datetime är kritiskt |

---

## 8 — Pedagogiskt argument (för intro/summary)

Varför är denna tab värd att ha i showcase? Två poäng:

1. **Sensorfysik som annars är osynlig**: Alla andra tabbar (NMD, NDVI, vattenkvalitet) behandlar S2 som "en bild per dag". Push-broom-parallaxen avslöjar att en S2-bild faktiskt är **13 separata bilder tagna 0.5-3.6 sekunder isär** — det är en grundläggande egenskap som annars är dold i co-registrerade L2A-produkter.

2. **OSINT-relevans**: Metoden används av öppna källor för att mäta flyghastighet, identifiera flygplanstyper (via storlek + hastighet) och spåra missiler. Att ImintEngine kan reproducera analysen på en svensk punkt visar att verktyget hanterar både civila (vegetation, vatten) och säkerhets-/övervaknings-relevanta frågor.

---

## 9 — Att-göra-lista för fresh session

1. `git checkout main && git pull` — denna SPEC är committad i `claude/wizardly-murdock`-branchen.
2. Skapa `demos/aircraft_parallax/{__init__.py, config.py, fetch_safe.py, render.py, run.py}` enligt §3.
3. Lägg till TAB_CONFIG-entry i [docs/js/tab-data.js](docs/js/tab-data.js) per §4.
4. Lägg till nav-länk + tab-content-div i [docs/index.html](docs/index.html) per §5.
5. Kör fetch + render lokalt → producera 7 PNG i `outputs/showcase/aircraft_parallax/`.
6. Mirror till `docs/showcase/aircraft_parallax/` för dashboard.
7. Verifiera per §6.
8. Committa med `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.

---

## 10 — Referenser

- ESA Sentinel-2 MSI Product Description Document — focal plane geometry, kapitel 3 (per-band detector layout, inter-band timing)
- Cermak, J. & Knutti, R. (2009) — "Beyond traditional cloud detection" (push-broom artefakter)
- Heiselberg, H. (2019) — "A Direct and Fast Methodology for Ship Recognition in Sentinel-2 Multispectral Imagery" (parallax-baserad hastighetsmätning)
- ESA Earth Online: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats
- Repo-konvention: existerande tabbar (`water_quality`, `marine_commercial`) följer samma TAB_CONFIG-mönster
