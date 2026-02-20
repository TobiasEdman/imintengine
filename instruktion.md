# IMINT Engine — Instruktioner för Claude Code

## Vad är det här projektet?

IMINT Engine är en modulär analysmotor för molnfria Sentinel-2-satellitbilder. Den är byggd som en fristående komponent ovanpå SDL 3.0-infrastrukturen (Space Data Lab 3.0) och Digital Earth Sweden (DES).

Huvudsyftet är att ta emot en molnfri satellitbild och köra ett valfritt antal analyser på den — förändringsdetektering, spektralanalys, objektdetektering — och exportera resultaten som PNG, GeoTIFF, GeoJSON och JSON.

Projektet är fristående från ColonyOS. ColonyOS är en möjlig exekveringsplattform, inte ett hårt beroende.

---

## Arkitektur — det viktigaste att förstå

Det finns en **hård gräns** mellan engine och executor. Håll den alltid.

```
executors/          ← Hur jobbet schemaläggs och körs (ColonyOS, lokalt, cron, etc.)
  base.py           ← Abstrakt BaseExecutor — alla executors implementerar detta
  local.py          ← Kör lokalt från terminal eller notebook
  colonyos.py       ← Kör inuti ett ColonyOS-containerjobb

imint/              ← Kärnan — vet ingenting om executors
  job.py            ← IMINTJob (input) och IMINTResult (output) — datamodellerna
  engine.py         ← run_job() — enda ingångspunkten för alla executors
  analyzers/        ← En fil per analyzer
    base.py         ← Abstrakt BaseAnalyzer + AnalysisResult
    change_detection.py
    spectral.py
    object_detection.py
  exporters/
    export.py       ← PNG, GeoTIFF, GeoJSON, JSON-hjälpfunktioner

config/
  analyzers.yaml    ← Aktivera/inaktivera analyzers och justera parametrar
```

### Flödet

```
Executor (local / ColonyOS / annat)
    → hämtar satellitdata + kör molndetektering
    → bygger ett IMINTJob med rgb + bands ifyllt
    → anropar run_job(job)
    → engine kör alla aktiverade analyzers
    → exporterar resultat till output_dir
    → returnerar IMINTResult
```

---

## Datamodeller

### IMINTJob
```python
@dataclass
class IMINTJob:
    date: str                        # ISO-datum, t.ex. "2022-06-15"
    coords: dict                     # {"west": ..., "south": ..., "east": ..., "north": ...}
    rgb: np.ndarray | None           # (H, W, 3) float32 i intervallet [0, 1]
    bands: dict[str, np.ndarray]     # {"B02": arr, "B03": arr, "B04": arr, "B08": arr, "B11": arr}
    output_dir: str                  # Katalog för outputfiler
    config_path: str                 # Sökväg till analyzers.yaml
    job_id: str | None               # Valfritt jobb-ID från exekveringsplattformen
```

### AnalysisResult
```python
@dataclass
class AnalysisResult:
    analyzer: str       # Namn på analyzern
    success: bool
    outputs: dict       # Namngivna outputs — arrayer, sökvägar, skalärer
    metadata: dict      # Metadata — tröskelvärden, modellnamn, etc.
    error: str | None
```

---

## Hur man lägger till en ny analyzer

Det är det vanligaste utvecklingssteget. Gör så här:

**1. Skapa filen** `imint/analyzers/min_analyzer.py`:
```python
from .base import BaseAnalyzer, AnalysisResult
import numpy as np

class MinAnalyzer(BaseAnalyzer):
    name = "min_analyzer"

    def analyze(self, rgb, bands=None, date=None, coords=None, output_dir="../outputs") -> AnalysisResult:
        # Din logik här
        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={"resultat": 42.0},
            metadata={"date": date},
        )
```

**2. Registrera den** i `imint/engine.py`:
```python
from .analyzers.min_analyzer import MinAnalyzer

ANALYZER_REGISTRY = {
    ...
    "min_analyzer": MinAnalyzer,
}
```

**3. Lägg till konfigblock** i `config/analyzers.yaml`:
```yaml
min_analyzer:
  enabled: true
  min_parameter: 0.5
```

Inga andra filer behöver ändras.

---

## Hur man lägger till en ny executor

Gör detta om du vill köra engine på en annan plattform (Airflow, cron, REST API, etc.):

**1. Skapa filen** `executors/min_executor.py`:
```python
from executors.base import BaseExecutor
from imint.job import IMINTJob, IMINTResult

class MinExecutor(BaseExecutor):
    def build_job(self, **kwargs) -> IMINTJob:
        # Hämta data, kör molndetektering, bygg IMINTJob
        ...

    def handle_result(self, result: IMINTResult) -> None:
        # Logga, notifiera, spara till extern lagring, etc.
        ...
```

**2. Använd den:**
```python
executor = MinExecutor()
result = executor.execute(date="2022-06-15", coords={...})
```

Engine (`run_job`) vet ingenting om den nya executorn.

---

## Befintliga analyzers

### change_detection
Jämför aktuell bild mot en lagrad baslinjesbild per säsong och geografiskt område.
- Första körningen för ett område sparar bilden som baslinje — returnerar 0 förändringar
- Efterföljande körningar flaggar pixlar som ändrats mer än `threshold`
- Morfologisk rensning tar bort brus
- Sammankopplade regioner identifieras med bounding boxes
- Output: `change_fraction`, `n_regions`, `regions` (lista med bbox + pixelantal), `change_mask`

### spectral
Beräknar spektrala index från Sentinel-2-band:
- **NDVI** — vegetationshälsa: `(B08 - B04) / (B08 + B04)`
- **NDWI** — vattendetektering: `(B03 - B08) / (B03 + B08)`
- **NDBI** — bebyggelse: `(B11 - B08) / (B11 + B08)`
- **EVI** — förstärkt vegetation
- Klassificerar varje pixel: vatten / vegetation / bebyggelse / bar mark
- Faller tillbaka på RGB-approximationer om band saknas

### object_detection
Två lägen:
- `heatmap` (standard) — variansbaserad anomalidetektion per patch, kräver ingen modell
- `model` — YOLO-modell tränad på satellitbilder (t.ex. xView). Kräver `ultralytics` och en `.pt`-fil

---

## Satellitband — referens

| Band | Våglängd | Upplösning | Användning |
|------|----------|------------|------------|
| B02  | Blå      | 10m        | EVI, RGB   |
| B03  | Grön     | 10m        | NDWI, RGB  |
| B04  | Röd      | 10m        | NDVI, RGB  |
| B08  | NIR      | 10m        | NDVI, NDWI, NDBI, EVI |
| B11  | SWIR1    | 20m        | NDBI       |

RGB-composite: B04=R, B03=G, B02=B, normaliserat till [0, 1].

---

## Köra lokalt (utan DES-konto)

```bash
# Installera beroenden
pip install -r requirements.txt

# Kör med syntetisk testdata
python executors/local.py \
  --date 2022-06-15 \
  --west 14.5 --south 56.0 --east 15.5 --north 57.0
```

Output hamnar i `outputs/2022-06-15/`.

## Köra med riktig DES-data

```bash
PYTHONPATH=../ai-pipelines-poc python executors/local.py \
  --date 2022-06-15 \
  --west 14.5 --south 56.0 --east 15.5 --north 57.0
```

---

## Kodkonventioner

- Alla analyzers returnerar alltid `AnalysisResult` — de kastar aldrig undantag. Fel fångas i `BaseAnalyzer.run()`.
- `run_job()` i `engine.py` är enda ingångspunkten — anropa aldrig analyzers direkt från executors.
- Outputfiler namnges alltid `{date}_{beskrivning}.{ext}`, t.ex. `2022-06-15_ndvi.png`.
- `analyzers.yaml` styr vilka analyzers som körs — ingen kodändring krävs för att aktivera/inaktivera.
- Sensitive data (credentials, `.cdsapirc`) committeas aldrig — se `.gitignore`.

---

## Externa beroenden och kontakter

| System | Beskrivning | Kontakt |
|--------|-------------|---------|
| Digital Earth Sweden (DES) | openEO-backend för Sentinel-2-data | henrik.forsgren@ri.se |
| ai-pipelines-poc | Molndetektering via ML-modell (Aleksis Pirinen/RISE) | erik.kallman@ri.se |
| metafilter | ERA5-väderfiltrering av datum | erik.kallman@ri.se |
| ColonyOS | Distribuerad jobbschemaläggning via Docker | — |

ML-modellen för molndetektering finns i `aleksispi/ml-cloud-opt-thick` — licens för kommersiellt bruk ej klargjord, kontrollera med Aleksis Pirinen (aleksis.pirinen@ri.se) innan produktionsdrift.

---

## Naturliga nästa steg

1. **Fler analyzers** — t.ex. vattenytesförändring med NDWI över tid, branddetektering med NBR
2. **Riktiga objektdetekteringsmodeller** — YOLO tränad på xView eller DOTA
3. **Kedja med metafilter** — ERA5-filtrering → molnverifiering → IMINT-analys
4. **Tester** — enhetstester per analyzer i `tests/`, kör med `pytest`
5. **Notebook** — `notebooks/explore.ipynb` för interaktiv utveckling av nya analyzers
