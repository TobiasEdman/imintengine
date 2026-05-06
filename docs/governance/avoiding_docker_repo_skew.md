# Att undvika Docker-repo-skew i ImintEngine

> *Rapport: governance-regler för att förhindra att Docker-images, körprotokoll
> och output-artefakter glider ifrån varandra över commits.*
>
> Författare: governance-utredning, 2026-05-05.
> Trigger: rekonstruktion av `imint-snap-c2rcc:latest` för
> `SPEC_lilla_karlso_birds.md` misslyckades på grund av saknad Dockerfile.

---

## 1. Problembeskrivning

I commit `52d19ae` (`feat(water_quality): real ESA C2RCC + Pahlevan MDN`)
levererades nio chl-a/TSM/CDOM-PNG i `docs/showcase/water_quality/2026/` och
spegling i `outputs/showcase/water_quality/2026/`. Bilderna producerades
genom att köra **ESA SNAP-grafen** `c2rcc.msi` i en lokal Docker-image
`imint-snap-c2rcc:latest` (1.39 GB) på Sentinel-2 L1C SAFE-arkiv. Tre
artefaktklasser tillkom på disk samtidigt:

1. PNG-filer (committade) under `outputs/showcase/water_quality/2026/*`
2. BEAM-DIMAP-utdata (icke-committade, gitignorerade) under
   `outputs/c2rcc_runs/c2rcc_2026_04_08.{dim,data/}`,
   `outputs/c2rcc_runs_t32vpk/`, `outputs/c2rcc_runs_t33vue/` —
   ~90 `.hdr/.img`-par per körning
3. Docker-imagen `imint-snap-c2rcc:latest` — i lokal Docker-daemon

**Problemet:** av dessa tre är endast (1) versionerad. (2) går inte att
återskapa utan (3), och (3) finns inte representerad i repot:

| Resurs | Versionerad? | Status |
|---|---|---|
| `imint-snap-c2rcc:latest` Docker-image | nej | Endast lokal daemon |
| Dockerfile för imagen | nej | Saknas helt |
| SNAP graph-XML (`Read→Resample→Subset→c2rcc.msi→Write`) | nej | Saknas helt |
| `docker run`-wrapper-skript (build.sh / run.sh) | nej | Saknas helt |
| Output-PNG (showcase) | ja | `52d19ae` |
| Output-DIMAP (rådata) | nej | Gitignored, men inte producerbar |

Samma commit-meddelande nämner `c2rcc.msi via SNAP 13 in custom Docker
image` och `netSet=C2X-Nets` — men dessa körparametrar finns bara i
prosa-text, inte i en versionerad config.

Detta är klassisk **code-vs-binary skew**: deliverables (PNG) är
versionerade, men processen (Dockerfile + graph + run-args) som
producerade dem är inte. Sex månader senare när SPEC_lilla_karlso_birds
behöver samma pipeline för en ny AOI/period måste imagen reverse-
engineeras — vilket `SPEC_lilla_karlso_birds.md:91-129` redan har
tvingats dokumentera som ett rekonstruktionsbesvär:

> KRITISKT: Det finns en lokal Docker-image som heter
> `imint-snap-c2rcc:latest` (1.39 GB) — repot-grep hittar **ingen**
> referens till hur den invocades. Antagligen kördes den manuellt med
> ad-hoc `docker run`-kommandon i en tidigare session.

---

## 2. Inventory: andra ställen med samma risk i repot

### 2.1 Bra exempel — följ detta mönster

`docker/cloud-models/` är **rätt struktur**:

| Fil | Roll |
|---|---|
| `docker/cloud-models/Dockerfile` | Build-recept (Python 3.11 + s2cloudless + omnicloudmask) |
| `docker/cloud-models/run_cloud_models.py` | Versionerad ENTRYPOINT |
| `demos/cloud_models/run_comparison.py:53` | `CONTAINER_IMG = "imint/cloud-models:latest"` |
| `demos/cloud_models/run_comparison.py:160-167` | Versionerat `docker run`-anrop |
| `docs/index.html:688` | Build-instruktion: `docker build -t imint/cloud-models:latest docker/cloud-models` |

Här går det att gå från en **frisk klon** till en producerande pipeline
genom att följa repot. Imagen kan byggas om från scratch, hostskripten
hänvisar till en känd image-tagg, och alla parametrar är i kod.

### 2.2 Risk-fall (skew eller potentiell skew)

| # | Image / körning | Hittas | Saknas | Allvar |
|---|---|---|---|---|
| 1 | `imint-snap-c2rcc:latest` | endast i lokal daemon | Dockerfile, graph-XML, run-skript, build-instruktion | **Hög** — aktivt blockerar SPEC_lilla_karlso_birds |
| 2 | `outputs/c2rcc_runs/`, `outputs/c2rcc_runs_t32vpk/`, `outputs/c2rcc_runs_t33vue/` | på lokal disk | hela produktionsreceptet | **Hög** — output-rådata är "föräldralös" |
| 3 | `outputs/safe_archives/` | tom dir committad | namngivningskonvention för cachade SAFE-arkiv | Låg — bara konvention |
| 4 | `imint-engine:latest` (root `Dockerfile`) | i repo | inget — Makefile bygger den | OK |
| 5 | `imint-engine:cuda` (`Dockerfile.cuda`) | i repo | inget — `Makefile:47` bygger den | OK |
| 6 | `imint-engine-api:v0.1.0` (`Dockerfile.api`) | i repo | byggs ej i Makefile, men docs nämner det | Låg |
| 7 | `localhost:5000/imint-engine:latest` (`scripts/submit_seasonal_jobs.py:127`) | refereras | implicit — push-steg saknas i kod | Medel |

`outputs/baselines/`, `outputs/water_quality_cache/`, `outputs/sr_cache/`,
`outputs/grazing_model/` m.fl. innehåller troligen liknande
"föräldralös rådata" — inget av dem är committat (gitignorerat), men
inte heller refererat till en versionerad processdefinition.

### 2.3 Sammanfattning av problemets storlek

Av sju identifierade Docker-images har **två (#1, #2)** skew. Båda hör
till samma fall (C2RCC). De övriga är välordnade, så roten är inte
"vi gör fel överallt" utan "vi har en lucka för third-party tunga
images där en *interaktiv* SNAP-konfig konsumerades men aldrig
versionerades".

---

## 3. Föreslagna regler — copy-paste-redo för `CLAUDE.md`

```markdown
## Docker-och processversionering — obligatorisk vid alla pipeline-images

Varje Docker-image som körs i en analys-/datapipeline måste vara helt
återskapbar från repot. Konkret innebär detta:

### 3.1 Sju-punkts-checklista per image

För varje image som processar data (inte utvecklarmiljö-images), skapa
ett dedikerat directory `docker/<namn>/` innehållande:

1. **Dockerfile** — explicit FROM-tagg (ALDRIG `:latest` på basen),
   pin alla `pip install` versioner.
2. **build.sh** (eller Makefile-target) — exakt en rad: hur
   imagen byggs, inklusive `--platform`-flag.
3. **run.sh** (eller motsvarande Python-driver) — visar hur
   imagen körs, vilka volymer som mountas, vilka args som skickas.
4. **graph/ eller config/** — SNAP-XML, ACOLITE settings, MLflow
   model-spec eller motsvarande deklarativ konfig.
5. **README.md** — 5-radig "vad gör den, vilken commit producerade
   senaste image-tag, vilken Sentinel-version".
6. **VERSIONS** (eller längst ner i README.md) — image-tag → git-SHA
   mapping.
7. **smoke_test.sh** — minimal körning som verifierar att imagen
   bygger och kör utan input-data (eller mot syntetisk fixture).

### 3.2 Regler (nolltolerans)

- **Aldrig `:latest` i Dockerfile FROM eller i körnings-skript.**
  Använd `python:3.11-slim` → pinna till `python:3.11.9-slim-bookworm`.
- **`docker run` i Python eller shell måste ha en motsvarande
  Dockerfile inom samma repo.** Lint-regel: om en commit lägger till
  `subprocess.run(["docker", "run", ...])` utan att samtidigt
  innehålla en Dockerfile för den image-tagg som körs → blockera.
- **Output-artefakter får inte committas innan processen som
  producerade dem är committad.** Order: process först, sedan output.
- **Image-tag måste pekas tillbaka till git-SHA.** I commit-meddelandet
  som producerade artefakter, eller i en sidecar-fil
  `<output-dir>/MANIFEST.json` med `{image, image_digest, git_sha,
  run_args, input_data_hash, produced_at}`.
- **Om en image bara finns lokalt och inte i repot — använd den ALDRIG
  i en pipeline.** Bygg om från Dockerfile först. Bygger inte → image
  får inte användas i produktion.

### 3.3 Förväntat directory-mönster

`docker/cloud-models/` är referensimplementation. Replikera den för
varje ny pipeline-image (jfr. `docker/c2rcc-snap/` för C2RCC-fallet).
```

---

## 4. C2RCC-rekonstruktionsplan

Konkret recept för att återskapa `imint-snap-c2rcc:latest` från lokal
daemon, dokumentera processen i repot och bevisa ekvivalens med
befintlig output i `outputs/c2rcc_runs/c2rcc_2026_04_08.dim`.

### Steg 1 — Extrahera build-historiken

```bash
# Hela image-konfigurationen (entrypoint, env, exposed ports, labels)
docker inspect imint-snap-c2rcc:latest \
  | jq '.[0] | {Config, RootFS, Architecture, Os, Size}' \
  > docker/c2rcc-snap/inspect.json

# Build-stegen (från senaste FROM och nedåt)
docker history --no-trunc --format \
  '{{.CreatedBy}}\t{{.Size}}' imint-snap-c2rcc:latest \
  > docker/c2rcc-snap/history.txt

# Layers att rotbestämma — om imagen bygger på esa/snap-base
docker history imint-snap-c2rcc:latest --format '{{.ID}} {{.CreatedBy}}'
```

### Steg 2 — Identifiera SNAP-versionen i imagen

```bash
docker run --rm imint-snap-c2rcc:latest /usr/local/snap/bin/gpt --version
docker run --rm imint-snap-c2rcc:latest \
  /usr/local/snap/bin/gpt -h c2rcc.msi | head -50
```

`SPEC_lilla_karlso_birds.md:101` antyder att `gpt -h | head -20` ger
SNAP-versionen. Notera om imagen är ESAs officiella `esa-snap` eller
en community-version (mundialis, snap-stamps etc.).

### Steg 3 — Skriv `docker/c2rcc-snap/Dockerfile`

Mall (anpassa efter steg 1–2):

```dockerfile
# docker/c2rcc-snap/Dockerfile
# ESA SNAP 13 + c2rcc.msi GPT-runner.
# Image-tagg: imint-snap-c2rcc:13.0.0-1
FROM mundialis/esa-snap:13.0.0  # ELLER: esa/snap:13.0.0 — bekräfta
                                 # från `docker history`-utdata.

# Inget Python-lager behövs — graph konsumeras direkt av gpt.
# Snabb sanity-check vid build-tid:
RUN /usr/local/snap/bin/gpt -h c2rcc.msi >/dev/null

WORKDIR /work
ENTRYPOINT ["/usr/local/snap/bin/gpt"]
```

### Steg 4 — Skriv graph-XML

`outputs/c2rcc_runs/c2rcc_2026_04_08.dim` har banden
`iop_apig`, `iop_agelb`, `iop_bpart`, `iop_bwit`, `kd489`, `kdmin`,
`rhow_B*`, `c2rcc_flags` — vilket bekräftar `outputRtosa=false`,
default-banduppsättning, och med stor sannolikhet `netSet=C2X-Nets`
(Bohuslän är hög-CDOM = Case-2 eXtreme).

```xml
<!-- docker/c2rcc-snap/c2rcc_msi_graph.xml -->
<graph id="C2RCC-MSI-L1C">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters><file>${inputSafe}</file></parameters>
  </node>
  <node id="Resample">
    <operator>Resample</operator>
    <sources><source>Read</source></sources>
    <parameters><referenceBand>B2</referenceBand></parameters>
  </node>
  <node id="Subset">
    <operator>Subset</operator>
    <sources><source>Resample</source></sources>
    <parameters>
      <geoRegion>${aoiWkt}</geoRegion>
      <copyMetadata>true</copyMetadata>
    </parameters>
  </node>
  <node id="C2RCC">
    <operator>c2rcc.msi</operator>
    <sources><source>Subset</source></sources>
    <parameters>
      <netSet>C2X-Nets</netSet>
      <outputRtosa>false</outputRtosa>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><source>C2RCC</source></sources>
    <parameters>
      <file>${outputDim}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>
```

### Steg 5 — `docker/c2rcc-snap/run.sh`

```bash
#!/usr/bin/env bash
# run.sh — wrap docker run for c2rcc.msi
set -euo pipefail
SAFE="${1:?usage: run.sh <SAFE.zip-or-dir> <output.dim> <wkt>}"
OUT="${2:?need output .dim path}"
WKT="${3:?need AOI WKT polygon}"

docker run --rm \
  -v "$(dirname "$SAFE"):/in:ro" \
  -v "$(dirname "$OUT"):/out" \
  -v "$(pwd)/docker/c2rcc-snap:/graph:ro" \
  imint-snap-c2rcc:13.0.0-1 \
  /graph/c2rcc_msi_graph.xml \
    -PinputSafe="/in/$(basename "$SAFE")" \
    -PoutputDim="/out/$(basename "$OUT")" \
    -PaoiWkt="$WKT" \
    -e
```

### Steg 6 — Bevisa ekvivalens

Reproduktionstest (placera i `tests/integration/test_c2rcc_repro.py`):

1. Hämta SAFE-arkivet för `S2B_MSIL1C_20260408T103029_*` via
   `imint.fetch.fetch_l1c_safe_from_gcp`.
2. Kör `docker/c2rcc-snap/run.sh` mot AOI v3 från
   `scripts/generate_water_quality_showcase.py:42-47`.
3. Jämför resultat mot `outputs/c2rcc_runs/c2rcc_2026_04_08.dim`:
   ```python
   import numpy as np
   from imint.exporters.dimap import read_dimap  # eller motsv.

   ref = read_dimap("outputs/c2rcc_runs/c2rcc_2026_04_08.dim")
   new = read_dimap("outputs/c2rcc_runs_repro/c2rcc_2026_04_08.dim")
   for band in ("iop_apig", "iop_agelb", "iop_bpart", "kd489"):
       diff = np.abs(ref[band] - new[band])
       valid = np.isfinite(ref[band]) & np.isfinite(new[band])
       assert np.percentile(diff[valid], 99) < 1e-4, (
           f"C2RCC repro skiljer på band {band}")
   ```

Om p99-diff < 1e-4 är rekonstruktionen bit-ekvivalent (NN-deterministiska,
inga RNG-frön i SNAP). Om större diff: kontrollera `netSet`, SNAP-version
(checkpoint-vikter ändras mellan minor-versioner), och `Resample`-default
(B2 vs `referenceResolution=10`).

### Steg 7 — Bygg Sentinel-2 graph + bind till SPEC

Lägg till i Makefile:

```makefile
build-c2rcc:  ## Build SNAP C2RCC image
	docker build --platform linux/amd64 \
	    -t imint-snap-c2rcc:13.0.0-1 docker/c2rcc-snap

c2rcc-smoke:  ## Quick smoke test on a synthetic SAFE
	./docker/c2rcc-snap/smoke_test.sh
```

Uppdatera `SPEC_lilla_karlso_birds.md` så att `Steg 3 — C2RCC via SNAP
Docker` (rad 89–129) refererar `make build-c2rcc` istället för
"reverse-engineer from history". Då försvinner blockaren.

---

## 5. Förebyggande tooling

### 5.1 Pre-commit hook — varna vid PNG i outputs/showcase utan process-ändring

```bash
#!/usr/bin/env bash
# .git/hooks/pre-commit (eller hook-skript anslutet via husky/pre-commit)
# Varnar om en commit innehåller binära artefakter under outputs/showcase
# men inte modifierar någon process-definition.

staged_outputs=$(git diff --cached --name-only --diff-filter=A \
                 | grep -E '^outputs/showcase/.*\.(png|tif|npz|dim)$' \
                 | wc -l)

staged_process=$(git diff --cached --name-only \
                 | grep -E '^(docker/|scripts/|imint/|demos/|.*\.Dockerfile|Dockerfile.*)' \
                 | wc -l)

if [ "$staged_outputs" -gt 0 ] && [ "$staged_process" -eq 0 ]; then
  echo "VARNING: commit lägger till $staged_outputs output-fil(er) i"
  echo "outputs/showcase/ men ändrar ingen process-definition (docker/,"
  echo "scripts/, imint/, demos/). Är du säker på att processen som"
  echo "producerade dem är versionerad? Avbryt med Ctrl-C eller"
  echo "fortsätt med Enter."
  read -r _
fi
exit 0
```

Hängs in på samma sätt som `~/.claude/hooks/co-authored-by.sh`.

### 5.2 Lint-regel — `docker run` utan Dockerfile

Lägg till i `tests/test_repo_hygiene.py`:

```python
# tests/test_repo_hygiene.py
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def test_every_docker_run_has_a_dockerfile():
    """Every image referenced by `docker run X` must have a Dockerfile
    in the repo (either at repo root, in docker/<name>/, or in the same
    directory as the calling script)."""
    pattern = re.compile(
        r'docker\s+run[^"\']*?["\']?([\w./-]+:[^\s"\']+|\w[\w./-]+)["\']?'
    )
    referenced_images: set[str] = set()
    for ext in ("*.py", "*.sh", "Makefile"):
        for f in REPO.rglob(ext):
            if any(part in str(f) for part in (".venv", "node_modules", ".git")):
                continue
            for m in pattern.finditer(f.read_text(errors="ignore")):
                tag = m.group(1)
                # Skip generic CLI invocations like `docker run --help`
                if tag.startswith("-"):
                    continue
                referenced_images.add(tag.split(":")[0])

    have_dockerfiles = {
        "imint-engine", "imint-engine-cuda", "imint-engine-api",
        "imint/cloud-models",
        # New entries: append when adding new pipeline images
        "imint-snap-c2rcc",
    }
    missing = referenced_images - have_dockerfiles - {
        # Allowlist for upstream/registry images we don't build ourselves
        "postgres", "redis", "minio/minio",
    }
    assert not missing, (
        f"docker run refereras till {missing}, men ingen Dockerfile "
        f"hittades. Lägg till Dockerfile under docker/<namn>/ eller "
        f"uppdatera have_dockerfiles."
    )
```

Körs i CI på varje PR. Misslyckas testet → PR-blockerad tills
processen committas.

### 5.3 GitHub Actions — auto-build pipeline-images

```yaml
# .github/workflows/build-pipeline-images.yml
name: Build pipeline images
on:
  push:
    paths:
      - 'docker/**/Dockerfile'
      - 'docker/**/*.py'
      - 'docker/**/*.xml'
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        image: [cloud-models, c2rcc-snap]
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t imint/${{ matrix.image }}:${{ github.sha }} \
                          docker/${{ matrix.image }}
      - run: docker run --rm imint/${{ matrix.image }}:${{ github.sha }} \
                          --help
      # Optional: push to GHCR with both :sha and :latest tags
```

### 5.4 Run-manifest sidecar

Kompletterande: varje pipeline-skript som producerar artefakter i
`outputs/showcase/` ska skriva en `MANIFEST.json` bredvid utdata:

```json
{
  "produced_at": "2026-04-28T20:34:11Z",
  "git_sha": "52d19ae",
  "image": "imint-snap-c2rcc:13.0.0-1",
  "image_digest": "sha256:abc...",
  "graph": "docker/c2rcc-snap/c2rcc_msi_graph.xml",
  "graph_sha256": "...",
  "run_args": {
    "aoi_wkt": "POLYGON((11.31 58.04,...))",
    "input_safes": ["S2B_MSIL1C_20260408T103029_N0510_R108_T32VPK.SAFE",
                    "S2B_MSIL1C_20260408T103029_N0510_R108_T33VUE.SAFE"],
    "netset": "C2X-Nets"
  },
  "input_data_hash": "sha256:...",
  "outputs": ["chlorophyll_a_c2rcc.png", "tsm_c2rcc.png", "cdom_c2rcc.png"]
}
```

Genereras automatiskt av en helper i `imint/exporters/manifest.py`.

---

## 6. Pre-existing best practice — kort referens

| Praxis | Vad det löser | Tillämplighet på ImintEngine |
|---|---|---|
| **Reproducible Docker builds** (pinned base, locked deps) | NN-vikter / SNAP-version-skew mellan builds | Direkt — punkt 3.2 nolltolerans-listan |
| **DVC (Data Version Control)** | Stora datapipelines (input → process → output) versionerade tillsammans | Tung; lämplig för `data/lulc_full/` om det blir relevant |
| **Pachyderm** | Cluster-skalig data-lineage med Docker som steg | Overkill — ingen multi-tenant pipeline |
| **MLflow Tracking** | Run-manifest med modellartefakter, params, metrics | Light-weight motsvarighet täcks av §5.4 (`MANIFEST.json`) |
| **GitHub Actions auto-build** | Image-tag → git-SHA mappning automatiserad | Ja — §5.3, 30 min att sätta upp |
| **`docker buildx` + SBOM** | Supply-chain audit av Java/SNAP-base-images | Optionellt; senare när image:erna stabiliseras |
| **Sigstore/cosign signing** | Image-immutability mellan repo och kluster | Optionellt; först när imagi pushas till delad registry |

Slutsats: vi behöver **inte** importera DVC eller Pachyderm. Det räcker
med disciplin i `docker/<namn>/`-konvention + den sju-punkts-checklista
som §3.1 listar + en CI-build i §5.3 + ett pre-commit-skydd i §5.1.

---

## 7. Sammanfattning — vad ska göras nu?

1. **Akut blockare för SPEC_lilla_karlso_birds:** kör §4-receptet,
   skapa `docker/c2rcc-snap/{Dockerfile,c2rcc_msi_graph.xml,run.sh,
   smoke_test.sh,README.md}`, verifiera bit-ekvivalens mot
   `outputs/c2rcc_runs/c2rcc_2026_04_08.dim`.

2. **Klipp in §3 i `CLAUDE.md`** under en ny sektion "Docker- och
   processversionering".

3. **Lägg till `tests/test_repo_hygiene.py`** (§5.2) i nästa commit
   så regeln blir self-enforcing.

4. **Stoppa pre-commit hook** §5.1 lokalt via
   `~/.claude/hooks/` (samma mekanism som `co-authored-by.sh`).

5. **(Optionellt)** sätt upp §5.3 GH Actions när `docker/c2rcc-snap/`
   är på plats. Mer värde än kostnad — varje commit som rör Dockerfile
   verifieras att bygga.

Inga `outputs/`-filer tas bort. Detta är en framåtriktad regelinförsel
— rekonstruktion görs först, sedan håller reglerna nya commits säkra.

---

*Källor i repot:*
- `SPEC_lilla_karlso_birds.md:89-129` — explicit beskrivning av blockaren
- `docker/cloud-models/Dockerfile` + `docker/cloud-models/run_cloud_models.py` — referens-implementation
- `demos/cloud_models/run_comparison.py:53,156-167` — exempel på korrekt host→container-bro
- `outputs/c2rcc_runs/c2rcc_2026_04_08.{dim,data/}` — befintlig referens-output
- `imint/analyzers/water_quality/c2rcc_wrapper.py` — alternativt ACOLITE-spår (separat fall, fungerande)
- `Makefile:38-50` — referens-mönster för image-build-targets
- `.gitignore:2-6` — varför `outputs/showcase/` är committat men `outputs/c2rcc_runs/` inte
