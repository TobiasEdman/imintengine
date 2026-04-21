# SDL 3.0 — Use Case Notebooks

Interactive Jupyter notebooks presenting Swedish Space Data Lab (SDL 3.0) use cases to the community. Each notebook wraps an [ImintEngine](https://github.com/TobiasEdman/imintengine) showcase as a reproducible end-to-end analysis.

**Status:** Staging area. Notebooks mature here and migrate to [`DigitalEarthSweden/digital-earth-sweden-community`](https://github.com/DigitalEarthSweden/digital-earth-sweden-community) once reviewed.

**Naming convention:** Files follow the DES community repo pattern `NNN_CATEGORY_NNN-Descriptive-Name.ipynb`. The `070_CASE_` prefix is a proposed new category slot (next after `060_EO_`) — the final prefix will be confirmed with the DES team during the migration PR.

## Run locally

```bash
cd notebooks
conda env create -f environment.yml
conda activate sdl3-cases
jupyter lab
```

Or with pip:

```bash
pip install -r requirements.txt
jupyter lab
```

## Run on Binder

Each notebook has a Binder badge that launches a hosted environment with all dependencies pre-installed. No DES account required — fallback to synthetic data if authentication fails.

## Notebooks

| # | Notebook | Område | Analyzers | Partners |
|---|----------|--------|-----------|----------|
| 010 | [Getting Started](sdl3-cases/070_CASE_010-Getting-Started.ipynb) | Gotland (template) | `spectral` | RISE, AI Sweden, LTU |
| 020 | [Wildfire — Kårböle](sdl3-cases/070_CASE_020-Wildfire-Karbole.ipynb) | Ljusdals kommun, Gävleborg | `spectral` (dNBR), `change_detection`, `prithvi` | Skogsstyrelsen, RISE, MSB |
| 030 | [Marine Vessels — Sotenäs](sdl3-cases/070_CASE_030-Marine-Vessels-Sotenas.ipynb) | Hunnebostrand, Bohuslän | `marine_vessels`, `ai2_vessels`, `object_detection` | SSC, Sjöfartsverket, PandionAI |
| 040 | [Grazing — Munkarp](sdl3-cases/070_CASE_040-Grazing-Munkarp.ipynb) | Höörs kommun, Skåne | `grazing`, `nmd`, `spectral` | Jordbruksverket, Naturvårdsverket, RISE |
| 050 | [Coastline — Ystad](sdl3-cases/070_CASE_050-Coastline-Ystad.ipynb) | Ystads kommun, Skåne | `shoreline` (CoastSat + SegFormer) | SGI, SMHI, SU |
| 060 | [Vegetation Edge](sdl3-cases/070_CASE_060-Vegetation-Edge.ipynb) | Småländska kustlandet | `vegetation_edge`, `nmd` | Skogsstyrelsen, Naturvårdsverket, SU |
| 070 | [Commercial Marine — Öresund](sdl3-cases/070_CASE_070-Commercial-Marine-Oresund.ipynb) | Öresund | `marine_vessels`, `ai2_vessels` | Maxar, PandionAI, SSC |
| 080 | [COT Cloud Filtering (SDL 2.0-arv)](sdl3-cases/070_CASE_080-COT-Cloud-Filtering.ipynb) | Norrland | `cot` (Pirinen MLP5) | RISE, SMHI, Skogsstyrelsen |
| 090 | [Multitemporal LULC — Prithvi](sdl3-cases/070_CASE_090-Multitemporal-LULC.ipynb) | Skåne | `prithvi`, `nmd` | NV, SJV, SKS, RISE, IBM/NASA |
| 100 | [Custom Analyzer (tutorial)](sdl3-cases/070_CASE_100-Custom-Analyzer.ipynb) | Tutorial | `BaseAnalyzer` | Community |

## Structure per notebook

1. **Titel** — beskrivning, partners, datakällor, licens\n2. **Metod** — vilka analyzers, varför, förväntade output\n3. **Setup** — imports, DES-auth, AOI-definition, datum\n4. **Analys** — kör `run_job()` via LocalExecutor\n5. **Visualisering** — folium-karta + matplotlib\n6. **Tolkning** — nästa steg, länkar till källkod

## Koppling till SDL 3.0-ansökans leverabler

| WP2-leverans | Notebook-täckning |
|-------------|-------------------|
| Library of interactive notebooks | ✅ 10 notebooks |
| Presenting use cases to community | ✅ Publik på GitHub + Binder |
| Documented on DES platform | ✅ Migreras till DES community-repo |
| Methods from collaborative partners | ✅ Partner-attribution i varje notebook |
| Seminarier och events | ⏳ Planerat: Rymddagen 2026, Hackathon 4–5 juni 2026 |

## CI

Every PR runs automated checks via [.github/workflows/notebooks.yml](../.github/workflows/notebooks.yml):
- JSON-validering av alla `.ipynb`
- Kontroll att alla notebooks har sektion 1–5
- nbconvert-parsning för att fånga syntaxfel

Manual smoke-test with papermill:

```bash
cd notebooks
./ci/test_notebooks.sh
```

## License

Notebook content: CC0 1.0 Universal (public domain).
Dependencies retain their original licenses — see upstream repos.
