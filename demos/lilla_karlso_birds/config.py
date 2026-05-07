"""Konfiguration för Lilla Karlsö C2RCC-tidsserie över sillgrissle-säsongen 2025.

Specificerat i SPEC_lilla_karlso_birds.md. AOI utvidgad till 22×22 km
runt kolonin för att fånga foderhabitat (skarpsill/sill 5–15 km från ön).
"""
from __future__ import annotations

import os
from pathlib import Path


# ── AOI ──────────────────────────────────────────────────────────────────
AOI_NAME = "Lilla Karlsö"
AOI_LAT = 57.311
AOI_LON = 18.061

# 22×22 km runt kolonin (utvidgad från lilla_karlso/-demos 7×5 km).
# Helt inom UTM 33N (T33VWE) — ingen MGRS-zon-gräns.
BBOX_WGS84 = {
    "west":  17.91,
    "south": 57.21,
    "east":  18.21,
    "north": 57.41,
}


# ── Period ───────────────────────────────────────────────────────────────
# Sillgrissle-häckningssäsong:
#   april 15: ägg läggs på klipphyllor
#   juni 15: ägg kläcks
#   juli 31: ungar lämnar kolonin
PERIOD_START = "2025-04-15"
PERIOD_END = "2025-07-31"


# ── Fetch-strategi ──────────────────────────────────────────────────────
MAX_AOI_CLOUD = 0.15       # SCL-stack-tröskel (15%) — Östersjön sommar är notoriskt molnig
                           # 3 datum träffar äggläggning/kläckning/ungar-lämnar perfekt
SCENE_CLOUD_MAX = 30.0     # STAC-cc backup
N_WORKERS_FETCH = 6        # Parallella GCP HTTPS-workers
DATE_WINDOW = 0            # Exakt-datum-match — optimal_fetch redan bestämt


# ── C2RCC ────────────────────────────────────────────────────────────────
DOCKER_IMAGE = os.environ.get(
    "C2RCC_IMAGE",
    # Default: GHCR signerad image (CI-byggd). Kan överridas via env för
    # att använda lokalt-byggd imint-snap-c2rcc:latest.
    "ghcr.io/tobiasedman/imint-c2rcc-snap:latest",
)


# ── Render ───────────────────────────────────────────────────────────────
# DES brand-palett för alla retrieval-cmaps (chl, TSM, CDOM)
DES_CMAP_STOPS = [
    (0.00, "#1a4338"),  # mörk-grön (DES primary)
    (0.25, "#cff8e4"),  # mint
    (0.50, "#fdd5c2"),  # persika
    (0.80, "#ff826c"),  # röd
    (1.00, "#ffffff"),  # vit
]

# Färgskalor — fasta så samma färg betyder samma värde över datum.
# log10-skala för chl-a (typ 0.5–25 mg/m³).
CHL_VMIN, CHL_VMAX = 0.5, 25.0     # mg/m³, log10
TSM_VMIN, TSM_VMAX = 0.0, 20.0     # g/m³, linjär
CDOM_VMIN, CDOM_VMAX = 0.0, 2.0    # m⁻¹, linjär


# ── Output paths ─────────────────────────────────────────────────────────
DEMO_ROOT = Path(__file__).parent
REPO_ROOT = DEMO_ROOT.parents[1]

# I k8s-podden mountas CephFS som /data; dessa overrides via env-var.
DATA_ROOT = Path(os.environ.get("C2RCC_DATA_ROOT", DEMO_ROOT / "cache"))
SAFE_CACHE = DATA_ROOT / "l1c_safes"
DIMAP_OUT = DATA_ROOT / "c2rcc_runs"
DOCS_OUT = REPO_ROOT / "docs" / "showcase" / "lilla_karlso_birds"
FRAMES_DIR = DOCS_OUT / "frames"
