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

# 10×22 km — Lilla Karlsö som östgräns, havsremsan väster om kolonin.
# Sillgrisslornas foderhabitat ligger 5–15 km väster om ön (pelagisk skarpsill/
# sill); Gotlands fastland (>18.13°E) är inte intressant för C2RCC-analysen.
# Boxen är shiftad västerut jämfört med tidigare 22×22 km kring ön — hela
# AOI är nu väster om/på Lilla Karlsö, inget Gotland-fastland inkluderat.
# 18.075 = östra spetsen av Lilla Karlsö; 17.775 = ~10 km västerut.
# OBS: AOI korsar fortfarande 18.0°E (UTM 33/34-gränsen) — separat tile-zon-issue.
BBOX_WGS84 = {
    "west":  17.775,
    "south": 57.21,
    "east":  18.075,
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
# Pin till digest, INTE :latest — pause-incident 2026-05-07 visade att
# CI byggde en :latest från en mundialis-baserad Dockerfile som råkar ge
# SNAP 9 (tot 2025 SAFE-format saknar reader). Plus: feature-branches
# tag:as inte som :latest överhuvudtaget (workflow enable on main only).
#
# Pinnad till CI-byggd image från commit f5337a9 (claude/blissful-leakey),
# verifierad 2026-05-07 i k8s-diag mot Lilla Karlsö-SAFE 2025-06-13
# (T33VXD): Ubuntu 24.04 base, SNAP 13 i /opt/snap, S2OrthoProductReaderPlugIn
# registrerad för EPSG:32633, Read exekverar (OOM på 4 GiB testpod, 16 GiB
# i produktionsjobb räcker enligt Mollösundscaset).
#
# När branchen mergas till main: CI bygger ny image med ny digest. Uppdatera
# denna pin till nya digesten (CI tag:ar ny som :latest på main, sha-tag på
# alla branches).
DOCKER_IMAGE = os.environ.get(
    "C2RCC_IMAGE",
    "ghcr.io/tobiasedman/imint-c2rcc-snap"
    "@sha256:4a2c3217d156a69adddf371eee5511747551911f1cb40fbddf2764a13c0e1aef",
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
