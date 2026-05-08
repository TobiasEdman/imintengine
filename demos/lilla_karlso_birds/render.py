"""Steg 4–6 av Lilla Karlsö-pipelinen: läs DIMAP, beräkna chl/TSM/CDOM,
render 4 PNG per scen, aggregera tidsserie.

Output:
    docs/showcase/lilla_karlso_birds/
        manifest.json                 — pipeline-state, datum, stats
        timeseries.json               — chl_p50, chl_p90, tsm_p50, cdom_p50/datum
        frames/<date>/rgb.png         — Sentinel-2 RGB
        frames/<date>/chl.png         — Klorofyll-a (mg/m³, log10)
        frames/<date>/tsm.png         — TSM (g/m³)
        frames/<date>/cdom.png        — CDOM (m⁻¹)

Använder DES brand-palett (config.DES_CMAP_STOPS).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, LogNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.exporters.manifest import write_manifest  # noqa: E402

from demos.lilla_karlso_birds import config  # noqa: E402


# DES brand-palett-cmap
DES_CMAP = LinearSegmentedColormap.from_list("des", config.DES_CMAP_STOPS)


# ── DIMAP-läsning ────────────────────────────────────────────────────────


def read_dimap_band(dimap_path: Path, band_name: str) -> np.ndarray:
    """Läs ett ENVI-band från en BEAM-DIMAP .data/-katalog."""
    img = dimap_path.with_suffix(".data") / f"{band_name}.img"
    if not img.is_file():
        raise FileNotFoundError(f"missing band: {img}")
    with rasterio.open(img) as src:
        return src.read(1)


def compute_retrievals(dimap_path: Path) -> dict[str, np.ndarray]:
    """Beräkna chl/TSM/CDOM från IOPs i DIMAP enligt Brockmann-formler."""
    apig = read_dimap_band(dimap_path, "iop_apig")
    bpart = read_dimap_band(dimap_path, "iop_bpart")
    bwit = read_dimap_band(dimap_path, "iop_bwit")
    agelb = read_dimap_band(dimap_path, "iop_agelb")

    # Brockmann 2016 standardrelationer
    chl = (np.maximum(apig, 0) ** 1.04) * 21.0  # mg/m³
    tsm = 1.72 * np.maximum(bpart, 0) + 3.1 * np.maximum(bwit, 0)  # g/m³
    cdom = np.maximum(agelb, 0)  # m⁻¹

    # NN-floor-rens: pixlar där apig < 0.001 är klampade outliers
    nn_floor = apig < 0.001
    chl[nn_floor] = np.nan

    # Land-mask: alla band 0 = utanför vatten (SNAP klipper land till 0)
    water = (apig != 0) | (bpart != 0) | (agelb != 0)
    chl[~water] = np.nan
    tsm[~water] = np.nan
    cdom[~water] = np.nan

    return {"chl": chl, "tsm": tsm, "cdom": cdom, "water": water}


# RGB-bakgrund hämtas från DES openEO L2A True-color via separat
# k8s-jobb (k8s/lilla-karlso-rgb-l2a-job.yaml). Den rhow-baserade
# RGB-vägen som tidigare bodde här (build_rgb_from_dimap) tog C2RCC:s
# water-leaving reflectance och stretchade p2/p98 — gav extremt mörk
# bild eftersom rhow har snäv dynamisk range över öppet vatten, plus
# att Karlsöarna helt försvann (NaN över land).
#
# L2A TCI ger samma typ av RGB som Mollösund-tabben (pcell 2/98 stretch
# på B04/B03/B02), och Karlsöarna syns tydligt mot havet.
#
# render_scene() använder inte längre build_rgb_from_dimap — rgb.png
# kommer från l2a-jobbets output direkt. Behåll funktionen kommentar­erad
# som dokumentation för varför vi övergav rhow.


# ── PNG-rendering ────────────────────────────────────────────────────────


def save_overlay(arr: np.ndarray, out_path: Path, *,
                 vmin: float, vmax: float, log: bool = False) -> None:
    """Spara transparent PNG där NaN är genomskinligt."""
    h, w = arr.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    if log:
        norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = DES_CMAP.copy()
    cmap.set_bad((0, 0, 0, 0))  # NaN → transparent
    masked = np.ma.masked_invalid(arr)
    ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=100, pad_inches=0, transparent=True)
    plt.close(fig)


def save_rgb(rgb: np.ndarray, out_path: Path) -> None:
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb)
    ax.axis("off")
    fig.savefig(out_path, dpi=100, pad_inches=0)
    plt.close(fig)


# ── Per-scen + totalrapport ──────────────────────────────────────────────


def render_scene(date: str, dimap_path: Path) -> dict:
    """Generera 4 PNG + per-scen-stats för ett datum."""
    out_dir = config.FRAMES_DIR / date
    out_dir.mkdir(parents=True, exist_ok=True)

    retrievals = compute_retrievals(dimap_path)

    # OBS: rgb.png skrivs INTE här — kommer från k8s/lilla-karlso-rgb-l2a-job.yaml
    # (L2A True-color via DES openEO). Render-jobbet rör inte befintlig
    # rgb.png om den finns; bara chl/tsm/cdom regenereras.
    save_overlay(retrievals["chl"], out_dir / "chl.png",
                 vmin=config.CHL_VMIN, vmax=config.CHL_VMAX, log=True)
    save_overlay(retrievals["tsm"], out_dir / "tsm.png",
                 vmin=config.TSM_VMIN, vmax=config.TSM_VMAX)
    save_overlay(retrievals["cdom"], out_dir / "cdom.png",
                 vmin=config.CDOM_VMIN, vmax=config.CDOM_VMAX)

    # Stats för tidsserien
    chl = retrievals["chl"]
    tsm = retrievals["tsm"]
    cdom = retrievals["cdom"]
    water = retrievals["water"]
    n_water = int(water.sum())
    chl_valid = chl[np.isfinite(chl) & (chl > 0)]
    tsm_valid = tsm[np.isfinite(tsm) & (tsm > 0)]
    cdom_valid = cdom[np.isfinite(cdom) & (cdom > 0)]

    return {
        "date": date,
        "n_water_pixels": n_water,
        "n_valid_chl": int(chl_valid.size),
        "chl_p50": float(np.percentile(chl_valid, 50)) if chl_valid.size else None,
        "chl_p90": float(np.percentile(chl_valid, 90)) if chl_valid.size else None,
        "chl_mean": float(chl_valid.mean()) if chl_valid.size else None,
        "tsm_p50": float(np.percentile(tsm_valid, 50)) if tsm_valid.size else None,
        "cdom_p50": float(np.percentile(cdom_valid, 50)) if cdom_valid.size else None,
        "frames": {
            "rgb": f"frames/{date}/rgb.png",
            "chl": f"frames/{date}/chl.png",
            "tsm": f"frames/{date}/tsm.png",
            "cdom": f"frames/{date}/cdom.png",
        },
    }


def main() -> int:
    plan_path = config.SAFE_CACHE.parent / "plan.json"
    plan = json.loads(plan_path.read_text())

    config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for date in plan["dates"]:
        dimap = config.DIMAP_OUT / f"{date}.dim"
        if not dimap.is_file():
            print(f"[skip] {date} — saknar {dimap}")
            continue
        print(f"[render] {date}")
        rec = render_scene(date, dimap)
        records.append(rec)
        print(f"   chl_p50={rec['chl_p50']}, n_water={rec['n_water_pixels']}")

    # Tidsserie
    timeseries_path = config.DOCS_OUT / "timeseries.json"
    timeseries_path.write_text(json.dumps({
        "aoi": config.BBOX_WGS84,
        "period": [config.PERIOD_START, config.PERIOD_END],
        "records": records,
    }, indent=2, ensure_ascii=False))
    print(f"\ntimeseries: {timeseries_path}")

    # MANIFEST.json sidecar (governance-regel §3)
    write_manifest(
        config.DOCS_OUT,
        image=config.DOCKER_IMAGE,
        process_files=[
            "docker/c2rcc-snap/Dockerfile",
            "docker/c2rcc-snap/c2rcc_msi_graph.xml",
            "docker/c2rcc-snap/run.sh",
            "demos/lilla_karlso_birds/render.py",
            "demos/lilla_karlso_birds/run_c2rcc.py",
            "demos/lilla_karlso_birds/fetch_safes.py",
            "demos/lilla_karlso_birds/config.py",
        ],
        run_args={
            "aoi": config.BBOX_WGS84,
            "period": [config.PERIOD_START, config.PERIOD_END],
            "max_aoi_cloud": config.MAX_AOI_CLOUD,
            "netset": "C2X-Nets",
        },
        outputs=[f"frames/{r['date']}/{k}.png" for r in records
                 for k in ("rgb", "chl", "tsm", "cdom")] +
                ["timeseries.json"],
        extra={"n_scenes": len(records),
               "fenology_window": "ägg → kläckning → ungar lämnar"},
    )
    print(f"manifest: {config.DOCS_OUT / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
