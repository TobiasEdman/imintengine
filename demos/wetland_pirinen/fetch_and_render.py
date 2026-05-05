"""
demos/wetland_pirinen/fetch_and_render.py

Reproducera Aleksis Pirinens 8-kanals input-stack från 2023-rapporten
*"A pre-study about using artificial intelligence for semantic
segmentation of Swedish wetland types"* (RISE / Naturvårdsverket) över
AOI Stormyran (Jämtland aapamyr, 64.296°N, 15.938°E).

Demon hämtar och visualiserar 8 raster — INGEN modell-inferens. Pirinen
släppte aldrig pre-tränade vikter; vi visar att ImintEngine kan
reproducera hans fullständiga input-stack från svenska öppna geodata.

8 lager:
    1. NMD basskikt (DES openEO)
    2. NMD markfuktighetsindex (lokal extraherad zip via nvv_smi)
    3. SLU markfukt (SKS ImageServer via slu_markfukt)
    4. NVV objekthöjd 0.5–5 m / busk-höjd (lokal via nvv_objektdata)
    5. NVV objekttäckning 0.5–5 m / busk-täckning (lokal)
    6. Skogsstyrelsen trädhöjd (SKG ImageServer via skg_height)
    7. NVV objekttäckning 5–45 m / träd-täckning (lokal)
    8. Copernicus DEM GLO-30 (S3 COG via copernicus_dem)

Lager #2, #4, #5, #7 kräver att ``k8s/prefetch-nvv-aux-job.yaml`` har
laddat NVV-zip:arna till PVC. Om de saknas lokalt rendererar demon en
placeholder-frame och flaggar lagret som ``status: "missing-data"``
i manifest.json. De övriga 4 lagren fungerar utan PVC.

Outputs:
    docs/showcase/wetland_pirinen/
        manifest.json          — per-lager record (stats, source, status)
        frames/<layer>.jpg     — single-layer heatmap per lager
        frames/overview.jpg    — 2×4-grid med alla 8 lager

Run:
    python demos/wetland_pirinen/fetch_and_render.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.training.copernicus_dem import fetch_dem_tile  # noqa: E402
from imint.training.nvv_objektdata import (  # noqa: E402
    fetch_bush_cover_tile,
    fetch_bush_height_tile,
    fetch_tree_cover_tile,
)
from imint.training.nvv_smi import fetch_smi_tile  # noqa: E402
from imint.training.skg_height import (  # noqa: E402
    _get_height_url,
    _to_nmd_grid_bounds,
)
from imint.training.slu_markfukt import fetch_markfukt_tile  # noqa: E402


# ── AOI / parametrar ─────────────────────────────────────────────────────
AOI_NAME = "Stormyran (Jämtland)"
AOI_LAT = 64.29581
AOI_LON = 15.93770
AOI_HALF_KM = 5.0    # 10×10 km totalt
SIZE_PX = 1000       # 10 m upplösning över 10 km

WGS84_BBOX = {
    "west":  AOI_LON - AOI_HALF_KM / (111.0 * np.cos(np.radians(AOI_LAT))),
    "east":  AOI_LON + AOI_HALF_KM / (111.0 * np.cos(np.radians(AOI_LAT))),
    "south": AOI_LAT - AOI_HALF_KM / 111.0,
    "north": AOI_LAT + AOI_HALF_KM / 111.0,
}

# ── Output paths ─────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DOCS_OUT = REPO_ROOT / "docs" / "showcase" / "wetland_pirinen"
FRAMES_DIR = DOCS_OUT / "frames"
LAYER_CACHE = HERE / "cache_layers"
DOCS_OUT.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
LAYER_CACHE.mkdir(parents=True, exist_ok=True)


# ── Layer specification ──────────────────────────────────────────────────
@dataclass
class LayerSpec:
    key: str                       # filename + manifest key
    title: str                     # display title
    pirinen_idx: int               # Pirinen 2023 lager-index (1-10)
    source: str                    # short source description
    fetcher: Callable[..., np.ndarray]
    cmap: str                      # matplotlib cmap name
    units: str                     # "%", "dm", "m", "class", etc
    vmin: float | None = None      # fixed colour range; None = auto
    vmax: float | None = None
    description: str = ""
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# Re-project once to EPSG:3006 + snap to 10 m grid; reused by every fetcher
PROJECTED = _to_nmd_grid_bounds(WGS84_BBOX)
W, S, E, N = PROJECTED["west"], PROJECTED["south"], PROJECTED["east"], PROJECTED["north"]


def _fetch_smi(**kw):
    return fetch_smi_tile(W, S, E, N, size_px=SIZE_PX, cache_dir=LAYER_CACHE, **kw)


def _fetch_slu_markfukt(**kw):
    return fetch_markfukt_tile(W, S, E, N, size_px=SIZE_PX, cache_dir=LAYER_CACHE, **kw)


def _fetch_bush_h(**kw):
    return fetch_bush_height_tile(W, S, E, N, size_px=SIZE_PX, cache_dir=LAYER_CACHE, **kw)


def _fetch_bush_c(**kw):
    return fetch_bush_cover_tile(W, S, E, N, size_px=SIZE_PX, cache_dir=LAYER_CACHE, **kw)


def _fetch_tree_h_skg(**kw):
    """SKG trädhöjd utan THF-mosaicRule (THF saknas norr om limes Norrlandicus).

    `imint.training.skg_height.fetch_height_tile` filtrerar på
    `ProductName = 'THF'` (Trädhöjd Flygbild). Den produkten täcker
    bara södra Sverige; norr om ~62°N returnerar servern endast
    nollor. För demon (Stormyran 64°N) hämtar vi default-mosaiken
    direkt från ImageServer utan filter.
    """
    import io
    import urllib.parse, urllib.request
    import rasterio
    cache_key = f"tree_height_skg_nofilter_{int(W)}_{int(S)}_{int(E)}_{int(N)}.npy"
    cache_path = LAYER_CACHE / cache_key
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (SIZE_PX, SIZE_PX):
            return cached
    params = {
        "bbox": f"{W},{S},{E},{N}",
        "bboxSR": "3006", "imageSR": "3006",
        "size": f"{SIZE_PX},{SIZE_PX}",
        "format": "tiff", "pixelType": "S16", "f": "image",
    }
    url = _get_height_url() + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ImintEngine/1.0")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with rasterio.io.MemoryFile(data) as mf:
        with mf.open() as ds:
            arr = ds.read(1).astype(np.float32) / 10.0  # dm → m
    arr = np.clip(arr, 0.0, None)
    np.save(cache_path, arr)
    return arr


def _fetch_tree_c(**kw):
    return fetch_tree_cover_tile(W, S, E, N, size_px=SIZE_PX, cache_dir=LAYER_CACHE, **kw)


def _fetch_dem(**kw):
    return fetch_dem_tile(W, S, E, N, size_px=SIZE_PX, cache_dir=LAYER_CACHE, **kw)


LAYERS: list[LayerSpec] = [
    LayerSpec(
        key="smi",
        title="NMD Markfuktighetsindex",
        pirinen_idx=3,
        source="Naturvårdsverket NMD2018 (lokal extraherad zip)",
        fetcher=_fetch_smi,
        cmap="YlGnBu",
        units="0–100",
        vmin=0, vmax=100,
        description="Topografisk fuktindex härlett från NMD-produktionen.",
    ),
    LayerSpec(
        key="slu_markfukt",
        title="SLU Markfuktighetskarta",
        pirinen_idx=4,
        source="SLU (Lidberg & Ågren) via Skogsstyrelsen ImageServer",
        fetcher=_fetch_slu_markfukt,
        cmap="YlGnBu",
        units="%",
        vmin=0, vmax=101,
        description="Markfuktighet predicerad från höjdmodell (DTW + STI).",
    ),
    LayerSpec(
        key="bush_height",
        title="Objekthöjd 0.5–5 m (busk-höjd)",
        pirinen_idx=6,
        source="Naturvårdsverket NMD2018 tilläggsskikt (lokal)",
        fetcher=_fetch_bush_h,
        cmap="YlGn",
        units="dm",
        vmin=0, vmax=50,
        description="Höjd för objekt i intervallet 0.5–5 m.",
    ),
    LayerSpec(
        key="bush_cover",
        title="Objekttäckning 0.5–5 m (busk-täckning)",
        pirinen_idx=7,
        source="Naturvårdsverket NMD2018 tilläggsskikt (lokal)",
        fetcher=_fetch_bush_c,
        cmap="Greens",
        units="%",
        vmin=0, vmax=100,
        description="Täckningsgrad för objekt i intervallet 0.5–5 m.",
    ),
    LayerSpec(
        key="tree_height",
        title="Trädhöjd (laser)",
        pirinen_idx=8,
        source="Skogsstyrelsen Trädhöjd 3.1 (ImageServer proxy)",
        fetcher=_fetch_tree_h_skg,
        cmap="viridis",
        units="m",
        vmin=0, vmax=30,
        description="Trädhöjd från ALS, ~2 m native, resamplad till 10 m.",
    ),
    LayerSpec(
        key="tree_cover",
        title="Objekttäckning 5–45 m (träd-täckning)",
        pirinen_idx=9,
        source="Naturvårdsverket NMD2018 tilläggsskikt (lokal)",
        fetcher=_fetch_tree_c,
        cmap="Greens",
        units="%",
        vmin=0, vmax=100,
        description="Täckningsgrad för objekt i intervallet 5–45 m.",
    ),
    LayerSpec(
        key="dem",
        title="Höjdmodell (Copernicus DEM GLO-30)",
        pirinen_idx=10,
        source="ESA Copernicus DEM GLO-30 (S3 COG)",
        fetcher=_fetch_dem,
        cmap="terrain",
        units="m",
        description="Topografi, ~30 m native, resamplad till 10 m.",
    ),
]


# Note: Pirinen lager #5 (NMD basskikt) hämtas separat via DES openEO i
# imint/fetch.py:fetch_nmd_data, vilket kräver DES-credentials. Det
# läggs till i ett senare steg när env-config bekräftats.


# ── Rendering ────────────────────────────────────────────────────────────
def _stats(arr: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(arr) & (arr != 0)
    if not valid.any():
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "land_pct": 0.0}
    nz = arr[valid]
    return {
        "min": float(nz.min()),
        "max": float(nz.max()),
        "mean": round(float(nz.mean()), 2),
        "land_pct": round(100.0 * valid.mean(), 1),
    }


def _render_layer(spec: LayerSpec, arr: np.ndarray, status: str) -> Path:
    """Render a single-layer heatmap with title + colorbar + stats."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#0f0f0f")
    fig.subplots_adjust(left=0.02, right=0.92, top=0.90, bottom=0.05)

    if status == "ok":
        vmin = spec.vmin if spec.vmin is not None else float(np.nanmin(arr))
        vmax = spec.vmax if spec.vmax is not None else float(np.nanmax(arr))
        im = ax.imshow(arr, cmap=spec.cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(spec.units, color="white", fontsize=8)
        cbar.ax.tick_params(colors="white", labelsize=7)
        cbar.outline.set_edgecolor("#444")
        st = _stats(arr)
        subtitle = f"{spec.units}  ·  mean {st['mean']}  ·  land {st['land_pct']}%"
        title_color = "#27ae60"
    else:
        ax.text(
            0.5, 0.5,
            f"saknad data\n\n{status}\n\nKör k8s/prefetch-nvv-aux-job.yaml\nför att fylla PVC.",
            ha="center", va="center", color="#e67e22", fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_facecolor("#1a1a1a")
        subtitle = "ej tillgänglig — väntar på prefetch"
        title_color = "#e67e22"

    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"#{spec.pirinen_idx}  {spec.title}",
        color=title_color, fontsize=11, fontweight="bold", y=0.97,
    )
    ax.set_title(subtitle, color="#bbb", fontsize=9, pad=4)

    out = FRAMES_DIR / f"{spec.key}.jpg"
    fig.savefig(out, dpi=110, facecolor="#0f0f0f", bbox_inches="tight")
    plt.close(fig)
    return out


def _render_overview(records: list[dict]) -> Path:
    """2×4 grid of all 7 layers (8 if NMD added) for tab thumbnail."""
    n = len(records)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows),
                              facecolor="#0f0f0f")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.10, hspace=0.20)
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, rec in zip(flat, records):
        spec = next(s for s in LAYERS if s.key == rec["key"])
        if rec["status"] == "ok" and rec.get("_arr") is not None:
            arr = rec["_arr"]
            vmin = spec.vmin if spec.vmin is not None else float(np.nanmin(arr))
            vmax = spec.vmax if spec.vmax is not None else float(np.nanmax(arr))
            ax.imshow(arr, cmap=spec.cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(f"#{spec.pirinen_idx} {spec.title}",
                         color="#27ae60", fontsize=9, pad=4)
        else:
            ax.set_facecolor("#1a1a1a")
            ax.text(0.5, 0.5, "saknad", ha="center", va="center",
                    color="#e67e22", fontsize=11, transform=ax.transAxes)
            ax.set_title(f"#{spec.pirinen_idx} {spec.title}",
                         color="#e67e22", fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide any extra axes
    for ax in flat[len(records):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Pirinen 2023 input-stack  ·  {AOI_NAME}  ·  {SIZE_PX}×{SIZE_PX} px @ 10 m",
        color="#27ae60", fontsize=13, fontweight="bold", y=0.99,
    )
    out = FRAMES_DIR / "overview.jpg"
    fig.savefig(out, dpi=100, facecolor="#0f0f0f", bbox_inches="tight")
    plt.close(fig)
    return out


# ── Main pipeline ────────────────────────────────────────────────────────
def main() -> int:
    print(f"=== Pirinen 2023 wetland-stack demo — {AOI_NAME} ===")
    print(f"AOI WGS84: {WGS84_BBOX}")
    print(f"AOI 3006:  W={W} S={S} E={E} N={N}  ({(E-W)/1000:.1f}×{(N-S)/1000:.1f} km)")
    print(f"Output: {DOCS_OUT}")
    print()

    records: list[dict] = []
    t_total = time.time()

    for spec in LAYERS:
        t0 = time.time()
        print(f"[{spec.key:18s}] fetching #{spec.pirinen_idx} {spec.title} ...")
        rec: dict = {
            "key": spec.key,
            "pirinen_idx": spec.pirinen_idx,
            "title": spec.title,
            "source": spec.source,
            "units": spec.units,
            "description": spec.description,
        }
        try:
            arr = spec.fetcher()
            elapsed = round(time.time() - t0, 2)
            rec.update({
                "status": "ok",
                "elapsed_s": elapsed,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "stats": _stats(arr),
            })
            rec["_arr"] = arr  # transient, not serialized
            print(f"    ok in {elapsed}s — {rec['stats']}")
        except FileNotFoundError as e:
            rec.update({
                "status": "missing-data",
                "elapsed_s": round(time.time() - t0, 2),
                "error": str(e).splitlines()[0],
            })
            print(f"    SKIP (missing data): {rec['error']}")
        except Exception as e:
            rec.update({
                "status": "error",
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "traceback": traceback.format_exc()[:1000],
            })
            print(f"    ERROR: {rec['error']}")

        # Render frame regardless of status (placeholder if not ok)
        frame_path = _render_layer(spec, rec.get("_arr", np.zeros((10, 10))),
                                    rec["status"])
        rec["frame_path"] = str(frame_path.relative_to(REPO_ROOT / "docs"))
        records.append(rec)

    # Overview grid
    overview = _render_overview(records)
    print(f"\n[overview] {overview.relative_to(REPO_ROOT)}")

    # Manifest (strip transient _arr)
    manifest = {
        "aoi": {
            "name": AOI_NAME,
            "lat": AOI_LAT, "lon": AOI_LON,
            "half_km": AOI_HALF_KM,
            "wgs84": WGS84_BBOX,
            "epsg3006": {"west": W, "south": S, "east": E, "north": N},
            "size_px": SIZE_PX,
        },
        "elapsed_s": round(time.time() - t_total, 2),
        "n_layers_ok":     sum(1 for r in records if r["status"] == "ok"),
        "n_layers_missing": sum(1 for r in records if r["status"] == "missing-data"),
        "n_layers_error":  sum(1 for r in records if r["status"] == "error"),
        "overview_path":   str(overview.relative_to(REPO_ROOT / "docs")),
        "layers": [{k: v for k, v in r.items() if k != "_arr"} for r in records],
        "reference": {
            "paper": "Pirinen, A. (2023). A pre-study about using AI for "
                     "semantic segmentation of Swedish wetland types.",
            "code": "https://github.com/aleksispi/ai-swetlands",
            "funder": "Naturvårdsverket",
        },
    }
    manifest_path = DOCS_OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {manifest_path}")

    print(f"\n=== summary: {manifest['n_layers_ok']} ok, "
          f"{manifest['n_layers_missing']} missing, "
          f"{manifest['n_layers_error']} error "
          f"in {manifest['elapsed_s']}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
