"""
demos/wetland_pirinen/make_overlays.py

Generates Sentinel-2 RGB background + 7 transparent PNG-overlays for the
Pirinen 2023 wetland-stack tab. Output goes to ``docs/showcase/wetland_pirinen/``
and is consumed by the renderTabDynamic-mall (tab-data.js + app.js).

Output files (all aligned to AOI bbox, identical pixel dimensions):
    rgb.png                  Sentinel-2 RGB (B04/B03/B02), 8-bit, opaque
    smi.png                  Markfuktighetsindex (NMD2018), transparent
    slu_markfukt.png         SLU markfukt (Lidberg)
    bush_height.png
    bush_cover.png
    tree_height.png
    tree_cover.png
    dem.png

Run:
    python demos/wetland_pirinen/make_overlays.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.config.env import load_env
load_env()

from imint.fetch import S2L2A_SPECTRAL_BANDS, fetch_seasonal_image  # noqa: E402
from imint.training.optimal_fetch import optimal_fetch_dates  # noqa: E402

# Reuse layer specs + AOI from fetch_and_render
from demos.wetland_pirinen.fetch_and_render import (  # noqa: E402
    AOI_NAME, E, LAYERS, LAYER_CACHE, N, S, SIZE_PX, W, WGS84_BBOX,
)

DOCS_OUT = REPO_ROOT / "docs" / "showcase" / "wetland_pirinen"
DOCS_OUT.mkdir(parents=True, exist_ok=True)
RGB_CACHE = Path(__file__).parent / "cache_rgb"
RGB_CACHE.mkdir(exist_ok=True)

# Search window for clear-sky scene over Stormyran. Aapamyr-AOI; peak
# greenness Jul–Aug. Wider window = more candidates for ERA5 prefilter.
PERIOD_START = "2024-06-01"
PERIOD_END = "2024-09-15"
MAX_AOI_CLOUD = 0.05   # SCL-stack: max 5% AOI cloud-pixels


# ── Sentinel-2 RGB ───────────────────────────────────────────────────────


def fetch_s2_rgb() -> tuple[np.ndarray, str]:
    """Pick a clear-sky Sentinel-2 scene via the ERA5→SCL pipeline.

    Stage 1: ERA5 atmosphere prefilter (free, AOI-aware ~30 km grid)
    Stage 2: SCL-stack screen via DES openEO (1 server-side call)
    Stage 3: Spectral fetch only on the cleanest surviving date

    Returns (RGB array (SIZE_PX, SIZE_PX, 3) uint8, ISO date string).
    """
    print(f"  [optimal_fetch] mode=era5_then_scl  "
          f"period={PERIOD_START}..{PERIOD_END}")
    plan = optimal_fetch_dates(
        bbox_wgs84=WGS84_BBOX,
        date_start=PERIOD_START,
        date_end=PERIOD_END,
        mode="era5_then_scl",
        max_aoi_cloud=MAX_AOI_CLOUD,
    )
    print(f"  [optimal_fetch] candidates_per_stage={plan.n_candidates_after}")
    print(f"  [optimal_fetch] elapsed={plan.elapsed_s}")
    if not plan.dates:
        raise RuntimeError(
            f"optimal_fetch_dates returned 0 dates for {PERIOD_START}..{PERIOD_END}"
        )

    # plan.dates is already sorted by cleanliness (best first per Atmosfär demo)
    for date in plan.dates:
        cache = RGB_CACHE / f"{date}.npz"
        if cache.exists():
            print(f"  [cache hit] {date}")
            arr = np.load(cache)["arr"]
            return _to_rgb(arr), date
        print(f"  [fetch spectral] {date} ...", end=" ", flush=True)
        try:
            result = fetch_seasonal_image(date=date, coords=WGS84_BBOX,
                                          source="des")
            if result is None:
                print("None — skipping")
                continue
            arr, _ = result
            np.savez_compressed(cache, arr=arr.astype(np.float32))
            print(f"ok shape={arr.shape}")
            return _to_rgb(arr), date
        except Exception as e:
            print(f"ERR {type(e).__name__}: {str(e)[:120]}")
            continue
    raise RuntimeError(
        f"All {len(plan.dates)} optimal-fetch dates failed to fetch"
    )


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Per-scene p2/p98 stretch to 8-bit RGB."""
    band = {b: arr[i] for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
    rgb = np.stack([band["B04"], band["B03"], band["B02"]], axis=-1)
    valid = np.isfinite(rgb).all(axis=-1)
    if not valid.any():
        return np.zeros(rgb.shape, dtype=np.uint8)
    lo = np.percentile(rgb[valid], 2)
    hi = np.percentile(rgb[valid], 98)
    if hi <= lo:
        hi = lo + 1e-3
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


# ── PNG overlay rendering ────────────────────────────────────────────────


def save_rgb_png(rgb: np.ndarray, out_path: Path) -> None:
    """Save RGB as opaque PNG, exactly SIZE_PX×SIZE_PX, no axes."""
    fig = plt.figure(figsize=(SIZE_PX / 100, SIZE_PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=100, pad_inches=0, bbox_inches=None)
    plt.close(fig)


def save_overlay_png(arr: np.ndarray, spec, out_path: Path) -> None:
    """Save raster as transparent PNG overlay. nodata pixels (==0) → alpha=0.

    The output is exactly SIZE_PX×SIZE_PX with no padding, suitable for
    Leaflet ImageOverlay aligned to the AOI bbox.
    """
    if arr.shape != (SIZE_PX, SIZE_PX):
        # Defensive: scale to SIZE_PX if mismatched (shouldn't happen)
        from scipy.ndimage import zoom
        zy = SIZE_PX / arr.shape[0]
        zx = SIZE_PX / arr.shape[1]
        arr = zoom(arr, (zy, zx), order=0)

    vmin = spec.vmin if spec.vmin is not None else float(np.nanmin(arr))
    vmax = spec.vmax if spec.vmax is not None else float(np.nanmax(arr))
    cmap = plt.get_cmap(spec.cmap)
    norm = (np.clip(arr, vmin, vmax) - vmin) / max(vmax - vmin, 1e-9)
    rgba = (cmap(norm) * 255).astype(np.uint8)  # (H, W, 4)
    rgba[arr == 0, 3] = 0  # nodata → transparent

    fig = plt.figure(figsize=(SIZE_PX / 100, SIZE_PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgba, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=100, pad_inches=0, bbox_inches=None,
                transparent=True)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    print(f"=== make_overlays — {AOI_NAME} ===")
    print(f"AOI 3006: W={W} S={S} E={E} N={N}  ({(E-W)/1000}×{(N-S)/1000} km)")
    print(f"Output: {DOCS_OUT}")
    print()

    print("[1/8] Sentinel-2 RGB (ERA5→SCL pipeline) ...")
    rgb, rgb_date = fetch_s2_rgb()
    save_rgb_png(rgb, DOCS_OUT / "rgb.png")
    print(f"   rgb.png written ({(DOCS_OUT/'rgb.png').stat().st_size:,} bytes) "
          f"— scen: {rgb_date}")

    for i, spec in enumerate(LAYERS, start=2):
        print(f"[{i}/8] {spec.key} ({spec.title}) ...")
        try:
            arr = spec.fetcher()
            out = DOCS_OUT / f"{spec.key}.png"
            save_overlay_png(arr, spec, out)
            print(f"   {out.name} written ({out.stat().st_size:,} bytes)")
        except FileNotFoundError as e:
            print(f"   SKIP (missing data): {str(e).splitlines()[0][:100]}")
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {str(e)[:120]}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
