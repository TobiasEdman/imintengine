"""Layout of the Sentinel-1 vessel reference record, shared by the check scripts.

The record holds rasters (hundreds of MB) and stays outside git. Its files and
their sha256 are listed in ``docs/marine/s1_port_results/large_files.sha256``;
every script takes the record root as ``--record``:

    inputs/                         AI2 reference-run inputs: radar-geometry GRD
                                    crops with GCPs, Satlas infra GeoJSON
    run1/reference.geojson          AI2's own output (the reference)
    run1/scratch_dir/windows/detector_predict/0/
                                    AI2's materialised model input, float32 DN
    run1/scratch_dir/windows/attribute_predict/0_<col>_<row>/
                                    AI2's 128 px attribute windows
    s1port/cache/                   the port's fetch cache (``cache_dir`` of
                                    ``imint.fetch.fetch_s1_vessel_scenes``)

Run the scripts from the repository root, e.g.
``python scripts/marine/s1_port/compare_to_reference.py --help``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# AI2 layer names per scene, in the analyzer's band-key order [vv, vh].
SCENE_LAYERS = {"target": "sentinel1", "historical1": "sentinel1_historical"}
POLS = ("vv", "vh")
SKAGEN_BBOX = (10.50, 57.55, 11.70, 58.05)
SKAGEN_DATE = "2026-06-28"
TARGET_ID = "S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF"
HISTORICAL_ID = "S1D_IW_GRDH_1SDV_20260523T053133_20260523T053158_002907_004F80"
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def add_record_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--record", type=Path, required=True,
                    help="root of the reference record (see scripts/marine/s1_port/README.md)")


def ai2_window_dir(record: Path) -> Path:
    return record / "run1" / "scratch_dir" / "windows" / "detector_predict" / "0"


def ai2_window_bounds(record: Path) -> tuple[int, int, int, int]:
    """AI2's detector window in absolute 10 m pixel units (col0, row0, col1, row1)."""
    return tuple(json.loads((ai2_window_dir(record) / "metadata.json").read_text())["bounds"])


def ai2_band_path(record: Path, scene: str, pol: str) -> Path:
    return ai2_window_dir(record) / "layers" / SCENE_LAYERS[scene] / pol / "geotiff.tif"


def read_ai2_stack(record: Path) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """AI2's model input as (4, H, W) float32 [target vv, vh, historical vv, vh] + bounds."""
    import rasterio

    bands = []
    for scene in SCENE_LAYERS:
        for pol in POLS:
            with rasterio.open(ai2_band_path(record, scene, pol)) as src:
                bands.append(src.read(1))
    return np.stack(bands), ai2_window_bounds(record)


def reference_properties(record: Path) -> list[dict]:
    """Properties of AI2's reference detections (``run1/reference.geojson``)."""
    path = record / "run1" / "reference.geojson"
    return [f["properties"] for f in json.loads(path.read_text())["features"]]


def read_port_input(tif: Path) -> tuple[np.ndarray, tuple[int, int], dict]:
    """A cached port model input: (4, H, W) uint16, its absolute pixel origin, its record."""
    import rasterio

    with rasterio.open(tif) as src:
        stack = src.read()
        origin = (round(src.transform.c / 10), round(src.transform.f / -10))
    return stack, origin, json.loads(tif.with_suffix(".json").read_text())


def write_json(path: Path | None, obj) -> None:
    """Print ``obj`` and, when ``path`` is given, write it there."""
    text = json.dumps(obj, indent=1, default=float)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
