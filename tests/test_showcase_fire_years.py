"""The fire tab's year-by-year land cover and NMD layers (scripts/showcase_fire_years.py)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from PIL import Image  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/showcase_fire_years.py"
AOI = json.loads((REPO / "config/lclu/fire_karbole.json").read_text())["aoi"]
HEIGHT, WIDTH = AOI["shape"]
PINE, BURNT = "#6d8b05", "#cdcd66"  # two colours of the service legend


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    return {"sha256": _sha(path), "size": path.stat().st_size}


def _raster(path: Path, codes: np.ndarray) -> Path:
    """A GeoTIFF on the AOI's 10 m lattice with a 10 px margin around the AOI."""
    west, _, _, north = AOI["bounds_3006"]
    with rasterio.open(path, "w", driver="GTiff", width=codes.shape[1], height=codes.shape[0],
                       count=1, dtype=codes.dtype.name, crs="EPSG:3006",
                       transform=Affine(10, 0, west - 100, 0, -10, north + 100)) as dst:
        dst.write(codes, 1)
    return path


def _inputs(tmp_path: Path, *, nmd2023_code: int = 118, tamper: bool = False) -> list[str]:
    run = tmp_path / "run"
    legend = [{"id": 1, "name": "Tallskog på fastmark", "color": PINE},
              {"id": 6, "name": "Temporärt ej skog på fastmark", "color": BURNT},
              {"id": 255, "name": "Data saknas", "color": None}]
    run.mkdir()
    (run / "legend.json").write_text(json.dumps(legend))
    years = {}
    for year in (2024, 2025):
        rgba = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
        rgba[..., 3] = 255
        rgba[: HEIGHT // 2, :, :3] = list(bytes.fromhex(PINE[1:]))
        rgba[HEIGHT // 2:, :, :3] = list(bytes.fromhex(BURNT[1:]))
        (run / str(year)).mkdir()
        png = run / str(year) / "classification.png"
        Image.fromarray(rgba).save(png)
        years[str(year)] = {
            "tessera_year": year, "vpp_year": 2024, "vpp_year_substituted": year == 2025,
            "outputs": {"classification.png": _identity(png)},
            "class_counts": {"1": 1, "6": 1},
            **({"note": "TESSERA 2025 with HR-VPP 2024"} if year == 2025 else {}),
        }
    if tamper:
        Image.new("RGBA", (WIDTH, HEIGHT), (*bytes.fromhex(PINE[1:]), 255)).save(
            run / "2024" / "classification.png")
    manifest = {
        "aoi": AOI, "legend": _identity(run / "legend.json"),
        "run_args": {"run_id": "RUN", "output_location": "pvc:/fire"},
        "produced_at": "2026-11-07T00:00:00+00:00", "git_sha": "a" * 40,
        "image": "python@sha256:" + "b" * 64, "image_digest": "sha256:" + "b" * 64,
        "model": {"name": "TESSERA R4", "checkpoint": {"sha256": "c" * 64, "size": 1}},
        "historical_note": "Fasta stödlager.",
        "vpp_year_substitutions": [{"year": 2025, "vpp_year": 2024, "label": "växtfenologi från 2024",
                                    "note": "not copied"}],
        "years": years,
    }
    (run / "MANIFEST.json").write_text(json.dumps(manifest))
    shape = (HEIGHT + 20, WIDTH + 20)
    nmd2018 = _raster(tmp_path / "nmd2018.tif", np.full(shape, 111, np.uint8))
    codes_2023 = np.full(shape, 118, np.uint16)
    codes_2023[10, 10] = nmd2023_code  # the AOI's north-west pixel
    nmd2023 = _raster(tmp_path / "nmd2023.tif", codes_2023)
    return ["--tessera-run", str(run), "--nmd2018-raster", str(nmd2018),
            "--nmd2023-raster", str(nmd2023), "--out-dir", str(tmp_path / "out")]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO)


def test_layers_are_bound_to_their_sources_and_name_no_input_path(tmp_path: Path) -> None:
    result = _run(_inputs(tmp_path, nmd2023_code=4211))
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    text = (out / "MANIFEST.json").read_text()
    assert str(tmp_path) not in text
    manifest = json.loads(text)
    years = manifest["tessera_r4"]["years"]
    assert {y: v["label"] for y, v in years.items()} == {"2024": "2024", "2025": "2025 · växtfenologi från 2024"}
    run = tmp_path / "run"
    for year, record in years.items():
        assert _sha(out / record["file"]) == record["sha256"] == _sha(run / year / "classification.png")
        assert [e["color"] for e in record["legend"]] == [PINE, BURNT]
    assert manifest["tessera_r4"]["source"]["manifest"] == _identity(run / "MANIFEST.json")
    nmd18, nmd23 = manifest["nmd_2018"], manifest["nmd_2023"]
    assert nmd18["level1"] == {"forest": {"pixels": HEIGHT * WIDTH, "fraction": 1.0}}
    assert nmd18["raster"] == {"file": "nmd2018.tif", **_identity(tmp_path / "nmd2018.tif")}
    assert nmd23["codes"] == {"118": HEIGHT * WIDTH - 1, "4211": 1}
    assert [(e["name"], e["color"]) for e in nmd23["legend"]] == [
        ("Temporärt ej skog på fastmark", "#cdcd66"), ("Torr buskdominerad mark", "#ccd69e")]
    # The NMD 2023 PNG keeps its own classes: the north-west pixel is 4211, not a 2018 fold.
    nw = Image.open(out / nmd23["file"]).convert("RGB").getpixel((0, 0))
    assert "#%02x%02x%02x" % nw == "#ccd69e"
    for layer in (nmd18, nmd23):
        assert _sha(out / layer["file"]) == layer["sha256"]


def test_a_year_that_differs_from_its_run_manifest_is_refused_before_writing(tmp_path: Path) -> None:
    result = _run(_inputs(tmp_path, tamper=True))
    assert result.returncode != 0 and "2024: classification.png differs from the run MANIFEST" in result.stderr
    assert not (tmp_path / "out").exists()


def test_an_nmd_2023_code_outside_its_table_is_refused_before_writing(tmp_path: Path) -> None:
    result = _run(_inputs(tmp_path, nmd2023_code=999))
    assert result.returncode != 0 and "codes outside its legend: [999]" in result.stderr
    assert not (tmp_path / "out").exists()
