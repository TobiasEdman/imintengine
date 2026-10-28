"""The teaser thumbnails align RGB with the maps and name no tile."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/showcase_nfi_teaser.py"
WATER = (0, 0, 255)  # unified class 10, vatten


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=REPO)


def _tile(tmp_path: Path) -> Path:
    src = tmp_path / "holdoutval_SECRET_tile"
    src.mkdir()
    rgb = Image.new("RGB", (512, 512), (0, 0, 0))  # 4 px border outside the map window
    rgb.paste(Image.new("RGB", (504, 504), (10, 120, 30)), (4, 4))
    rgb.save(src / "rgb.png")
    return src


def test_rgb_is_cropped_to_the_map_window_and_manifest_names_no_path(tmp_path: Path) -> None:
    src = _tile(tmp_path)
    for model in ("secretmodel_a", "secretmodel_b"):
        Image.new("RGB", (504, 504), WATER).save(src / f"{model}.png")
    out = tmp_path / "out"
    result = _run("--example", str(src / "rgb.png"), str(src / "secretmodel_a.png"),
                  str(src / "secretmodel_b.png"), "--map-names", "a", "b",
                  "--out-dir", str(out), "--size", "64")
    assert result.returncode == 0, result.stderr
    rgb = Image.open(out / "example_1_rgb.png")
    assert rgb.size == (64, 64)
    assert rgb.getpixel((0, 0)) == rgb.getpixel((63, 63)) == (10, 120, 30)
    assert Image.open(out / "example_1_b.png").getpixel((5, 5)) == WATER
    text = (out / "MANIFEST.json").read_text()
    assert "SECRET" not in text and "secretmodel" not in text and str(tmp_path) not in text
    manifest = json.loads(text)
    assert manifest["classes_present"] == {"vatten": 1.0}
    assert len(manifest["examples"][0]["a"]["input_sha256"]) == 64


def test_a_colour_outside_the_unified_palette_is_refused(tmp_path: Path) -> None:
    src = _tile(tmp_path)
    Image.new("RGB", (504, 504), (1, 2, 3)).save(src / "map.png")
    result = _run("--example", str(src / "rgb.png"), str(src / "map.png"),
                  "--map-names", "a", "--out-dir", str(tmp_path / "out"))
    assert result.returncode != 0 and "not in the unified palette" in result.stderr


def test_an_example_missing_a_map_is_refused(tmp_path: Path) -> None:
    src = _tile(tmp_path)
    result = _run("--example", str(src / "rgb.png"), "--map-names", "a",
                  "--out-dir", str(tmp_path / "out"))
    assert result.returncode != 0 and "expected RGB + 1 maps" in result.stderr
