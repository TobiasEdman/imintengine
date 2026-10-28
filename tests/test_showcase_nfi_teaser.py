"""The NFI-tab layers share the model window and name no tile."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")  # save_nmd_overlay's module
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/showcase_nfi_teaser.py"
WATER = (0, 0, 255)  # unified class 10, vatten
GREEN = (10, 120, 30)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=REPO)


def _tile(tmp_path: Path, nmd_codes: tuple[int, int] = (111, 4233),
          map_colour: tuple[int, int, int] = WATER) -> list[str]:
    src = tmp_path / "holdoutval_SECRET_tile"
    src.mkdir()
    rgb = Image.new("RGB", (512, 512), (0, 0, 0))  # 4 px ring outside the model window
    rgb.paste(Image.new("RGB", (504, 504), GREEN), (4, 4))
    rgb.save(src / "rgb.png")
    raw = np.zeros((512, 512), dtype=np.uint16)
    raw[4:508, 4:256], raw[4:508, 256:508] = nmd_codes
    np.savez(src / "nmd2023.npz", nmd_label_raw=raw)
    Image.new("RGB", (504, 504), map_colour).save(src / "secretmodel.png")
    return ["--rgb", str(src / "rgb.png"), "--nmd2023", str(src / "nmd2023.npz"),
            "--map", f"a={src / 'secretmodel.png'}", "--out-dir", str(tmp_path / "out")]


def test_layers_share_the_model_window_and_manifest_names_no_path(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path))
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    rgb = Image.open(out / "rgb.png")
    assert rgb.size == (504, 504) and rgb.getpixel((0, 0)) == rgb.getpixel((503, 503)) == GREEN
    nmd = Image.open(out / "nmd2023.png").convert("RGB")
    assert nmd.size == (504, 504)
    assert nmd.getpixel((0, 0)) == (0, 100, 0)          # 111 tallskog
    assert nmd.getpixel((503, 503)) == (210, 180, 140)  # 4233 → 42 öppen mark med vegetation
    text = (out / "MANIFEST.json").read_text()
    assert "SECRET" not in text and "secretmodel" not in text and str(tmp_path) not in text
    manifest = json.loads(text)
    assert manifest["window_px"] == 504
    assert manifest["map_classes_present"] == {"vatten": 1.0}
    assert set(manifest["nmd2023_display_classes"]) == {"forest_pine", "open_land_vegetated"}
    assert len(manifest["layers"]["nmd2023"]["input_sha256"]) == 64


def test_an_nmd_code_without_a_display_mapping_is_refused(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path, nmd_codes=(111, 999)))
    assert result.returncode != 0 and "without a display mapping: [999]" in result.stderr


def test_a_map_colour_outside_the_unified_palette_is_refused(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path, map_colour=(1, 2, 3)))
    assert result.returncode != 0 and "not in the unified palette" in result.stderr


def test_published_tab_matches_its_manifest_and_names_no_tile() -> None:
    tab_data = (REPO / "docs/js/tab-data.js").read_text()
    index = (REPO / "docs/index.html").read_text()
    manifest_text = (REPO / "docs/showcase/nfi_validation/MANIFEST.json").read_text()
    manifest = json.loads(manifest_text)
    legend_block = re.search(r"unified_nfi: \[(.*?)\]", tab_data, re.S).group(1)
    legend = {label.lower() for label in re.findall(r"label:'([^']+)'", legend_block)}
    assert set(manifest["map_classes_present"]) <= legend
    config = tab_data[tab_data.index("nfi_validation: {"):tab_data.index("marine_leisure: {")]
    images = re.findall(r"'(showcase/nfi_validation/[^']+)'", config)
    assert len(images) == 4
    for src in images:
        assert (REPO / "docs" / src).is_file(), src
    for text in (tab_data, index, manifest_text):
        assert not re.search(r"holdoutval_|nfi20\d\d_", text)
