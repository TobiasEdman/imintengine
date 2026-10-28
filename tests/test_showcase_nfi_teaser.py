"""The NFI-tab layers share the model window, draw honest scales and name no tile."""
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
FULL, WINDOW = 40, 32  # a 4 px ring outside the model window, like 512 → 504


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=REPO)


def _tile(tmp_path: Path, nmd_codes: tuple[int, int] = (111, 4233),
          map_colour: tuple[int, int, int] = WATER, drop: str | None = None,
          crops_in_parcels: bool = True) -> list[str]:
    src = tmp_path / "holdoutval_SECRET_tile"
    src.mkdir()
    rgb = Image.new("RGB", (FULL, FULL), (0, 0, 0))
    rgb.paste(Image.new("RGB", (WINDOW, WINDOW), GREEN), (4, 4))
    rgb.save(src / "rgb.png")
    raw = np.zeros((FULL, FULL), dtype=np.uint16)
    raw[4:36, 4:20], raw[4:36, 20:36] = nmd_codes
    # Training label: pine on the left, barley (an LPIS parcel) top right, a
    # clear-cut bottom right, and an unlabelled 4 px corner.
    label = np.zeros((FULL, FULL), dtype=np.uint8)
    label[4:36, 4:20], label[4:20, 20:36], label[20:36, 20:36] = 1, 12, 22
    label[4:8, 4:8] = 0
    parcels = np.zeros((FULL, FULL), dtype=np.float32)
    if crops_in_parcels:
        parcels[4:20, 20:36] = 2.5
    harvest = np.zeros((FULL, FULL), dtype=np.uint8)
    harvest[20:36, 20:36] = 1
    sidecar = {"nmd_label_raw": raw, "label": label, "parcel_area_ha": parcels, "harvest_mask": harvest}
    sidecar.pop(drop, None)
    np.savez(src / "nmd2023.npz", **sidecar)
    ramp = np.tile(np.linspace(0, 100, FULL, dtype=np.float32), (FULL, 1))
    height = ramp.copy()
    height[:, :20] = 0                                   # no trees in the left half
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(128, FULL, FULL)).astype(np.float16)
    embedding[:, 10:14, 10:14] = 0                       # an embedding gap
    layers = {"dem": ramp, "height": height, "vpp_length": ramp, "tessera": embedding}
    layers.pop(drop, None)
    np.savez(src / "tile.npz", **layers)
    Image.new("RGB", (WINDOW, WINDOW), map_colour).save(src / "secretmodel.png")
    return ["--rgb", str(src / "rgb.png"), "--nmd2023", str(src / "nmd2023.npz"),
            "--tile-npz", str(src / "tile.npz"),
            "--map", f"a={src / 'secretmodel.png'}", "--out-dir", str(tmp_path / "out")]


def test_layers_share_the_model_window_and_manifest_names_no_path(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path))
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    rgb = Image.open(out / "rgb.png")
    assert rgb.size == (WINDOW, WINDOW) and rgb.getpixel((0, 0)) == rgb.getpixel((31, 31)) == GREEN
    nmd = Image.open(out / "nmd2023.png").convert("RGB")
    assert nmd.getpixel((0, 0)) == (0, 100, 0)          # 111 tallskog
    assert nmd.getpixel((31, 31)) == (210, 180, 140)    # 4233 → 42 öppen mark med vegetation
    for name in ("dem", "height", "vpp_length", "tessera_embedding", "training_label", "lpis", "harvest"):
        assert Image.open(out / f"{name}.png").size == (WINDOW, WINDOW), name
    text = (out / "MANIFEST.json").read_text()
    assert "SECRET" not in text and "secretmodel" not in text and str(tmp_path) not in text
    manifest = json.loads(text)
    assert manifest["window_px"] == WINDOW
    assert manifest["map_classes_present"] == {"vatten": 1.0}
    assert set(manifest["nmd2023_display_classes"]) == {"forest_pine", "open_land_vegetated"}
    tile_digest = manifest["layers"]["dem"]["input_sha256"]
    assert {manifest["layers"][k]["input_sha256"]
            for k in ("dem", "height", "vpp_length", "tessera_embedding")} == {tile_digest}


def test_scalar_scales_round_outward_and_zero_means_transparent(tmp_path: Path) -> None:
    assert _run(*_tile(tmp_path)).returncode == 0
    out = tmp_path / "out"
    scales = json.loads((out / "MANIFEST.json").read_text())["background_scales"]
    dem = scales["dem"]
    assert dem["vmin"] % 10 == 0 and dem["vmax"] % 10 == 0 and dem["vmin"] < dem["vmax"]
    assert len(dem["legend"]) == 5 and dem["legend"][0]["label"] == f"{dem['vmin']:g} m"
    assert scales["height"]["zero_transparent"] and not dem["zero_transparent"]
    height = np.asarray(Image.open(out / "height.png"))
    assert height.shape[-1] == 4
    assert (height[:, :16, 3] == 0).all() and (height[:, 16:, 3] == 255).all()


def test_embedding_gap_is_transparent_and_rest_is_drawn(tmp_path: Path) -> None:
    assert _run(*_tile(tmp_path)).returncode == 0
    rgba = np.asarray(Image.open(tmp_path / "out" / "tessera_embedding.png"))
    gap = np.zeros(rgba.shape[:2], bool)
    gap[6:10, 6:10] = True                               # tile 10:14 minus the 4 px ring
    assert (rgba[gap, 3] == 0).all() and (rgba[~gap, 3] == 255).all()
    meta = json.loads((tmp_path / "out" / "MANIFEST.json").read_text())["embedding"]
    assert meta["channels"] == 128 and meta["valid_fraction"] == round(1 - 16 / WINDOW ** 2, 4)



def test_label_layers_show_only_their_own_pixels(tmp_path: Path) -> None:
    assert _run(*_tile(tmp_path)).returncode == 0
    out = tmp_path / "out"
    alpha = {n: np.asarray(Image.open(out / f"{n}.png"))[..., 3] > 0
             for n in ("training_label", "lpis", "harvest")}
    barley = np.zeros((WINDOW, WINDOW), bool)
    barley[0:16, 16:32] = True
    cut = np.zeros((WINDOW, WINDOW), bool)
    cut[16:32, 16:32] = True
    assert (alpha["lpis"] == barley).all() and (alpha["harvest"] == cut).all()
    assert not alpha["training_label"][0:4, 0:4].any() and alpha["training_label"][4:, 4:].all()
    assert tuple(np.asarray(Image.open(out / "harvest.png"))[20, 20, :3]) == (0, 206, 209)
    meta = json.loads((out / "MANIFEST.json").read_text())["label_layers"]
    assert set(meta["training_label"]["classes_present"]) == {"tallskog", "korn", "hygge"}
    assert set(meta["lpis"]["classes_present"]) == {"korn"}
    assert meta["harvest"]["fraction"] == 0.25


def test_crops_outside_lpis_parcels_are_refused_before_writing(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path, crops_in_parcels=False))
    assert result.returncode != 0 and "crop classes outside LPIS parcels" in result.stderr
    assert not (tmp_path / "out").exists()


def test_a_sidecar_missing_a_label_array_is_refused_before_writing(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path, drop="harvest_mask"))
    assert result.returncode != 0 and "label sidecar lacks arrays: ['harvest_mask']" in result.stderr
    assert not (tmp_path / "out").exists()

def test_a_tile_missing_a_background_layer_is_refused_before_writing(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path, drop="vpp_length"))
    assert result.returncode != 0 and "tile npz lacks layers: ['vpp_length']" in result.stderr
    assert not (tmp_path / "out").exists()


def test_an_nmd_code_without_a_display_mapping_is_refused_before_writing(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path, nmd_codes=(111, 999)))
    assert result.returncode != 0 and "without a display mapping: [999]" in result.stderr
    assert not (tmp_path / "out").exists()


def test_a_map_colour_outside_the_unified_palette_is_refused_before_writing(tmp_path: Path) -> None:
    result = _run(*_tile(tmp_path, map_colour=(1, 2, 3)))
    assert result.returncode != 0 and "not in the unified palette" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("maps, error", [
    (["rgb"], "reserved for a reference layer"),
    (["nmd2023"], "reserved for a reference layer"),
    (["height"], "reserved for a reference layer"),
    (["tessera_embedding"], "reserved for a reference layer"),
    (["lpis"], "reserved for a reference layer"),
    (["a", "a"], "may appear once"),
    (["../escaped"], "NAME of [a-z0-9_]"),
    (["Tessera"], "NAME of [a-z0-9_]"),
])
def test_a_bad_map_name_is_refused_before_any_output_changes(
        tmp_path: Path, maps: list[str], error: str) -> None:
    args = _tile(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    existing = {name: f"previous {name}".encode() for name in ("rgb.png", "nmd2023.png", "MANIFEST.json")}
    for name, content in existing.items():
        (out / name).write_bytes(content)
    map_png = args[args.index("--map") + 1].split("=", 1)[1]
    base = args[:args.index("--map")] + args[args.index("--map") + 2:]
    result = _run(*base, *[a for name in maps for a in ("--map", f"{name}={map_png}")])
    assert result.returncode != 0 and error in result.stderr
    assert {p.name: p.read_bytes() for p in out.iterdir()} == existing
    assert not (tmp_path / "escaped.png").exists()


def _legend(tab_data: str, key: str) -> list[tuple[str, str]]:
    """Coloured items of LEGENDS[key]: a literal block, or else its assignment line."""
    block = (re.search(rf"\n    {key}: \[(.*?)\n    \]", tab_data, re.S)
             or re.search(rf"\nLEGENDS\.{key} = (.*?);\n", tab_data))
    return re.findall(r"\{color:'([^']+)',label:'([^']+)'\}", block.group(1))


def test_published_tab_matches_its_manifest_and_names_no_tile() -> None:
    tab_data = (REPO / "docs/js/tab-data.js").read_text()
    index = (REPO / "docs/index.html").read_text()
    manifest_text = (REPO / "docs/showcase/nfi_validation/MANIFEST.json").read_text()
    manifest = json.loads(manifest_text)
    legend = {label.lower() for _, label in _legend(tab_data, "unified_nfi")}
    assert set(manifest["map_classes_present"]) <= legend
    for key, scale in manifest["background_scales"].items():
        assert _legend(tab_data, f"nfi_{key}") == [(s["color"], s["label"]) for s in scale["legend"]], key
    config = tab_data[tab_data.index("nfi_validation: {"):tab_data.index("marine_leisure: {")]
    images = re.findall(r"'(showcase/nfi_validation/[^']+)'", config)
    assert sorted(Path(p).name for p in images) == sorted(l["file"] for l in manifest["layers"].values())
    for src in images:
        assert (REPO / "docs" / src).is_file(), src
    # Every overlay that can sit on the NMD2023 background shows both legends.
    assert config.count("legend:['unified_nfi','nmd2023']") == 2
    for key in ("nfi_dem", "nfi_height", "nfi_vpp_length", "nfi_embedding",
                "nfi_lpis", "nfi_harvest", "nfi_label"):
        assert f"legend:['{key}','nmd2023']" in config, key
    labels = manifest["label_layers"]
    lpis_legend = {label.lower() for _, label in _legend(tab_data, "nfi_lpis")}
    assert lpis_legend == set(labels["lpis"]["classes_present"])
    assert set(labels["training_label"]["classes_present"]) <= legend
    for text in (tab_data, index, manifest_text):
        assert not re.search(r"holdoutval_|nfi20\d\d_", text)


def test_every_inspected_colour_has_a_legend_name() -> None:
    """Click-to-inspect names a pixel by exact colour; no opaque colour may be unnamed."""
    tab_data = (REPO / "docs/js/tab-data.js").read_text()
    config = tab_data[tab_data.index("nfi_validation: {"):tab_data.index("marine_leisure: {")]
    images = dict(re.findall(r"'(nfi-[a-z]+)':\s*'(showcase/nfi_validation/[^']+)'", config))
    specs = re.findall(r"\{label:'([^']+)',\s*image:'(nfi-[a-z]+)',\s*legend:'([a-z0-9_]+)'", config)
    assert len(specs) == 6
    for label, image, legend_key in specs:
        names = {colour.upper() for colour, _ in _legend(tab_data, legend_key)}
        rgba = np.asarray(Image.open(REPO / "docs" / images[image]).convert("RGBA")).reshape(-1, 4)
        opaque = {"#%02X%02X%02X" % tuple(c[:3]) for c in np.unique(rgba[rgba[:, 3] > 0], axis=0)}
        assert opaque <= names, (label, sorted(opaque - names))
