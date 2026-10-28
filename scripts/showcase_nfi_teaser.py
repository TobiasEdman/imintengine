#!/usr/bin/env python3
"""Images for the NFI field-validation tab (TAB_CONFIG.nfi_validation).

One example tile: the Sentinel-2 RGB and one class map per --map, as rendered
by scripts/ladder_inference_matrix.py, plus NMD2023 from the tile's label
sidecar (key ``nmd_label_raw``) and background layers from the tile itself
(--tile-npz): elevation, tree height, VPP season length and the TESSERA
embedding. The maps cover the model's centred inference window (504 px of a
512 px tile), so every layer is centre-cropped to that window and written at
native resolution into --out-dir, with MANIFEST.json.

NMD2023 is drawn like every other NMD layer in the showcase: raw codes →
``nmd_raster_to_l2`` → ``save_nmd_overlay`` (the ``nmd`` legend). NMD2023 adds
codes that the level-2 table predates; NMD2023_DISPLAY_CODES folds them into
their NMD2018 counterparts for display only. An unknown code is refused.

Scalar layers get a colour scale from the 2nd-98th percentile of their valid
pixels, rounded outward to tens; the scale and its legend stops are recorded
in the manifest so the page legend can be checked against them. Where 0 means
"nothing here" (no trees, no growing season) the pixel is transparent. The
embedding is shown as its first three principal components (RGB).

The manifest binds every output to its input by sha256 and records no input
path: paths carry the tile name, which the public page must not reveal
(Tobias, 2026-10-28). The script cannot see which tile it was given, so it
claims nothing about it; a private provenance record, kept outside this
repository, maps the digests to the tile and records the tile's check
against the frozen NFI evaluation set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.analyzers.nmd import L2_CLASSES, NMD_LEVEL2, nmd_raster_to_l2  # noqa: E402
from imint.exporters.export import save_nmd_overlay  # noqa: E402
from imint.schema.unified_schema import UNIFIED_CLASSES, UNIFIED_COLORS  # noqa: E402

COLOR_TO_CLASS = {rgb: cls for cls, rgb in UNIFIED_COLORS.items()}

#: NMD2023 basskikt v2.1 codes absent from the NMD2018 level-2 table → the
#: NMD2018 code drawn in their place.
NMD2023_DISPLAY_CODES = {
    **{c: 2 for c in (200, *range(211, 219), *range(221, 229))},   # öppen våtmark, finindelad
    **{c: 41 for c in (411, 412, 413)},                             # utan vegetation, glaciär, snöfält
    **{c: 42 for c in (421, 422, 423, 4211, 4212, 4213, 4221, 4222, 4223,
                       4231, 4232, 4233)},                          # busk-, ris-, gräsdominerad
    43: 115, 23: 125, 230: 125,                                      # låg fjällskog → lövskog
    54: 52,                                                          # torvtäkt → anlagd mark
}
NMD2018_CODES = set().union(*NMD_LEVEL2.values())

#: Scalar background layers from the tile npz: key → (colormap, unit, whether
#: 0 means "nothing here" and is drawn transparent).
SCALAR_LAYERS = {
    "dem": ("YlOrBr", "m", False),
    "height": ("YlGn", "m", True),
    "vpp_length": ("viridis", "dagar", True),
}
EMBEDDING_KEY, EMBEDDING_LAYER = "tessera", "tessera_embedding"
LEGEND_STOPS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip())
    return sha, dirty


def centre_crop(array: np.ndarray, side: int, *, channels_first: bool = False) -> np.ndarray:
    """Centre-crop the spatial axes to ``side``: (H, W[, C]), or (C, H, W) if channels_first."""
    height, width = array.shape[-2:] if channels_first else array.shape[:2]
    if height != width or (width - side) % 2:
        raise ValueError(f"cannot centre-crop {(height, width)} to {side}: non-square or odd margin")
    m = (width - side) // 2
    if channels_first:
        return array[..., m:m + side, m:m + side]
    return array[m:m + side, m:m + side]


def nmd2023_to_display_l2(raw: np.ndarray) -> np.ndarray:
    unknown = set(np.unique(raw).tolist()) - NMD2018_CODES - set(NMD2023_DISPLAY_CODES) - {0}
    if unknown:
        raise ValueError(f"NMD2023 codes without a display mapping: {sorted(unknown)}")
    folded = raw.copy()
    for code, shown in NMD2023_DISPLAY_CODES.items():
        folded[raw == code] = shown
    return nmd_raster_to_l2(folded)


def unified_class_counts(rgb: np.ndarray) -> Counter:
    colours, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    result = Counter()
    for colour, count in zip(map(tuple, colours.tolist()), counts.tolist()):
        if colour not in COLOR_TO_CLASS:
            raise ValueError(f"map colour {colour} is not in the unified palette")
        result[COLOR_TO_CLASS[colour]] += count
    return result


def hex_colour(rgba: tuple) -> str:
    return "#%02X%02X%02X" % tuple(int(round(c * 255)) for c in rgba[:3])


def render_scalar(name: str, values: np.ndarray, colormap: str, unit: str,
                  zero_is_empty: bool) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(values) & ((values != 0) if zero_is_empty else True)
    if not valid.any():
        raise ValueError(f"layer {name!r} has no valid pixels")
    lo, hi = np.percentile(values[valid], [2, 98])
    vmin, vmax = float(np.floor(lo / 10) * 10), float(np.ceil(hi / 10) * 10)
    if vmax <= vmin:
        raise ValueError(f"layer {name!r} is constant; no colour scale")
    cmap = colormaps[colormap]
    rgba = cmap(np.clip((np.nan_to_num(values) - vmin) / (vmax - vmin), 0, 1))
    rgba[..., 3] = valid
    stops = np.linspace(vmin, vmax, LEGEND_STOPS)
    scale = {"colormap": colormap, "unit": unit, "vmin": vmin, "vmax": vmax,
             "zero_transparent": zero_is_empty, "valid_fraction": round(float(valid.mean()), 4),
             "legend": [{"color": hex_colour(cmap((s - vmin) / (vmax - vmin))),
                         "label": f"{s:g} {unit}"} for s in stops]}
    return (rgba * 255).round().astype(np.uint8), scale


def render_embedding(embedding: np.ndarray) -> tuple[np.ndarray, dict]:
    """First three principal components of a (C, H, W) embedding as RGBA."""
    channels, height, width = embedding.shape
    flat = embedding.reshape(channels, -1).T.astype(np.float32)
    valid = np.any(flat != 0, axis=1)
    if valid.sum() < 3:
        raise ValueError("embedding has fewer than three non-zero pixels")
    centred = flat[valid] - flat[valid].mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = vt[:3] * np.sign(vt[np.arange(3), np.abs(vt[:3]).argmax(axis=1)])[:, None]
    comps = centred @ axes.T                       # sign fixed: largest loading positive
    lo, hi = np.percentile(comps, [2, 98], axis=0)
    rgba = np.zeros((height * width, 4), dtype=np.float32)
    rgba[valid, :3] = np.clip((comps - lo) / (hi - lo), 0, 1)
    rgba[valid, 3] = 1
    meta = {"method": "PCA over valid pixels; components 1-3 → R, G, B; 2-98 % stretch",
            "channels": channels, "valid_fraction": round(float(valid.mean()), 4)}
    return (rgba.reshape(height, width, 4) * 255).round().astype(np.uint8), meta


def write_layers(rgb_png: Path, sidecar: Path, tile_npz: Path, maps: dict[str, Path],
                 out_dir: Path) -> dict:
    model_maps = {name: np.asarray(Image.open(p).convert("RGB")) for name, p in maps.items()}
    rgb = np.asarray(Image.open(rgb_png).convert("RGB"))
    with np.load(sidecar, allow_pickle=False) as z:
        nmd_raw = z["nmd_label_raw"]
    with np.load(tile_npz, allow_pickle=False) as z:
        missing = [k for k in (*SCALAR_LAYERS, EMBEDDING_KEY) if k not in z.files]
        if missing:
            raise ValueError(f"tile npz lacks layers: {missing}")
        scalars = {k: z[k].astype(np.float32) for k in SCALAR_LAYERS}
        embedding = z[EMBEDDING_KEY]
    side = min(min(a.shape[:2]) for a in (rgb, nmd_raw, *model_maps.values()))

    # Crop, validate and render every layer before --out-dir is touched, so a
    # refused run (odd margin, unknown NMD code, off-palette map, empty or
    # constant layer) writes nothing.
    rgb_crop = centre_crop(rgb, side)
    l2 = nmd2023_to_display_l2(centre_crop(nmd_raw, side))
    map_crops = {name: centre_crop(array, side) for name, array in model_maps.items()}
    classes = sum((unified_class_counts(crop) for crop in map_crops.values()), Counter())
    background, scales = {}, {}
    for key, (colormap, unit, zero_is_empty) in SCALAR_LAYERS.items():
        background[key], scales[key] = render_scalar(
            key, centre_crop(scalars[key], side), colormap, unit, zero_is_empty)
    background[EMBEDDING_LAYER], embedding_meta = render_embedding(
        centre_crop(embedding, side, channels_first=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_crop).save(out_dir / "rgb.png", optimize=True)
    save_nmd_overlay(l2, str(out_dir / "nmd2023.png"))
    layers = {"rgb": (out_dir / "rgb.png", rgb_png), "nmd2023": (out_dir / "nmd2023.png", sidecar)}
    for name, crop in map_crops.items():
        Image.fromarray(crop).save(out_dir / f"{name}.png", optimize=True)
        layers[name] = (out_dir / f"{name}.png", maps[name])
    for name, rgba in background.items():
        Image.fromarray(rgba, mode="RGBA").save(out_dir / f"{name}.png", optimize=True)
        layers[name] = (out_dir / f"{name}.png", tile_npz)

    total = sum(classes.values())
    l2_ids, l2_counts = np.unique(l2, return_counts=True)
    return {
        "window_px": side,
        "layers": {kind: {"file": out.name, "sha256": sha256(out), "input_sha256": sha256(src)}
                   for kind, (out, src) in layers.items()},
        "background_scales": scales,
        "embedding": embedding_meta,
        "nmd2023_display_classes": {("unclassified" if i == 0 else L2_CLASSES[i - 1]):
                                    round(n / l2.size, 4)
                                    for i, n in zip(l2_ids.tolist(), l2_counts.tolist())},
        "map_classes_present": {UNIFIED_CLASSES[c]: round(n / total, 4)
                                for c, n in sorted(classes.items())},
    }


#: Output stems the reference and background layers own; a model map may not take them.
RESERVED_LAYER_NAMES = frozenset({"rgb", "nmd2023", *SCALAR_LAYERS, EMBEDDING_LAYER})
#: A map name becomes a file stem in --out-dir, so no separators or dots.
MAP_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")


def parse_map(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not path or not MAP_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"expected NAME=PATH with NAME of [a-z0-9_], got {value!r}")
    if name in RESERVED_LAYER_NAMES:
        raise argparse.ArgumentTypeError(f"map name {name!r} is reserved for a reference layer")
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rgb", type=Path, required=True, help="Sentinel-2 RGB PNG of the tile")
    ap.add_argument("--nmd2023", type=Path, required=True,
                    help="the tile's NMD2023 label sidecar (.npz with nmd_label_raw)")
    ap.add_argument("--tile-npz", type=Path, required=True,
                    help="the tile itself (.npz with dem, height, vpp_length and tessera)")
    ap.add_argument("--map", type=parse_map, action="append", required=True, dest="maps",
                    metavar="NAME=PNG", help="class map of one model cell, e.g. tessera_r4=...")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    maps = dict(args.maps)
    if len(maps) != len(args.maps):
        ap.error("each --map name may appear once")
    root = Path(__file__).resolve().parents[1]
    sha, dirty = git_state(root)  # before any output exists in the tree
    record = write_layers(args.rgb, args.nmd2023, args.tile_npz, maps, args.out_dir)
    manifest = {
        "description": ("Layers for the NFI field-validation tab for one example tile, withheld "
                        "by name: Sentinel-2 summer RGB, NMD2023, one class map per model cell, "
                        "and background layers from the tile (elevation, tree height, VPP "
                        "season length, TESSERA embedding). Inputs are bound by sha256; a private "
                        "provenance record maps them to the tile and to its check against the "
                        "frozen NFI evaluation set."),
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by": "scripts/showcase_nfi_teaser.py",
        "git_sha": sha, "git_dirty": dirty,
        "run_args": {"maps": [name for name, _ in args.maps],
                     "crop": "centre crop to the model window", "resample": "none"},
        **record,
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "window_px": record["window_px"],
                      "map_classes": len(record["map_classes_present"])}))


if __name__ == "__main__":
    main()
