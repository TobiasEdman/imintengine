#!/usr/bin/env python3
"""Images for the NFI field-validation tab (TAB_CONFIG.nfi_validation).

One example tile: the Sentinel-2 RGB and one class map per --map, as rendered
by scripts/ladder_inference_matrix.py, plus NMD2023 from the tile's label
sidecar (key ``nmd_label_raw``). The maps cover the model's centred inference
window (504 px of a 512 px tile), so every layer is centre-cropped to that
window and written at native resolution: rgb.png, nmd2023.png and
<map name>.png, with MANIFEST.json, in --out-dir.

NMD2023 is drawn like every other NMD layer in the showcase: raw codes →
``nmd_raster_to_l2`` → ``save_nmd_overlay`` (the ``nmd`` legend). NMD2023 adds
codes that the level-2 table predates; NMD2023_DISPLAY_CODES folds them into
their NMD2018 counterparts for display only. An unknown code is refused.

The manifest binds every output to its input by sha256 and records no input
path: paths carry the tile name, which the public page must not reveal
(Tobias, 2026-10-28). The private experiments repository maps the digests to
the tile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip())
    return sha, dirty


def centre_crop(array: np.ndarray, side: int) -> np.ndarray:
    height, width = array.shape[:2]
    if height != width or (width - side) % 2:
        raise ValueError(f"cannot centre-crop {array.shape[:2]} to {side}: non-square or odd margin")
    margin = (width - side) // 2
    return array[margin:margin + side, margin:margin + side]


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


def write_layers(rgb_png: Path, sidecar: Path, maps: dict[str, Path], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_maps = {name: np.asarray(Image.open(p).convert("RGB")) for name, p in maps.items()}
    rgb = np.asarray(Image.open(rgb_png).convert("RGB"))
    with np.load(sidecar, allow_pickle=False) as z:
        nmd_raw = z["nmd_label_raw"]
    side = min(min(a.shape[:2]) for a in (rgb, nmd_raw, *model_maps.values()))

    layers = {}
    Image.fromarray(centre_crop(rgb, side)).save(out_dir / "rgb.png", optimize=True)
    layers["rgb"] = (out_dir / "rgb.png", rgb_png)
    l2 = nmd2023_to_display_l2(centre_crop(nmd_raw, side))
    save_nmd_overlay(l2, str(out_dir / "nmd2023.png"))
    layers["nmd2023"] = (out_dir / "nmd2023.png", sidecar)
    classes = Counter()
    for name, array in model_maps.items():
        cropped = centre_crop(array, side)
        classes += unified_class_counts(cropped)
        Image.fromarray(cropped).save(out_dir / f"{name}.png", optimize=True)
        layers[name] = (out_dir / f"{name}.png", maps[name])

    total = sum(classes.values())
    l2_ids, l2_counts = np.unique(l2, return_counts=True)
    return {
        "window_px": side,
        "layers": {kind: {"file": out.name, "sha256": sha256(out), "input_sha256": sha256(src)}
                   for kind, (out, src) in layers.items()},
        "nmd2023_display_classes": {("unclassified" if i == 0 else L2_CLASSES[i - 1]):
                                    round(n / l2.size, 4)
                                    for i, n in zip(l2_ids.tolist(), l2_counts.tolist())},
        "map_classes_present": {UNIFIED_CLASSES[c]: round(n / total, 4)
                                for c, n in sorted(classes.items())},
    }


def parse_map(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rgb", type=Path, required=True, help="Sentinel-2 RGB PNG of the tile")
    ap.add_argument("--nmd2023", type=Path, required=True,
                    help="the tile's NMD2023 label sidecar (.npz with nmd_label_raw)")
    ap.add_argument("--map", type=parse_map, action="append", required=True, dest="maps",
                    metavar="NAME=PNG", help="class map of one model cell, e.g. tessera_r4=...")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    sha, dirty = git_state(root)  # before any output exists in the tree
    record = write_layers(args.rgb, args.nmd2023, dict(args.maps), args.out_dir)
    manifest = {
        "description": ("Layers for the NFI field-validation tab: Sentinel-2 summer RGB, NMD2023 "
                        "and one class map per model cell for one example tile. The tile is not "
                        "in the NFI evaluation set and is withheld by name; inputs are bound by "
                        "sha256."),
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
