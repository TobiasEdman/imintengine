#!/usr/bin/env python3
"""Thumbnails for the NFI field-validation teaser (docs/nfi-utvardering.html).

Each --example takes a Sentinel-2 RGB PNG followed by one class-map PNG per
--map-names entry, as rendered by scripts/ladder_inference_matrix.py. The maps
cover the model's centred inference window (504 px of a 512 px tile), so every
image is centre-cropped to the smallest side in its example before resizing
(bilinear for RGB, nearest for class maps). Outputs are example_<n>_rgb.png and
example_<n>_<map name>.png plus MANIFEST.json in --out-dir.

The manifest binds every output to its input by sha256 and records no input
path: paths carry the tile name, which the public page must not reveal
(Tobias, 2026-10-28). The private experiments repository maps the digests to
tiles. It also lists the unified classes present in the maps, so the page
legend can be checked against them.
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

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.schema.unified_schema import UNIFIED_CLASSES, UNIFIED_COLORS  # noqa: E402

COLOR_TO_CLASS = {rgb: cls for cls, rgb in UNIFIED_COLORS.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip())
    return sha, dirty


def centre_crop(image: Image.Image, side: int) -> Image.Image:
    width, height = image.size
    if width != height or (width - side) % 2:
        raise ValueError(f"cannot centre-crop {image.size} to {side}: non-square or odd margin")
    margin = (width - side) // 2
    return image.crop((margin, margin, margin + side, margin + side))


def class_counts(image: Image.Image) -> Counter:
    counts = Counter()
    for count, rgb in image.getcolors(maxcolors=256 ** 3):
        if rgb not in COLOR_TO_CLASS:
            raise ValueError(f"map colour {rgb} is not in the unified palette")
        counts[COLOR_TO_CLASS[rgb]] += count
    return counts


def write_examples(examples: list[list[Path]], map_names: list[str], out_dir: Path,
                   size: int) -> tuple[list[dict], Counter]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records, classes = [], Counter()
    for n, paths in enumerate(examples, start=1):
        if len(paths) != 1 + len(map_names):
            raise ValueError(f"example {n}: expected RGB + {len(map_names)} maps, got {len(paths)} files")
        images = [Image.open(p).convert("RGB") for p in paths]
        side = min(min(im.size) for im in images)
        record = {}
        for kind, src, image in zip(["rgb", *map_names], paths, images):
            resample = Image.BILINEAR if kind == "rgb" else Image.NEAREST
            out = centre_crop(image, side).resize((size, size), resample)
            if kind != "rgb":
                classes += class_counts(out)
            dest = out_dir / f"example_{n}_{kind}.png"
            out.save(dest, optimize=True)
            record[kind] = {"file": dest.name, "sha256": sha256(dest), "input_sha256": sha256(src)}
        records.append(record)
    return records, classes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example", nargs="+", action="append", type=Path, required=True,
                    metavar="PNG", help="RGB PNG, then one class-map PNG per --map-names entry")
    ap.add_argument("--map-names", nargs="+", required=True,
                    help="model cell of each map column, e.g. prithvi600m_r4 tessera_r4")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    sha, dirty = git_state(root)  # before any output exists in the tree
    records, classes = write_examples(args.example, args.map_names, args.out_dir, args.size)
    total = sum(classes.values())
    manifest = {
        "description": ("Thumbnails for the NFI field-validation teaser: a Sentinel-2 summer RGB "
                        "and one class map per model cell for each example tile. Tiles are not "
                        "in the NFI evaluation set and are withheld by name; inputs are bound "
                        "by sha256."),
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by": "scripts/showcase_nfi_teaser.py",
        "git_sha": sha, "git_dirty": dirty,
        "run_args": {"examples": len(records), "map_names": args.map_names, "size": args.size,
                     "crop": "centre crop to the smallest side per example",
                     "resample": {"rgb": "bilinear", "map": "nearest"}},
        "examples": records,
        "classes_present": {UNIFIED_CLASSES[c]: round(n / total, 4)
                            for c, n in sorted(classes.items())},
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "examples": len(records),
                      "classes": len(classes)}))


if __name__ == "__main__":
    main()
