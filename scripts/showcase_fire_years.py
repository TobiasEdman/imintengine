#!/usr/bin/env python3
"""Land cover year by year and NMD 2018/2023 for the fire tab (docs/showcase/fire/).

The fire tab (``TAB_CONFIG.fire`` in docs/js/tab-data.js) draws every layer
on the AOI grid of config/lclu/fire_karbole.json: 267x559 px at 10 m in
EPSG:3006. This script writes three kinds of layer there:

- ``lulc_year_<year>.png``: the TESSERA R4 map of each year of one
  scripts/render_fire_tessera_years.py run (task 0151), copied byte for byte
  once its sha256 matches the run's MANIFEST. The run's ``legend.json`` names
  the colours. A year whose HR-VPP comes from another year is labelled with
  the service's wording, e.g. "2025 · växtfenologi från 2024".
- ``nmd_2018.png``: NMD 2018 basskikt, the baseline before the fire, drawn
  from raw codes in Naturvårdsverket's own classes and colours: the style
  shipped with the raster, pinned in docs/verification/nmd2018/. The
  engine's NMD 2018 level-2 table is not used; it folds 113 barrblandskog
  into lövskog.
- ``nmd_2023.png``: NMD 2023 basskikt v2.1 with Naturvårdsverket's own classes
  and colours (``NMD2023_LEGEND``, palette ``nmd2023_l2``).

Both NMD windows are read from the national rasters on their own 10 m
lattice, never resampled. A code outside the product's classes is refused.
Level-1 shares come from the engine's level-1 tables. Everything is
validated before the first file is written.

MANIFEST.json binds each file by sha256 to its source: the TESSERA run (id,
git sha, image digest, the run MANIFEST's sha256) or the NMD raster (file
name, size, sha256) and the window read from it. It lists, per layer, the
classes present with the colour and name the tab's legend must show, and the
NMD code counts and level-1 shares the tab's text quotes. No input path is
recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.analyzers.nmd import NMD2023_LEGEND, nmd_legend  # noqa: E402
from imint.exporters.export import save_nmd_overlay  # noqa: E402
from imint.fetch import read_nmd2023_window  # noqa: E402
from imint.schema.nmd2023 import (  # noqa: E402
    NMD2023_CLASSES, NMD2023_OUTSIDE, NMD2023_OUTSIDE_NAME, NMD2023_QML_SHA256)

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "config/lclu/fire_karbole.json"
OUT_DIR = ROOT / "docs/showcase/fire"
RES = 10

#: Naturvårdsverket's style for NMD 2018 basskikt, from the official archive;
#: docs/verification/nmd2018/README.md records the source and retrieval.
NMD2018_QML = ROOT / "docs/verification/nmd2018/nmd2018bas_ogeneraliserad_v1_1.qml"
NMD2018_QML_SHA256 = "8faa67351c71a641a7ee2f4e4d868c076e44e1810b9dfeec99241e3769752dfd"

#: The two NMD layers: product, raster dtype and the product as Naturvårdsverket names it.
NMD_LAYERS = {
    "nmd_2018": (2018, "uint8", "NMD 2018 basskikt, ogeneraliserad, v1.1"),
    "nmd_2023": (2023, "uint16", "NMD 2023 basskikt v2.1"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def git_state(root: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip())
    return sha, dirty


def aoi_bbox(aoi: dict) -> dict:
    west, south, east, north = aoi["bounds_3006"]
    if aoi["crs"] != "EPSG:3006" or any(v % RES for v in (west, south, east, north)):
        raise ValueError(f"AOI must be EPSG:3006 on the {RES} m lattice: {aoi}")
    if list(aoi["shape"]) != [(north - south) // RES, (east - west) // RES]:
        raise ValueError(f"AOI shape {aoi['shape']} does not match its bounds at {RES} m")
    return {"west": west, "south": south, "east": east, "north": north}


def read_nmd2018_window(path: Path, bbox: dict):
    """Read ``bbox`` from the NMD 2018 raster, on its 10 m lattice.

    ``imint.fetch.read_nmd2023_window`` does this for NMD 2023 only (it
    requires uint16), and ``fetch_nmd_label_local`` reads square tiles only,
    so the 267x559 NMD 2018 window is read here. The window must lie inside
    the raster: NMD 2018 has no "outside coverage" value to fill with.

    Returns:
        ``(raster, transform)`` of the window.
    """
    import rasterio
    from rasterio.windows import Window, from_bounds

    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 3006 or src.dtypes[0] != "uint8":
            raise ValueError(f"{path.name}: expected EPSG:3006 uint8, got {src.crs} {src.dtypes[0]}")
        win = from_bounds(bbox["west"], bbox["south"], bbox["east"], bbox["north"], src.transform)
        parts = (win.col_off, win.row_off, win.width, win.height)
        if any(abs(v - round(v)) > 1e-6 for v in parts):
            raise ValueError(f"{path.name}: bbox {bbox} is off the raster lattice (window {parts})")
        window = Window(*(round(v) for v in parts))
        if (window.col_off < 0 or window.row_off < 0 or window.col_off + window.width > src.width
                or window.row_off + window.height > src.height):
            raise ValueError(f"{path.name}: bbox {bbox} reaches outside the raster")
        return src.read(1, window=window), src.window_transform(window)


def opaque_colours(rgba: np.ndarray) -> set[str]:
    pixels = rgba.reshape(-1, rgba.shape[-1])
    if rgba.shape[-1] == 4:
        pixels = pixels[pixels[:, 3] > 0]
    return {"#%02x%02x%02x" % tuple(c[:3]) for c in np.unique(pixels[:, :3], axis=0)}


# ── TESSERA R4 years ─────────────────────────────────────────────────────────

def plan_tessera(run: Path, aoi: dict) -> dict:
    """Check the run against its MANIFEST and the AOI; return the copy plan and records."""
    manifest_path = run / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["aoi"]["bounds_3006"] != aoi["bounds_3006"] or \
            manifest["aoi"]["shape"] != aoi["shape"]:
        raise ValueError(f"run AOI {manifest['aoi']} is not the fire AOI {aoi}")
    if identity(run / "legend.json") != {k: manifest["legend"][k] for k in ("sha256", "size")}:
        raise ValueError("legend.json differs from the run MANIFEST")
    by_colour = {entry["color"]: entry for entry in json.loads((run / "legend.json").read_text())
                 if entry["color"]}
    substitutions = {s["year"]: s for s in manifest["vpp_year_substitutions"]}
    height, width = aoi["shape"]
    copies, years = [], {}
    for year_key, record in sorted(manifest["years"].items()):
        year = int(year_key)
        source = run / year_key / "classification.png"
        expected = record["outputs"]["classification.png"]
        if identity(source) != {k: expected[k] for k in ("sha256", "size")}:
            raise ValueError(f"{year}: classification.png differs from the run MANIFEST")
        if record["vpp_year_substituted"] != (year in substitutions):
            raise ValueError(f"{year}: HR-VPP substitution disagrees with vpp_year_substitutions")
        with Image.open(source) as image:
            if image.size != (width, height) or image.mode != "RGBA":
                raise ValueError(f"{year}: expected a {width}x{height} RGBA PNG, got "
                                 f"{image.size} {image.mode}")
            colours = opaque_colours(np.asarray(image))
        unnamed = sorted(colours - set(by_colour))
        if unnamed:
            raise ValueError(f"{year}: colours without a legend entry: {unnamed}")
        present = sorted((by_colour[c] for c in colours), key=lambda e: e["id"])
        if [e["id"] for e in present] != sorted(int(k) for k in record["class_counts"]):
            raise ValueError(f"{year}: PNG classes differ from the run's class_counts")
        name = f"lulc_year_{year}.png"
        copies.append((source, name))
        years[year_key] = {
            "file": name,
            "label": f"{year} · {substitutions[year]['label']}" if year in substitutions else str(year),
            "tessera_year": record["tessera_year"],
            "vpp_year": record["vpp_year"],
            "vpp_year_substituted": record["vpp_year_substituted"],
            **({"vpp_note": record["note"]} if "note" in record else {}),
            "legend": [{"id": e["id"], "color": e["color"], "name": e["name"]} for e in present],
        }
    source = {
        "run_id": manifest["run_args"]["run_id"],
        "output_location": manifest["run_args"]["output_location"],
        "produced_by": "scripts/render_fire_tessera_years.py render (k8s/fire-tessera-render-job.yaml)",
        "produced_at": manifest["produced_at"],
        "git_sha": manifest["git_sha"],
        "image": manifest["image"],
        "image_digest": manifest["image_digest"],
        "manifest": identity(manifest_path),
        "legend": {k: manifest["legend"][k] for k in ("sha256", "size")},
        "model": manifest["model"]["name"],
        "checkpoint": manifest["model"]["checkpoint"],
    }
    return {"copies": copies, "record": {
        "source": source,
        "historical_note": manifest["historical_note"],
        "vpp_year_substitutions": [{k: s[k] for k in ("year", "vpp_year", "label")}
                                   for s in manifest["vpp_year_substitutions"]],
        "years": years,
    }}


# ── NMD 2018 and 2023 ────────────────────────────────────────────────────────

def nmd2018_classes() -> dict[int, tuple[str, str]]:
    """code → (name, colour) of every named class in the pinned NMD 2018 style, in its order.

    The style lists all 256 raster values; the classes are the entries whose
    label carries a name after the code. The others are not classes.
    """
    if sha256_file(NMD2018_QML) != NMD2018_QML_SHA256:
        raise ValueError(f"{NMD2018_QML.name} is not the pinned official style")
    classes = {}
    for entry in ET.parse(NMD2018_QML).iter("paletteEntry"):
        code, _, name = entry.attrib["label"].partition(" ")
        if int(code) != int(entry.attrib["value"]):
            raise ValueError(f"{NMD2018_QML.name}: label {entry.attrib['label']!r} "
                             f"does not start with its value {entry.attrib['value']}")
        if name.strip():
            classes[int(code)] = (name.strip(), entry.attrib["color"].lower())
    return classes


def plan_nmd(name: str, raster_path: Path, aoi: dict) -> dict:
    product, dtype, title = NMD_LAYERS[name]
    bbox = aoi_bbox(aoi)
    if product == 2023:
        raw, transform, _crs = read_nmd2023_window(bbox, str(raster_path))
    else:
        raw, transform = read_nmd2018_window(raster_path, bbox)
    if raw.shape != tuple(aoi["shape"]) or list(transform)[:6] != list(aoi["transform"]):
        raise ValueError(f"{name}: window {raw.shape} {list(transform)[:6]} is not the AOI grid")
    if raw.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype} codes, got {raw.dtype}")
    codes, counts = np.unique(raw, return_counts=True)
    present = dict(zip(codes.tolist(), counts.tolist()))

    if product == 2023:
        NMD2023_LEGEND.check(raw)  # refuses any code outside Naturvårdsverket's table
        classes, qml_sha256 = NMD2023_CLASSES, NMD2023_QML_SHA256
        l2 = NMD2023_LEGEND.l2_raster(raw)
        palette = "nmd2023_l2"
        draw = lambda path: save_nmd_overlay(l2, str(path), palette=palette)  # noqa: E731
        drawing = ("level 2 of imint.analyzers.nmd.NMD2023_LEGEND, "
                   f"imint.exporters.export.save_nmd_overlay(palette={palette!r})")
    else:
        classes, qml_sha256 = nmd2018_classes(), NMD2018_QML_SHA256
        unknown = sorted(set(present) - set(classes))
        if unknown:
            raise ValueError(f"NMD 2018 window holds codes without a class in "
                             f"{NMD2018_QML.name}: {unknown}")
        lut = np.zeros((256, 3), dtype=np.uint8)
        for code, (_name, colour) in classes.items():
            lut[code] = list(bytes.fromhex(colour[1:]))
        draw = lambda path: Image.fromarray(lut[raw]).save(path)  # noqa: E731
        drawing = f"raw codes in the colours of {NMD2018_QML.relative_to(ROOT)}"
    legend = [{"key": "outside", "color": "#808080", "name": NMD2023_OUTSIDE_NAME,
               "pixels": present[NMD2023_OUTSIDE]}
              ] if product == 2023 and NMD2023_OUTSIDE in present else []
    legend += [{"key": str(code), "color": colour, "name": class_name, "pixels": present[code]}
               for code, (class_name, colour) in classes.items() if code in present]

    tables = nmd_legend(product)  # level 1 only: 2018 groups, 2023 klassgrupper
    total = raw.size
    l1_keys = list(tables.level1)
    l1_counts = np.bincount(tables.l1_raster(raw).ravel(), minlength=len(l1_keys) + 1)
    if l1_counts[0] and product == 2018:
        raise ValueError("NMD 2018 window holds pixels outside the level-1 table")
    level1 = {key: {"pixels": int(n), "fraction": round(int(n) / total, 4)}
              for key, n in zip([tables.unmapped_key, *l1_keys], l1_counts.tolist()) if n}
    return {
        "draw": draw,
        "record": {
            "file": f"{name}.png",
            "product": title,
            "raster": {"file": raster_path.name, **identity(raster_path)},
            "window": {"dtype": dtype,
                       "sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()},
            "qml_sha256": qml_sha256,
            "drawing": drawing,
            "pixels": int(total),
            "codes": {str(code): int(n) for code, n in present.items()},
            "level1": level1,
            "legend": legend,
        },
    }


# ── Output ───────────────────────────────────────────────────────────────────

def write_outputs(out_dir: Path, tessera: dict, nmd: dict) -> dict[str, dict]:
    """Write every layer, reopen it and return its identity, keyed by file name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for source, name in tessera["copies"]:
        shutil.copyfile(source, out_dir / name)
        if sha256_file(out_dir / name) != sha256_file(source):
            raise RuntimeError(f"{name}: copy differs from its source")
        written[name] = identity(out_dir / name)
    for layer in nmd.values():
        path = out_dir / layer["record"]["file"]
        layer["draw"](path)
        with Image.open(path) as image:
            shown = opaque_colours(np.asarray(image))
        named = {entry["color"] for entry in layer["record"]["legend"]}
        if shown != named:
            raise RuntimeError(f"{path.name}: drawn colours {sorted(shown)} != legend {sorted(named)}")
        written[path.name] = identity(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tessera-run", type=Path, required=True,
                    help="output directory of one render_fire_tessera_years.py run "
                         "(MANIFEST.json, legend.json, <year>/classification.png)")
    ap.add_argument("--nmd2018-raster", type=Path, required=True,
                    help="nmd2018bas_ogeneraliserad_v1_1.tif")
    ap.add_argument("--nmd2023-raster", type=Path, required=True, help="NMD2023bas_v2_1.tif")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    sha, dirty = git_state(ROOT)  # before any output exists in the tree
    aoi = json.loads(SCOPE.read_text())["aoi"]
    tessera = plan_tessera(args.tessera_run, aoi)
    nmd = {"nmd_2018": plan_nmd("nmd_2018", args.nmd2018_raster, aoi),
           "nmd_2023": plan_nmd("nmd_2023", args.nmd2023_raster, aoi)}

    written = write_outputs(args.out_dir, tessera, nmd)
    record = tessera["record"]
    for year in record["years"].values():
        year.update(written[year["file"]])
    manifest = {
        "description": ("Fire tab (Kårbölebranden) layers on the AOI grid: TESSERA R4 land cover "
                        "per year, copied from one render run, and NMD 2018 (before the fire) "
                        "and NMD 2023 drawn from the national rasters. Files written by "
                        "scripts/showcase_fire_years.py; the tab's other images predate it."),
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by": "scripts/showcase_fire_years.py",
        "git_sha": sha, "git_dirty": dirty,
        "aoi": {key: aoi[key] for key in ("crs", "bounds_3006", "shape", "transform")},
        "tessera_r4": record,
        **{name: {**layer["record"], **written[layer["record"]["file"]]}
           for name, layer in nmd.items()},
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "files": sorted(written),
                      "years": sorted(record["years"])}))


if __name__ == "__main__":
    main()
