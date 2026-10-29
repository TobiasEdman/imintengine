"""Annual TESSERA R4 serving, independent of exact-day imagery acquisition.

Prepared inputs are immutable, identity-pinned canonical 512px NPZ bundles.
Only public-area bundles belong in a serving catalogue; never point this at a
training or held-out evaluation directory. See scripts/prepare_lclu.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable

import numpy as np
from affine import Affine
from pyproj import Transformer
import rasterio
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from imint.job import GeoContext
from imint.schema.unified_schema import UNIFIED_CLASSES, UNIFIED_COLORS
from imint.training.tile_config import TileConfig
from scripts.cache_predict import atomic_write, contract_pin, file_identity

YEARS = tuple(range(2018, 2026))
SOURCE_SIZE, OUTPUT_SIZE, CROP, NODATA = 512, 504, 4, 255
EMBEDDING = {"producer": "geotessera", "producer_version": "0.10.2",
             "dataset": "v1", "variant": "vultr"}
AUX_NAMES = ["height", "volume", "basal_area", "diameter", "dem", "vpp_sosd",
             "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv", "markfukt"]
HISTORICAL_NOTE = (
    "Året gäller TESSERA och HR-VPP. Skogliga grunddata använder senaste "
    "tillgängliga skanning i den angivna utgåvan. Trädhöjd, markfukt och DEM "
    "är fasta stödlager, inte observationer från det valda året."
)
ROOT = Path(__file__).resolve().parents[2]


class InputUnavailable(ValueError):
    """No verified input for the exact requested grid and year."""


@dataclass(frozen=True)
class Grid:
    east: int
    north: int

    @classmethod
    def from_lonlat(cls, lon: float, lat: float) -> "Grid":
        # Coarse admission envelope only; actual Swedish source coverage is
        # required by the catalogue/preparer. This is not a country polygon.
        if not (10 <= lon <= 25 and 55 <= lat <= 70):
            raise InputUnavailable("Välj en plats inom Sveriges datatäckning.")
        east, north = Transformer.from_crs(4326, 3006, always_xy=True).transform(lon, lat)
        bbox = TileConfig(SOURCE_SIZE).bbox_from_center(east, north)
        return cls((bbox["west"] + bbox["east"]) // 2,
                   (bbox["south"] + bbox["north"]) // 2)

    def bounds(self, size: int = SOURCE_SIZE) -> dict:
        return TileConfig(size).bbox_from_center(self.east, self.north)

    def transform(self, size: int = SOURCE_SIZE) -> Affine:
        b = self.bounds(size)
        return from_bounds(b["west"], b["south"], b["east"], b["north"], size, size)

    def describe(self) -> dict:
        b = self.bounds(OUTPUT_SIZE)
        tx = Transformer.from_crs(3006, 4326, always_xy=True)
        # Densify each projected edge; draw the actual projected square.
        points = []
        corners = [(b["west"], b["south"]), (b["east"], b["south"]),
                   (b["east"], b["north"]), (b["west"], b["north"])]
        for start, end in zip(corners, corners[1:] + corners[:1]):
            for fraction in np.linspace(0, 1, 17)[:-1]:
                points.append(list(tx.transform(start[0] + fraction * (end[0] - start[0]),
                                                start[1] + fraction * (end[1] - start[1]))))
        points.append(points[0])
        return {"crs": "EPSG:3006", "center": [self.east, self.north],
                "source_shape": [512, 512], "shape": [504, 504], "resolution_m": 10,
                "crop": [4, 508, 4, 508], "bounds": b,
                "transform": list(self.transform(OUTPUT_SIZE))[:6],
                "footprint": {"type": "Polygon", "coordinates": [points]}}


def public_identity(identity: dict) -> dict:
    digest, size = identity["sha256"], identity["size"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid SHA256 identity")
    if type(size) is not int or size < 0:
        raise ValueError("Invalid byte size")
    return {"sha256": digest, "size": size}


def public_provenance(entry: dict) -> dict:
    """Project operator metadata into a closed public structure.

    No free text, filenames, paths, URLs or unknown nested fields are copied.
    Content hashes identify sources without revealing local inventory names.
    Dataset names and temporal policy are defined by the serving contract.
    """
    def digest(value):
        return public_identity({"sha256": value, "size": 0})["sha256"]

    def count(value):
        if type(value) is not int or value < 0:
            raise ValueError("Invalid provenance count")
        return value

    raw = entry["auxiliary_provenance"]
    aux = {}
    if "vpp" in raw:
        vpp = raw["vpp"]
        if type(vpp["year"]) is not int or vpp["year"] != entry["year"] or vpp["season"] != 1:
            raise ValueError("Provenance must name the exact HR-VPP year and season")
        aux["vpp"] = {"year": vpp["year"], "season": 1,
                      "index": public_identity(vpp["index"]),
                      "files": [public_identity(r) for r in vpp["files"]]}
    if "forestry" in raw:
        forest = raw["forestry"]
        aux["forestry"] = {
            "mirror_manifest_sha256": digest(forest["mirror_manifest_sha256"]),
            "grunddata_omdrev_pixels": {
                variable: {str(r): count(rounds[str(r)]) for r in (0, 1, 2, 3)}
                for variable, rounds in forest["grunddata_omdrev_pixels"].items()
                if variable in ("volym", "grundyta", "medeldiameter")},
            "rasters": [{"sha256": digest(r["sha256"]),
                         "source_sha256": digest(r["source_sha256"])}
                        for r in forest["rasters"].values()]}
    if "dem" in raw:
        aux["dem"] = {"files": [public_identity(r) for r in raw["dem"]["files"]]}
    if not aux:
        raise ValueError("No publishable source identities in auxiliary provenance")
    embedding = {}
    raw_embedding = entry.get("embedding_provenance", {})
    for name in ("registry", "landmasks", "cog_index"):
        if name in raw_embedding:
            embedding[name] = public_identity(raw_embedding[name])
    if "files" in raw_embedding:
        embedding["files"] = [public_identity(r) for r in raw_embedding["files"]]
    for name in ("download_budget_bytes", "uncached_source_bytes", "blocks"):
        if name in raw_embedding:
            embedding[name] = count(raw_embedding[name])
    return {"auxiliary_provenance": aux, "embedding_provenance": embedding}


def verify_file(record: dict, base: Path) -> Path:
    path = (base / record["path"]).resolve()
    actual = file_identity(path)
    if public_identity(actual) != public_identity(record):
        raise ValueError("Pinned file identity mismatch; rebuild the serving manifest")
    return path


def write_json(path: Path, value: dict | list) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()
    atomic_write(path, lambda stream: stream.write(data + b"\n"))


def validate_bundle(path: Path, record: dict, grid: Grid, year: int) -> dict:
    """Validate before canonical normalization can conceal absent inputs."""
    from scripts.inference_comparison import _load_npz_for_inference

    data, buffer = _load_npz_for_inference(
        path, expected_sha256=record["sha256"], expected_size=record["size"])
    try:
        if any(int(data[key]) != year for key in ("year", "tessera_year", "vpp_year")):
            raise ValueError("Input years must exactly match the requested year")
        if (str(data["crs"].item()) != "EPSG:3006"
                or not np.array_equal(data["transform"], list(grid.transform())[:6])):
            raise ValueError("Input affine/CRS does not match the snapped 512px grid")
        embedding = data["tessera"]
        if embedding.shape != (128, 512, 512) or embedding.dtype != np.float16:
            raise ValueError("Expected canonical float16 TESSERA (128,512,512)")
        valid = data["valid_embedding"]
        if valid.shape != (512, 512) or valid.dtype != np.bool_:
            raise ValueError("Explicit boolean embedding validity mask is required")
        if not np.isfinite(embedding).all() or np.any(valid & ~np.any(embedding != 0, axis=0)):
            raise ValueError("Invalid or zero embeddings labelled available")
        if not valid[4:508, 4:508].any():
            raise InputUnavailable("TESSERA saknar giltiga pixlar för platsen och året.")
        missing = {}
        for name in AUX_NAMES:
            arr = data[name]  # absence fails; never silently invent a channel
            if arr.shape != (512, 512) or arr.dtype != np.float32 or np.isinf(arr).any():
                raise ValueError(f"Invalid auxiliary shape/type/values: {name}")
            if name not in ("height", "markfukt") and np.isnan(arr).any():
                raise ValueError(f"Noncanonical auxiliary nodata: {name}")
            if name in ("height", "markfukt"):
                missing[name] = int(np.isnan(arr[4:508, 4:508]).sum())
            elif name.startswith("vpp_"):
                missing[name] = int((arr[4:508, 4:508] == 0).sum())
        return {"aux_nodata_or_zero_pixels": missing,
                "embedding_valid_pixels": int(valid[4:508, 4:508].sum())}
    finally:
        data.close()
        if buffer is not None:
            buffer.close()


class BundleCatalog:
    """A qualified set of exact-grid public-area inputs, never a year fallback."""

    def __init__(self, entries: list[dict], base: Path):
        self.entries = {}
        self.base = base
        for entry in entries:
            if entry["year"] not in YEARS or entry["embedding"] != EMBEDDING:
                raise ValueError("Unsupported annual input or embedding producer")
            if not entry.get("auxiliary_provenance") or entry.get("public_area") is not True:
                raise ValueError("Serving requires public-area inputs with auxiliary provenance")
            grid = Grid(*entry["center_3006"])
            if grid.east % 10 or grid.north % 10:
                raise ValueError("Catalogue centres must be on the 10m grid")
            key = (grid.east, grid.north, entry["year"])
            if key in self.entries:
                raise ValueError("Ambiguous catalogue entry")
            public_provenance(entry)
            path = verify_file(entry["input"], base)
            validate_bundle(path, entry["input"], grid, entry["year"])
            self.entries[key] = entry

    def years(self, grid: Grid) -> list[int]:
        return [y for y in YEARS if (grid.east, grid.north, y) in self.entries]

    def entry(self, grid: Grid, year: int) -> dict:
        try:
            return self.entries[(grid.east, grid.north, year)]
        except KeyError:
            raise InputUnavailable("Verifierade årsdata saknas för denna plats. Inget annat år används.") from None

    def prepare(self, grid: Grid, year: int) -> tuple[Path, dict]:
        entry = self.entry(grid, year)
        path = verify_file(entry["input"], self.base)
        validate_bundle(path, entry["input"], grid, year)
        return path, entry

    def identity(self, grid: Grid, year: int) -> dict:
        entry = self.entry(grid, year)
        return {"input": public_identity(entry["input"]),
                **public_provenance(entry)}

    def locations(self) -> list[dict]:
        tx = Transformer.from_crs(3006, 4326, always_xy=True)
        return [{"lon": tx.transform(e, n)[0], "lat": tx.transform(e, n)[1]}
                for e, n in sorted({(e, n) for e, n, _ in self.entries})]

    def areas(self) -> list[dict]:
        return []


class TesseraR4:
    """Resident, strict-loaded existing head; never initialize replacement weights."""

    def __init__(self, record: dict, base: Path, device: str):
        import torch
        from scripts.inference_comparison import load_model, inference_aux_names

        path = verify_file(record, base)
        self.device = torch.device(device)
        self.model, _, _, _ = load_model(
            str(path), self.device, backbone_name="tessera_v1", img_size=504,
            expected_checkpoint_sha256=record["sha256"],
            expected_checkpoint_size=record["size"])
        names = inference_aux_names(self.model.ck_cfg, self.model.n_aux_channels)
        if (names != AUX_NAMES or self.model.num_classes != 28
                or self.model.embed_dim != 128 or self.model.aux_fusion != "gated"
                or self.model.frac_head is None):
            raise ValueError("Checkpoint does not satisfy the TESSERA R4 serving contract")
        self.model.eval()
        self.contract = {"checkpoint": public_identity(record), "family": "tessera",
                         "aux_names": names, "num_classes": 28, "aux_fusion": "gated",
                         "fraction_head_present": True, "device": str(self.device)}

    def predict(self, path: Path, record: dict) -> np.ndarray:
        import torch
        from scripts.inference_comparison import (
            _build_inference_inputs, _close_inference_inputs, _forward_from_inputs)
        inputs = _build_inference_inputs(
            str(path), self.device, 504, AUX_NAMES, family="tessera",
            expected_tile_sha256=record["sha256"], expected_tile_size=record["size"])
        try:
            if not torch.isfinite(inputs["aux"]).all():
                raise ValueError("Nonfinite normalized auxiliary inputs")
            with torch.inference_mode():
                logits = _forward_from_inputs(self.model, inputs, self.device)
                if logits.shape != (1, 28, 504, 504) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid model logits")
                return logits.argmax(1).squeeze(0).cpu().numpy()
        finally:
            _close_inference_inputs(inputs)


def legend() -> list[dict]:
    return [{"id": key, "name": name, "color": "#%02x%02x%02x" % UNIFIED_COLORS[key]}
            for key, name in UNIFIED_CLASSES.items()] + [
                {"id": NODATA, "name": "Data saknas", "color": None}]


def export_map(prediction: np.ndarray, valid: np.ndarray, grid: Grid, output: Path) -> dict:
    from imint.exporters.export import save_geotiff
    from PIL import Image

    if (prediction.shape != (504, 504) or valid.shape != (504, 504)
            or not np.issubdtype(prediction.dtype, np.integer)
            or np.any((prediction < 0) | (prediction >= 28))):
        raise ValueError("Invalid model classification output")
    classes = np.where(valid, prediction, NODATA).astype(np.uint8)
    geo = GeoContext("EPSG:3006", grid.transform(504), grid.bounds(504), {}, (504, 504))
    path = output / "classification.tif"
    save_geotiff(classes, str(path), geo=geo)
    with rasterio.open(path, "r+") as dst:
        dst.nodata = NODATA
        dst.write_colormap(1, {**{k: (*v, 255) for k, v in UNIFIED_COLORS.items()},
                              NODATA: (0, 0, 0, 0)})
    # Verify the actual bytes after reopening, including the 40 m crop shift.
    with rasterio.open(path) as src:
        expected = grid.transform() * Affine.translation(4, 4)
        if (src.crs.to_epsg() != 3006 or src.transform != expected
                or src.shape != (504, 504) or src.nodata != 255
                or tuple(src.bounds) != array_bounds(504, 504, expected)
                or not np.array_equal(src.read(1), classes)):
            raise ValueError("GeoTIFF round-trip geometry/data verification failed")
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        display = np.full((height, width), NODATA, np.uint8)
        reproject(classes, display, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs="EPSG:3857", src_nodata=NODATA,
                  dst_nodata=NODATA, resampling=Resampling.nearest)
    palette = np.zeros((256, 4), np.uint8)
    for key, color in UNIFIED_COLORS.items():
        palette[key] = (*color, 255)  # class 0 remains opaque black
    Image.fromarray(palette[display]).save(output / "overlay.png")
    west, south, east, north = transform_bounds(
        "EPSG:3857", "EPSG:4326", *array_bounds(height, width, transform))
    write_json(output / "legend.json", legend())
    return {"overlay_bounds": [[south, west], [north, east]],
            "display_crs": "EPSG:3857", "display_resampling": "nearest",
            "valid_pixels": int(valid.sum()), "nodata_pixels": int((~valid).sum()),
            "class_counts": {str(k): int((classes == k).sum()) for k in UNIFIED_CLASSES}}


def implementation_identity() -> dict:
    # Hash the numerical path and serving code, including uncommitted changes.
    files = [ROOT / "scripts/inference_comparison.py", ROOT / "scripts/cache_predict.py",
             ROOT / "imint/training/unified_dataset.py", ROOT / "imint/training/tile_config.py",
             ROOT / "imint/exporters/export.py", ROOT / "imint/schema/unified_schema.py",
             Path(__file__), ROOT / "imint/inference/lclu_sources.py",
             ROOT / "imint/inference/lclu_cog.py",
             ROOT / "scripts/prepare_lclu.py"] + sorted((ROOT / "imint/fm").rglob("*.py"))
    return {str(p.relative_to(ROOT)): file_identity(p)["sha256"] for p in files}


class LcluService:
    def __init__(self, model, sources, output_root: Path):
        self.model, self.sources = model, sources
        self.output_root = output_root / "lclu"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.code = implementation_identity()
        self.identity = {"version": 1, "model": model.contract, "implementation": self.code,
                         "runtime": {p: version(p) for p in ("torch", "numpy", "rasterio", "pyproj", "Pillow", "scipy")},
                         "embedding": EMBEDDING, "legend": legend(), "crop": [4, 508, 4, 508]}

    @classmethod
    def from_manifest(cls, path: Path, output_root: Path) -> "LcluService":
        manifest = json.loads(path.read_text())
        if manifest["version"] != 1:
            raise ValueError("Unsupported serving manifest")
        if "preparation" in manifest:
            from imint.inference.lclu_sources import RepositorySources
            sources = RepositorySources(manifest["preparation"], path.parent, output_root)
        else:
            sources = BundleCatalog(manifest["inputs"], path.parent)
        model = TesseraR4(manifest["checkpoint"], path.parent, manifest.get("device", "cpu"))
        return cls(model, sources, output_root)

    def key(self, grid: Grid, year: int) -> str:
        if type(year) is not int or year not in YEARS:
            raise InputUnavailable("Året stöds inte. Ingen årsersättning tillåts.")
        return contract_pin({"inference": self.identity, "grid": grid.describe(), "year": year,
                             "source": self.sources.identity(grid, year)})

    def cached(self, key: str) -> dict | None:
        directory = self.output_root / key
        if not directory.exists():
            return None
        try:
            result = json.loads((directory / "result.json").read_text())
            if result["cache_key"] != key:
                raise ValueError("Cache key mismatch")
            for name, identity in result["files"].items():
                if name not in ("classification.tif", "overlay.png", "legend.json", "provenance.json"):
                    raise ValueError("Unexpected result artifact")
                if public_identity(file_identity(directory / name)) != identity:
                    raise ValueError("Incomplete or corrupt cached artifact")
            if set(result["files"]) != {"classification.tif", "overlay.png", "legend.json", "provenance.json"}:
                raise ValueError("Incomplete cached result")
            return result
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError("Cached map failed integrity verification; operator repair required") from exc

    def run(self, grid: Grid, year: int, phase: Callable[[str], None]) -> dict:
        start = time.perf_counter()
        if implementation_identity() != self.code:
            raise RuntimeError("Serving code changed; restart the API before inference")
        request_key = self.key(grid, year)
        phase("preparing")
        path, entry = self.sources.prepare(grid, year)
        quality = validate_bundle(path, entry["input"], grid, year)
        sources = public_provenance(entry)
        key = contract_pin({"request": request_key, "input": public_identity(entry["input"])})
        cached = self.cached(key)
        if cached is not None:
            return {**cached, "cache_hit": True, "request_ms": (time.perf_counter() - start) * 1000}
        prepared = time.perf_counter()
        phase("classifying")
        pred = self.model.predict(path, entry["input"])
        inferred = time.perf_counter()
        with np.load(path, allow_pickle=False) as data:
            valid = data["valid_embedding"][4:508, 4:508].copy()
        phase("exporting")
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.output_root))
        try:
            details = export_map(pred, valid, grid, temporary)
            provenance = {"year": year, "geometry": grid.describe(), "inference": self.identity,
                          "input": public_identity(entry["input"]), "embedding": entry["embedding"],
                          **sources,
                          "historical_note": HISTORICAL_NOTE, "quality": quality,
                          "created_at": datetime.now(timezone.utc).isoformat(),
                          "timing_ms": {"preparation": (prepared - start) * 1000,
                                        "inference": (inferred - prepared) * 1000,
                                        "export": (time.perf_counter() - inferred) * 1000}}
            write_json(temporary / "provenance.json", provenance)
            files = {name: public_identity(file_identity(temporary / name)) for name in
                     ("classification.tif", "overlay.png", "legend.json", "provenance.json")}
            result = {"cache_key": key, "year": year, "geometry": grid.describe(),
                      "files": files, "quality": quality, **details,
                      "timing_ms": provenance["timing_ms"]}
            write_json(temporary / "result.json", result)
            os.rename(temporary, self.output_root / key)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return {**result, "cache_hit": False, "request_ms": (time.perf_counter() - start) * 1000}
