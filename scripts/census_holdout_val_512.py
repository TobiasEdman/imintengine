#!/usr/bin/env python3
"""Read-only census of the legacy ``holdout_val_512`` tiles (multi-year NFI control plan, step 1).

Answers, in aggregate only, whether the August-2026 off-footprint holdout can
serve as evaluation imagery for the 2018-2024 field-truth years:

  (a) key and flag consistency per tile, the way nfi2024-key-census2 did it
      (key presence, all-zero arrays, positive flag on an empty array);
  (b) per-tile spectral year, tessera_source, frame_2016, s1_vv_vh_2016 and
      vpp_year completeness against the 2024 campaign bar;
  (c) an NFI plot index over the holdout tiles with the same identity
      (TractID, PlotID, Year), the same year-exact colocation as index v4, the
      same 496 px common crop and the same teacher-pool exclusion as the
      task-0103 tile sizing (``imint.eval.fieldtruth`` at the sizing's process
      commit, union of the seven r2 feature tables);
  (d) a never-trained-on check: holdout tile names against every split file,
      feature table and plot index on the volume, plus footprint overlap
      with the training cohort;
  (e) eligible plot-years per year, with the tiles a non-root process cannot
      read (the 84 PermissionError files of the sizing run) kept separate.

Raw plot identities and coordinates never leave this process: the report
carries counts, year tables, tile names and file paths only. Nothing is
written under the data root; the Job mounts it read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from imint.training.nfi_colocate import colocate_plots, tile_year  # noqa: E402
from imint.training.slu_nfi import load_nfi_plots  # noqa: E402
from imint.training.tile_config import TileConfig  # noqa: E402
from imint.training.tile_time import resolve_tile_year  # noqa: E402

SCHEMA = "nfi-holdout-census-v1"
TILE = TileConfig(size_px=512)
CROP_PX = 496  # smallest model reader crop; identical to the task-0103 sizing
PAD = (TILE.size_px - CROP_PX) // 2
YEARS = tuple(range(2018, 2025))
TARGET_PER_YEAR = 1000  # decision D2, Tobias 2026-10-11

# imint/eval/fieldtruth.py at ba31235c6bb5f4c2fa9d5c71929926a0023d062b, the
# process commit of docs/experiments/nfi_fieldtruth_0103_tile_sizing.md. The
# 841-plot-year exclusion union is defined by that file's observation_keys and
# resolve_observation_year; a different file is a different identity.
FIELDTRUTH_SHA256 = "3f7143f2fceda3104ea61925000547f6cd6baac49d9112acce6c49be208ad68c"

# Same lists as k8s/nfi2024-key-census-job.yaml, so the two censuses compare.
CHECK_KEYS = ("spectral", "label", "s1_vv_vh", "s1_vv_vh_2016", "tessera",
              "height", "volume", "basal_area", "diameter", "dem",
              "markfukt", "frame_2016",
              "vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv")
FLAG_PAIRS = (("tessera", "has_tessera"),
              ("frame_2016", "has_frame_2016"),
              ("s1_vv_vh", "has_s1"))
AUX_KEYS = ("vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv",
            "dem", "height", "volume", "basal_area", "diameter", "markfukt")
FORESTRY_KEYS = ("height", "volume", "basal_area", "diameter")
META_KEYS = ("bbox_3006", "easting", "northing", "tile_size_px", "year",
             "lpis_year", "dates", "temporal_mask", "tessera_source",
             "tessera_year", "has_tessera", "has_frame_2016", "frame_2016_year",
             "has_s1", "s1_enrich_v", "vpp_year")
PROMOTED_TESSERA_PREFIX = "geotessera-0.10"
MARGIN = 0.02  # a key is BEHIND the 2024 bar only beyond this share, as in census2

SPLIT_GLOBS = (
    "unified_v2_512/train.txt", "unified_v2_512/val.txt", "unified_v2_512/test.txt",
    "unified_v2_512/split_*.txt", "unified_v2_512/manifest.json",
    "unified_v2/train.txt", "unified_v2/val.txt", "unified_v2/test.txt",
    "distill/*.json", "distill/**/*split*.json", "distill/heads/*_split.json",
    "distill/crop_split/**/*.json",
    "checkpoints/**/*split*.json", "checkpoints/**/*train*.json", "checkpoints/**/*val*.json",
    "checkpoints/**/*test*.json", "checkpoints/**/*tiles*.json", "checkpoints/**/*.txt",
    "distill/**/*.txt", "ladder_inference/tiles.json",
)
# Text-scan hits by top-level directory. Anything else is training territory
# or unknown and makes the verdict inconclusive until a human classifies it.
PROVENANCE_DIRS = {"audits", "debug", "manifests", "nfi"}
VISUALISATION_DIRS = {"ladder_inference"}
TEXT_SCAN_DIRS = ("checkpoints", "distill", "manifests", "audits",
                  "ladder_inference", "nfi", "debug")
TEXT_SUFFIXES = {".json", ".txt", ".yaml", ".yml", ".md", ".log", ".csv", ".jsonl"}
TEXT_MAX_BYTES = 64 * 1024 * 1024
HOLDOUT_DIR_TOKEN = "holdout_val_512"
_TILE_TOKEN = re.compile(r"[A-Za-z0-9]+(?:_\d+){2,3}")


# --------------------------------------------------------------------------- helpers
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fieldtruth(path: Path) -> ModuleType:
    """Import Codex's identity module from a file and refuse any other version."""
    digest = sha256_path(path)
    if digest != FIELDTRUTH_SHA256:
        raise SystemExit(f"{path}: sha256 {digest} is not the task-0103 process-commit "
                         f"fieldtruth.py ({FIELDTRUTH_SHA256}); refusing a different identity")
    spec = importlib.util.spec_from_file_location("fieldtruth_pinned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scalar(data, key):
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.ndim != 0 and value.size != 1:
        return None
    item = value.reshape(-1)[0]
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8", "replace")
    if isinstance(item, (np.integer, int)):
        return int(item)
    if isinstance(item, (np.floating, float)):
        return float(item)
    return str(item)


def _nonzero(array) -> bool:
    a = np.asarray(array)
    if a.dtype.kind in "iufb":
        return bool(np.any(a))
    return a.size > 0


def world_readable(st: os.stat_result) -> bool:
    """What a non-root, non-owner process (the sizing Job, uid 2000) could open."""
    return bool(st.st_mode & stat.S_IROTH)


def read_manifest(path: Path) -> tuple[dict[str, dict], str]:
    payload = path.read_bytes()
    records = json.loads(payload)
    by_name = {r["name"]: r for r in records}
    if len(by_name) != len(records):
        raise SystemExit(f"{path}: duplicate tile names")
    return by_name, sha256_bytes(payload)


# --------------------------------------------------------------------------- per tile
def read_tile(path: Path, manifest: dict | None) -> dict:
    """One tile's metadata, key census and consistency verdicts. Never writes."""
    rec: dict = {"name": path.stem, "bytes": path.stat().st_size}
    st = path.stat()
    rec["world_readable"] = world_readable(st)
    rec["mode"] = oct(stat.S_IMODE(st.st_mode))
    rec["uid"] = st.st_uid
    rec["in_manifest"] = manifest is not None
    rec["manifest_year"] = int(manifest["year"]) if manifest else None
    try:
        with np.load(path, allow_pickle=True) as data:
            files = set(data.files)
            rec["keys"] = files
            meta = {k: data[k] for k in META_KEYS if k in files}
            present, zero = {}, {}
            for key in CHECK_KEYS:
                if key not in files:
                    continue
                array = data[key]
                if getattr(array, "ndim", 0) < 2:
                    continue
                present[key] = True
                zero[key] = not _nonzero(array)
            if "spectral" in files:
                shape = tuple(data["spectral"].shape)
                rec["spectral_shape"] = list(shape)
                rec["spectral_ok"] = (len(shape) == 3 and shape[0] == 24
                                      and shape[1] == shape[2]
                                      and data["spectral"].dtype.kind in "iuf")
            else:
                rec["spectral_ok"] = False
    except PermissionError as exc:
        rec.update(readable=False, admitted=False, error=f"PermissionError: {exc.strerror}")
        return rec
    except Exception as exc:  # truncated zip, bad pickle, ... — counted, never fatal
        rec.update(readable=False, admitted=False, error=f"{type(exc).__name__}: {exc}")
        return rec
    rec["readable"] = True
    rec["present"] = present
    rec["zero"] = zero
    rec["flag_on_empty"] = sorted(
        key for key, flag in FLAG_PAIRS
        if present.get(key) and _scalar(meta, flag) == 1 and zero.get(key))

    # geometry — bbox_3006 is authoritative (Codex sizing), centre derived from it
    bbox = None
    if "bbox_3006" in meta:
        values = np.asarray(meta["bbox_3006"]).reshape(-1)
        if values.shape == (4,) and np.isfinite(values.astype(float)).all():
            bbox = dict(zip(("west", "south", "east", "north"), (int(v) for v in values)))
    size_px = _scalar(meta, "tile_size_px")
    if bbox is not None:
        ext = bbox["east"] - bbox["west"]
        if ext > 0 and ext % 10 == 0:
            size_px = ext // 10
    rec["size_px"] = int(size_px) if size_px else None
    rec["bbox_on_grid"] = bool(bbox and all(v % int(TILE.gsd_m) == 0 for v in bbox.values())
                               and bbox["east"] - bbox["west"] == bbox["north"] - bbox["south"]
                               == TILE.size_m)
    rec["bbox_matches_manifest"] = bool(
        bbox and manifest and all(abs(bbox[k] - int(manifest["bbox_3006"][k])) <= 1 for k in bbox))
    rec["_bbox"] = bbox  # internal: used for colocation and overlap, never reported

    # year — lenient order (index v4) and strict (sizing); both recorded
    rec["year_lenient"] = tile_year(meta)
    try:
        rec["year_strict"] = resolve_tile_year(meta)
        rec["year_conflict"] = False
    except ValueError:
        rec["year_strict"] = None
        rec["year_conflict"] = True
    rec["year_matches_manifest"] = (rec["manifest_year"] is not None
                                    and rec["year_lenient"] == rec["manifest_year"])
    # Quarantine: a tile whose explicit year contradicts its spectral dates, or
    # whose year cannot be resolved, is neither colocated nor counted as ready.
    # Year-exact colocation on a contradictory year would be a silent mismatch.
    rec["admitted"] = not rec["year_conflict"] and rec["year_lenient"] is not None
    mask = meta.get("temporal_mask")
    rec["temporal_mask"] = ("".join(str(int(x)) for x in np.asarray(mask).reshape(-1)[:4])
                            if mask is not None else None)

    # modalities against the 2024 bar
    rec["tessera_source"] = _scalar(meta, "tessera_source")
    rec["tessera_year"] = _scalar(meta, "tessera_year")
    rec["has_tessera"] = _scalar(meta, "has_tessera")
    rec["has_frame_2016"] = _scalar(meta, "has_frame_2016")
    rec["frame_2016_year"] = _scalar(meta, "frame_2016_year")
    rec["has_s1"] = _scalar(meta, "has_s1")
    rec["s1_enrich_v"] = _scalar(meta, "s1_enrich_v")
    rec["vpp_year"] = _scalar(meta, "vpp_year")
    rec["vpp_year_ok"] = rec["vpp_year"] is not None and rec["vpp_year"] == rec["year_lenient"]
    rec["aux_complete"] = all(k in files for k in AUX_KEYS)
    rec["label_present"] = "label" in files
    usable = {k: bool(present.get(k)) and not zero.get(k, True) for k in CHECK_KEYS}
    rec["usable"] = usable
    rec["bar_modalities"] = bool(
        rec["spectral_ok"] and rec["label_present"] and rec["aux_complete"]
        and usable["s1_vv_vh"] and rec["has_s1"] == 1
        and usable["tessera"] and rec["has_tessera"] == 1
        and usable["frame_2016"] and rec["has_frame_2016"] == 1
        and usable["s1_vv_vh_2016"]
        and rec["vpp_year_ok"]
        and all(usable[k] for k in FORESTRY_KEYS)
        and not rec["flag_on_empty"])
    rec["tessera_promoted"] = bool(rec["tessera_source"]
                                   and str(rec["tessera_source"]).startswith(PROMOTED_TESSERA_PREFIX))
    rec["bar_full"] = rec["bar_modalities"] and rec["tessera_promoted"]
    rec["_meta"] = meta  # internal: colocation input
    return rec


# --------------------------------------------------------------------------- aggregates
def census_tables(tiles: list[dict], bar_keys: Counter, bar_n: int) -> dict:
    readable = [t for t in tiles if t.get("readable")]
    n = len(readable)
    presence = Counter()
    for t in readable:
        presence.update(t["keys"])
    behind = {}
    for key, c in sorted(bar_keys.items()):
        share_bar = c / max(bar_n, 1)
        share_here = presence.get(key, 0) / max(n, 1)
        if share_bar - share_here > MARGIN:
            behind[key] = {"holdout": presence.get(key, 0), "holdout_share": round(share_here, 4),
                           "nfi2024_share": round(share_bar, 4)}
    zero = Counter()
    seen = Counter()
    for t in readable:
        for key, is_present in t["present"].items():
            seen[key] += 1
            if t["zero"].get(key):
                zero[key] += 1
    flag_on_empty = Counter()
    for t in readable:
        for key in t["flag_on_empty"]:
            flag_on_empty[key] += 1
    return {
        "tiles_readable": n,
        "key_presence": dict(sorted(presence.items())),
        "keys_behind_nfi2024_bar": behind,
        "all_zero": {k: {"present": seen[k], "all_zero": zero[k]} for k in CHECK_KEYS if seen[k]},
        "flag_on_empty_array": dict(flag_on_empty),
        "spectral_ok": sum(1 for t in readable if t["spectral_ok"]),
        "bbox_on_grid": sum(1 for t in readable if t["bbox_on_grid"]),
        "bbox_matches_manifest": sum(1 for t in readable if t["bbox_matches_manifest"]),
    }


def _by_year(tiles: list[dict], predicate) -> dict[str, int]:
    out = Counter()
    for t in tiles:
        if predicate(t):
            out[str(t["year_lenient"])] += 1
    return dict(sorted(out.items()))


def modality_tables(tiles: list[dict]) -> dict:
    """Readiness over ADMITTED tiles only; quarantined tiles are listed under files."""
    readable = [t for t in tiles if t.get("admitted")]
    return {
        "tiles_admitted": len(readable),
        "tiles_by_year": _by_year(readable, lambda t: True),
        "year_matches_manifest": _by_year(readable, lambda t: t["year_matches_manifest"]),
        "year_conflict_strict": sum(1 for t in readable if t["year_conflict"]),
        "year_unknown": sum(1 for t in readable if t["year_lenient"] is None),
        "temporal_mask": dict(Counter(t["temporal_mask"] for t in readable).most_common()),
        "tessera_source": dict(Counter(str(t["tessera_source"] or "unstamped") for t in readable)),
        "tessera_year_equals_tile_year": sum(
            1 for t in readable if t["tessera_year"] is not None and t["tessera_year"] == t["year_lenient"]),
        "has_tessera": dict(Counter(str(t["has_tessera"]) for t in readable)),
        "tessera_usable": _by_year(readable, lambda t: t["usable"]["tessera"] and t["has_tessera"] == 1),
        "has_frame_2016": dict(Counter(str(t["has_frame_2016"]) for t in readable)),
        "frame_2016_usable": _by_year(readable, lambda t: t["usable"]["frame_2016"] and t["has_frame_2016"] == 1),
        "frame_2016_year": dict(Counter(str(t["frame_2016_year"]) for t in readable)),
        "s1_vv_vh_2016_usable": _by_year(readable, lambda t: t["usable"]["s1_vv_vh_2016"]),
        "s1_enrich_v": dict(Counter(str(t["s1_enrich_v"]) for t in readable)),
        "vpp_year_present": sum(1 for t in readable if t["vpp_year"] is not None),
        "vpp_year_equals_tile_year": _by_year(readable, lambda t: t["vpp_year_ok"]),
        "aux_11_complete": _by_year(readable, lambda t: t["aux_complete"]),
        "label_present": _by_year(readable, lambda t: t["label_present"]),
        "bar_modalities": _by_year(readable, lambda t: t["bar_modalities"]),
        "bar_full_promoted_tessera": _by_year(readable, lambda t: t["bar_full"]),
    }


# --------------------------------------------------------------------------- index
def build_index(tiles: list[dict], plots: pd.DataFrame) -> pd.DataFrame:
    """Year-exact colocation of NFI plots onto the readable holdout tiles (index-v4 path)."""
    frames = []
    for t in tiles:
        if not t.get("admitted") or t["_bbox"] is None:
            continue
        bbox = t["_bbox"]
        meta = dict(t["_meta"])
        meta["easting"] = (bbox["west"] + bbox["east"]) // 2
        meta["northing"] = (bbox["south"] + bbox["north"]) // 2
        tile = TileConfig(size_px=t["size_px"] or TILE.size_px)
        if tile.bbox_from_center(meta["easting"], meta["northing"]) != bbox:
            raise SystemExit(f"{t['name']}: bbox_3006 is not a {tile.size_m} m tile on the lattice")
        got = colocate_plots(plots, name=t["name"], npz_data=meta, tile=tile)
        if got.empty:
            continue
        got.insert(0, "tile_name", t["name"])
        frames.append(got)
    if not frames:
        return plots.iloc[0:0].assign(tile_name=pd.Series(dtype="object"),
                                      row=pd.Series(dtype="int64"), col=pd.Series(dtype="int64"))
    return pd.concat(frames, ignore_index=True)


def exposure_union(ft: ModuleType, data_root: Path, inputs: dict) -> pd.MultiIndex:
    """The teacher-pool plot-years, exactly as the task-0103 sizing computed them."""
    source = _read_hashed(data_root / "nfi/nfi_index_unified_v2_512.parquet", inputs)
    features = []
    for model in ft.NFI_MODELS:
        frame = _read_hashed(data_root / f"distill/heads/{model}_r2_plot_features.parquet", inputs)
        features.append(ft.resolve_observation_year(frame, source)[ft.NFI_KEY])
    exposure = pd.concat(features).drop_duplicates(ft.NFI_KEY)
    return ft.observation_keys(exposure)


def _read_hashed(path: Path, inputs: dict) -> pd.DataFrame:
    payload = path.read_bytes()
    inputs[str(path)] = {"bytes": len(payload), "sha256": sha256_bytes(payload)}
    return pd.read_parquet(io.BytesIO(payload))


def valid_rows(frame: pd.DataFrame) -> np.ndarray:
    """Same value gate as the sizing: finite volumes/coordinates, no negative volume."""
    cols = ["VolPine", "VolContorta", "VolSpruce", "VolBirch", "VolOtherDec", "Easting", "Northing"]
    values = frame[cols].to_numpy(float)
    return np.isfinite(values).all(axis=1) & (values[:, :5] >= 0).all(axis=1)


def eligibility(ft: ModuleType, index: pd.DataFrame, tiles: list[dict], exposure: pd.MultiIndex,
                v4_known: pd.MultiIndex, years: tuple[int, ...]) -> dict:
    by_name = {t["name"]: t for t in tiles}
    if index.empty:
        return {str(y): {} for y in years}
    idx = index.copy()
    idx["in_crop"] = idx["row"].between(PAD, PAD + CROP_PX - 1) & idx["col"].between(PAD, PAD + CROP_PX - 1)
    keys = ft.observation_keys(idx)
    idx["exposed"] = keys.isin(exposure)
    idx["valid"] = valid_rows(idx)
    idx["already_v4_crop"] = keys.isin(v4_known)
    idx["restricted"] = ~idx["tile_name"].map(lambda n: by_name[n]["world_readable"]).astype(bool)
    idx["bar_modalities"] = idx["tile_name"].map(lambda n: by_name[n]["bar_modalities"]).astype(bool)
    idx["bar_full"] = idx["tile_name"].map(lambda n: by_name[n]["bar_full"]).astype(bool)
    idx["same_year_overlap"] = idx["tile_name"].map(
        lambda n: by_name[n].get("same_year_overlap", False)).astype(bool)
    assert not ft.observation_keys(idx.loc[~idx["exposed"]]).isin(exposure).any()

    def distinct(mask) -> int:
        return int(len(ft.observation_keys(idx.loc[mask]).unique())) if mask.any() else 0

    out = {}
    for year in years:
        y = idx["Year"] == year
        open_ = y & ~idx["restricted"]
        elig = y & idx["in_crop"] & ~idx["exposed"] & idx["valid"]
        row = {
            "rows": int(y.sum()),
            "plot_years_on_tiles": distinct(y),
            "in_common_crop": distinct(y & idx["in_crop"]),
            "excluded_teacher_pool": distinct(y & idx["in_crop"] & idx["exposed"]),
            "invalid_values": distinct(y & idx["in_crop"] & ~idx["exposed"] & ~idx["valid"]),
            "eligible": distinct(elig),
            "eligible_world_readable": distinct(elig & ~idx["restricted"]),
            "eligible_restricted_files": distinct(elig & idx["restricted"]),
            "eligible_on_bar_modalities": distinct(elig & idx["bar_modalities"]),
            "eligible_on_bar_full": distinct(elig & idx["bar_full"]),
            "eligible_also_in_v4_crop": distinct(elig & idx["already_v4_crop"]),
            "eligible_new_vs_v4": distinct(elig & ~idx["already_v4_crop"]),
            "tiles_with_plots": int(idx.loc[y, "tile_name"].nunique()),
            "tiles_world_readable_with_plots": int(idx.loc[open_, "tile_name"].nunique()),
        }
        keep = elig & ~idx["same_year_overlap"]
        row["tiles_excluded_same_year_overlap"] = int(
            idx.loc[y & idx["same_year_overlap"], "tile_name"].nunique())
        row["eligible_excl_same_year_overlap"] = distinct(keep)
        row["eligible_excl_same_year_overlap_world_readable"] = distinct(keep & ~idx["restricted"])
        row["eligible_excl_same_year_overlap_on_bar_modalities"] = distinct(keep & idx["bar_modalities"])
        row["gap_to_target_all_files"] = max(0, TARGET_PER_YEAR - row["eligible"])
        row["gap_to_target_world_readable"] = max(0, TARGET_PER_YEAR - row["eligible_world_readable"])
        row["gap_to_target_bar_full"] = max(0, TARGET_PER_YEAR - row["eligible_on_bar_full"])
        row["gap_to_target_excl_same_year_overlap"] = max(
            0, TARGET_PER_YEAR - row["eligible_excl_same_year_overlap"])
        row["gap_to_target_excl_same_year_overlap_world_readable"] = max(
            0, TARGET_PER_YEAR - row["eligible_excl_same_year_overlap_world_readable"])
        out[str(year)] = row
    return out


# --------------------------------------------------------------------------- never trained on
def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in (_strings(k) + _strings(v))]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    return []


def _stems(values: list[str]) -> set[str]:
    out = set()
    for v in values:
        base = os.path.basename(v.strip())
        if base.endswith(".npz"):
            base = base[:-4]
        if base:
            out.add(base)
    return out


def scan_split_files(data_root: Path, holdout: set[str]) -> list[dict]:
    seen: dict[Path, dict] = {}
    for pattern in SPLIT_GLOBS:
        for path in sorted(data_root.glob(pattern)):
            if not path.is_file() or path in seen or path.stat().st_size > TEXT_MAX_BYTES:
                continue
            payload = path.read_bytes()
            try:
                if path.suffix == ".json":
                    names = _stems(_strings(json.loads(payload)))
                else:
                    names = _stems(payload.decode("utf-8", "replace").splitlines())
            except (ValueError, UnicodeDecodeError) as exc:
                seen[path] = {"path": str(path), "error": f"{type(exc).__name__}"}
                continue
            seen[path] = {"path": str(path), "bytes": len(payload), "sha256": sha256_bytes(payload),
                          "tile_names": len(names), "holdout_names_found": len(names & holdout),
                          "_holdout_names": names & holdout}
    return [seen[p] for p in sorted(seen)]


def scan_text_mentions(data_root: Path, holdout: set[str]) -> list[dict]:
    """Every text-like file under the scan roots that names the holdout dir or a holdout tile."""
    hits = []
    for sub in TEXT_SCAN_DIRS:
        root = data_root / sub
        if not root.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                path = Path(dirpath) / fname
                if path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                try:
                    if path.stat().st_size > TEXT_MAX_BYTES:
                        continue
                    text = path.read_bytes().decode("utf-8", "replace")
                except OSError:
                    continue
                dir_hits = text.count(HOLDOUT_DIR_TOKEN)
                tokens = set(_TILE_TOKEN.findall(text)) & holdout
                if dir_hits or tokens:
                    hits.append({"path": str(path), "mentions_holdout_dir": dir_hits,
                                 "holdout_tile_names": len(tokens), "_tokens": tokens})
    return sorted(hits, key=lambda h: h["path"])


def cohort_geometry(cohort_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(cohort_dir.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                files = set(data.files)
                meta = {k: data[k] for k in ("bbox_3006", "easting", "northing", "year", "lpis_year", "dates")
                        if k in files}
        except Exception:
            rows.append({"name": path.stem, "readable": False})
            continue
        bbox = None
        if "bbox_3006" in meta:
            v = np.asarray(meta["bbox_3006"]).reshape(-1)
            if v.shape == (4,):
                bbox = [int(x) for x in v]
        elif "easting" in meta and "northing" in meta:
            b = TILE.bbox_from_center(int(meta["easting"]), int(meta["northing"]))
            bbox = [b["west"], b["south"], b["east"], b["north"]]
        rows.append({"name": path.stem, "readable": True, "year": tile_year(meta),
                     "west": bbox[0] if bbox else None, "south": bbox[1] if bbox else None,
                     "east": bbox[2] if bbox else None, "north": bbox[3] if bbox else None})
    return pd.DataFrame(rows)


def require_cohort_geometry(cohort_dir: Path) -> pd.DataFrame:
    """The training cohort's geometry, or a refusal: a missing or incomplete
    cohort would silently turn the same-year exclusion into a no-op."""
    if not cohort_dir.is_dir():
        raise SystemExit(f"REFUSING: training cohort directory {cohort_dir} is missing; "
                         f"the same-year footprint exclusion cannot be computed")
    geo = cohort_geometry(cohort_dir)
    if geo.empty:
        raise SystemExit(f"REFUSING: no tiles under {cohort_dir}")
    bad = geo.loc[~geo["readable"].astype(bool) | geo["west"].isna() | geo["year"].isna(), "name"].tolist() \
        if "west" in geo else geo["name"].tolist()
    if bad:
        raise SystemExit(f"REFUSING: {len(bad)} cohort tiles without readable geometry or year "
                         f"(first: {bad[:3]}); the footprint exclusion would be incomplete")
    return geo


def footprint_overlap(tiles: list[dict], cohort: pd.DataFrame) -> dict:
    """Mark every holdout tile's overlap with the training cohort and summarise.

    Sets ``any_year_overlap`` and ``same_year_overlap`` on each readable tile.
    Decision 2026-10-12 (Tobias, after PR #71): a holdout tile that overlaps a
    training tile of the SAME year is excluded from the evaluation; D6 allows
    other-year overlap only. The excluded names are listed so the evaluation
    copy can leave them out reproducibly.
    """
    geo = cohort.dropna(subset=["west"]) if len(cohort) else cohort
    for t in tiles:
        t["any_year_overlap"] = t["same_year_overlap"] = False
    tiles = [t for t in tiles if t.get("admitted", t.get("readable")) and t.get("_bbox") is not None]
    if geo.empty:
        return {"cohort_tiles_with_geometry": 0, "excluded_same_year_overlap_names": []}
    W, S, E, N = (geo[c].to_numpy(float) for c in ("west", "south", "east", "north"))
    Y = geo["year"].to_numpy()
    any_year, same_year, tiles_any, tiles_same = Counter(), Counter(), Counter(), Counter()
    excluded = []
    for t in tiles:
        b = t["_bbox"]
        hit = (W < b["east"]) & (E > b["west"]) & (S < b["north"]) & (N > b["south"])
        same = hit & (Y == t["year_lenient"])
        yr = str(t["year_lenient"])
        t["any_year_overlap"] = bool(hit.any())
        t["same_year_overlap"] = bool(same.any())
        if hit.any():
            tiles_any[yr] += 1
            any_year[yr] += int(hit.sum())
        if same.any():
            tiles_same[yr] += 1
            same_year[yr] += int(same.sum())
            excluded.append(t["name"])
    return {"cohort_tiles_with_geometry": int(len(geo)),
            "holdout_tiles_overlapping_any_year_training_tile": dict(sorted(tiles_any.items())),
            "holdout_tiles_overlapping_same_year_training_tile": dict(sorted(tiles_same.items())),
            "training_tile_overlaps_any_year": dict(sorted(any_year.items())),
            "training_tile_overlaps_same_year": dict(sorted(same_year.items())),
            "excluded_same_year_overlap_names": sorted(excluded)}


def classify_text_hit(path: str, data_root: Path, parsed_names: dict[str, set[str]],
                      tokens: set[str] | None = None, dir_mentions: int = 0) -> tuple[str, int]:
    """Class of a text-scan hit and how many of its tile names no parser accounted for.

    A file that parsed as a split file accounts only for the exact names its
    parser extracted; a tile name embedded in a sentence or a config value is
    still unaccounted text evidence, and so is any mention of the holdout
    directory itself. Directory evidence is judged independently of whether
    the file parsed: a parser extracts tile names, never directories, so a
    ``train_dir`` pointing at the holdout stays unclassified until a human
    classifies it. Only the provenance and visualisation directories are
    exempt.
    """
    tokens = tokens or set()
    top = Path(path).relative_to(data_root).parts[0]
    if top in PROVENANCE_DIRS:
        return "provenance_record", 0
    if top in VISUALISATION_DIRS:
        return "visualisation", 0
    unaccounted = tokens - parsed_names[path] if path in parsed_names else tokens
    if path in parsed_names and not unaccounted and not dir_mentions:
        return "parsed_split_file", 0
    return "unclassified", len(unaccounted)


def never_trained_on(ft: ModuleType, data_root: Path, cohort_dir: Path, staging_dir: Path,
                     tiles: list[dict], exposure_tables: dict[str, pd.DataFrame],
                     indexes: dict[str, pd.DataFrame], footprint: dict) -> dict:
    """Named-tile evidence of training or selection use, failing closed.

    The verdict is about tile NAMES in split files, feature tables, plot
    indexes and the training directories. It is not an imagery-independence
    proof: footprint overlap with training tiles is reported beside it as an
    unresolved spatial question (decision 2026-10-12 excludes the same-year
    overlaps). A split file that cannot be parsed, or a text file in training
    territory that names a holdout tile without being a parsed split file,
    makes the verdict inconclusive rather than clean.
    """
    holdout = {t["name"] for t in tiles}
    cohort_names = {p.stem for p in cohort_dir.glob("*.npz")} if cohort_dir.is_dir() else set()
    staging_names = {p.stem for p in staging_dir.glob("*.npz")} if staging_dir.is_dir() else set()
    split_files = scan_split_files(data_root, holdout)
    parsed_names = {f["path"]: f.pop("_holdout_names") for f in split_files if "error" not in f}
    scan_errors = [f for f in split_files if "error" in f]
    text_hits = scan_text_mentions(data_root, holdout)
    for h in text_hits:
        h["class"], h["unaccounted_tile_names"] = classify_text_hit(
            h["path"], data_root, parsed_names, h.pop("_tokens"), h["mentions_holdout_dir"])
    feature_hits = {name: int(frame["tile_name"].isin(holdout).sum()) for name, frame in exposure_tables.items()}
    index_hits = {name: int(frame["tile_name"].isin(holdout).sum()) for name, frame in indexes.items()}
    split_hits = [f for f in split_files if f.get("holdout_names_found")]
    # The ladder inference matrix renders predictions on frozen holdout tiles for
    # the dashboard. It chooses nothing (checkpoints are selected on the training
    # val split), so it is reported but not counted as selection use.
    visual_only = [f for f in split_hits if classify_text_hit(f["path"], data_root, {})[0] == "visualisation"]
    selection_hits = [f for f in split_hits if f not in visual_only]
    # Outside the provenance and visualisation directories, every mention the
    # parsers did not account for keeps the verdict open: a tile name in a
    # sentence or a config value, or the holdout directory itself as a
    # train_dir. Nothing here is counted as use; it is counted as unresolved.
    unclassified = [h for h in text_hits if h["class"] == "unclassified"
                    and (h["unaccounted_tile_names"] or h["mentions_holdout_dir"])]
    inconclusive = ([f"split file not parsed: {f['path']} ({f['error']})" for f in scan_errors]
                    + [f"unaccounted text evidence ({h['unaccounted_tile_names']} holdout tile names, "
                       f"{h['mentions_holdout_dir']} holdout-directory mentions): {h['path']}"
                       for h in unclassified])
    named_use = bool((holdout & cohort_names) or selection_hits
                     or any(feature_hits.values()) or any(index_hits.values()))
    if named_use:
        verdict = "named_tile_use_detected"
    elif inconclusive:
        verdict = "inconclusive"
    else:
        verdict = "no_named_tile_use_detected"
    same_year = footprint.get("holdout_tiles_overlapping_same_year_training_tile", {})
    return {
        "verdict": verdict,
        "verdict_basis": "tile names in split files, feature tables, plot indexes and training directories; "
                         "not an imagery-independence proof",
        "named_tile_use_detected": named_use,
        "scan_inconclusive_reasons": inconclusive,
        "spatial_independence": {
            "status": "unresolved_by_this_check",
            "holdout_tiles_overlapping_same_year_training_tile": same_year,
            "excluded_same_year_overlap_tiles": len(footprint.get("excluded_same_year_overlap_names", [])),
            "note": "decision 2026-10-12: tiles overlapping a same-year training tile are excluded; "
                    "other-year overlap is allowed (D6)",
        },
        "holdout_tile_names": len(holdout),
        "names_in_training_cohort_dir": len(holdout & cohort_names),
        "names_in_nfi2024_staging_dir": len(holdout & staging_names),
        "split_files_scanned": split_files,
        "split_files_not_parsed": [f["path"] for f in scan_errors],
        "split_files_naming_holdout_tiles": [f["path"] for f in selection_hits],
        "visualisation_files_naming_holdout_tiles": [f["path"] for f in visual_only],
        "teacher_feature_tables_naming_holdout_tiles": feature_hits,
        "plot_indexes_naming_holdout_tiles": index_hits,
        "text_files_mentioning_holdout": text_hits,
        "footprint": footprint,
    }


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("/data"))
    ap.add_argument("--holdout-dir", type=Path, default=None, help="default <data-root>/holdout_val_512")
    ap.add_argument("--manifest", type=Path, default=None, help="default <data-root>/audits/holdout_val_phaseA.json")
    ap.add_argument("--cohort-dir", type=Path, default=None, help="default <data-root>/unified_v2_512")
    ap.add_argument("--staging-dir", type=Path, default=None, help="default <data-root>/unified_v2_512_nfi2024_staging")
    ap.add_argument("--v4-index", type=Path, default=None, help="default <data-root>/nfi/nfi_index_v4_2026-09-30.parquet")
    ap.add_argument("--fieldtruth-file", type=Path, default=REPO / "imint/eval/fieldtruth.py")
    ap.add_argument("--expected-sizing", type=Path, default=None,
                    help="task-0103 tile-sizing JSON; its input hashes and 841 union must match")
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    ap.add_argument("--out-json", type=Path, required=True, help="aggregate report (outside the data root)")
    args = ap.parse_args(argv)

    root = args.data_root
    holdout_dir = args.holdout_dir or root / "holdout_val_512"
    manifest_path = args.manifest or root / "audits/holdout_val_phaseA.json"
    cohort_dir = args.cohort_dir or root / "unified_v2_512"
    staging_dir = args.staging_dir or root / "unified_v2_512_nfi2024_staging"
    v4_path = args.v4_index or root / "nfi/nfi_index_v4_2026-09-30.parquet"
    if args.out_json.resolve().is_relative_to(root.resolve()):
        raise SystemExit("--out-json must not be under the data root (read-only census)")
    years = tuple(sorted(set(args.years)))
    t0 = time.time()
    inputs: dict = {}

    ft = load_fieldtruth(args.fieldtruth_file)
    inputs[str(args.fieldtruth_file)] = {"sha256": FIELDTRUTH_SHA256}
    manifest, manifest_sha = read_manifest(manifest_path)
    inputs[str(manifest_path)] = {"bytes": manifest_path.stat().st_size, "sha256": manifest_sha}

    # ---- tiles
    paths = sorted(p for p in holdout_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    print(f"holdout files: {len(paths):,} under {holdout_dir}", flush=True)
    tiles = []
    for i, path in enumerate(paths, 1):
        tiles.append(read_tile(path, manifest.get(path.stem)))
        if i % 100 == 0:
            print(f"  {i:,}/{len(paths):,} tiles read ({time.time() - t0:.0f}s)", flush=True)
    readable = [t for t in tiles if t.get("readable")]
    quarantined = [t for t in readable if not t.get("admitted")]
    restricted = [t for t in tiles if not t["world_readable"]]
    unreadable = [t for t in tiles if not t.get("readable")]
    planned_missing = Counter(str(r["year"]) for n, r in manifest.items() if n not in {t["name"] for t in tiles})

    # ---- nfi2024 staging as the bar
    bar_keys, bar_n = Counter(), 0
    for p in sorted(staging_dir.glob("*.npz")) if staging_dir.is_dir() else []:
        try:
            with np.load(p, allow_pickle=True) as d:
                bar_keys.update(d.files)
        except Exception:
            bar_keys["__unreadable__"] += 1
        bar_n += 1
    print(f"nfi2024 staging bar: {bar_n:,} tiles", flush=True)

    # ---- plots, exposure, index
    plots_path = root / "nfi/nfi_plots.parquet"
    inputs[str(plots_path)] = {"bytes": plots_path.stat().st_size, "sha256": sha256_path(plots_path)}
    plots = load_nfi_plots(data_dir=root / "nfi", years=years)
    assert not ft.observation_keys(plots).has_duplicates
    exposure = exposure_union(ft, root, inputs)
    v4 = _read_hashed(v4_path, inputs)
    v4_crop = v4.loc[v4["row"].between(PAD, PAD + CROP_PX - 1) & v4["col"].between(PAD, PAD + CROP_PX - 1)]
    v4_known = ft.observation_keys(v4_crop).unique()
    index = build_index(tiles, plots)
    print(f"index: {len(index):,} rows on {index['tile_name'].nunique() if len(index) else 0:,} tiles", flush=True)
    footprint = footprint_overlap(tiles, require_cohort_geometry(cohort_dir))
    print(f"footprint: {len(footprint['excluded_same_year_overlap_names'])} holdout tiles overlap a "
          f"same-year training tile", flush=True)
    elig = eligibility(ft, index, tiles, exposure, v4_known, years)

    # ---- never trained on
    pinned = pd.read_parquet(io.BytesIO((root / "nfi/nfi_index_unified_v2_512.parquet").read_bytes()))
    feature_tables = {m: pd.read_parquet(root / f"distill/heads/{m}_r2_plot_features.parquet") for m in ft.NFI_MODELS}
    provenance = never_trained_on(ft, root, cohort_dir, staging_dir, tiles, feature_tables,
                                  {"pinned_index": pinned, "v4_index": v4}, footprint)

    # ---- sizing cross-check
    sizing_check = None
    if args.expected_sizing:
        sizing = json.loads(args.expected_sizing.read_bytes())
        mismatches = []
        for cephfs_path, rec in sizing["inputs"].items():
            local = str(root / Path(cephfs_path).relative_to("/cephfs"))
            if local in inputs and inputs[local].get("sha256") != rec["sha256"]:
                mismatches.append(local)
        sizing_check = {"file": str(args.expected_sizing), "inputs_compared": len(sizing["inputs"]),
                        "input_hash_mismatches": mismatches,
                        "teacher_feature_union_expected": sizing["teacher_feature_union"],
                        "teacher_feature_union_here": int(len(exposure)),
                        "legacy_permission_errors_expected": sizing["legacy_metadata_issues"].get("metadata_PermissionError"),
                        "legacy_files_expected": sizing["legacy_existing_files"]}
        if mismatches or len(exposure) != sizing["teacher_feature_union"]:
            raise SystemExit(f"sizing cross-check failed: {sizing_check}")

    report = {
        "schema": SCHEMA,
        "read_only": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "years": list(years),
        "identity": list(ft.NFI_KEY),
        "common_crop_px": CROP_PX,
        "target_per_year": TARGET_PER_YEAR,
        "holdout_dir": str(holdout_dir),
        "inputs": inputs,
        "teacher_feature_union": int(len(exposure)),
        "files": {
            "present": len(tiles),
            "readable": len(readable),
            "unreadable": [{"name": t["name"], "error": t.get("error")} for t in unreadable],
            "admitted": len(readable) - len(quarantined),
            "quarantined_year_conflict": sorted(t["name"] for t in quarantined),
            "quarantined_by_manifest_year": dict(sorted(Counter(str(t["manifest_year"]) for t in quarantined).items())),
            "not_world_readable": len(restricted),
            "not_world_readable_names": sorted(t["name"] for t in restricted),
            "not_world_readable_by_year": dict(sorted(Counter(str(t["manifest_year"]) for t in restricted).items())),
            "not_world_readable_modes": dict(Counter(f"{t['mode']} uid={t['uid']}" for t in restricted)),
            "not_in_manifest": sorted(t["name"] for t in tiles if not t["in_manifest"]),
            "planned_in_manifest": len(manifest),
            "planned_but_missing_by_year": dict(sorted(planned_missing.items())),
        },
        "census": census_tables(tiles, bar_keys, bar_n),
        "modalities": modality_tables(tiles),
        "eligibility_by_year": elig,
        "provenance": provenance,
        "sizing_cross_check": sizing_check,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, sort_keys=False))
    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    f, c, m = report["files"], report["census"], report["modalities"]
    print("\n=== files ===")
    print(f"present {f['present']:,}  readable {f['readable']:,}  unreadable {len(f['unreadable'])}  "
          f"not world-readable {f['not_world_readable']} {f['not_world_readable_by_year']}")
    print(f"admitted {f['admitted']:,}; quarantined for contradictory year: {len(f['quarantined_year_conflict'])} "
          f"{f['quarantined_by_manifest_year']}")
    print(f"planned {f['planned_in_manifest']:,}, planned but missing by year {f['planned_but_missing_by_year']}")
    print("\n=== key census (readable tiles) ===")
    for key, n in c["key_presence"].items():
        print(f"  {key:28s} {n:>6,}/{c['tiles_readable']:<6,}")
    print(f"keys behind the nfi2024 bar: {c['keys_behind_nfi2024_bar'] or 'none'}")
    print(f"all-zero arrays: { {k: v['all_zero'] for k, v in c['all_zero'].items() if v['all_zero']} or 'none'}")
    print(f"positive flag on empty array: {c['flag_on_empty_array'] or 'none'}")
    print("\n=== modalities by year ===")
    for key in ("tiles_by_year", "year_matches_manifest", "tessera_usable", "frame_2016_usable",
                "s1_vv_vh_2016_usable", "vpp_year_equals_tile_year", "aux_11_complete",
                "label_present", "bar_modalities", "bar_full_promoted_tessera"):
        print(f"  {key:28s} {m[key]}")
    print(f"  tessera_source: {m['tessera_source']}   temporal_mask: {m['temporal_mask']}")
    print("\n=== eligible plot-years (distinct TractID,PlotID,Year; 496 px crop; teacher pool excluded) ===")
    hdr = ("year", "on_tiles", "in_crop", "excluded", "eligible", "world_rd", "restricted", "bar_full",
           "gap_all", "gap_rd", "sameyr_tl", "elig_excl", "gap_excl")
    print("  " + " ".join(f"{h:>10s}" for h in hdr))
    for year, r in report["eligibility_by_year"].items():
        if not r:
            print(f"  {year:>10s}  (no plots)")
            continue
        print("  " + " ".join(f"{v:>10}" for v in (
            year, r["plot_years_on_tiles"], r["in_common_crop"], r["excluded_teacher_pool"], r["eligible"],
            r["eligible_world_readable"], r["eligible_restricted_files"], r["eligible_on_bar_full"],
            r["gap_to_target_all_files"], r["gap_to_target_world_readable"],
            r["tiles_excluded_same_year_overlap"], r["eligible_excl_same_year_overlap"],
            r["gap_to_target_excl_same_year_overlap"])))
    p = report["provenance"]
    print("\n=== never trained on ===")
    print(f"names in cohort dir {p['names_in_training_cohort_dir']}, in nfi2024 staging {p['names_in_nfi2024_staging_dir']}")
    print(f"split files scanned {len(p['split_files_scanned'])}, not parsed {p['split_files_not_parsed'] or 'none'}, "
          f"naming holdout tiles: {p['split_files_naming_holdout_tiles'] or 'none'}")
    print(f"visualisation-only files naming holdout tiles: {p['visualisation_files_naming_holdout_tiles'] or 'none'}")
    print(f"teacher feature tables naming holdout tiles: {p['teacher_feature_tables_naming_holdout_tiles']}")
    print(f"plot indexes naming holdout tiles: {p['plot_indexes_naming_holdout_tiles']}")
    print(f"text files mentioning the holdout: {len(p['text_files_mentioning_holdout'])}")
    for h in p["text_files_mentioning_holdout"]:
        print(f"    {h['path']}  dir={h['mentions_holdout_dir']} names={h['holdout_tile_names']} "
              f"unaccounted={h['unaccounted_tile_names']} [{h['class']}]")
    fp = dict(p.get("footprint") or {})
    excluded = fp.pop("excluded_same_year_overlap_names", [])
    print(f"footprint: {fp}")
    print(f"holdout tiles excluded for same-year training overlap: {len(excluded)}")
    print(f"\nVERDICT (named-tile evidence only): {p['verdict']}")
    for r in p["scan_inconclusive_reasons"]:
        print(f"  inconclusive: {r}")
    print(f"spatial independence: {p['spatial_independence']['status']}; same-year overlap tiles "
          f"{p['spatial_independence']['holdout_tiles_overlapping_same_year_training_tile']}")
    if report["sizing_cross_check"]:
        print(f"sizing cross-check: {report['sizing_cross_check']}")


if __name__ == "__main__":
    main()
