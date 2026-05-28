"""Build train / val / test splits for each evaluation axis.

One canonical source of tile names — the unified_v2 / unified_v2_512
directory listings — gets sliced by different policies depending on
which shift we're measuring. All splits are deterministic given a
``seed``, and their tile lists are written to JSON so a later run can
reproduce them exactly.

Output schema (one file per split):

    /checkpoints/eval_splits/<axis>_<variant>/<split>.txt
        — one tile name per line

    /checkpoints/eval_splits/<axis>_<variant>/manifest.json
        — { axis, variant, seed, criteria, split_sizes, created_at }
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


@dataclass
class SplitManifest:
    """Self-describing record of how a split was built.

    Reproducibility-critical: every choice that affects which tiles go
    into which bucket is recorded here so a later "why was tile X in
    train?" question is answerable from a single file.
    """

    axis: str                       # "temporal" | "geographic" | "sensor" | "phenology" | "in_distribution"
    variant: str                    # e.g. "train_2018_2022_test_2023_2024"
    seed: int
    criteria: dict = field(default_factory=dict)
    split_sizes: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "manifest.json").open("w") as f:
            json.dump(self.__dict__, f, indent=2)


# ── Tile metadata cache ─────────────────────────────────────────────────────


def load_tile_metadata(tiles_dir: str) -> list[dict]:
    """Read minimal metadata (year, bbox-centre, type) for every .npz tile.

    Caches the result at ``<tiles_dir>/.eval_metadata.json`` so re-runs
    are instant. Cache is invalidated when ANY source .npz has a more
    recent ``mtime`` than the cache itself — same heuristic the rest of
    the project uses for derived artefacts.

    Returns:
        List of dicts with keys ``name``, ``year``, ``centre_x_3006``,
        ``centre_y_3006``, ``tile_type``. Unreadable .npz files are
        skipped; year stays ``None`` if no year-bearing field is
        present.
    """
    import json
    import numpy as np

    if not os.path.isdir(tiles_dir):
        raise FileNotFoundError(f"tiles_dir does not exist: {tiles_dir}")

    npz_files = [
        f for f in os.listdir(tiles_dir)
        if f.endswith(".npz") and not f.startswith(".")
    ]
    if not npz_files:
        return []

    cache_path = os.path.join(tiles_dir, ".eval_metadata.json")
    newest_npz_mtime = max(
        os.path.getmtime(os.path.join(tiles_dir, f)) for f in npz_files
    )
    if (
        os.path.exists(cache_path)
        and os.path.getmtime(cache_path) > newest_npz_mtime
    ):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if isinstance(cached, list) and cached and "name" in cached[0]:
                return cached
        except Exception:
            pass  # treat as cache miss

    # Cache miss — rebuild. ``allow_pickle=True`` to read object arrays
    # (some legacy tiles store date strings as objects).
    entries: list[dict] = []
    for fname in sorted(npz_files):
        path = os.path.join(tiles_dir, fname)
        name = fname[:-4]
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            continue

        year: int | None = None
        for key in ("tessera_year", "lpis_year", "year"):
            if key in data.files:
                try:
                    year = int(data[key])
                    break
                except Exception:
                    pass
        if year is None and "dates" in data.files:
            for d in data["dates"]:
                s = str(d)
                if len(s) >= 4 and s[:4].isdigit():
                    year = int(s[:4])
                    break

        cx: int | None = None
        cy: int | None = None
        if "bbox_3006" in data.files:
            try:
                bbox = data["bbox_3006"]
                cx = int((float(bbox[0]) + float(bbox[2])) / 2)
                cy = int((float(bbox[1]) + float(bbox[3])) / 2)
            except Exception:
                pass

        if name.startswith("crop_"):
            tile_type = "crop"
        elif name.startswith("urban_"):
            tile_type = "urban"
        elif name.startswith("tile_"):
            tile_type = "lulc"
        else:
            tile_type = "other"

        entries.append({
            "name":           name,
            "year":           year,
            "centre_x_3006":  cx,
            "centre_y_3006":  cy,
            "tile_type":      tile_type,
        })

    # Best-effort cache write — non-fatal if the dir is read-only.
    try:
        with open(cache_path, "w") as f:
            json.dump(entries, f)
    except Exception:
        pass
    return entries


# ── Standard in-distribution split ──────────────────────────────────────────


def make_in_distribution_split(
    tiles_dir: str,
    *,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    stratify_by: tuple[str, ...] = ("tile_type", "year"),
) -> dict[str, list[str]]:
    """Stratified random 80/10/10 split.

    For each unique tuple of ``stratify_by`` values, tiles are shuffled
    with ``seed`` and partitioned in proportion to ``train_frac`` /
    ``val_frac`` / (1 - train_frac - val_frac). Floor rounding on
    train+val means the remainder always lands in test, never the
    other way around — keeps held-out evaluation honest.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` with tile
        names (no ``.npz`` extension).
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must leave room for test "
            f"(got {train_frac} + {val_frac})"
        )

    entries = load_tile_metadata(tiles_dir)
    if not entries:
        return {"train": [], "val": [], "test": []}

    buckets: dict[tuple, list[str]] = {}
    for e in entries:
        key = tuple(e.get(k) for k in stratify_by)
        buckets.setdefault(key, []).append(e["name"])

    rng = random.Random(seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        names = list(buckets[key])
        rng.shuffle(names)
        n = len(names)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # Floor rounding on train+val → remainder to test.
        splits["train"].extend(names[:n_train])
        splits["val"].extend(names[n_train:n_train + n_val])
        splits["test"].extend(names[n_train + n_val:])
    return splits


# ── Temporal shift splits ───────────────────────────────────────────────────


def make_temporal_split(
    tiles_dir: str,
    *,
    train_years: tuple[int, ...],
    test_years: tuple[int, ...],
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Train on ``train_years``, test on ``test_years``. No leakage between sets.

    Validation is sampled inside ``train_years``. Use this for the
    chronological-holdout phase 2a tests.
    """
    raise NotImplementedError


def make_year_leave_one_out(
    tiles_dir: str,
    *,
    held_out_year: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Hold out one year; train on all others.

    Tests whether any single year is anomalous (drought, sensor
    calibration shift, etc.).
    """
    raise NotImplementedError


# ── Geographic shift splits ─────────────────────────────────────────────────


def make_geographic_split(
    tiles_dir: str,
    *,
    split_axis: str,                 # "north_south" | "coast_inland" | "skog_slatt"
    seed: int = 42,
) -> dict[str, list[str]]:
    """Train and test bands separated along a known geographic axis.

    Implementation per axis:
      * north_south:   easting/northing partition at e.g. y=6.7e6
      * coast_inland:  distance to Sweden coastline polygon < 5 km
      * skog_slatt:    NMD-derived forest-fraction per tile > / < 60 %

    TODO: distance-to-coast needs a Lantmäteriet polygon cached at
    ``/cephfs/aux/sweden_coastline.geojson``. NMD-fraction needs a
    quick scan of each tile's label histogram.
    """
    raise NotImplementedError


# ── Sensor shift splits ─────────────────────────────────────────────────────


def make_l2a_to_l1c_split(
    tiles_dir_l2a: str,
    tiles_dir_l1c: str,
    *,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Pair tiles by name across BOA (L2A) and TOA (L1C) sources.

    Train on L2A, test on L1C. Measures atmospheric-correction
    sensitivity — the most direct proxy for sensor shift we have when
    everything is Sentinel-2.
    """
    raise NotImplementedError


def make_frame_dropout_split(
    tiles_dir: str,
    *,
    drop_slots: tuple[int, ...],
    seed: int = 42,
) -> dict[str, list[str]]:
    """Synthetic shift: same tiles, but some slots zeroed at eval time.

    Returns a single list (no train/val) because the dropout is applied
    online by the eval loader, not by changing which tiles are read.
    The split sizes tell the report generator how big the comparison was.
    """
    raise NotImplementedError


# ── Phenology shift splits ──────────────────────────────────────────────────


def make_phenology_shift_split(
    tiles_dir: str,
    *,
    shift_days: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Eval-time VPP windows shifted by ``shift_days``.

    Same tiles as in-distribution; the VPP window shift is applied at
    refetch time (separate offline pre-step) or as a post-processing
    swap on cached frames. Implementation choice deferred.
    """
    raise NotImplementedError


# ── Helpers ─────────────────────────────────────────────────────────────────


def write_split_files(
    split_dict: dict[str, list[str]],
    out_dir: Path,
) -> None:
    """Write each split's tile names to a flat ``.txt`` (one per line)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, tile_names in split_dict.items():
        with (out_dir / f"{split_name}.txt").open("w") as f:
            f.write("\n".join(sorted(tile_names)))
            f.write("\n")
