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
    """Read minimal metadata (year, bbox, name) for every tile in ``tiles_dir``.

    Caches result in ``<tiles_dir>/.eval_metadata.json`` so re-runs are
    instant; metadata is invalidated by mtime on the source .npz files.

    Returns:
        List of ``{"name", "year", "centre_x_3006", "centre_y_3006",
        "tile_type", "has_aux", ...}``. ``tile_type`` parsed from the
        tile name prefix (``crop_*`` / ``urban_*`` / ``tile_*``).

    TODO: implement the mtime-keyed cache; today the dataset loader
    already reads every .npz on every epoch — we don't want to do
    that for split-building.
    """
    raise NotImplementedError


# ── Standard in-distribution split ──────────────────────────────────────────


def make_in_distribution_split(
    tiles_dir: str,
    *,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    # test_frac is the remainder
    stratify_by: tuple[str, ...] = ("tile_type", "year"),
) -> dict[str, list[str]]:
    """Stratified random 80/10/10 split.

    Stratification keys default to (tile_type, year) so each test bucket
    has the same class-mix as train. ``seed`` is the only knob that
    changes which tiles land where.

    TODO: pull metadata via load_tile_metadata, group by stratify_by
    tuple, sample within each group.
    """
    raise NotImplementedError


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
