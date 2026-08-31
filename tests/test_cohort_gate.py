"""The cohort gate holds the tile set constant across label sources.

docs/experiments/label_source_ladder.md compares a model trained on the
in-tile 23-class label (rung 1) against the same model trained on NMD2023
sidecars (rung 2). The sidecars cover ~94.5% of tiles, so without a gate rung 1
silently trains on a superset and the rung delta mixes a label change with a
cohort change. `--cohort-dir` restricts the tile set to a directory's entries
WITHOUT reading labels from it.
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.training.unified_dataset import UnifiedDataset

H = W = 32
N_CLASSES = 23


def _write_tile(path):
    np.savez_compressed(
        str(path),
        spectral=(np.random.rand(24, H, W) * 0.4).astype(np.float32),
        label=np.random.randint(0, N_CLASSES, (H, W)).astype(np.int64),
        doy=np.array([260, 130, 190, 220], dtype=np.float32),
        year=np.int32(2022),
        easting=np.float32(500000.0), northing=np.float32(6500000.0),
    )


@pytest.fixture
def tiles(tmp_path):
    """5 tiles; only 3 of them have a sidecar in the cohort dir."""
    d = tmp_path / "tiles"
    d.mkdir()
    names = [f"tile_{i:03d}.npz" for i in range(5)]
    for n in names:
        _write_tile(d / n)
    (d / "split_train.txt").write_text("\n".join(names) + "\n")
    (d / "split_val.txt").write_text("\n".join(names) + "\n")

    cohort = tmp_path / "cohort"
    cohort.mkdir()
    for n in names[:3]:
        _write_tile(cohort / n)
    return d, cohort


def _ds(tile_dir, **kw):
    return UnifiedDataset(lulc_dir=tile_dir, split="train", patch_size=H,
                          augment_override=False, **kw)


def test_without_gate_all_tiles_are_used(tiles):
    tile_dir, _ = tiles
    assert len(_ds(tile_dir)) == 5


def test_cohort_dir_restricts_to_its_entries(tiles):
    tile_dir, cohort = tiles
    assert len(_ds(tile_dir, cohort_dir=cohort)) == 3


def test_cohort_dir_does_not_supply_labels(tiles):
    """The gate filters; it must not become a label source.

    Labels still come from the source tile, so a rung-1 run gated on the
    NMD2023 sidecar dir keeps training on the in-tile 23-class label.
    """
    tile_dir, cohort = tiles
    gated = _ds(tile_dir, cohort_dir=cohort)
    assert gated.label_dir is None
    ungated = _ds(tile_dir)
    name = gated._entries[0]["name"]
    match = [e for e in ungated._entries if e["name"] == name][0]
    assert gated._entries[0]["path"] == match["path"]


def test_gate_and_label_dir_compose(tiles):
    """Both gates apply — the intersection, not the last one to run."""
    tile_dir, cohort = tiles
    labels = cohort.parent / "labels"
    labels.mkdir()
    for n in [e for e in sorted(p.name for p in cohort.iterdir())][:2]:
        _write_tile(labels / n)
    ds = _ds(tile_dir, cohort_dir=cohort, label_dir=labels)
    assert len(ds) == 2
