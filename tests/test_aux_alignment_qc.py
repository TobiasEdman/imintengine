"""Tests for the cross-raster aux-alignment QC gate.

Proves the gate distinguishes a correctly-placed forestry aux (footprint on the
NMD forest mask) from the 256/512 wrong-grid case (footprint disjoint from /
uncorrelated with forest, or a literal shape mismatch).
"""
import numpy as np

from imint.training.aux_alignment_qc import (
    FORESTRY_AUX,
    check_tile_alignment,
    forest_mask,
    phi_coefficient,
)


def _tile(*, aligned: bool = True, px: int = 64) -> dict:
    """Synthetic tile: a forest blob in the top-left, forestry aux either on it
    (aligned) or on a disjoint region (the wrong-grid signature)."""
    label = np.full((px, px), 10, np.uint8)          # water everywhere…
    label[4:28, 4:28] = 1                            # …except a tallskog blob
    forest = np.isin(label, [1, 2, 3, 4, 5, 6])
    if aligned:
        vol = np.where(forest, 150.0, 0.0).astype(np.float32)
    else:
        vol = np.zeros((px, px), np.float32)
        vol[36:60, 36:60] = 150.0                    # disjoint from forest
    return {"label": label, "volume": vol, "height": vol * 0.1,
            "basal_area": vol * 0.2, "diameter": vol * 0.1}


def test_phi_perfect_opposite_and_independent():
    a = np.zeros((8, 8), bool); a[:4] = True
    assert phi_coefficient(a, a) == 1.0
    assert phi_coefficient(a, ~a) == -1.0
    rng = np.random.default_rng(0)
    x = rng.random((128, 128)) > 0.5
    y = rng.random((128, 128)) > 0.5
    assert abs(phi_coefficient(x, y)) < 0.05         # independent ⇒ ~0


def test_aligned_tile_passes():
    r = check_tile_alignment(_tile(aligned=True))
    assert r["status"] == "pass", r
    assert all(v >= 0.15 for v in r["phi"].values())


def test_misaligned_tile_fails():
    r = check_tile_alignment(_tile(aligned=False))
    assert r["status"] == "fail", r
    assert set(r["failed_aux"]) == set(FORESTRY_AUX)


def test_shape_mismatch_fails():
    # A literal 256-in-512 case: aux is the wrong size → cannot align → fail.
    t = _tile(aligned=True)
    t["volume"] = np.full((32, 32), 150.0, np.float32)
    r = check_tile_alignment(t)
    assert r["status"] == "fail" and "volume" in r["failed_aux"]


def test_no_label_is_skipped_not_passed():
    t = _tile(aligned=False)
    del t["label"]
    assert forest_mask(t) is None
    assert check_tile_alignment(t)["status"] == "skipped"


def test_uninformative_tiles_skipped():
    px = 64
    all_water = {"label": np.full((px, px), 10, np.uint8),
                 "volume": np.zeros((px, px), np.float32)}
    assert check_tile_alignment(all_water)["status"] == "skipped"
    all_forest = {"label": np.ones((px, px), np.uint8),
                  "volume": np.full((px, px), 150.0, np.float32)}
    assert check_tile_alignment(all_forest)["status"] == "skipped"
