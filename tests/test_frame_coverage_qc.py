"""Tests for the per-frame spectral coverage QC.

Covers the shared primitive (used by both the gate and fetch_spectral) and the
tile verdict — proving a swath-edge partial frame and an empty-but-present frame
both fail, while an absent (masked-out) frame is correctly excluded.
"""
import numpy as np

from imint.training.frame_coverage_qc import (
    check_frame_coverage,
    frame_valid_fraction,
    frame_valid_fractions,
)


def _spec(fracs, px: int = 10, nb: int = 6) -> np.ndarray:
    """(nf*nb, px, px) cube where frame f has ``fracs[f]`` of its rows non-zero."""
    nf = len(fracs)
    spec = np.zeros((nf * nb, px, px), np.float32)
    for f, fr in enumerate(fracs):
        rows = int(round(fr * px))
        spec[f * nb:(f + 1) * nb, :rows, :] = 0.2
    return spec


def _tile(fracs, mask=None):
    nf = len(fracs)
    return {"spectral": _spec(fracs), "num_bands": 6,
            "temporal_mask": np.array(mask if mask is not None else [1] * nf,
                                      np.uint8)}


def test_frame_valid_fraction_primitive():
    nb, px = 6, 10
    assert frame_valid_fraction(np.full((nb, px, px), 0.2, np.float32)) == 1.0
    assert frame_valid_fraction(np.zeros((nb, px, px), np.float32)) == 0.0
    half = np.zeros((nb, px, px), np.float32); half[:, :5, :] = 0.2
    assert frame_valid_fraction(half) == 0.5


def test_all_frames_covered_passes():
    assert check_frame_coverage(_tile([1.0, 1.0, 1.0, 1.0]))["status"] == "pass"


def test_partial_swath_frame_fails():
    r = check_frame_coverage(_tile([1.0, 0.6, 1.0, 1.0]), min_valid_frac=0.9)
    assert r["status"] == "fail"
    assert (1, 0.6) in r["bad_frames"]


def test_empty_present_frame_fails():
    # slot 2 is ~100% no-data but marked present — the empty-post-crop gap.
    r = check_frame_coverage(_tile([1.0, 1.0, 0.0, 1.0]))
    assert r["status"] == "fail"
    assert any(f == 2 for f, _ in r["bad_frames"])


def test_absent_frame_is_excluded_not_failed():
    # Same empty slot 2, but correctly masked absent → not evaluated → pass.
    r = check_frame_coverage(_tile([1.0, 1.0, 0.0, 1.0], mask=[1, 1, 0, 1]))
    assert r["status"] == "pass", r
    assert 2 not in r["valid_frac"]
    assert [f for f, _ in frame_valid_fractions(_tile([1, 1, 0, 1],
                                                      mask=[1, 1, 0, 1]))] == [0, 1, 3]


def test_no_spectral_skipped():
    assert check_frame_coverage({"label": np.zeros((4, 4))})["status"] == "skipped"
