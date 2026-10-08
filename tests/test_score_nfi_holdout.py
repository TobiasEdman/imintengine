"""Paired ranking cannot gain accuracy by changing its observation support."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_nfi_holdout import paired_report


def test_shared_coverage_ranking_and_uncertainty_are_reproducible():
    held = pd.DataFrame({
        "TractID": [1, 2, 3, 4], "PlotID": [1]*4, "Year": [2024]*4,
        "tile_name": ["a", "b", "c", "d"],
        "nfi_forest": [1, 2, -1, 4],
        "Easting": [100000, 200000, 300000, 400000],
        "Northing": [6500000]*4,
    })
    predictions = {"model_a": np.array([1, 3, 0, 4]),
                   "model_b": np.array([1, 2, 0, 1])}
    baselines = {"NMD2023": (np.array([1, 2, 0, 1]), np.array([111, 112, 0, 113]))}
    protocol = {"bootstrap_seed": 123, "bootstrap_samples": 100,
                "block_km": 50, "sesoi": 0.02}
    result = paired_report(held, predictions, baselines, protocol)
    assert result == paired_report(held, predictions, baselines, protocol)
    assert result["compared_observations"] == 3
    assert result["excluded_for_nmd_coverage"] == 1
    assert {s["n"] for s in result["scores"].values()} == {3}
    assert result["highest_point_estimate"] == ["model_a", "model_b"]
    assert len(result["pairs"]) == 3
    assert all(p["n_shared"] == 3 for p in result["pairs"].values())
    assert not any(p["difference_supported"] for p in result["pairs"].values())


def test_no_nmd_coverage_cannot_produce_winner():
    held = pd.DataFrame({"nfi_forest": [1]})
    with pytest.raises(ValueError, match="no common"):
        paired_report(held, {"a": np.array([1])},
                      {"NMD2023": (np.array([0]), np.array([0]))}, {})


def test_block_test_does_not_treat_repeated_plots_as_independent():
    from score_nfi_holdout import block_signflip_pvalue
    original = block_signflip_pvalue(np.array([1, 1]), np.array([0, 1]))
    repeated = block_signflip_pvalue(np.ones(200), np.repeat([0, 1], 100))
    assert original == repeated == 0.5


def test_undefined_metrics_become_null_before_output_creation():
    import json
    from score_nfi_holdout import json_metrics
    held = pd.DataFrame({"TractID": [1, 2], "PlotID": [1, 1], "Year": [2024, 2024],
                         "nfi_forest": [1, 1], "Easting": [1, 100001],
                         "Northing": [6500000, 6500000]})
    result = paired_report(
        held, {"model": np.ones(2)},
        {"NMD2023": (np.ones(2), np.full(2, 111))},
        {"bootstrap_seed": 1, "bootstrap_samples": 100, "block_km": 50, "sesoi": .02})
    encoded = json.dumps(json_metrics(result), allow_nan=False)
    assert json.loads(encoded)["scores"]["model"]["cohen_kappa"] is None


def test_frozen_baseline_roles_cannot_be_swapped_or_omitted(tmp_path):
    from score_nfi_holdout import validate_baseline_paths
    from prepare_nfi_holdout import file_identity
    a, b = tmp_path / "nmd23", tmp_path / "nmd18"
    a.write_bytes(b"year23")
    b.write_bytes(b"year18")
    manifest = {"baselines": {"NMD2023": file_identity(a), "NMD2018": file_identity(b)}}
    validate_baseline_paths(manifest, {"NMD2023": a, "NMD2018": b})
    with pytest.raises(ValueError, match="frozen role"):
        validate_baseline_paths(manifest, {"NMD2023": b, "NMD2018": a})
    with pytest.raises(ValueError, match="roster"):
        validate_baseline_paths(manifest, {"NMD2023": a})


def test_exact_block_tail_has_no_monte_carlo_resolution_floor():
    from score_nfi_holdout import block_signflip_pvalue
    assert block_signflip_pvalue(np.ones(22), np.arange(22)) == 2 ** -21


def test_integer_block_distribution_matches_exhaustive_label_swaps():
    import itertools
    from score_nfi_holdout import block_signflip_pvalue
    rng = np.random.default_rng(41)
    blocks = np.repeat(np.arange(7), 4)
    signs = np.array(list(itertools.product([-1, 1], repeat=7)))
    for _ in range(20):
        difference = rng.choice([-1, 0, 1], size=len(blocks))
        sums = np.array([difference[blocks == b].sum() for b in range(7)])
        exhaustive = np.mean(np.abs(signs @ sums) >= abs(sums.sum()))
        assert block_signflip_pvalue(difference, blocks) == exhaustive


def test_tract_year_centroids_keep_cluster_whole_and_preserve_sampling_coordinates():
    from score_nfi_holdout import tract_block_frame
    from race_rigor_stats import block_ids
    held = pd.DataFrame({"TractID": [1, 1, 1], "Year": [2024, 2024, 2022],
                         "Easting": [49999., 50001., 160000.], "Northing": [6500000.] * 3})
    original = held.copy(deep=True)
    statistical = tract_block_frame(held)
    blocks = block_ids(statistical, 50)
    assert blocks[0] == blocks[1] and blocks[0] != blocks[2]
    pd.testing.assert_frame_equal(held, original)


def test_tile_role_scores_keep_the_same_common_coverage():
    held = pd.DataFrame({"TractID": [1,2,3], "PlotID": [1]*3, "Year": [2024]*3,
                         "tile_role": ["campaign", "cohort", "campaign"], "nfi_forest": [1,2,3],
                         "Easting": [100000.,200000.,300000.], "Northing": [6500000.]*3})
    result = paired_report(held, {"model": np.array([1,1,3])},
        {"NMD2023": (np.array([2,2,3]), np.array([112,112,0]))},
        {"bootstrap_seed": 1, "bootstrap_samples": 100, "block_km": 50, "sesoi": .02, "primary_population": "all"})
    assert result["tile_role_scores"]["campaign"]["model"]["n"] == 1
    assert result["tile_role_scores"]["campaign"]["model"]["overall_exact"] == 1.
    assert result["tile_role_scores"]["cohort"]["model"]["overall_exact"] == 0.
    assert result["compared_observations"] == 2
    matrix = result["tile_role_scores"]["campaign"]["model"]["confusion_matrix"]
    assert np.asarray(matrix).shape == (5, 5) and np.asarray(matrix).sum() == 1
    assert matrix[1][1] == 1
    assert np.asarray(result["scores"]["model"]["confusion_matrix"]).sum() == 2
