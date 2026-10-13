"""Equal-year selection must survive ordering, duplicate tiles and NMD coverage."""
import json

import numpy as np
import pandas as pd
import pytest

from imint.eval.fieldtruth import (
    NFI_KEY, NFI_YEAR_SAMPLING, load_frozen_holdout, sha256_file,
    validate_year_selection,
)
from scripts import prepare_nfi_holdout as prep
from scripts.score_nfi_holdout import paired_report
from tests.nfi_freeze_fixtures import complete_manifest


def selection(years=(2023, 2024), count=2):
    return dict(NFI_YEAR_SAMPLING, years=list(years), observations_per_year=count)


def candidates():
    return pd.DataFrame([
        dict(TractID=t, PlotID=1, Year=y, tile_name=f"tile-{y}-{t}",
             tile_role="campaign", nfi_forest=t % 4 + 1,
             Easting=t * 100000., Northing=6500000.)
        for y in (2022, 2023, 2024) for t in (1, 2, 3, 4)
    ])


def test_equal_selection_is_order_and_score_independent():
    frame = candidates()
    got, counts = prep.select_balanced_years(frame, selection())
    changed = frame.sample(frac=1, random_state=51).assign(nfi_forest=0, model_pred=999)
    again, _ = prep.select_balanced_years(changed, selection())
    pd.testing.assert_frame_equal(got[NFI_KEY], again[NFI_KEY])
    assert got.Year.value_counts().to_dict() == {2023: 2, 2024: 2}
    assert counts == {"pre_balance_observations": 12, "outside_selected_years": 4,
                      "eligible_by_selected_year": {"2023": 4, "2024": 4},
                      "excluded_by_year_quota": 4}
    assert set(got.tile_name) <= set(frame.tile_name)
    assert "_year_rank" not in got


def test_quota_never_silently_shrinks_or_drops_a_year():
    for requested in (selection(count=5), selection(years=(2021, 2024))):
        with pytest.raises(ValueError, match="insufficient eligible"):
            prep.select_balanced_years(candidates(), requested)


def test_duplicate_tile_rows_cannot_inflate_annual_counts():
    frame = candidates()
    duplicate = frame.iloc[[0]].assign(tile_name="another-tile")
    with pytest.raises(ValueError, match="unique plot-years"):
        prep.select_balanced_years(pd.concat([frame, duplicate]), selection())


@pytest.mark.parametrize("change", [
    {"years": None}, {"years": [2024]}, {"years": [2024, 2023]},
    {"years": [2023, 2023]}, {"years": [2023, True]},
    {"observations_per_year": 0}, {"observations_per_year": True},
    {"observations_per_year": 1.5}, {"seed": 7}, {"seed": 20261008.0}, {"method": "random"},
])
def test_balance_requires_explicit_valid_prespecified_policy(change):
    with pytest.raises(ValueError, match="balanced year"):
        validate_year_selection({"primary_population": "balanced", "year_selection": {**selection(), **change}})


def test_reader_rejects_correctly_hashed_but_unbalanced_table(tmp_path):
    held, _ = prep.select_balanced_years(candidates(), selection())
    path, training = tmp_path / "holdout.parquet", tmp_path / "training.parquet"
    held.to_parquet(path, index=False)
    held[NFI_KEY].iloc[:0].to_parquet(training, index=False)
    manifest = complete_manifest(dict(schema="nfi-fieldtruth-freeze-v1",
        holdout=dict(file=path.name, observations=4, sha256=sha256_file(path)),
        training=dict(file=training.name, observations=0, sha256=sha256_file(training)),
        campaign={"training_tiles": 0}))
    manifest["protocol"].update(primary_population="balanced", year_selection=selection())
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    load_frozen_holdout(mp, sha256_file(mp))
    held.loc[held.Year == 2023, "Year"] = 2022
    held.to_parquet(path, index=False)
    manifest["holdout"]["sha256"] = sha256_file(path)
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="year balance differs"):
        load_frozen_holdout(mp, sha256_file(mp))


def test_balanced_comparison_cannot_lose_year_quota_to_nmd_nodata():
    held, _ = prep.select_balanced_years(candidates(), selection())
    protocol = dict(primary_population="balanced", year_selection=selection(),
                    bootstrap_seed=1, bootstrap_samples=100, block_km=50, sesoi=.02)
    pred = held.nfi_forest.to_numpy()
    raw = np.full(len(held), 111)
    result = paired_report(held, {"model": pred}, {"NMD2023": (pred, raw)}, protocol)
    assert result["compared_year_support"] == {"2023": 2, "2024": 2}
    assert result["year_selection"] == selection()
    raw[0] = 0
    with pytest.raises(ValueError, match="year balance differs"):
        paired_report(held, {"model": pred}, {"NMD2023": (pred, raw)}, protocol)
