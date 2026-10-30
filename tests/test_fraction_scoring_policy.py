"""Real policy/NPZ/parquet I/O; all observations and predictions are synthetic."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import fraction_scoring_policy as policy
from scripts import prediction_cache as cache
from scripts import score_against_truth as scoring


def fixture_contract():
    return {"checkpoint": {"sha256": "1" * 64, "size_bytes": 99},
            "model": {"fraction_head": True}}


def fixture_policy(decision="eligible"):
    return {"schema": policy.SCHEMA, "entries": [{
        "cell": "synthetic_r4", "checkpoint": fixture_contract()["checkpoint"],
        "decision": decision, "reason": "Synthetic test review; not campaign evidence",
        "head_enabled": True, "fraction_loss_signal": "recorded_nonzero",
        "training_log_sha256": "2" * 64, "basis_report_sha256": "3" * 64,
        "target_lineage_review_sha256": "4" * 64 if decision == "eligible" else None,
    }]}


def write_policy(tmp_path, value):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    return {"policy_path": path,
            "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("mode", ["requested", "automatic"])
def test_head_presence_never_replaces_policy(mode):
    with pytest.raises(cache.CacheError, match="reviewed policy"):
        policy.select_fraction_mode(fixture_contract(), **{mode: True})


def test_class_scoring_needs_no_supervision_policy():
    selected = policy.select_fraction_mode(fixture_contract())
    assert not selected["enabled"] and selected["decision"] == "not_requested"


@pytest.mark.parametrize("decision", ["unverified", "ineligible"])
def test_auto_records_denial_and_explicit_request_fails(tmp_path, decision):
    options = write_policy(tmp_path, fixture_policy(decision))
    result = policy.select_fraction_mode(fixture_contract(), automatic=True, **options)
    assert not result["enabled"] and result["decision"] == decision
    assert result["policy_sha256"] == options["policy_sha256"]
    with pytest.raises(cache.CacheError, match="not eligible"):
        policy.select_fraction_mode(fixture_contract(), requested=True, **options)


def test_changing_only_decision_cannot_promote_unverified_evidence(tmp_path):
    value = fixture_policy("unverified")
    value["entries"][0]["decision"] = "eligible"
    with pytest.raises(cache.CacheError, match="lack supervision"):
        policy.select_fraction_mode(
            fixture_contract(), automatic=True, **write_policy(tmp_path, value))


@pytest.mark.parametrize("field,value", [("sha256", "5" * 64), ("size_bytes", 100)])
def test_decision_cannot_transfer_to_another_checkpoint(tmp_path, field, value):
    contract = fixture_contract()
    contract["checkpoint"][field] = value
    with pytest.raises(cache.CacheError, match="checkpoint"):
        policy.select_fraction_mode(
            contract, automatic=True, **write_policy(tmp_path, fixture_policy()))


def test_head_contradiction_does_not_silently_fall_back(tmp_path):
    contract = fixture_contract()
    contract["model"]["fraction_head"] = False
    with pytest.raises(cache.CacheError, match="disagree"):
        policy.select_fraction_mode(
            contract, automatic=True, **write_policy(tmp_path, fixture_policy()))


def test_modified_policy_rejected_before_parsing(tmp_path):
    options = write_policy(tmp_path, fixture_policy())
    options["policy_path"].write_text("not even JSON")
    with pytest.raises(cache.CacheError, match="digest mismatch"):
        policy.select_fraction_mode(fixture_contract(), requested=True, **options)


def test_symlink_policy_rejected(tmp_path):
    options = write_policy(tmp_path, fixture_policy())
    link = tmp_path / "link.json"
    link.symlink_to(options["policy_path"])
    options["policy_path"] = link
    with pytest.raises(cache.CacheError):
        policy.select_fraction_mode(fixture_contract(), requested=True, **options)


@pytest.mark.parametrize("duplicate", ["checkpoint", "json_key"])
def test_ambiguous_decisions_fail_even_with_matching_digest(tmp_path, duplicate):
    value = fixture_policy()
    if duplicate == "checkpoint":
        value["entries"].append(copy.deepcopy(value["entries"][0]))
    options = write_policy(tmp_path, value)
    if duplicate == "json_key":
        raw = options["policy_path"].read_text().replace(
            '"decision": "eligible"', '"decision": "ineligible", "decision": "eligible"')
        options["policy_path"].write_text(raw)
        options["policy_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    with pytest.raises(cache.CacheError, match="duplicate"):
        policy.select_fraction_mode(fixture_contract(), requested=True, **options)


def build_cache(tmp_path):
    tile = tmp_path / "tile.npz"
    np.savez(tile, spectral=np.ones((6, 3, 3), np.float32), year=2022)
    contract = {
        **fixture_contract(), "schema": cache.SCHEMA,
        "model": {"family": "prithvi", "fraction_head": True},
        "source": {"test": True}, "runtime": {"test": True},
        "options": {"img_size": 3, "num_classes": 28, "batch_size": 1},
        "tiles": {"tile": {"input": cache.file_identity(tile), "crop_size": 3}},
    }
    directory = cache.initialize(tmp_path / "cache", contract)
    probs = np.zeros((28, 3, 3), np.float16)
    probs[2] = 1  # Hard head says spruce; fraction head says pine.
    fracs = np.zeros((4, 3, 3), np.float16)
    fracs[0] = 1
    cache.publish_prediction(directory, contract, "tile", {
        "pred": np.full((3, 3), 2, np.uint8), "probs": probs, "fracs": fracs}, "test")
    return directory, tile


def score_args(directory, tile, index):
    return SimpleNamespace(
        cache_dir=directory.parent, ckpt_sha=directory.name, img_size=3, num_classes=28,
        use_fraction_head=False, auto_fraction=True, fraction_policy=None,
        fraction_policy_sha256=None, plot_index=index, lucas_index=index,
        data_dir=tile.parent, min_support=1, dominant_frac=0.7, forest_floor=0.1,
        dump_per_plot=None, dump_per_point=None)


@pytest.mark.parametrize("truth", ["nfi", "lucas"])
def test_scoring_refuses_missing_policy_before_reading_truth(tmp_path, truth):
    directory, tile = build_cache(tmp_path)
    args = score_args(directory, tile, tmp_path / "truth-must-not-be-read.parquet")
    with pytest.raises(cache.CacheError, match="reviewed policy"):
        getattr(scoring, f"score_{truth}")(args)


@pytest.mark.parametrize("truth", ["nfi", "lucas"])
@pytest.mark.parametrize("decision", ["eligible", "unverified"])
def test_scoring_policy_controls_actual_fraction_use(tmp_path, truth, decision):
    directory, tile = build_cache(tmp_path)
    index = tmp_path / "index.parquet"
    row = {"tile_name": "tile", "tile_path": str(tile), "row": 1, "col": 1,
           "Year": 2022, "point_id": "synthetic-1", "unified_class": 1,
           "forest_dominant": "tall", "is_mixed": False, "split": "test",
           "VolPine": 100, "VolContorta": 0, "VolSpruce": 0,
           "VolBirch": 0, "VolOtherDec": 0, "Maturityclass": 41}
    pd.DataFrame([row]).to_parquet(index, index=False)
    args = score_args(directory, tile, index)
    options = write_policy(tmp_path, fixture_policy(decision))
    args.fraction_policy = options["policy_path"]
    args.fraction_policy_sha256 = options["policy_sha256"]
    output = tmp_path / "scored.parquet"
    if truth == "nfi":
        args.dump_per_plot = output
    else:
        args.dump_per_point = output
    result = getattr(scoring, f"score_{truth}")(args)
    eligible = decision == "eligible"
    assert result["_meta"]["fraction_scoring"]["enabled"] is eligible
    assert result["_meta"]["fraction_scoring"]["policy_sha256"] == options["policy_sha256"]
    scored = pd.read_parquet(output)
    assert len(scored) == 1
    if truth == "nfi":
        assert scored["model_pred"].iloc[0] == (1 if eligible else 2)
    else:
        assert ("l2a_forest_fraction" in result) is eligible
        assert ("frac_tall" in scored.columns) is eligible
        if eligible:
            assert scored["frac_tall"].iloc[0] == 1
            assert result["l2a_forest_fraction"]["dominant_argmax"]["agreement"] == 1


def test_current_campaign_policy_never_authorizes_unverified_fractions():
    path = Path(__file__).resolve().parents[1] / "docs/experiments/fm_fraction_scoring_policy.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = policy.load_policy(path, digest)
    assert len(actual["entries"]) == 28
    assert {entry["decision"] for entry in actual["entries"]} == {"unverified", "ineligible"}
    for entry in actual["entries"]:
        contract = {"checkpoint": entry["checkpoint"],
                    "model": {"fraction_head": entry["head_enabled"]}}
        result = policy.select_fraction_mode(
            contract, automatic=True, policy_path=path, policy_sha256=digest)
        assert not result["enabled"]
    affected = next(e for e in actual["entries"] if e["cell"] == "prithvi600m_r4")
    assert affected["decision"] == "ineligible" and not affected["head_enabled"]


@pytest.mark.parametrize("field,value", [
    ("head_enabled", False),
    ("fraction_loss_signal", "recorded_zero"),
    ("fraction_loss_signal", "not_recorded"),
])
def test_eligible_requires_each_supervision_condition(tmp_path, field, value):
    record = fixture_policy()
    record["entries"][0][field] = value
    with pytest.raises(cache.CacheError, match="lack supervision"):
        policy.select_fraction_mode(
            fixture_contract(), requested=True, **write_policy(tmp_path, record))


def test_explicit_eligible_request_succeeds(tmp_path):
    options = write_policy(tmp_path, fixture_policy())
    result = policy.select_fraction_mode(fixture_contract(), requested=True, **options)
    assert result["enabled"] is True and result["mode"] == "explicit"
    assert result["target_lineage_review_sha256"] == "4" * 64
