"""Metadata-only crop embargo checks; no real holdout features or labels."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import crop_holdout_guard as guard

SOURCE = "a" * 40


def write_freeze(tmp_path, *, mutate=None):
    train = [("train", 10), ("train", 11)]
    held = [("held", 20), ("held", 21)]
    prior = [("prior", 30)]
    split = {
        "git_sha": SOURCE, "key_cols": ["tile_name", "point_id"],
        "key_digest_format": "sha256-jsonl-v1",
        "plots": [{"tile_name": t, "point_id": p} for t,p in train],
        "holdout_plots": [{"tile_name": t, "point_id": p} for t,p in held],
        "prior_test_plots": [{"tile_name": t, "point_id": p} for t,p in prior],
        "holdout_tiles": ["held"], "prior_test_tiles": ["prior"],
        "n_distill": 2, "n_holdout": 2, "n_qualified": 4,
        "prior_test_point_count": 1, "prior_test_tile_count": 1,
        "distill_keys_sha256": guard.key_digest(train),
        "holdout_keys_sha256": guard.key_digest(held),
        "qualified_keys_sha256": guard.key_digest(train + held),
        "partition_sha256": guard.key_digest([(t,p,"distill") for t,p in train]
                                              + [(t,p,"holdout") for t,p in held]),
        "prior_test_keys_sha256": guard.key_digest(prior),
        "prior_test_tiles_sha256": hashlib.sha256(b"prior").hexdigest(),
    }
    if mutate:
        mutate(split)
    raw = json.dumps(split).encode()
    (tmp_path / guard.SPLIT).write_bytes(raw)
    manifest = {k:v for k,v in split.items() if k.endswith("sha256") or k in (
        "git_sha", "n_distill", "n_holdout", "n_qualified",
        "prior_test_point_count", "prior_test_tile_count")}
    manifest["artifacts"] = {guard.SPLIT: hashlib.sha256(raw).hexdigest(),
                             guard.VALIDATOR: "e" * 64, guard.DISTILL: "f" * 64}
    raw = json.dumps(manifest).encode()
    (tmp_path / guard.MANIFEST).write_bytes(raw)
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "source_git_sha": SOURCE}


def test_frozen_metadata_does_not_touch_validator_or_features(tmp_path, monkeypatch):
    options = write_freeze(tmp_path)
    # These are not valid artifacts and must never be opened, statted or parsed.
    (tmp_path / "lucas_crop_validator_holdout_index.parquet").mkdir()
    (tmp_path / "held.npz").write_bytes(b"not a feature file")
    seen = []
    original = guard.read_pinned
    def capture(path, *a, **kw):
        seen.append(Path(path).name)
        return original(path, *a, **kw)
    monkeypatch.setattr(guard, "read_pinned", capture)
    actual = guard.load_crop_holdout(tmp_path, **options)
    assert seen == [guard.MANIFEST, guard.SPLIT]
    assert actual["protected_point_ids"] == {20,21,30}
    assert actual["protected_tiles"] == {"held","prior"}
    assert actual["summary"]["counts"]["n_holdout"] == 2
    assert actual["summary"]["cleared_for_scoring"] is False


def test_digest_format_matches_existing_freeze_producer():
    from scripts import build_lucas_crop_split as producer
    keys = [("åker", 2), ("tile", 1)]
    assert guard.key_digest(keys) == producer._key_digest(keys)
    assert guard.key_digest([("tile",1,"distill"),("åker",2,"holdout")]) == producer._partition_digest(
        [("tile",1)], [("åker",2)])


@pytest.mark.parametrize("target", [guard.MANIFEST, guard.SPLIT])
def test_changed_metadata_fails_before_json(tmp_path, target):
    options = write_freeze(tmp_path)
    (tmp_path / target).write_text("invalid JSON")
    with pytest.raises(guard.HoldoutError, match="digest mismatch"):
        guard.load_crop_holdout(tmp_path, **options)


@pytest.mark.parametrize("field,value", [
    ("n_holdout", 1), ("key_cols", ["point_id"]),
    ("holdout_keys_sha256", "0" * 64), ("holdout_tiles", ["wrong"]),
])
def test_self_consistent_byte_pins_do_not_replace_identity_checks(tmp_path, field, value):
    options = write_freeze(tmp_path, mutate=lambda s: s.update({field:value}))
    with pytest.raises(guard.HoldoutError):
        guard.load_crop_holdout(tmp_path, **options)


def test_duplicate_frozen_key_rejected(tmp_path):
    options = write_freeze(tmp_path, mutate=lambda s: s["holdout_plots"].append(s["holdout_plots"][0]))
    with pytest.raises(guard.HoldoutError, match="duplicate"):
        guard.load_crop_holdout(tmp_path, **options)


@pytest.mark.parametrize("target,field", [
    (20, "distill_crop_holdout_point_ids"),
    (30, "distill_prior_test_point_ids"),
])
def test_same_physical_point_in_other_partition_is_rejected(tmp_path, target, field):
    def change(s):
        s["plots"][0]["point_id"] = target
        train = [(r["tile_name"],r["point_id"]) for r in s["plots"]]
        held = [(r["tile_name"],r["point_id"]) for r in s["holdout_plots"]]
        s["distill_keys_sha256"] = guard.key_digest(train)
        s["qualified_keys_sha256"] = guard.key_digest(train + held)
        s["partition_sha256"] = guard.key_digest(
            [(t,p,"distill") for t,p in train] + [(t,p,"holdout") for t,p in held])
    options = write_freeze(tmp_path, mutate=change)
    with pytest.raises(guard.HoldoutError, match="crosses distill") as error:
        guard.load_crop_holdout(tmp_path, **options)
    diagnostic = error.value.diagnostics
    assert diagnostic["identity_metadata_verified"] is True
    assert diagnostic["overlap_details"][field] == 1
    assert diagnostic["distill_protected_point_overlap"] == 1
    assert diagnostic["distill_protected_tile_overlap"] == 0
    assert diagnostic["cleared_for_scoring"] is False


def test_manifest_symlink_and_wrong_source_are_rejected(tmp_path):
    options = write_freeze(tmp_path)
    with pytest.raises(guard.HoldoutError, match="source identity"):
        guard.load_crop_holdout(tmp_path, **{**options, "source_git_sha":"b"*40})
    p = tmp_path / guard.MANIFEST
    p.rename(tmp_path / "real.json")
    p.symlink_to(tmp_path / "real.json")
    with pytest.raises(OSError):
        guard.load_crop_holdout(tmp_path, **options)


def test_protected_identity_survives_tile_renaming_and_year_change(tmp_path):
    actual = guard.load_crop_holdout(tmp_path, **write_freeze(tmp_path))
    report = guard.audit_candidates(actual, [
        {"tile_name":"new_tile_name", "point_id":"20", "Year":2030},
        {"tile_name":"held", "point_id":999},
        {"tile_name":"new_tile_name", "point_id":30},
        {"tile_name":"development", "point_id":888},
    ], source="lucas")
    assert report["rows"] == 4 and report["blocked_rows"] == 3
    assert report["protected_point_hits"] == 2 and report["protected_tile_hits"] == 1
    assert "2030" not in json.dumps(report) and "new_tile_name" not in json.dumps(report)


def test_nfi_namespace_uses_tile_exclusion_not_lucas_point_numbers(tmp_path):
    actual = guard.load_crop_holdout(tmp_path, **write_freeze(tmp_path))
    result = guard.audit_candidates(actual, [
        {"tile_name":"development", "point_id":20}, {"tile_name":"held"}], source="nfi")
    assert result["protected_point_hits"] == 0 and result["blocked_rows"] == 1


@pytest.mark.parametrize("bad", [None, True, 20.0, "020", "+20", "20.0", " 20", -20])
def test_malformed_point_identity_cannot_bypass_embargo(tmp_path, bad):
    actual = guard.load_crop_holdout(tmp_path, **write_freeze(tmp_path))
    with pytest.raises(guard.HoldoutError):
        guard.audit_candidates(actual, [{"tile_name":"other", "point_id":bad}], source="lucas")


def test_empty_candidate_population_is_not_clear(tmp_path):
    actual = guard.load_crop_holdout(tmp_path, **write_freeze(tmp_path))
    with pytest.raises(guard.HoldoutError, match="empty"):
        guard.audit_candidates(actual, [], source="lucas")


def test_cli_reads_only_parquet_identity_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    options = write_freeze(tmp_path)
    candidate = tmp_path / "candidate.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"tile_name":"renamed", "point_id":"20", "unused_truth":"do-not-export"},
        {"tile_name":"other", "point_id":"999", "unused_truth":"do-not-export"},
    ]), candidate)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    script = Path(__file__).resolve().parents[1] / "scripts/crop_holdout_guard.py"
    result = subprocess.run([sys.executable, str(script), "--split-dir", str(tmp_path),
        "--manifest-sha256", options["manifest_sha256"], "--source-git-sha", SOURCE,
        "--candidate-index", str(candidate), "--candidate-index-sha256", digest,
        "--candidate-source", "lucas"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["candidates"]["blocked_rows"] == 1
    assert report["candidate_columns_read"] == ["tile_name", "point_id"]
    assert report["files_read"] == [guard.MANIFEST, guard.SPLIT, str(candidate)]
    assert "do-not-export" not in result.stdout


@pytest.mark.parametrize("named_validator", [True, False])
def test_validator_candidate_is_refused_before_open(tmp_path, monkeypatch, named_validator):
    actual = guard.load_crop_holdout(tmp_path, **write_freeze(tmp_path))
    filename = guard.VALIDATOR if named_validator else "renamed-validator.parquet"
    expected = "0" * 64 if named_validator else actual["validator_sha256"]
    def forbidden(*args, **kwargs):
        raise AssertionError("validator candidate was opened")
    monkeypatch.setattr(guard, "read_pinned", forbidden)
    with pytest.raises(guard.HoldoutError, match="validator index"):
        guard.read_candidate_identities(actual, tmp_path / filename, expected, source="lucas")
