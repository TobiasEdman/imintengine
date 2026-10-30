"""Disk/content and interruption tests for frozen prediction caches; no model weights."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import prediction_cache as cache
from scripts.cache_predict import make_cached_predict_fn


def make_contract(tmp_path, *, fractions=False, count=2):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tiles = {}
    for i in range(count):
        p = inputs / f"tile{i}.npz"
        np.savez(p, spectral=np.full((6, 3, 3), i, dtype=np.float32))
        tiles[p.stem] = {"input": cache.file_identity(p), "crop_size": 3}
    return {
        "schema": cache.SCHEMA,
        "checkpoint": {"sha256": "1" * 64, "size_bytes": 99},
        "source": {"git_sha": "a" * 40, "files": {"inference.py": {
            "sha256": "2" * 64, "size_bytes": 20}}},
        "runtime": {"python": "test", "packages": [["numpy", "test", None]]},
        "options": {"img_size": 3, "num_classes": 2, "batch_size": 2},
        "model": {"family": "test", "fraction_head": fractions},
        "tiles": tiles,
    }


def arrays(fractions=False):
    probs = np.empty((2, 3, 3), np.float16)
    probs[0], probs[1] = 0.25, 0.75
    out = {"pred": np.ones((3, 3), np.uint8), "probs": probs}
    if fractions:
        out["fracs"] = np.full((4, 3, 3), 0.3, np.float16)
    return out


def populated(tmp_path, fractions=False, count=2):
    contract = make_contract(tmp_path, fractions=fractions, count=count)
    directory = cache.initialize(tmp_path / "cache", contract)
    for stem in contract["tiles"]:
        assert cache.publish_prediction(directory, contract, stem, arrays(fractions), "test")
    return contract, directory


@pytest.mark.parametrize("fractions", [False, True])
def test_manifest_receipts_and_arrays_round_trip(tmp_path, fractions):
    contract, directory = populated(tmp_path, fractions)
    manifest = cache.load_manifest(directory.parent, directory.name)
    assert manifest["contract"] == contract
    predict = make_cached_predict_fn(directory.parent, directory.name, want_fracs=fractions)
    actual = predict(tmp_path / "inputs/tile0.npz")
    expected = arrays(fractions)
    np.testing.assert_array_equal(actual[0], expected["pred"])
    np.testing.assert_array_equal(actual[1], expected["probs"])
    if fractions:
        np.testing.assert_array_equal(actual[2], expected["fracs"])
    before = (directory / "tile0.json").read_bytes()
    assert not cache.publish_prediction(directory, contract, "tile0", expected, "later")
    assert (directory / "tile0.json").read_bytes() == before
    assert cache.initialize(directory.parent, contract) == directory


@pytest.mark.parametrize("section,field,value", [
    ("checkpoint", "sha256", "3" * 64),
    ("source", "git_sha", "b" * 40),
    ("runtime", "python", "different"),
    ("options", "img_size", 4),
    ("options", "batch_size", 1),
    ("model", "fraction_head", True),
    ("model", "num_frames", 4),
])
def test_every_recipe_axis_invalidates_key(tmp_path, section, field, value):
    original = make_contract(tmp_path)
    changed = copy.deepcopy(original)
    changed[section][field] = value
    assert cache.contract_key(original) != cache.contract_key(changed)


def test_changed_input_and_actual_source_bytes_invalidate_key(tmp_path):
    contract = make_contract(tmp_path)
    changed = copy.deepcopy(contract)
    p = tmp_path / "inputs/tile0.npz"
    np.savez(p, spectral=np.full((6, 3, 3), 88, np.float32))
    changed["tiles"]["tile0"]["input"] = cache.file_identity(p)
    assert cache.contract_key(changed) != cache.contract_key(contract)
    source_root = tmp_path / "src"
    (source_root / "imint").mkdir(parents=True)
    source_file = source_root / "imint/a.py"
    source_file.write_text("value = 1\n")
    first = cache.source_identity(source_root, "a" * 40)
    source_file.write_text("value = 2\n")
    second = cache.source_identity(source_root, "a" * 40)
    assert first != second


def test_missing_tile_blocks_scoring_even_when_requested_tile_exists(tmp_path):
    contract = make_contract(tmp_path)
    directory = cache.initialize(tmp_path / "cache", contract)
    cache.publish_prediction(directory, contract, "tile0", arrays(), "test")
    with pytest.raises(cache.CacheError):
        make_cached_predict_fn(directory.parent, directory.name, want_fracs=False)
    assert cache.load_manifest(directory.parent, directory.name, require_complete=False)


def test_legacy_and_tampered_manifest_are_rejected(tmp_path):
    with pytest.raises(cache.CacheError, match="legacy"):
        cache.load_manifest(tmp_path, "a" * 16)
    contract, directory = populated(tmp_path)
    path = directory / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["contract"]["options"]["img_size"] = 4
    path.write_text(json.dumps(manifest))
    with pytest.raises(cache.CacheError, match="identity mismatch"):
        cache.load_manifest(directory.parent, directory.name)
    with pytest.raises(cache.CacheError, match="differs"):
        cache.initialize(directory.parent, contract)


def test_output_corruption_never_becomes_a_hit_or_gets_overwritten(tmp_path):
    contract, directory = populated(tmp_path)
    path = directory / "tile0.npz"
    bad = bytearray(path.read_bytes())
    bad[len(bad) // 2] ^= 1
    path.write_bytes(bad)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        cache.has_prediction(directory, contract, "tile0")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        cache.publish_prediction(directory, contract, "tile0", arrays(), "retry")
    assert path.read_bytes() == bad


def test_current_input_bytes_checked_by_consumer(tmp_path):
    _, directory = populated(tmp_path)
    predict = make_cached_predict_fn(directory.parent, directory.name, want_fracs=False)
    path = tmp_path / "inputs/tile0.npz"
    np.savez(path, spectral=np.full((6, 3, 3), 5, np.float32))
    with pytest.raises(cache.CacheError, match="input tile identity changed"):
        predict(path)


def test_receipt_swap_rejected(tmp_path):
    contract, directory = populated(tmp_path)
    (directory / "tile0.json").write_bytes((directory / "tile1.json").read_bytes())
    with pytest.raises(cache.CacheError, match="receipt identity mismatch"):
        cache.read_prediction(directory, contract, "tile0")


@pytest.mark.parametrize("case", [
    "shape", "dtype", "nan", "range", "sum", "pred_range", "pred_conflict",
    "unexpected_fractions", "missing_probs",
])
def test_invalid_arrays_not_published(tmp_path, case):
    contract = make_contract(tmp_path)
    directory = cache.initialize(tmp_path / "cache", contract)
    values = arrays()
    if case == "shape":
        values["probs"] = values["probs"][:, :2, :]
    elif case == "dtype":
        values["probs"] = values["probs"].astype(np.float32)
    elif case == "nan":
        values["probs"][0, 0, 0] = np.nan
    elif case == "range":
        values["probs"][0, 0, 0] = -0.1
    elif case == "sum":
        values["probs"][:] = 0
    elif case == "pred_range":
        values["pred"][:] = 255
    elif case == "pred_conflict":
        values["pred"][:] = 0
    elif case == "unexpected_fractions":
        values["fracs"] = np.zeros((4, 3, 3), np.float16)
    else:
        del values["probs"]
    with pytest.raises(cache.CacheError):
        cache.publish_prediction(directory, contract, "tile0", values, "test")
    assert not (directory / "tile0.npz").exists()


def test_float16_tie_allows_original_float32_argmax(tmp_path):
    contract = make_contract(tmp_path)
    values = arrays()
    values["probs"][:] = np.float16(0.5)
    # np.argmax(float16 probs) is 0, but float32's winning class may have been 1.
    cache.validate_arrays(values, contract, "tile0")


def test_interrupted_publish_is_incomplete_not_a_cache_hit(tmp_path, monkeypatch):
    contract = make_contract(tmp_path)
    directory = cache.initialize(tmp_path / "cache", contract)

    def interrupted(*args, **kwargs):
        raise OSError("simulated receipt publication failure")
    monkeypatch.setattr(cache, "_publish_json", interrupted)
    with pytest.raises(OSError, match="simulated"):
        cache.publish_prediction(directory, contract, "tile0", arrays(), "test")
    assert (directory / "tile0.npz").exists()
    assert not (directory / "tile0.json").exists()
    with pytest.raises(cache.CacheError):
        cache.has_prediction(directory, contract, "tile0")


def test_concurrent_manifest_and_disjoint_tile_publication(tmp_path):
    contract = make_contract(tmp_path)
    root = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=2) as executor:
        directories = list(executor.map(lambda _: cache.initialize(root, contract), range(2)))
    assert directories[0] == directories[1]
    directory = directories[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda stem: cache.publish_prediction(directory, contract, stem, arrays(), "test"),
            ("tile0", "tile1")))
    assert results == [True, True]
    cache.load_manifest(root, directory.name)
    for stem in contract["tiles"]:
        assert cache.has_prediction(directory, contract, stem)


def test_scoring_rejects_wrong_crop_or_class_count(tmp_path):
    from scripts.score_against_truth import _validate_cache_options
    _, directory = populated(tmp_path)
    args = SimpleNamespace(cache_dir=directory.parent, ckpt_sha=directory.name,
                           img_size=3, num_classes=2)
    _validate_cache_options(args)
    args.img_size = 2
    with pytest.raises(cache.CacheError, match="img_size"):
        _validate_cache_options(args)
    args.img_size, args.num_classes = 3, 23
    with pytest.raises(cache.CacheError, match="num_classes"):
        _validate_cache_options(args)


def test_metadata_hash_checked_before_header_or_year_access(tmp_path, monkeypatch):
    contract = make_contract(tmp_path)
    path = tmp_path / "inputs/tile0.npz"
    np.savez(path, spectral=np.full((6, 3, 3), 99, np.float32), year=2022)
    accessed = []

    def forbidden(*args):
        accessed.append(True)
        raise AssertionError("metadata accessed before identity verification")
    monkeypatch.setattr(cache, "_spatial_shape", forbidden)
    with pytest.raises(RuntimeError, match="mismatch"):
        cache.tile_metadata(path, contract["tiles"]["tile0"]["input"],
                            family="test", include_year=True)
    assert accessed == []


def test_authenticated_rectangular_geometry_and_year(tmp_path):
    import pandas as pd
    from scripts.score_against_truth import _prepare_cache_index
    contract = make_contract(tmp_path, count=1)
    path = tmp_path / "inputs/tile0.npz"
    np.savez(path, spectral=np.zeros((6, 7, 9), np.float32), year=np.int32(2022))
    contract["tiles"]["tile0"]["input"] = cache.file_identity(path)
    directory = cache.initialize(tmp_path / "cache", contract)
    cache.publish_prediction(directory, contract, "tile0", arrays(), "test")
    manifest = cache.load_manifest(directory.parent, directory.name)
    index = pd.DataFrame([
        {"tile_name": "tile0", "tile_path": str(path), "row": 2, "col": 3},
        {"tile_name": "tile0", "tile_path": str(path), "row": 6, "col": 8},
    ])
    cropped, years = _prepare_cache_index(index, manifest, include_year=True)
    assert len(cropped) == 1
    assert tuple(cropped[["row", "col"]].iloc[0]) == (0, 0)
    assert years == {"tile0": 2022}


def _write_pickle_marker(path):
    Path(path).write_text("unsafe deserialization")


def test_metadata_does_not_unpickle_object_year(tmp_path):
    marker = tmp_path / "pickle-marker"

    class ObjectYear:
        def __reduce__(self):
            return _write_pickle_marker, (str(marker),)

    path = tmp_path / "object.npz"
    np.savez(path, spectral=np.zeros((6, 3, 3), np.float32),
             year=np.array(ObjectYear(), dtype=object))
    identity = cache.file_identity(path)
    with pytest.raises(ValueError, match="Object arrays"):
        cache.tile_metadata(path, identity, family="test", include_year=True)
    assert not marker.exists()
