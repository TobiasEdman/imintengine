"""Contain tile-local extraction failures without weakening the OOF gate."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import extract_plot_features as epf


def _row(tile_name: str, tile_path: Path, point_id: str) -> dict:
    return {
        "tile_name": tile_name,
        "tile_path": str(tile_path),
        "point_id": point_id,
        "row": 0,
        "col": 0,
        "unified_class": 1,
    }


def test_required_npz_key_filters_stale_tiles_before_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = tmp_path / "valid.npz"
    missing = tmp_path / "missing.npz"
    stale = tmp_path / "stale.npz"
    unreadable = tmp_path / "unreadable.npz"
    np.savez(valid, spectral=np.zeros(1), s1_vv_vh=np.zeros(1),
             s1_enrich_v=np.int32(4))
    np.savez(missing, spectral=np.zeros(1))
    np.savez(stale, spectral=np.zeros(1), s1_vv_vh=np.zeros(1),
             s1_enrich_v=np.int32(0))
    unreadable.write_bytes(b"not an npz")
    index = pd.DataFrame([
        _row("valid", valid, "p-valid"),
        _row("missing", missing, "p-missing"),
        _row("stale", stale, "p-stale"),
        _row("unreadable", unreadable, "p-unreadable"),
    ])
    messages: list[str] = []

    filtered = epf.filter_index_by_npz_requirements(
        index, ("s1_vv_vh",), log=messages.append
    )

    assert filtered["point_id"].tolist() == ["p-valid"]
    assert any("stale-version=1" in line for line in messages)
    assert any("DROP stale.npz: stale version" in line for line in messages)

    forwarded: list[str] = []
    store = {"feat": "stale-from-an-earlier-forward"}

    class Inference:
        @staticmethod
        def run_inference(_model, path, _device, **_kwargs):
            forwarded.append(Path(path).name)
            store["feat"] = 1

    monkeypatch.setattr(
        epf,
        "_sample_feature",
        lambda feat, rows, cols, crop: np.full((len(rows), 1), feat),
    )
    records, failures, _ = epf.extract_feature_records(
        filtered,
        model=object(),
        device="cpu",
        img_size=1,
        crop_size=1,
        infcmp=Inference(),
        store=store,
        feat_cols=["f000"],
        aux_names=None,
        truth_col="unified_class",
        dominant_frac=0.7,
        log=messages.append,
    )
    assert failures == 0
    assert [record["point_id"] for record in records] == ["p-valid"]
    assert forwarded == ["valid.npz"]


def test_tile_errors_continue_and_never_reuse_stale_hook_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {name: tmp_path / f"{name}.npz"
             for name in ("first", "raises", "hookless", "last")}
    index = pd.DataFrame([
        _row(name, path, f"p-{name}") for name, path in paths.items()
    ])
    store = {"feat": "preexisting-stale-feature"}
    forwarded: list[str] = []

    class Inference:
        @staticmethod
        def run_inference(_model, path, _device, **_kwargs):
            name = Path(path).stem
            forwarded.append(name)
            if name == "raises":
                raise ValueError("cannot build SAR input")
            if name == "hookless":
                return
            store["feat"] = {"first": 11, "last": 44}[name]

    monkeypatch.setattr(
        epf,
        "_sample_feature",
        lambda feat, rows, cols, crop: np.full((len(rows), 1), feat),
    )
    messages: list[str] = []

    records, failure_count, details = epf.extract_feature_records(
        index,
        model=object(),
        device="cpu",
        img_size=1,
        crop_size=1,
        infcmp=Inference(),
        store=store,
        feat_cols=["f000"],
        aux_names=None,
        truth_col="unified_class",
        dominant_frac=0.7,
        log=messages.append,
    )

    assert forwarded == ["first", "raises", "hookless", "last"]
    assert [record["point_id"] for record in records] == ["p-first", "p-last"]
    assert [record["f000"] for record in records] == [11, 44]
    assert failure_count == 2
    assert any("raises: ValueError: cannot build SAR input" in d for d in details)
    assert any("hookless: RuntimeError: hook captured no feature" in d
               for d in details)
    assert any("tile failures: 2" in message for message in messages)


def test_zero_feature_records_fails_loudly(tmp_path: Path) -> None:
    index = pd.DataFrame([_row("bad", tmp_path / "bad.npz", "p-bad")])

    class NoHookInference:
        @staticmethod
        def run_inference(_model, _path, _device, **_kwargs):
            return

    messages: list[str] = []
    with pytest.raises(RuntimeError, match="zero records"):
        epf.extract_feature_records(
            index,
            model=object(),
            device="cpu",
            img_size=1,
            crop_size=1,
            infcmp=NoHookInference(),
            store={"feat": "must-be-cleared"},
            feat_cols=["f000"],
            aux_names=None,
            truth_col="unified_class",
            dominant_frac=0.7,
            log=messages.append,
        )
    assert any("ERROR bad: RuntimeError: hook captured no feature" in line
               for line in messages)


def test_base_exception_is_not_contained(tmp_path: Path) -> None:
    index = pd.DataFrame([_row("stop", tmp_path / "stop.npz", "p-stop")])

    class InterruptingInference:
        @staticmethod
        def run_inference(_model, _path, _device, **_kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        epf.extract_feature_records(
            index,
            model=object(),
            device="cpu",
            img_size=1,
            crop_size=1,
            infcmp=InterruptingInference(),
            store={"feat": None},
            feat_cols=["f000"],
            aux_names=None,
            truth_col="unified_class",
            dominant_frac=0.7,
        )


def test_failure_details_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_tiles = epf._MAX_FAILED_TILE_DETAILS + 3
    index = pd.DataFrame([
        _row(f"bad-{i}", tmp_path / f"bad-{i}.npz", f"p-{i}")
        for i in range(n_tiles)
    ])

    class MostlyBadInference:
        @staticmethod
        def run_inference(_model, path, _device, **_kwargs):
            if Path(path).stem == f"bad-{n_tiles - 1}":
                store["feat"] = 7

    store = {"feat": None}
    messages: list[str] = []
    monkeypatch.setattr(
        epf,
        "_sample_feature",
        lambda feat, rows, cols, crop: np.full((len(rows), 1), feat),
    )
    records, failure_count, details = epf.extract_feature_records(
        index,
        model=object(),
        device="cpu",
        img_size=1,
        crop_size=1,
        infcmp=MostlyBadInference(),
        store=store,
        feat_cols=["f000"],
        aux_names=None,
        truth_col="unified_class",
        dominant_frac=0.7,
        log=messages.append,
    )
    assert len(records) == 1
    assert failure_count == n_tiles - 1
    assert len(details) == epf._MAX_FAILED_TILE_DETAILS
    assert any("additional tile failures omitted" in line for line in messages)


def test_require_npz_key_cli_is_repeatable() -> None:
    tree = ast.parse((ROOT / "scripts" / "extract_plot_features.py").read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--require-npz-key"
    ]
    assert len(calls) == 1
    kwargs = {
        kw.arg: kw.value.value for kw in calls[0].keywords
        if kw.arg and isinstance(kw.value, ast.Constant)
    }
    assert kwargs["action"] == "append"


def _inventory_record(tile_name: str, path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "tile_name": tile_name,
        "file_name": path.name,
        "tile_path": str(path),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_frozen_inventory_must_exactly_cover_extract_index(
    tmp_path: Path,
) -> None:
    first = tmp_path / "tile-a.npz"
    second = tmp_path / "tile-b.npz"
    np.savez(first, spectral=np.zeros(1))
    np.savez(second, spectral=np.ones(1))
    index = pd.DataFrame([
        _row("tile-a", first, "p-a"),
        _row("tile-b", second, "p-b"),
    ])
    inventory = tmp_path / "lucas_crop_split.json"
    inventory.write_text(json.dumps({
        "distill_tile_inventory": [
            _inventory_record("tile-a", first),
            _inventory_record("tile-b", second),
        ]
    }))

    identities = epf.load_tile_inventory(
        inventory, partition="distill", index_df=index
    )

    assert set(identities) == {str(first), str(second)}
    document = json.loads(inventory.read_text())
    document["distill_tile_inventory"].pop()
    inventory.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not exactly match"):
        epf.load_tile_inventory(
            inventory, partition="distill", index_df=index
        )


def test_extraction_forwards_frozen_tile_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile = tmp_path / "tile-a.npz"
    np.savez(tile, spectral=np.zeros(1))
    identity = _inventory_record("tile-a", tile)
    index = pd.DataFrame([_row("tile-a", tile, "p-a")])
    observed: list[dict] = []
    store = {"feat": None}

    class Inference:
        @staticmethod
        def run_inference(_model, _path, _device, **kwargs):
            observed.append(kwargs)
            store["feat"] = 1

    monkeypatch.setattr(
        epf,
        "_sample_feature",
        lambda feat, rows, cols, crop: np.full((len(rows), 1), feat),
    )
    records, failures, _ = epf.extract_feature_records(
        index,
        model=object(),
        device="cpu",
        img_size=1,
        crop_size=1,
        infcmp=Inference(),
        store=store,
        feat_cols=["f000"],
        aux_names=None,
        truth_col="unified_class",
        dominant_frac=0.7,
        tile_identities={
            str(tile): (identity["size"], identity["sha256"])
        },
    )

    assert failures == 0
    assert len(records) == 1
    assert observed == [{
        "img_size": 1,
        "return_probs": True,
        "aux_channel_names": None,
        "expected_tile_size": identity["size"],
        "expected_tile_sha256": identity["sha256"],
    }]


def test_frozen_tile_identity_mismatch_is_not_contained(tmp_path: Path) -> None:
    tile = tmp_path / "tile-a.npz"
    tile.write_bytes(b"changed")
    index = pd.DataFrame([_row("tile-a", tile, "p-a")])

    class Inference:
        class TileIdentityError(RuntimeError):
            pass

        @staticmethod
        def run_inference(_model, _path, _device, **_kwargs):
            raise Inference.TileIdentityError("sha256 mismatch")

    with pytest.raises(Inference.TileIdentityError, match="sha256 mismatch"):
        epf.extract_feature_records(
            index,
            model=object(),
            device="cpu",
            img_size=1,
            crop_size=1,
            infcmp=Inference(),
            store={"feat": None},
            feat_cols=["f000"],
            aux_names=None,
            truth_col="unified_class",
            dominant_frac=0.7,
            tile_identities={str(tile): (len(b"changed"), "0" * 64)},
        )


def test_sealed_index_authenticates_inventory_before_any_tile_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile = tmp_path / "tile-a.npz"
    index = pd.DataFrame([_row("tile-a", tile, "p-a")])
    inventory = tmp_path / "split.json"
    expected_size = 123
    expected_sha256 = "a" * 64
    events: list[str] = []

    def inventory_first(*_args, **_kwargs):
        events.append("inventory")
        return {str(tile): (expected_size, expected_sha256)}

    monkeypatch.setattr(epf, "load_tile_inventory", inventory_first)
    monkeypatch.setattr(
        epf.os.path,
        "exists",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("sealed mode must not probe path existence")
        ),
    )
    monkeypatch.setattr(
        epf,
        "filter_index_by_npz_requirements",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed mode must not run unauthenticated NPZ probes")
        ),
    )
    monkeypatch.setattr(
        epf.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed mode must not call direct numpy.load")
        ),
    )

    class Sample:
        def __contains__(self, key):
            return key == "spectral"

        def __getitem__(self, key):
            assert key == "spectral"
            return np.zeros((6, 8, 8), dtype=np.float32)

        def close(self):
            events.append("sample-close")

    class Buffer:
        def close(self):
            events.append("buffer-close")

    class Inference:
        @staticmethod
        def _load_npz_for_inference(path, **kwargs):
            events.append("authenticated-sample")
            assert str(path) == str(tile)
            assert kwargs == {
                "expected_size": expected_size,
                "expected_sha256": expected_sha256,
            }
            return Sample(), Buffer()

    prepared, identities, crop_size = epf.prepare_plot_index(
        index,
        tile_inventory=inventory,
        tile_inventory_partition="distill",
        required_keys=("s1_vv_vh",),
        img_size=8,
        infcmp=Inference(),
        log=lambda _message: None,
    )

    assert events == [
        "inventory",
        "authenticated-sample",
        "sample-close",
        "buffer-close",
    ]
    assert identities == {str(tile): (expected_size, expected_sha256)}
    assert crop_size == 8
    assert prepared["point_id"].tolist() == ["p-a"]


def test_crop_parquet_output_uses_exclusive_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "features.parquet"
    frame = pd.DataFrame({"point_id": ["p-a"], "f000": [1.25]})
    payload = b"PAR1descriptor-backed-testPAR1"

    def write_to_open_file(_frame, destination, *, index):
        assert index is False
        assert not isinstance(destination, (str, Path))
        destination.write(payload)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_to_open_file)

    epf.write_parquet_create_only(frame, output)

    assert output.read_bytes() == payload
    with pytest.raises(RuntimeError, match="refusing to create crop output"):
        epf.write_parquet_create_only(frame, output)


def test_crop_parquet_output_round_trips_with_pyarrow(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    output = tmp_path / "features.parquet"
    frame = pd.DataFrame({"point_id": ["p-a"], "f000": [1.25]})

    epf.write_parquet_create_only(frame, output)

    assert pd.read_parquet(output).equals(frame)


def test_crop_parquet_output_rejects_precreated_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"do-not-overwrite")
    output = tmp_path / "features.parquet"
    output.symlink_to(target)
    frame = pd.DataFrame({"point_id": ["p-a"], "f000": [1.25]})

    with pytest.raises(RuntimeError, match="refusing to create crop output"):
        epf.write_parquet_create_only(frame, output)

    assert target.read_bytes() == b"do-not-overwrite"


def test_extractor_uses_one_safe_checkpoint_load() -> None:
    source = inspect.getsource(epf.main)
    assert "prepare_plot_index(" in source
    assert "filter_index_by_npz_requirements(" not in source
    assert "np.load(" not in source
    assert "os.path.exists" not in source
    assert "expected_checkpoint_size=args.checkpoint_size" in source
    assert "expected_checkpoint_sha256=args.checkpoint_sha256" in source
    assert "return_checkpoint_config=True" in source
    assert "torch.load" not in source
