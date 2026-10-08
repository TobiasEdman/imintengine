"""Consumed bytes retain their identity even if paths are replaced afterward."""
import json
import sys
import numpy as np
import pandas as pd
import pytest
from imint.eval import fieldtruth as ft
from scripts import compare_nmd2023_nfi as comparison, score_nfi_holdout as scoring
from tests.nfi_freeze_fixtures import complete_manifest


@pytest.fixture
def frozen(tmp_path):
    held = pd.DataFrame({"TractID": [1], "PlotID": [1], "Year": [2024], "tile_name": ["a"],
                         "nfi_forest": [1], "Easting": [500000.], "Northing": [6500000.]})
    manifest = complete_manifest({"schema": "nfi-fieldtruth-freeze-v1", "campaign": {"training_tiles": 0},
                                  "cells": {"clay_r1": {"checkpoint": {"sha256": "a"*64}}}})
    for name, frame in (("holdout", held), ("training", held[ft.NFI_KEY].iloc[:0])):
        p = tmp_path / (name + ".parquet"); frame.to_parquet(p)
        manifest[name] = {"file": p.name, "sha256": ft.sha256_file(p), "observations": len(frame)}
    raster = tmp_path / "nmd.tif"; raster.write_bytes(b"approved-raster-placeholder")
    manifest["baselines"] = {"NMD2023": {"path": str(raster), "bytes": raster.stat().st_size, "sha256": ft.sha256_file(raster)}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest)); digest = ft.sha256_file(path)
    for cell in manifest["cells"]:
        p = tmp_path / f"nfi-per-plot-{cell}.parquet"; held.assign(model_pred=1).to_parquet(p)
        meta = dict(ft.evaluation_runtime_identity(manifest), cell=cell, holdout_manifest_sha256=digest,
                    checkpoint_sha256="a"*64, status="success", prediction_sha256=ft.sha256_file(p))
        p.with_suffix(".parquet.meta.json").write_text(json.dumps(meta))
    return held, manifest, path, digest, raster, tmp_path / "nfi-per-plot-clay_r1.parquet"


def test_compare_rejects_raster_changed_during_sampling(frozen, tmp_path, monkeypatch):
    held, manifest, path, digest, raster, dump = frozen
    def replace(*args):
        raster.write_bytes(b"changed-raster")
        return np.array([1]), np.array([111])
    monkeypatch.setattr(comparison, "verify_evaluation_source", ft.evaluation_runtime_identity)
    monkeypatch.setattr(comparison, "sample_nmd_unified", replace)
    out = tmp_path / "compare.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--plots", "unused", "--nmd2023", str(raster),
        "--model-per-plot", str(dump), "--model-id", "clay_r1", "--holdout-manifest", str(path),
        "--expected-manifest-sha256", digest, "--out", str(out)])
    with pytest.raises(ValueError, match="frozen input content changed"):
        comparison.main()
    assert not out.exists()


def test_score_reports_consumed_dump_hash_not_replacement(frozen, tmp_path, monkeypatch):
    held, manifest, path, digest, raster, dump = frozen
    approved_dump = ft.sha256_file(dump)
    original = scoring.verify_prediction_dump
    def replace_after_read(p, *args, **kwargs):
        frame = original(p, *args, **kwargs)
        if p == dump: held.assign(model_pred=4).to_parquet(p)
        return frame
    monkeypatch.setattr(scoring, "verify_evaluation_source", ft.evaluation_runtime_identity)
    monkeypatch.setattr(scoring, "verify_prediction_dump", replace_after_read)
    monkeypatch.setattr(scoring, "sample_nmd_unified", lambda *a: (np.array([1]), np.array([111])))
    monkeypatch.setattr(scoring, "paired_report", lambda h,p,b,pr: {"compared_observations": 1, "consumed": p["clay_r1"].tolist()})
    out = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", ["score", "--holdout-manifest", str(path), "--expected-manifest-sha256", digest,
                                     "--dump-dir", str(tmp_path), "--nmd2023", str(raster), "--out", str(out)])
    scoring.main()
    result = json.loads(out.read_text())
    assert result["consumed"] == [1]
    assert result["prediction_sha256"]["clay_r1"] == approved_dump != ft.sha256_file(dump)


def test_dump_must_bind_runtime_identity(frozen):
    held, manifest, path, digest, raster, dump = frozen
    sidecar = dump.with_suffix(".parquet.meta.json")
    meta = json.loads(sidecar.read_text()); meta["source_payload_sha256"] = "f"*64
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenance"):
        ft.verify_prediction_dump(dump, path, "clay_r1", expected_manifest_sha256=digest)
