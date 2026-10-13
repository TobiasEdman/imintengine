"""An approved external digest and sealed runtime bind frozen evaluations."""
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from imint.eval import fieldtruth as ft
from tests.nfi_freeze_fixtures import complete_manifest


def fixture(tmp_path):
    held = pd.DataFrame({"TractID": [1], "PlotID": [1], "Year": [2024], "tile_name": ["a"]})
    trained = held[ft.NFI_KEY].iloc[:0]
    manifest = complete_manifest({"schema": "nfi-fieldtruth-freeze-v1", "campaign": {"training_tiles": 0}})
    for label, frame in (("holdout", held), ("training", trained)):
        path = tmp_path / (label + ".parquet"); frame.to_parquet(path)
        manifest[label] = {"file": path.name, "sha256": ft.sha256_file(path), "observations": len(frame)}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    return path, manifest


def test_edited_manifest_recomputed_internal_hashes_rejected(tmp_path):
    path, manifest = fixture(tmp_path)
    approved = ft.sha256_file(path)
    ft.load_frozen_holdout(path, approved)
    held = pd.read_parquet(tmp_path / "holdout.parquet").assign(PlotID=99)
    held.to_parquet(tmp_path / "holdout.parquet")
    manifest["holdout"]["sha256"] = ft.sha256_file(tmp_path / "holdout.parquet")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="approved SHA256"):
        ft.load_frozen_holdout(path, approved)
    with pytest.raises(ValueError, match="approved SHA256"):
        ft.load_frozen_holdout(path, None)


@pytest.mark.parametrize("field", ["source_sha256", "cells", "baselines", "protocol"])
def test_incomplete_freeze_rejected_even_with_matching_hash(tmp_path, field):
    path, manifest = fixture(tmp_path)
    manifest[field] = {}; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        ft.load_frozen_holdout(path, ft.sha256_file(path))


def test_full_execution_tree_and_runtime_checked(tmp_path, monkeypatch):
    from scripts import crop_distill_provenance as provenance
    _, manifest = fixture(tmp_path)
    frozen = manifest["preparation_runtime"]
    monkeypatch.setattr(provenance, "verify_runtime", lambda *a, **k: copy.deepcopy(frozen))
    monkeypatch.setattr(provenance, "snapshot_tree", lambda root: [])
    monkeypatch.setattr(provenance, "tree_payload_sha256", lambda files: frozen["source"]["payload_sha256"])
    assert ft.verify_evaluation_source(manifest)["runtime_image"] == manifest["runtime_image"]
    monkeypatch.setattr(provenance, "tree_payload_sha256", lambda files: "f" * 64)
    with pytest.raises(ValueError, match="execution source tree"):
        ft.verify_evaluation_source(manifest)
    changed = copy.deepcopy(frozen); changed["runtime_manifest"]["sha256"] = "e" * 64
    monkeypatch.setattr(provenance, "verify_runtime", lambda *a, **k: changed)
    with pytest.raises(ValueError, match="execution runtime"):
        ft.verify_evaluation_source(manifest)
