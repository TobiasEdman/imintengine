"""Synthetic CPU integration and fail-closed checks; no private data or GPUs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import r5_tessera_pilot as pilot


def rows(tile, offset, *, year=2022):
    return pd.DataFrame([dict(tile_name=tile, point_id=offset + i, row=8 + i,
                              col=10, unified_class=11 + i, Year=year)
                         for i in range(11)])


def population():
    train, hold = rows("synthetic_train", 0), rows("synthetic_hold", 100)
    overlapping = rows("synthetic_other_year", 100, year=2021)
    source = pd.concat([train, hold, overlapping], ignore_index=True)
    return pd.concat([train, overlapping], ignore_index=True), hold, source


def test_identity_exclusion_crosses_years_and_tiles():
    train, hold, source = population()
    clean, forbidden, counts = pilot.exclude_protected(train, hold, source, train.tile_name.unique())
    assert clean.point_id.tolist() == list(range(11))
    assert set(forbidden) == {"synthetic_hold", "synthetic_other_year"}
    assert counts["overlapping_point_ids"] == 11  # recomputed, never hardcoded 404
    assert counts["excluded_train_tiles"] == 1


@pytest.mark.parametrize("bad_id", ["0100", "100.0", True, 100.0, None])
def test_identity_aliases_fail(bad_id):
    train, hold, source = population()
    source["point_id"] = source.point_id.astype(object)
    source.loc[0, "point_id"] = bad_id
    with pytest.raises(ValueError, match="point_id"):
        pilot.exclude_protected(train, hold, source, train.tile_name.unique())


def test_removed_class_does_not_relax_protection():
    train, hold, source = population()
    train = train[~((train.tile_name == "synthetic_train") & (train.unified_class == 21))]
    with pytest.raises(ValueError, match="all 11"):
        pilot.exclude_protected(train, hold, source, train.tile_name.unique())


def test_source_truth_drift_fails():
    train, hold, source = population()
    source.loc[0, "unified_class"] = 12
    with pytest.raises(ValueError, match="mismatch"):
        pilot.exclude_protected(train, hold, source, train.tile_name.unique())


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Same path, model and geometry rules at a small CPU fixture resolution.
    monkeypatch.setattr(pilot, "TILE_SIZE", 64)
    monkeypatch.setattr(pilot, "IMAGE_SIZE", 56)
    train, hold, source = population()
    roots = {key: tmp_path / key for key in ("data_dir", "label_dir", "frac_dir")}
    for root in roots.values():
        root.mkdir()
    rng = np.random.default_rng(42)
    for index, name in enumerate(("synthetic_train", "synthetic_hold", "synthetic_val")):
        np.savez_compressed(roots["data_dir"] / f"{name}.npz",
                            tessera=rng.normal(size=(128, 64, 64)).astype(np.float16),
                            tessera_source=pilot.TESSERA_SOURCE, has_tessera=1,
                            tessera_year=2022, year=2022, has_frame_2016=1,
                            easting=index * 10000, northing=0,
                            spectral=np.ones((24, 64, 64), np.float32) * 0.1)
        label = np.full((64, 64), 11, np.uint8)
        label[20:24] = 2  # forest must survive the crop treatment
        np.savez_compressed(roots["label_dir"] / f"{name}.npz", label=label,
                            parcel_area_ha=np.ones((64, 64), np.float32),
                            label_mask=np.ones((64, 64), np.uint8),
                            custom_provenance=np.str_("synthetic-r4"))
        np.savez_compressed(roots["frac_dir"] / f"{name}.npz",
                            frac=np.ones((4, 64, 64), np.float32) * 25,
                            frac_unreliable=np.zeros((64, 64), bool))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for frame, name in ((train, pilot.INDEX_NAME), (hold, pilot.VALIDATOR_HOLDOUT_INDEX_NAME)):
        frame.drop(columns="Year").to_parquet(frozen / name, index=False)
    (frozen / "manifest.json").write_text(json.dumps({"artifacts": {
        name: pilot.identity(frozen / name)["sha256"]
        for name in (pilot.INDEX_NAME, pilot.VALIDATOR_HOLDOUT_INDEX_NAME)}}))
    source.to_parquet(tmp_path / "source.parquet", index=False)
    (tmp_path / "cohort.json").write_text(json.dumps({
        "train": ["synthetic_train", "synthetic_other_year"], "val": ["synthetic_val"]}))
    nfi = tmp_path / "nfi"
    nfi.mkdir()
    pd.DataFrame({"synthetic": [1]}).to_parquet(nfi / "holdout.parquet")
    (nfi / "manifest.json").write_text(json.dumps({"schema": "nfi-fieldtruth-freeze-v1",
        "holdout": {"observations": 1, "sha256": pilot.identity(nfi / "holdout.parquet")["sha256"]}}))
    import torch
    from imint.fm.registry import build_backbone
    from imint.fm.upernet import build_segmentation_from_spec
    torch.set_num_threads(1)
    torch.manual_seed(42)
    enc, spec = build_backbone("tessera_v1", num_frames=1, pretrained=False)
    model = build_segmentation_from_spec(spec, encoder=enc, num_classes=28, img_size=56,
        n_aux_channels=11, enable_tradslag_head=True, num_tradslag=4, aux_fusion="gated")
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "metrics": {"miou": 0.0},
                "config": {"backbone_name": "tessera_v1", "num_classes": 28,
                           "n_aux_channels": 11, "enabled_aux_names": ["height", "volume", "basal_area", "diameter", "dem",
                               "vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv", "markfukt"],
                           "num_temporal_frames": 1, "enable_tradslag_head": True,
                           "aux_fusion": "gated", "img_size": 56}}, checkpoint)
    inputs = {"crop_freeze": pilot.identity(frozen / "manifest.json"),
              "source_index": pilot.identity(tmp_path / "source.parquet"),
              "nfi_freeze": pilot.identity(nfi / "manifest.json"),
              "teacher_checkpoint": pilot.identity(checkpoint),
              "cohort": pilot.identity(tmp_path / "cohort.json"),
              **{key: str(path) for key, path in roots.items()}}
    out = tmp_path / "bundle"
    manifest = pilot.prepare(inputs, out)
    return inputs, out, manifest


def test_prepare_readback_and_immutable_bundle(prepared):
    inputs, out, manifest = prepared
    assert manifest["splits"]["train"] == ["synthetic_train"]
    assert manifest["counts"]["overlapping_point_ids"] == 11
    assert len(pd.read_parquet(out / "train.parquet")) == 11
    assert len(pd.read_parquet(out / "holdout.parquet")) == 11
    loaded = pilot.load_bundle(out / "manifest.json", pilot.identity(out / "manifest.json")["sha256"])
    assert loaded == manifest
    with pytest.raises(FileExistsError):
        pilot.prepare(inputs, out)
    path = Path(manifest["inventory"]["synthetic_train"]["fraction"]["path"])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        pilot.load_bundle(out / "manifest.json", pilot.identity(out / "manifest.json")["sha256"])


def test_real_cpu_extraction_fit_dense_overlay_readback(prepared, tmp_path):
    _, _, manifest = prepared
    result = pilot.produce(manifest, tmp_path / "produced", "cpu")
    assert result["state"] == "sidecars_produced_not_training_approved"
    assert list(result["training_labels"]) == ["synthetic_train"]
    assert not (tmp_path / "produced/labels/synthetic_hold.npz").exists()
    assert not (tmp_path / "produced/labels/synthetic_val.npz").exists()
    original_path = manifest["inventory"]["synthetic_train"]["label"]["path"]
    output_path = result["training_labels"]["synthetic_train"]["path"]
    with np.load(original_path) as before, np.load(output_path) as after:
        for key in before.files:
            if key != "label":
                np.testing.assert_array_equal(before[key], after[key])
        allowed = np.zeros((64, 64), bool)
        allowed[4:-4, 4:-4] = np.isin(before["label"][4:-4, 4:-4], pilot.CROP_CLASSES)
        np.testing.assert_array_equal(before["label"][~allowed], after["label"][~allowed])
        assert np.any(before["label"][allowed] != after["label"][allowed])
        assert np.isin(after["label"][allowed], pilot.CROP_CLASSES).all()
    features = pd.read_parquet(result["features"]["path"])
    assert len(features) == 11 and len([c for c in features if c.startswith("f")]) == 128


@pytest.mark.parametrize("change", ["source", "empty", "year", "frame"])
def test_v2_readiness_rejects_stale_inputs(prepared, change):
    _, _, manifest = prepared
    path = Path(manifest["inventory"]["synthetic_train"]["tile"]["path"])
    with np.load(path) as z:
        payload = dict(z)
    if change == "source":
        payload["tessera_source"] = "old-v1"
    elif change == "empty":
        payload["tessera"] = np.zeros_like(payload["tessera"])
    elif change == "year":
        payload["tessera_year"] = 2021
    else:
        payload["has_frame_2016"] = 0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError):
        pilot.validate_tile(path, training=True, crop_years={2022})


def score_pair():
    cm = np.zeros((28, 28), np.int64)
    for c in pilot.CROP_CLASSES:
        cm[c, c], cm[c, 0] = 9, 1
    base = {"schema": pilot.SCHEMA, "training_mode": "train", "embedding_source": pilot.TESSERA_SOURCE,
            "metric_version": pilot.SEGMENTATION_METRIC_VERSION,
            "crop_confusion": cm.tolist(), "lulc_confusion": np.eye(28, dtype=int).tolist()}
    for field in ("protocol_sha256", "training_cohort_sha256", "training_regime_sha256",
                  "common_class_stats_sha256", "runtime_sha256",
                  "vocabulary_sha256", "crop_keys_sha256", "crop_truth_sha256",
                  "lulc_samples_sha256", "lulc_masks_sha256", "lulc_labels_sha256",
                  "evaluation_inputs_sha256"):
        base[field] = pilot.digest(field.encode())
    r4, r5 = dict(base, arm="r4_control"), deepcopy(dict(base, arm="r5"))
    r5["crop_confusion"][11][11] += 1
    r5["crop_confusion"][11][0] -= 1
    return r4, r5


def test_acceptance_uses_fixed_success_criterion():
    r4, r5 = score_pair()
    assert pilot.compare(r4, r5)["passed"]
    r5["crop_confusion"] = r4["crop_confusion"]
    assert not pilot.compare(r4, r5)["passed"]  # strict improvement required
    r4, r5 = score_pair()
    r5["lulc_confusion"][1][1] = 0
    r5["lulc_confusion"][1][0] = 1
    assert not pilot.compare(r4, r5)["passed"]


@pytest.mark.parametrize("field", ["evaluation_inputs_sha256", "lulc_masks_sha256",
    "lulc_labels_sha256", "crop_keys_sha256", "vocabulary_sha256", "training_cohort_sha256"])
def test_acceptance_rejects_unmatched_samples_masks_labels(field):
    r4, r5 = score_pair()
    r5[field] = "a" * 64
    with pytest.raises(ValueError, match=field):
        pilot.compare(r4, r5)


def test_acceptance_rejects_historical_v1_and_support_mismatch():
    r4, r5 = score_pair()
    r4["embedding_source"] = "old-v1"
    with pytest.raises(ValueError, match="embedding"):
        pilot.compare(r4, r5)
    r4, r5 = score_pair()
    r5["crop_confusion"][11][11] += 1
    with pytest.raises(ValueError, match="support"):
        pilot.compare(r4, r5)


def test_crop_border_and_raw_sjv_metadata_remain_unchanged(monkeypatch):
    monkeypatch.setattr(pilot, "TILE_SIZE", 12)
    monkeypatch.setattr(pilot, "IMAGE_SIZE", 4)
    labels = np.full((12, 12), 11, dtype=np.uint8)
    mask = np.ones((12, 12), dtype=np.uint8)
    mask[5, 5] = 0
    original = {"label": labels, "label_mask": mask, "forest_head_sha": np.str_("keep")}
    result = pilot.crop_overlay(original, np.full((4, 4), 12, dtype=np.uint8))
    assert result["label"][5, 5] == 12  # label_mask stores SJV codes, not validity
    assert result["label"][4, 4] == 12
    np.testing.assert_array_equal(result["label"][:4], labels[:4])
    np.testing.assert_array_equal(original["label"], labels)
    np.testing.assert_array_equal(result["label_mask"], mask)
    assert result["forest_head_sha"] == "keep"


def test_acceptance_accepts_exact_half_point_boundary():
    r4, r5 = score_pair()
    baseline = np.eye(28, dtype=np.int64) * 2000
    treatment = baseline.copy()
    for c in range(1, 28):
        treatment[c, c] = 1990
        treatment[c, 0] = 10
    r4["lulc_confusion"], r5["lulc_confusion"] = baseline.tolist(), treatment.tolist()
    result = pilot.compare(r4, r5)
    assert result["passed"] and result["lulc_miou_delta"] == pytest.approx(-0.005)


def test_experiment_declaration_matches_executable_contract():
    declaration = json.loads((Path(__file__).parents[1] / "config/r5_tessera_pilot.json").read_text())
    assert declaration["schema"] == pilot.SCHEMA
    assert declaration["embedding_source"] == pilot.TESSERA_SOURCE
    assert declaration["metric_version"] == pilot.SEGMENTATION_METRIC_VERSION
    assert declaration["crop_classes"] == list(pilot.CROP_CLASSES)
    assert declaration["image_size"] == pilot.IMAGE_SIZE
    assert declaration["tile_size"] == pilot.TILE_SIZE
    assert declaration["students"]["arms"] == ["r4_control", "r5"]
    assert declaration["queue_integration"] == "none"


def test_provenance_and_one_epoch_paired_student_evaluation(prepared, tmp_path, monkeypatch):
    from scripts import run_r5_tessera as runner
    _, out, bundle = prepared
    bundle_sha = pilot.identity(out / "manifest.json")["sha256"]
    runtime = {"image": "synthetic-cpu-fixture", "source_git_sha": "test-only"}
    produced = pilot.produce(bundle, tmp_path / "produced", "cpu",
                             runtime=runtime, bundle_sha256=bundle_sha)
    persisted = json.loads((tmp_path / "produced/manifest.json").read_text())
    assert persisted["bundle_sha256"] == bundle_sha and persisted["runtime"] == runtime
    monkeypatch.setitem(runner.STUDENT_REGIME, "batch_size", 1)
    monkeypatch.setitem(runner.STUDENT_REGIME, "num_workers", 0)
    records = []
    for arm in ("r4_control", "r5"):
        trained = runner.train_arm(bundle, produced, arm, tmp_path / arm, device="cpu",
                                   probe=True, bundle_sha256=bundle_sha, runtime=runtime)
        assert trained["state"] == "probe_complete" and trained["completed_epochs"] == 1
        assert trained["production"]["manifest"] == produced
        assert trained["production"]["canonical_sha256"] == pilot.digest(pilot.json_bytes(produced))
        expected_label = (produced["training_labels"]["synthetic_train"] if arm == "r5"
                          else bundle["inventory"]["synthetic_train"]["label"])
        assert trained["training_labels"]["synthetic_train"] == expected_label
        import torch
        checkpoint = torch.load(trained["checkpoint"]["path"], weights_only=False)
        assert checkpoint["config"]["n_aux_channels"] == 11
        assert checkpoint["config"]["enabled_aux_names"] == list(EXPECTED_R4_AUX)
        record = runner.evaluate_arm(bundle, trained, tmp_path / (arm + "_eval"),
                                     bundle_sha256=bundle_sha, runtime=runtime, device="cpu")
        assert record["training_manifest"]["manifest"] == trained
        assert np.asarray(record["crop_confusion"]).sum() == 11
        assert np.asarray(record["lulc_confusion"]).sum() == 56 * 56
        assert (tmp_path / (arm + "_eval") / "record.json").is_file()
        records.append(record)
    for key in records[0]:
        if key.endswith("sha256"):
            assert records[0][key] == records[1][key], key
    with pytest.raises(ValueError):
        pilot.compare(*records)  # a one-epoch probe cannot certify pilot success


def test_explicit_dataset_membership_rejects_missing_and_no_retry(prepared):
    from imint.training.unified_dataset import UnifiedDataset
    _, _, bundle = prepared
    root = bundle["roots"]["data_dir"]
    with pytest.raises(ValueError, match="membership"):
        UnifiedDataset(lulc_dir=root, backbone_family="tessera", tile_names=["absent.npz"])
    ds = UnifiedDataset(lulc_dir=root, backbone_family="tessera", patch_size=56,
                        tile_names=["synthetic_train.npz", "synthetic_val.npz"])
    assert ds.tile_names == ["synthetic_train.npz", "synthetic_val.npz"]
    Path(root, "synthetic_train.npz").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="unreadable"):
        ds[0]


def test_spatial_identity_is_independent_of_tile_name_and_year():
    inventory = {
        "train": {"bbox_3006": {"west": 0, "east": 5120, "south": 0, "north": 5120}},
        "other_year": {"bbox_3006": {"west": 10, "east": 5130, "south": 0, "north": 5120}},
    }
    with pytest.raises(ValueError, match="footprint"):
        pilot.verify_spatial_partition(["train"], {"other_year"}, inventory)


def test_runtime_archive_and_ci_include_r5_payload():
    import importlib.util
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("pipeline_selector", root / ".github/scripts/select_pipeline_images.py")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    for path in ("scripts/r5_tessera_pilot.py", "scripts/run_r5_tessera.py",
                 "config/r5_tessera_pilot.json", "imint/training/unified_dataset.py"):
        for event in ("pull_request", "push"):
            assert selector.select(event, "refs/heads/main", "", [path]) == ["ladder-crop-distill"]
    source = (root / "docker/ladder-crop-distill/prepare_runtime.py").read_text()
    assert '"config",' in source and '"imint",' in source and '"scripts",' in source
    smoke = (root / "docker/ladder-crop-distill/runtime_smoke.py").read_text()
    assert "from scripts import r5_tessera_pilot, run_r5_tessera" in smoke


EXPECTED_R4_AUX = ("height", "volume", "basal_area", "diameter", "dem", "vpp_sosd",
                   "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv", "markfukt")


def test_student_aux_matches_r4_training_entrypoint():
    import ast
    from scripts import run_r5_tessera as runner
    from imint.training.config import TrainingConfig
    root = Path(__file__).parents[1]
    cfg = TrainingConfig()
    for key, value in runner.STUDENT_REGIME.items():
        setattr(cfg, key, value)
    assert cfg.enabled_aux_names == EXPECTED_R4_AUX
    declaration = json.loads((root / "config/r5_tessera_pilot.json").read_text())
    assert declaration["students"]["aux_channel_names"] == list(EXPECTED_R4_AUX)
    # The canonical R4 entrypoint must still enable these six groups.
    tree = ast.parse((root / "scripts/train_unified.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "TrainingConfig"
                and any(k.arg == "enable_height_channel" for k in n.keywords))
    enabled = {k.arg for k in call.keywords if isinstance(k.value, ast.Constant) and k.value.value is True}
    assert {"enable_height_channel", "enable_volume_channel", "enable_basal_area_channel",
            "enable_diameter_channel", "enable_dem_channel", "enable_vpp_channels"} <= enabled


def test_explicit_dataset_ignores_unbound_nested_tiles(prepared):
    from imint.training.unified_dataset import UnifiedDataset
    _, _, bundle = prepared
    root = Path(bundle["roots"]["data_dir"])
    nested = root / "tiles"
    nested.mkdir()
    (nested / "synthetic_train.npz").write_bytes(b"unbound corrupt decoy")
    ds = UnifiedDataset(lulc_dir=root, backbone_family="tessera", patch_size=56,
                        tile_names=["synthetic_train.npz"])
    assert Path(ds._entries[0]["path"]).resolve() == root / "synthetic_train.npz"
    assert ds[0]["spectral"].shape[-2:] == (56, 56)


@pytest.mark.parametrize("change", ["corrupt", "shape", "nonfinite", "range", "mask", "missing"])
def test_invalid_fraction_fails_preparation_and_explicit_loader(prepared, tmp_path, change):
    from imint.training.unified_dataset import UnifiedDataset
    inputs, _, bundle = prepared
    path = Path(bundle["inventory"]["synthetic_train"]["fraction"]["path"])
    with np.load(path) as z:
        payload = dict(z)
    if change == "corrupt":
        path.write_bytes(b"corrupt")
    else:
        if change == "shape":
            payload["frac"] = payload["frac"][:, :-1]
        elif change == "nonfinite":
            payload["frac"][0, 0, 0] = np.nan
        elif change == "range":
            payload["frac"][0, 0, 0] = 101
        elif change == "mask":
            payload["frac_unreliable"] = np.full((64, 64), 2, np.uint8)
        else:
            del payload["frac"]
        np.savez_compressed(path, **payload)
    with pytest.raises((ValueError, KeyError)):
        pilot.prepare(inputs, tmp_path / "reprepared")
    ds = UnifiedDataset(lulc_dir=bundle["roots"]["data_dir"], backbone_family="tessera", patch_size=56,
                        tile_names=["synthetic_train.npz"], frac_dir=bundle["roots"]["frac_dir"])
    with pytest.raises(ValueError, match="bound fraction"):
        ds[0]
