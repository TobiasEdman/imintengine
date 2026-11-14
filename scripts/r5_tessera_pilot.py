#!/usr/bin/env python3
"""Prepare and produce the TESSERA-only R5 crop-label treatment.

All outputs are private, create-only artifacts. This module never submits a
workload or starts student training. ``prepare`` binds the current v2 inputs;
``produce`` uses the existing extractor/MLP/dense inference pipeline. Production
execution requires a reviewed runtime and a separately approved GPU budget.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_lucas_crop_split import (
    CROP_CLASSES, INDEX_NAME, VALIDATOR_HOLDOUT_INDEX_NAME, _point_id_values,
)
from imint.metrics import SEGMENTATION_METRIC_VERSION, miou_from_confusion
from imint.training.tile_time import resolve_tile_year
from imint.training.tile_bbox import resolve_tile_bbox
from imint.training.tile_config import TileConfig
from imint.training.unified_dataset import validate_fraction_sidecar

SCHEMA = "r5-tessera-pilot/1"
TESSERA_SOURCE = "geotessera-0.10.2"
IMAGE_SIZE = 504
TILE_SIZE = 512
KEYS = ["tile_name", "point_id"]
INDEX_FIELDS = [*KEYS, "row", "col", "unified_class"]
SHA = re.compile(r"[0-9a-f]{64}")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("input changed while hashing")
    return {"path": str(path), "sha256": h.hexdigest(), "size": after.st_size}


def capture(record: dict) -> bytes:
    if not isinstance(record.get("sha256"), str) or not SHA.fullmatch(record["sha256"]):
        raise ValueError("a reviewed full SHA-256 is required for each input")
    payload = Path(record["path"]).read_bytes()
    if digest(payload) != record["sha256"]:
        raise ValueError("input bytes differ from the reviewed identity")
    return payload


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def canonical_index(frame: pd.DataFrame, *, crops_only: bool) -> pd.DataFrame:
    required = set(INDEX_FIELDS)
    if not required <= set(frame):
        raise ValueError("LUCAS index lacks identity, coordinates or labels")
    frame = frame.copy()
    ids, problem = _point_id_values(frame.point_id.tolist(), "LUCAS index")
    if problem:
        raise ValueError(problem)
    frame["point_id"] = ids
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("empty or duplicate LUCAS keys")
    for name in frame.tile_name:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("invalid tile name")
    for column in ("row", "col", "unified_class"):
        values = frame[column].to_numpy()
        if (not np.issubdtype(values.dtype, np.integer)
                or (values < 0).any()):
            raise ValueError(f"invalid integer {column}")
    if crops_only:
        if not set(frame.unified_class) <= set(CROP_CLASSES):
            raise ValueError("frozen crop truth must use classes 11..21")
        if not frame[["row", "col"]].apply(lambda c: c.between(4, 507)).all().all():
            raise ValueError("crop point outside the fixed 504 centre window")
    return frame.sort_values(KEYS).reset_index(drop=True)


def join_year(frozen: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    frozen = canonical_index(frozen, crops_only=True)
    merged = frozen[INDEX_FIELDS].merge(
        source[[*INDEX_FIELDS, "Year"]], on=KEYS, how="left",
        validate="one_to_one", suffixes=("", "_source"), indicator=True,
    )
    if not (merged._merge == "both").all():
        raise ValueError("frozen keys missing from the bound source index")
    for column in ("row", "col", "unified_class"):
        if not (merged[column] == merged[column + "_source"]).all():
            raise ValueError("frozen/source label or geometry mismatch")
    if not np.issubdtype(merged.Year.dtype, np.integer):
        raise ValueError("LUCAS Year must be an integer")
    return merged[[*INDEX_FIELDS, "Year"]]


def exclude_protected(distill: pd.DataFrame, holdout: pd.DataFrame,
                      source: pd.DataFrame, train_tiles: list[str]) -> tuple[pd.DataFrame, list[str], dict]:
    """Exclude point identity across years and tiles, then entire train tiles.

    Both student arms must use the resulting SAME train-tile set. The historic
    R4 checkpoint is only the feature teacher, not this matched R4 control.
    """
    source = canonical_index(source, crops_only=False)
    distill, holdout = join_year(distill, source), join_year(holdout, source)
    protected_ids = set(holdout.point_id)
    shared = set(distill.point_id) & protected_ids
    forbidden = set(source.loc[source.point_id.isin(protected_ids), "tile_name"])
    allowed = set(train_tiles) - forbidden
    clean = distill[~distill.point_id.isin(protected_ids) & distill.tile_name.isin(allowed)].copy()
    if set(clean.unified_class) != set(CROP_CLASSES):
        raise ValueError("clean training subset must cover all 11 crops; do not relax the holdout")
    if set(holdout.unified_class) != set(CROP_CLASSES):
        raise ValueError("protected holdout must cover all 11 crops")
    assert not set(clean.point_id) & protected_ids
    assert not set(clean.tile_name) & forbidden
    counts = {"original_distill_rows": len(distill), "protected_rows": len(holdout),
              "overlapping_point_ids": len(shared), "clean_training_rows": len(clean),
              "excluded_train_tiles": len(set(train_tiles) & forbidden)}
    return clean.reset_index(drop=True), sorted(forbidden), counts


def validate_tile(path: Path, *, training: bool, crop_years: set[int]) -> dict:
    with np.load(path, allow_pickle=False) as tile:
        emb = tile["tessera"]
        if (str(tile.get("tessera_source", "")) != TESSERA_SOURCE
                or int(tile.get("has_tessera", 0)) != 1
                or emb.shape != (128, TILE_SIZE, TILE_SIZE)
                or not np.isfinite(emb).all() or not np.any(emb)):
            raise ValueError("tile lacks complete promoted Tessera v2 embeddings")
        year = resolve_tile_year(tile)
        if year is None or int(tile.get("tessera_year", -1)) != year:
            raise ValueError("Tessera and spectral years must match exactly")
        if crop_years and crop_years != {year}:
            raise ValueError("LUCAS crop year and tile year differ")
        if training and int(tile.get("has_frame_2016", 0)) != 1:
            raise ValueError("training tile lacks required frame_2016")
        bbox = resolve_tile_bbox(name=path.stem, tile=TileConfig(size_px=TILE_SIZE), npz_data=tile)
        if bbox is None:
            raise ValueError("tile lacks geometry for spatial partition validation")
        return bbox


def verify_spatial_partition(train: list[str], protected: set[str], inventory: dict) -> None:
    """Reject positive-area footprint overlap, independent of year/name."""
    from scipy.spatial import cKDTree
    names = sorted(protected)
    def center(name):
        box = inventory[name]["bbox_3006"]
        return ((box["west"] + box["east"]) / 2, (box["south"] + box["north"]) / 2)
    centers = np.asarray([center(name) for name in names])
    tree = cKDTree(centers)
    width = TileConfig(size_px=TILE_SIZE).size_m
    for name in train:
        point = np.asarray(center(name))
        candidates = tree.query_ball_point(point, r=width, p=np.inf)
        if any(np.max(np.abs(point - centers[index])) < width for index in candidates):
            raise ValueError("training footprint overlaps validation/protected holdout; review the common cohort")


def prepare(inputs: dict, out: Path) -> dict:
    """Bind private source artifacts and emit a create-only preparation bundle."""
    if out.exists():
        raise FileExistsError("preparation output exists; never overwrite a bundle")
    records = {key: inputs[key] for key in (
        "crop_freeze", "source_index", "nfi_freeze", "teacher_checkpoint", "cohort",
    )}
    captured = {key: capture(record) for key, record in records.items()}
    freeze = json.loads(captured["crop_freeze"])
    nfi = json.loads(captured["nfi_freeze"])
    if nfi.get("schema") != "nfi-fieldtruth-freeze-v1" or not nfi.get("holdout", {}).get("observations", 0):
        raise ValueError("a completed nonempty NFI freeze is required")
    nfi_holdout = Path(records["nfi_freeze"]["path"]).parent / "holdout.parquet"
    capture({"path": str(nfi_holdout), "sha256": nfi["holdout"]["sha256"]})
    frozen_root = Path(records["crop_freeze"]["path"]).parent
    frames = {}
    for name in (INDEX_NAME, VALIDATOR_HOLDOUT_INDEX_NAME):
        bound = {"path": str(frozen_root / name), "sha256": freeze["artifacts"][name]}
        frames[name] = pd.read_parquet(io.BytesIO(capture(bound)))
        records[name] = bound
    source = canonical_index(pd.read_parquet(io.BytesIO(captured["source_index"])), crops_only=False)
    cohort = json.loads(captured["cohort"])
    for split in ("train", "val"):
        names = cohort[split]
        if not names or len(names) != len(set(names)) or any(
            not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names
        ):
            raise ValueError("cohort must contain unique safe tile stems in each split")
    if set(cohort["train"]) & set(cohort["val"]):
        raise ValueError("train/val tile overlap")
    clean, forbidden, counts = exclude_protected(
        frames[INDEX_NAME], frames[VALIDATOR_HOLDOUT_INDEX_NAME], source, cohort["train"],
    )
    train = sorted(set(cohort["train"]) - set(forbidden))
    holdout = join_year(frames[VALIDATOR_HOLDOUT_INDEX_NAME], source)
    roots = {key: str(Path(inputs[key]).resolve(strict=True))
             for key in ("data_dir", "label_dir", "frac_dir")}
    student_names = set(train) | set(cohort["val"])
    all_names = student_names | set(holdout.tile_name)
    crop_years = source[source.unified_class.isin(CROP_CLASSES)].groupby("tile_name").Year.agg(set).to_dict()
    inventory = {}
    for name in sorted(all_names):
        path = Path(roots["data_dir"]) / f"{name}.npz"
        years = crop_years.get(name, set())
        before = identity(path)
        bbox = validate_tile(path, training=name in train, crop_years=years)
        if identity(path) != before:
            raise ValueError("tile changed during preparation")
        entry = {"tile": before, "bbox_3006": bbox}
        if name in student_names:
            for kind, root in (("label", "label_dir"), ("fraction", "frac_dir")):
                entry[kind] = identity(Path(roots[root]) / f"{name}.npz")
            with np.load(io.BytesIO(capture(entry["fraction"])), allow_pickle=False) as fractions:
                validate_fraction_sidecar(fractions, TILE_SIZE, TILE_SIZE)
        inventory[name] = entry
    verify_spatial_partition(train, set(cohort["val"]) | set(holdout.tile_name), inventory)
    for frame in (clean, holdout):
        frame["tile_path"] = frame.tile_name.map(lambda name: str(Path(roots["data_dir"]) / f"{name}.npz"))
    out.mkdir(parents=True, exist_ok=False)
    clean.to_parquet(out / "train.parquet", index=False)
    holdout.to_parquet(out / "holdout.parquet", index=False)
    manifest = {
        "schema": SCHEMA, "state": "prepared_not_execution_approved", "backbone": "tessera_v1",
        "embedding_source": TESSERA_SOURCE, "image_size": IMAGE_SIZE,
        "metric_version": SEGMENTATION_METRIC_VERSION, "inputs": records, "roots": roots,
        "indices": {name: identity(out / name) for name in ("train.parquet", "holdout.parquet")},
        "splits": {"train": train, "val": sorted(cohort["val"])}, "forbidden_tiles": forbidden,
        "inventory": inventory, "counts": counts,
        "arms": {"r4_control": "R4 labels, matched filtered cohort, cold start",
                 "r5": "same regime and cohort; crop-only overlay on training labels"},
        "success": {"crop_accuracy_delta_strictly_above": 0.0, "max_lulc_miou_drop": 0.005},
    }
    for record in records.values():
        capture(record)
    (out / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def load_bundle(path: Path, expected_sha256: str) -> dict:
    bundle = json.loads(capture({"path": str(path), "sha256": expected_sha256}))
    if bundle.get("schema") != SCHEMA or bundle.get("embedding_source") != TESSERA_SOURCE:
        raise ValueError("unsupported pilot bundle")
    for record in bundle["inputs"].values():
        capture(record)
    for record in bundle["indices"].values():
        capture(record)
    for entry in bundle["inventory"].values():
        for key in ("tile", "label", "fraction"):
            if key not in entry:
                continue
            record = entry[key]
            if identity(Path(record["path"])) != record:
                raise ValueError("bound dataset changed after preparation")
    return bundle


def fit_head(features: pd.DataFrame, expected: pd.DataFrame) -> dict:
    """Existing 128-hidden standardized MLP, fitted only on the clean partition.

    No random point-wise early-stopping validation: all clean training rows
    fit the head; the protected validator is never used for model selection.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from train_distill_head import feature_cols, head_forward_npz

    ids, problem = _point_id_values(features.point_id.tolist(), "features")
    if problem:
        raise ValueError(problem)
    features = features.assign(point_id=ids).sort_values(KEYS).reset_index(drop=True)
    expected = expected.sort_values(KEYS).reset_index(drop=True)
    for column in (*KEYS, "unified_class"):
        if features[column].tolist() != expected[column].tolist():
            raise ValueError("feature keys/order/truth differ from clean training index")
    cols = feature_cols(features)
    if cols != [f"f{i:03d}" for i in range(128)]:
        raise ValueError("Tessera requires all 128 native features")
    X = features[cols].to_numpy(dtype=np.float32)
    y = expected.unified_class.to_numpy(dtype=np.int64)
    if not np.isfinite(X).all() or set(y) != set(CROP_CLASSES):
        raise ValueError("nonfinite features or incomplete crop vocabulary")
    scaler = StandardScaler().fit(X)
    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                          early_stopping=False, random_state=42)
    model.fit(scaler.transform(X), y)
    head = {"mean_": scaler.mean_.astype(np.float32), "scale_": scaler.scale_.astype(np.float32),
            "W1": model.coefs_[0].astype(np.float32), "b1": model.intercepts_[0].astype(np.float32),
            "W2": model.coefs_[1].astype(np.float32), "b2": model.intercepts_[1].astype(np.float32),
            "classes_": model.classes_.astype(np.int64)}
    if not np.array_equal(head_forward_npz(X, **head), model.predict(scaler.transform(X))):
        raise ValueError("exported numpy head differs from sklearn")
    return head


def crop_overlay(sidecar: dict, prediction: np.ndarray) -> dict:
    """Replace only original crop-class pixels in the centre inference window."""
    label = np.asarray(sidecar["label"])
    if label.shape != (TILE_SIZE, TILE_SIZE) or prediction.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("R5 requires a 512 label and a 504 centre prediction")
    if not np.issubdtype(label.dtype, np.integer) or not np.isin(label, range(28)).all():
        raise ValueError("invalid 28-class R4 labels")
    if not np.isin(prediction, CROP_CLASSES).all():
        raise ValueError("crop head predicted outside classes 11..21")
    result = {key: np.array(value, copy=True) for key, value in sidecar.items()}
    off = (TILE_SIZE - IMAGE_SIZE) // 2
    window = result["label"][off:-off, off:-off]
    mask = np.isin(window, CROP_CLASSES)
    window[mask] = prediction[mask]
    return result


def produce(bundle: dict, out: Path, device: str, *, runtime: dict | None = None,
            bundle_sha256: str | None = None) -> dict:
    """Extract R4 features, fit the crop head, and publish train-only sidecars."""
    if out.exists():
        raise FileExistsError("production output exists; use a fresh private directory")
    import torch
    from distill_forest_labels import _load_inference_comparison, _upsample_feat_to, head_predict_dense
    from extract_plot_features import extract_feature_records, register_preclassifier_hook

    infcmp = _load_inference_comparison()
    checkpoint = bundle["inputs"]["teacher_checkpoint"]
    model, _, _, size, config = infcmp.load_model(
        checkpoint["path"], torch.device(device), backbone_name="tessera_v1", img_size=IMAGE_SIZE,
        expected_checkpoint_size=Path(checkpoint["path"]).stat().st_size,
        expected_checkpoint_sha256=checkpoint["sha256"], return_checkpoint_config=True,
    )
    if size != IMAGE_SIZE or not config.get("enabled_aux_names"):
        raise ValueError("teacher must declare its 504 grid and ordered aux channels")
    store = register_preclassifier_hook(model)
    if store["n_features"] != 128:
        store["handle"].remove()
        raise ValueError("R4 teacher must expose 128 gated pre-classifier features")
    expected = pd.read_parquet(bundle["indices"]["train.parquet"]["path"])
    index = expected.copy()
    index[["row", "col"]] -= (TILE_SIZE - IMAGE_SIZE) // 2
    identities = {entry["tile"]["path"]: (entry["tile"]["size"], entry["tile"]["sha256"])
                  for entry in bundle["inventory"].values()}
    out.mkdir(parents=True, exist_ok=False)
    try:
        records, failures, failure_details = extract_feature_records(
            index, model=model, device=torch.device(device), img_size=IMAGE_SIZE,
            crop_size=IMAGE_SIZE, infcmp=infcmp, store=store,
            feat_cols=[f"f{i:03d}" for i in range(128)], aux_names=config["enabled_aux_names"],
            truth_col="unified_class", dominant_frac=0.7, tile_identities=identities,
        )
        features = pd.DataFrame(records)
        features.to_parquet(out / "features.parquet", index=False)
        if failures:
            (out / "extraction_failures.json").write_bytes(json_bytes({"failures": failure_details}))
            raise ValueError("feature extraction was incomplete; partial features persisted")
        head = fit_head(features, expected)
        np.savez(out / "crop_head.npz", **head)
        head_identity = identity(out / "crop_head.npz")
        labels = out / "labels"
        labels.mkdir()
        outputs = {}
        for name in bundle["splits"]["train"]:
            entry = bundle["inventory"][name]
            store["feat"] = None
            infcmp.run_inference(
                model, entry["tile"]["path"], torch.device(device), img_size=IMAGE_SIZE,
                return_probs=True, aux_channel_names=config["enabled_aux_names"],
                expected_tile_size=entry["tile"]["size"],
                expected_tile_sha256=entry["tile"]["sha256"],
            )
            if store["feat"] is None:
                raise ValueError("dense inference did not capture a feature map")
            prediction = head_predict_dense(_upsample_feat_to(store["feat"], IMAGE_SIZE), head)
            payload = capture(entry["label"])
            with np.load(io.BytesIO(payload), allow_pickle=False) as original:
                overlay = crop_overlay(dict(original), prediction)
            overlay["r5_crop_head_sha256"] = np.str_(head_identity["sha256"])
            np.savez_compressed(labels / f"{name}.npz", **overlay)
            outputs[name] = identity(labels / f"{name}.npz")
    finally:
        store["handle"].remove()
    report = {"schema": SCHEMA, "state": "sidecars_produced_not_training_approved",
              "head": head_identity, "features": identity(out / "features.parquet"),
              "training_labels": outputs, "validation_labels": bundle["roots"]["label_dir"],
              "bundle_sha256": bundle_sha256 or digest(json_bytes(bundle)),
              "runtime": runtime}
    (out / "manifest.json").write_bytes(json_bytes(report))
    return report


def compare(r4: dict, r5: dict) -> dict:
    """Fail closed unless the two score records bind the exact same evaluation."""
    for record, arm in ((r4, "r4_control"), (r5, "r5")):
        if (record.get("schema") != SCHEMA or record.get("arm") != arm
                or record.get("training_mode") != "train"
                or record.get("embedding_source") != TESSERA_SOURCE
                or record.get("metric_version") != SEGMENTATION_METRIC_VERSION):
            raise ValueError("wrong arm, embedding source, or metric version")
    for field in ("protocol_sha256", "training_cohort_sha256", "training_regime_sha256",
                  "common_class_stats_sha256", "runtime_sha256",
                  "vocabulary_sha256", "crop_keys_sha256", "crop_truth_sha256",
                  "lulc_samples_sha256", "lulc_masks_sha256", "lulc_labels_sha256",
                  "evaluation_inputs_sha256"):
        value = r4.get(field)
        if not isinstance(value, str) or not SHA.fullmatch(value) or value != r5.get(field):
            raise ValueError(f"comparison requires matching {field}")
    scores = {}
    for record in (r4, r5):
        cms = {}
        for key in ("crop_confusion", "lulc_confusion"):
            cm = np.asarray(record[key])
            if cm.shape != (28, 28) or not np.issubdtype(cm.dtype, np.integer) or (cm < 0).any():
                raise ValueError("invalid 28-class confusion matrix")
            cms[key] = cm
        crop = cms["crop_confusion"]
        if (crop.sum(axis=1)[list(CROP_CLASSES)] <= 0).any() or crop[:11].any() or crop[22:].any():
            raise ValueError("crop confusion must cover all 11 crops and only crop targets")
        if cms["lulc_confusion"][1:].sum() == 0:
            raise ValueError("LULC confusion has no scored targets")
        scores[record["arm"]] = {
            "crop_accuracy": float(np.trace(crop) / crop.sum()),
            "lulc_miou": miou_from_confusion(cms["lulc_confusion"])[0],
        }
    for key in ("crop_confusion", "lulc_confusion"):
        if not np.array_equal(np.asarray(r4[key]).sum(axis=1), np.asarray(r5[key]).sum(axis=1)):
            raise ValueError("paired confusion matrices have different target support")
    crop_delta = scores["r5"]["crop_accuracy"] - scores["r4_control"]["crop_accuracy"]
    miou_delta = scores["r5"]["lulc_miou"] - scores["r4_control"]["lulc_miou"]
    return {"schema": SCHEMA, "scores": scores, "crop_accuracy_delta": crop_delta,
            "lulc_miou_delta": miou_delta, "passed": bool(crop_delta > 0 and (
                miou_delta >= -0.005 or np.isclose(miou_delta, -0.005, rtol=0, atol=1e-12)))}


def production_runtime(path: Path, source_git_sha: str, image: str) -> dict:
    """Reuse the crop image's source seal and interpreter identity."""
    from crop_distill_provenance import verify_runtime, snapshot_tree, tree_payload_sha256
    runtime = verify_runtime(path, source_git_sha=source_git_sha, image_ref=image)
    root = Path(__file__).resolve().parents[1]
    if tree_payload_sha256(snapshot_tree(root)) != runtime["source"]["payload_sha256"]:
        raise ValueError("producer source differs from the sealed image payload")
    expected = Path(runtime["environments"]["model"]["python"]["path"]).absolute()
    if Path(sys.executable).absolute() != expected:
        raise ValueError("producer requires the image's pinned model interpreter")
    return {"manifest": identity(path), "source_git_sha": source_git_sha, "image": image}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--inputs", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prod = commands.add_parser("produce")
    prod.add_argument("--bundle", type=Path, required=True)
    prod.add_argument("--bundle-sha256", required=True)
    prod.add_argument("--out", type=Path, required=True)
    prod.add_argument("--device", choices=("cpu", "cuda"), required=True)
    prod.add_argument("--runtime-manifest", type=Path, required=True)
    prod.add_argument("--source-git-sha", required=True)
    prod.add_argument("--image", required=True)
    score = commands.add_parser("compare")
    score.add_argument("--r4", type=Path, required=True)
    score.add_argument("--r5", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(json.loads(args.inputs.read_text()), args.out)
        print(json.dumps({"state": result["state"], "counts": result["counts"]}))
    elif args.command == "produce":
        runtime = production_runtime(args.runtime_manifest, args.source_git_sha, args.image)
        bundle = load_bundle(args.bundle, args.bundle_sha256)
        result = produce(bundle, args.out, args.device, runtime=runtime,
                         bundle_sha256=args.bundle_sha256)
        print(json.dumps({"state": result["state"], "training_tiles": len(result["training_labels"])}))
    else:
        result = compare(json.loads(args.r4.read_text()), json.loads(args.r5.read_text()))
        with args.out.open("xb") as stream:
            stream.write(json_bytes(result))
        print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
