#!/usr/bin/env python3
"""Run one explicitly selected TESSERA pilot arm, or write its paired evaluation.

Uses the existing UnifiedDataset/LULCTrainer and strict inference loader. No
queue, Kubernetes submission, downloads, or implicit progression between stages.
Production CLI requires the sealed crop runtime and exact artifact hashes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts import r5_tessera_pilot as pilot
from imint.training.config import TrainingConfig
from imint.training.unified_dataset import UnifiedDataset

# R4 TESSERA training regime, shared verbatim by the two new student arms.
# Probe changes epochs only; test fixtures may reduce grid/batch explicitly.
STUDENT_REGIME = {
    "backbone_name": "tessera_v1", "num_classes": 28,
    "enable_multitemporal": False, "num_temporal_frames": 1,
    "enable_height_channel": True, "enable_volume_channel": True,
    "enable_basal_area_channel": True, "enable_diameter_channel": True,
    "enable_dem_channel": True, "enable_vpp_channels": True,
    "enable_markfukt_channel": True, "enable_tradslag_head": True,
    "aux_fusion": "gated", "lambda_frac": 1.0, "epochs": 30,
    "batch_size": 8, "lr": 1e-4, "weight_decay": 0.35,
    "loss_type": "focal_dice", "focal_gamma": 2.0,
    "weighting_method": "sqrt", "label_smoothing": 0.05,
    "lovasz_weight": 0.3, "early_stopping_patience": 20,
    "num_workers": 16, "seed": 42, "deterministic": True,
    "log_training_exposure": True, "log_confusion_every_epoch": True,
}


def declaration_identity() -> dict:
    return pilot.identity(Path(__file__).resolve().parents[1] / "config/r5_tessera_pilot.json")


def validate_production(bundle: dict, produced: dict, bundle_sha256: str) -> None:
    if (produced.get("schema") != pilot.SCHEMA
            or produced.get("bundle_sha256") != bundle_sha256
            or produced.get("state") != "sidecars_produced_not_training_approved"
            or set(produced["training_labels"]) != set(bundle["splits"]["train"])):
        raise ValueError("crop production does not match the prepared bundle")
    for name, record in produced["training_labels"].items():
        if Path(record["path"]).name != f"{name}.npz":
            raise ValueError("crop sidecar basename differs from its bound tile")
        if pilot.identity(Path(record["path"])) != record:
            raise ValueError("crop sidecar changed after production")


def datasets(bundle: dict, produced: dict, arm: str, config: TrainingConfig):
    if arm not in {"r4_control", "r5"}:
        raise ValueError("only the paired R4 control and R5 are supported")
    if arm == "r5":
        parents = {str(Path(record["path"]).parent) for record in produced["training_labels"].values()}
        if len(parents) != 1:
            raise ValueError("R5 sidecars must occupy one flat directory")
        train_labels = parents.pop()
    else:
        train_labels = bundle["roots"]["label_dir"]
    shared = dict(lulc_dir=bundle["roots"]["data_dir"], patch_size=config.patch_pixels,
                  backbone_family="tessera", enable_aux=True,
                  aux_channel_names=config.enabled_aux_names,
                  frac_dir=bundle["roots"]["frac_dir"])
    train = UnifiedDataset(**shared, split="train", label_dir=train_labels,
                           tile_names=[f"{name}.npz" for name in bundle["splits"]["train"]])
    val = UnifiedDataset(**shared, split="val", label_dir=bundle["roots"]["label_dir"],
                         tile_names=[f"{name}.npz" for name in bundle["splits"]["val"]])
    return train, val


def common_class_stats(bundle: dict) -> dict:
    """Freeze weights from control labels, shared by both treatment arms."""
    counts = np.zeros(28, dtype=np.int64)
    for name in bundle["splits"]["train"]:
        with np.load(io.BytesIO(pilot.capture(bundle["inventory"][name]["label"])), allow_pickle=False) as z:
            label = z["label"]
            if not np.issubdtype(label.dtype, np.integer) or not np.isin(label, range(28)).all():
                raise ValueError("control labels violate the 28-class vocabulary")
            counts += np.bincount(label.ravel(), minlength=28)
    # Uniform tile sampling is explicit, independent of changed crop labels.
    return {"class_counts": {str(c): int(n) for c, n in enumerate(counts)},
            "tile_weights": {f"{n}.npz": 1.0 for n in bundle["splits"]["train"]}}


def train_arm(bundle: dict, produced: dict, arm: str, out: Path, *, device: str,
              probe: bool, bundle_sha256: str, runtime: dict) -> dict:
    import torch
    from imint.training.trainer import LULCTrainer

    validate_production(bundle, produced, bundle_sha256)
    if produced.get("runtime") != runtime:
        raise ValueError("producer runtime differs from student runtime")
    if out.exists():
        raise FileExistsError("training output already exists; never restart implicitly")
    out.mkdir(parents=True)
    stats_root = out / "statistics"
    stats_root.mkdir()
    stats = common_class_stats(bundle)
    (stats_root / "class_stats.json").write_bytes(pilot.json_bytes(stats))
    cfg = TrainingConfig()
    # Set after __post_init__ so shell IMINT_* values cannot change this pilot.
    for key, value in STUDENT_REGIME.items():
        setattr(cfg, key, value)
    cfg.patch_pixels = cfg.img_size = pilot.IMAGE_SIZE
    cfg.data_dir = str(stats_root)
    cfg.checkpoint_dir = str(out / "checkpoints")
    cfg.device = device
    if probe:
        cfg.epochs = 1
        cfg.save_every_n_epochs = 1
        cfg.fixed_checkpoint_only = True
    train, val = datasets(bundle, produced, arm, cfg)
    if len(train) < cfg.batch_size:
        raise ValueError("training cohort is smaller than one full configured batch")
    protocol = asdict(cfg)
    for key in ("data_dir", "checkpoint_dir"):
        protocol.pop(key)
    selected_labels = (produced["training_labels"] if arm == "r5" else
                       {name: bundle["inventory"][name]["label"] for name in bundle["splits"]["train"]})
    # The canonical producer payload binds its fitted head and every output;
    # retaining the payload makes this chain independently inspectable.
    production = {"canonical_sha256": pilot.digest(pilot.json_bytes(produced)), "manifest": produced}
    execution = {"schema": pilot.SCHEMA, "arm": arm, "mode": "probe" if probe else "train",
                 "bundle_sha256": bundle_sha256, "runtime": runtime,
                 "production": production, "training_labels": selected_labels,
                 "regime": protocol, "regime_sha256": pilot.digest(pilot.json_bytes(protocol)),
                 "declaration": declaration_identity(), "common_class_stats_sha256": pilot.digest(pilot.json_bytes(stats)),
                 "train_tiles": train.tile_names, "val_tiles": val.tile_names}
    (out / "execution.json").write_bytes(pilot.json_bytes(execution))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    trainer = LULCTrainer(cfg)
    result = trainer.train(train, val)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    checkpoint = pilot.identity(Path(result["checkpoint"]))
    log = json.loads((Path(cfg.checkpoint_dir) / "training_log.json").read_text())
    if (not log["epochs"] or log.get("status") not in {"completed", "stopped"}
            or not all(np.isfinite(row["train_loss"]) for row in log["epochs"])):
        raise ValueError("student training did not complete with finite loss")
    report = {**execution, "state": "probe_complete" if probe else "training_complete",
              "checkpoint": checkpoint, "training_log": pilot.identity(Path(cfg.checkpoint_dir) / "training_log.json"),
              "elapsed_seconds": elapsed, "completed_epochs": len(log["epochs"]),
              "device": device,
              "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
    (out / "manifest.json").write_bytes(pilot.json_bytes(report))
    return report


def evaluate_arm(bundle: dict, trained: dict, out: Path, *, bundle_sha256: str,
                 runtime: dict, device: str) -> dict:
    import torch
    from distill_forest_labels import _load_inference_comparison
    from imint.metrics import compute_miou
    from imint.schema.unified_schema import UNIFIED_CLASSES

    if (trained.get("bundle_sha256") != bundle_sha256
            or trained.get("arm") not in {"r4_control", "r5"}
            or trained.get("state") not in {"training_complete", "probe_complete"}
            or trained.get("runtime") != runtime):
        raise ValueError("training identity/runtime does not match evaluation")
    if out.exists():
        raise FileExistsError("evaluation output exists")
    checkpoint = trained["checkpoint"]
    pilot.capture(checkpoint)
    infcmp = _load_inference_comparison()
    model, _, _, size, config = infcmp.load_model(
        checkpoint["path"], torch.device(device), backbone_name="tessera_v1", img_size=pilot.IMAGE_SIZE,
        expected_checkpoint_sha256=checkpoint["sha256"], expected_checkpoint_size=checkpoint["size"],
        return_checkpoint_config=True,
    )
    if size != pilot.IMAGE_SIZE or config.get("num_classes") != 28:
        raise ValueError("student checkpoint grid/vocabulary differs from the pilot")
    holdout = pd.read_parquet(io.BytesIO(pilot.capture(bundle["indices"]["holdout.parquet"])))
    holdout = holdout.sort_values(pilot.KEYS).reset_index(drop=True)
    off = (pilot.TILE_SIZE - pilot.IMAGE_SIZE) // 2
    val_names = set(bundle["splits"]["val"])
    all_names = sorted(val_names | set(holdout.tile_name))
    label_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    crop_cm, lulc_cm = np.zeros((28, 28), np.int64), np.zeros((28, 28), np.int64)
    point_records = []
    out.mkdir(parents=True)
    for name in all_names:
        entry = bundle["inventory"][name]
        pred = infcmp.run_inference(
            model, entry["tile"]["path"], torch.device(device), img_size=pilot.IMAGE_SIZE,
            aux_channel_names=config["enabled_aux_names"],
            expected_tile_sha256=entry["tile"]["sha256"], expected_tile_size=entry["tile"]["size"],
        )
        if pred.shape != (pilot.IMAGE_SIZE, pilot.IMAGE_SIZE):
            raise ValueError("prediction grid differs from frozen evaluation grid")
        # Cache every expensive inference before subsequent work can fail.
        np.save(out / f"{name}.npy", pred, allow_pickle=False)
        if name in val_names:
            with np.load(io.BytesIO(pilot.capture(entry["label"])), allow_pickle=False) as z:
                target = z["label"][off:-off, off:-off].astype("<i8")
            if target.shape != pred.shape or not np.isin(target, range(28)).all():
                raise ValueError("validation target violates the fixed vocabulary/grid")
            # Same v2 valid-target definition as trainer: background ignored.
            mask = target != 0
            label_hash.update(name.encode() + b"\0" + target.tobytes())
            mask_hash.update(name.encode() + b"\0" + mask.tobytes())
            lulc_cm += compute_miou(pred, target, 28)["confusion_matrix"]
        points = holdout[holdout.tile_name == name]
        for point in points.itertuples(index=False):
            predicted = int(pred[point.row - off, point.col - off])
            crop_cm[point.unified_class, predicted] += 1
            point_records.append({"tile_name": name, "point_id": int(point.point_id),
                                  "Year": int(point.Year), "truth": int(point.unified_class),
                                  "prediction": predicted})
    pd.DataFrame(point_records).to_parquet(out / "crop_predictions.parquet", index=False)
    key_columns = [*pilot.KEYS, "Year", "row", "col"]
    record = {
        "schema": pilot.SCHEMA, "arm": trained["arm"], "training_mode": trained["mode"],
        "embedding_source": pilot.TESSERA_SOURCE, "metric_version": pilot.SEGMENTATION_METRIC_VERSION,
        "protocol_sha256": bundle_sha256,
        "training_manifest": {"canonical_sha256": pilot.digest(pilot.json_bytes(trained)), "manifest": trained},
        "training_cohort_sha256": pilot.digest(pilot.json_bytes(bundle["splits"])),
        "training_regime_sha256": trained["regime_sha256"],
        "common_class_stats_sha256": trained["common_class_stats_sha256"],
        "runtime_sha256": pilot.digest(pilot.json_bytes(runtime)),
        "vocabulary_sha256": pilot.digest(pilot.json_bytes(UNIFIED_CLASSES)),
        "crop_keys_sha256": pilot.digest(holdout[key_columns].to_json(orient="records").encode()),
        "crop_truth_sha256": pilot.digest(holdout.unified_class.to_numpy(dtype="<i8").tobytes()),
        "lulc_samples_sha256": pilot.digest(pilot.json_bytes({"tiles": sorted(val_names), "grid": [pilot.IMAGE_SIZE, off]})),
        "lulc_masks_sha256": mask_hash.hexdigest(), "lulc_labels_sha256": label_hash.hexdigest(),
        "evaluation_inputs_sha256": pilot.digest(pilot.json_bytes({n: bundle["inventory"][n]["tile"] for n in all_names})),
        "checkpoint": checkpoint, "crop_confusion": crop_cm.tolist(), "lulc_confusion": lulc_cm.tolist(),
    }
    (out / "record.json").write_bytes(pilot.json_bytes(record))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "train", "evaluate"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--artifact", type=Path, required=True,
                        help="produce manifest for train/probe; training manifest for evaluate")
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--arm", choices=("r4_control", "r5"))
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runtime = pilot.production_runtime(args.runtime_manifest, args.source_git_sha, args.image)
    bundle = pilot.load_bundle(args.bundle, args.bundle_sha256)
    artifact = json.loads(pilot.capture({"path": str(args.artifact), "sha256": args.artifact_sha256}))
    if args.stage == "evaluate":
        result = evaluate_arm(bundle, artifact, args.out, bundle_sha256=args.bundle_sha256,
                              runtime=runtime, device=args.device)
        print(json.dumps({"arm": result["arm"], "evaluation_written": True}))
    else:
        if args.arm is None:
            parser.error("--arm is required for probe/train")
        result = train_arm(bundle, artifact, args.arm, args.out, device=args.device,
                           probe=args.stage == "probe", bundle_sha256=args.bundle_sha256, runtime=runtime)
        print(json.dumps({"state": result["state"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
