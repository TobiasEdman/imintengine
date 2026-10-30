#!/usr/bin/env python3
"""IM-023 TESSERA retrain: the R4 recipe with one variant switch.

The single source of the training flags for the smoke, main and control
jobs (k8s/im023/). The recipe is ``k8s/ladder/ladder-r4-tessera-job.yaml``;
a test holds it flag for flag against that manifest.

    main     29 classes, labels from relabel_barrblandskog.py (NMD 113 → 28)
    control  28 classes, R4's own distilled labels

Both run on code with the FractionLoss mask fix, which R4 did not have, so
main minus control is the effect of class 28 and control minus R4 the effect
of the fix.

    python scripts/im023_retrain.py --variant main --epochs 30 \\
        --checkpoint-dir /cephfs/checkpoints/im023/tessera_b28

Arguments after ``--`` are appended to the train_unified.py command (last
flag wins). They exist for the local CPU smoke; the cluster jobs pass none.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_barrblandskog import MANIFEST, NEW_CLASS, RULE  # noqa: E402

PATHS = {
    "data_dir": "/cephfs/unified_v2_512",
    "frac_dir": "/cephfs/tradslag_fracs",
    "r2_label_dir": "/cephfs/distill/tessera_r2",
    "b28_label_dir": "/cephfs/distill/tessera_r2_b28",
}
# variant → (num classes, label directory key)
VARIANTS = {"main": (29, "b28_label_dir"), "control": (28, "r2_label_dir")}
RECIPE = (
    "--backbone-name", "tessera_v1",
    "--enable-markfukt",
    "--enable-tradslag-head",
    "--aux-fusion", "gated",
    "--lambda-frac", "1.0",
    "--batch-size", "8",
    "--lr", "1e-4",
    "--weight-decay", "0.35",
    "--loss-type", "focal_dice",
    "--focal-gamma", "2.0",
    "--patch-size", "504",
    "--img-size", "504",
    "--weighting-method", "sqrt",
    "--label-smoothing", "0.05",
    "--lovasz-weight", "0.3",
    "--patience", "20",
    "--num-workers", "16",
    "--device", "cuda",
)
LABEL_SAMPLE = 20


def train_args(variant: str, epochs: int, checkpoint_dir: str, paths: dict[str, str]) -> list[str]:
    classes, label_key = VARIANTS[variant]
    return [*RECIPE,
            "--data-dirs", paths["data_dir"],
            "--frac-dir", paths["frac_dir"],
            "--label-dir", paths[label_key],
            "--num-classes", str(classes),
            "--epochs", str(epochs),
            "--checkpoint-dir", checkpoint_dir]


def npz_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("*.npz"))


def npz_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as z:
        return z.files


def label_max(path: Path) -> int:
    with np.load(path, allow_pickle=False) as z:
        return int(z["label"].max())


def preflight(variant: str, paths: dict[str, str]) -> list[str]:
    """Refusals that keep a run from training on the wrong or partial inputs."""
    data, fracs, r2, b28 = (Path(paths[k]) for k in ("data_dir", "frac_dir", "r2_label_dir", "b28_label_dir"))
    classes, label_key = VARIANTS[variant]
    labels = Path(paths[label_key])
    problems = []
    tiles = npz_names(data)
    if not tiles:
        return [f"no tiles in {data}"]
    without = [t for t in tiles[:3] if "tessera" not in npz_keys(data / t)]
    if without:
        problems.append(f"tiles without a tessera key: {without}")
    if not npz_names(fracs):
        problems.append(f"no fraction sidecars in {fracs}")
    if not (r2 / "_GATE_OK").is_file():
        problems.append(f"{r2} has no _GATE_OK: the R4 distillation is not complete")
    r2_names = npz_names(r2)
    if variant == "main":
        manifest_path = b28 / MANIFEST
        if not manifest_path.is_file():
            problems.append(f"{manifest_path} missing: run k8s/im023/relabel-b28-job.yaml first")
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest["rule"] != RULE:
                problems.append(f"labels made by rule {manifest['rule']}, expected {RULE}")
            if not manifest["distilled_tiles"] == len(npz_names(b28)) == len(r2_names):
                problems.append(f"cohort differs: manifest {manifest['distilled_tiles']}, "
                                f"{len(npz_names(b28))} in {b28}, {len(r2_names)} in {r2}")
            if manifest["class_28_pixels"] == 0:
                problems.append("relabelled labels hold no class 28")
    label_names = npz_names(labels)
    if not label_names:
        problems.append(f"no label sidecars in {labels}")
    for name in label_names[:LABEL_SAMPLE]:
        top = label_max(labels / name)
        if top >= classes:
            problems.append(f"{labels / name}: label {top} with {classes} classes")
    return problems


def class_weight_source(data_dir: str) -> str:
    """Where the trainer's class weights come from; it reads data_dir, not the label dir."""
    stats = Path(data_dir) / "class_stats.json"
    if not stats.is_file():
        return f"{stats} absent: uniform class weights"
    content = json.loads(stats.read_text())
    counts = content.get("class_counts") or content.get("pixel_counts") or {}
    has_28 = str(NEW_CLASS) in counts or "barrblandskog" in counts
    return (f"{stats} ({len(counts)} classes); class 28 "
            + ("counted" if has_28 else "absent → count 1 → weight at --max-class-weight"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    for key, default in PATHS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", default=default)
    args = ap.parse_args(own)
    paths = {key: getattr(args, key) for key in PATHS}

    if (Path(args.checkpoint_dir) / "best_model.pt").exists():
        print(f"REFUSED: {args.checkpoint_dir} already holds a model; move it before rerunning")
        return 1
    problems = preflight(args.variant, paths)
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems))
        return 1
    print(f"class weights: {class_weight_source(paths['data_dir'])}")
    command = [sys.executable, str(ROOT / "scripts" / "train_unified.py"),
               *train_args(args.variant, args.epochs, args.checkpoint_dir, paths), *extra]
    print("TRAIN:", " ".join(command), flush=True)
    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
