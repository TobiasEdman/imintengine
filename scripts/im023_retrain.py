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

Class weights come from ``<data_dir>/class_stats.json``, which predates class
28: read as is, 28 counts one pixel and gets ``--max-class-weight``. main
therefore trains on a copy in which 28 takes the share of class 3 that the
relabel moved (``b28_class_stats``); control reads the file as R4 did.

    python scripts/im023_retrain.py --variant main --epochs 30 \\
        --checkpoint-dir /cephfs/checkpoints/im023/tessera_b28

Arguments after ``--`` are appended to the train_unified.py command (last
flag wins). They exist for the local CPU smoke and the 2080 Ti chain check;
the H100 jobs pass none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from imint.schema.unified_schema import UNIFIED_CLASS_NAMES  # noqa: E402
from relabel_barrblandskog import MANIFEST, NEW_CLASS, OLD_CLASS, RULE  # noqa: E402

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
B28_STATS = "class_stats_b28.json"   # main's class weights, written into its checkpoint dir


def train_args(variant: str, epochs: int, checkpoint_dir: str, paths: dict[str, str]) -> list[str]:
    classes, label_key = VARIANTS[variant]
    weights = ["--class-stats", str(Path(checkpoint_dir) / B28_STATS)] if variant == "main" else []
    return [*RECIPE,
            "--data-dirs", paths["data_dir"],
            "--frac-dir", paths["frac_dir"],
            "--label-dir", paths[label_key],
            "--num-classes", str(classes),
            "--epochs", str(epochs),
            "--checkpoint-dir", checkpoint_dir,
            *weights]


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
        stats = data / "class_stats.json"
        if not stats.is_file() or "pixel_counts" not in json.loads(stats.read_text()):
            problems.append(f"{stats} has no pixel_counts: class 28's weight cannot be derived")
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


def label_counts(label_dir: Path, classes: tuple[int, ...]) -> dict[int, int]:
    """Pixels of each of ``classes`` over every label sidecar in ``label_dir``."""
    totals = np.zeros(max(classes) + 1, np.int64)
    for name in npz_names(label_dir):
        with np.load(label_dir / name, allow_pickle=False) as z:
            totals += np.bincount(z["label"].ravel(), minlength=totals.size)[:totals.size]
    return {c: int(totals[c]) for c in classes}


def b28_class_stats(data_dir: str, b28_label_dir: str) -> dict:
    """``<data_dir>/class_stats.json`` with class 28 carved out of class 3.

    Class 28 takes the share of class 3 that the relabel turned into 28 in the
    training labels. The total and every other count stay as R4 read them, so
    28 gets the weight the recipe computes for its frequency instead of the cap.
    """
    source = Path(data_dir) / "class_stats.json"
    stats = json.loads(source.read_text())
    counts = label_counts(Path(b28_label_dir), (OLD_CLASS, NEW_CLASS))
    share = counts[NEW_CLASS] / (counts[OLD_CLASS] + counts[NEW_CLASS])
    old, new = UNIFIED_CLASS_NAMES[OLD_CLASS], UNIFIED_CLASS_NAMES[NEW_CLASS]
    pixels = dict(stats["pixel_counts"])
    moved = round(pixels[old] * share)
    pixels[old] -= moved
    pixels[new] = moved
    return stats | {"pixel_counts": pixels, "im023_class_28": {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "label_dir": b28_label_dir, "rule": RULE,
        "label_pixels": {old: counts[OLD_CLASS], new: counts[NEW_CLASS]},
        "share": share, "moved": moved}}


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
    if args.variant == "main":
        derived = b28_class_stats(paths["data_dir"], paths["b28_label_dir"])
        target = Path(args.checkpoint_dir) / B28_STATS
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(derived, indent=2, ensure_ascii=False))
        info = derived["im023_class_28"]
        print(f"class weights: {target}: {info['source']} with class 28 = "
              f"{info['share']:.2%} of class {OLD_CLASS} ({info['moved']} px)")
    else:
        print(f"class weights: {class_weight_source(paths['data_dir'])}")
    command = [sys.executable, str(ROOT / "scripts" / "train_unified.py"),
               *train_args(args.variant, args.epochs, args.checkpoint_dir, paths), *extra]
    print("TRAIN:", " ".join(command), flush=True)
    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
