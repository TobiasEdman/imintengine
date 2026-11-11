#!/usr/bin/env python3
"""IM-023 Prithvi-600M retrain: the R4 recipe with the TESSERA variant switch.

The single source of the training flags for the smoke, main and control
jobs (k8s/im023/prithvi600m/). The recipe is
``k8s/ladder/ladder-r4-prithvi600m-job.yaml``; a test holds it flag for flag
against that manifest.

    main     29 classes, labels from relabel_distilled_b28.py (NMD 113 → 28)
    control  28 classes, R4's own distilled labels

R4 Prithvi-600M trained without the fraction head: the ladder generator adds
only ``--frac-dir`` for rung 4 and the base manifest has no head flag, so the
sidecars were read and never supervised. Both variants add the head
(``--enable-tradslag-head --lambda-frac 1.0``) on code with the FractionLoss
mask fix. main minus control is then the effect of class 28, and control
minus R4 the effect of the head. ``--no-fraction-head`` drops it, and control
is then R4's flags exactly.

The variants, class 28's weight (``b28_class_stats``) and the class weight
report are im023_retrain's, imported. That file is pinned by the TESSERA
jobs and stays as it ran; the preflight here is its own, without the
TESSERA key check and naming this backbone's relabel job.

    python scripts/im023_retrain_prithvi600m.py --variant main --epochs 30 \\
        --checkpoint-dir /cephfs/checkpoints/im023/prithvi600m_b28

A checkpoint directory that already holds a model or a resume state is
refused, because the trainer would silently continue from it. ``--resume``
continues an interrupted run instead (a 600M run is ~10 h; an eviction must
not cost it): the directory must hold ``last_checkpoint.pt`` and a training
log of an unfinished run with this variant's class count and head.

Arguments after ``--`` are appended to the train_unified.py command (last
flag wins). They exist for the local CPU smoke; the H100 jobs pass none.
Only the smoke's overrides (``SMOKE_EXTRAS``) are accepted there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from im023_retrain import (B28_STATS, LABEL_SAMPLE, VARIANTS, b28_class_stats,  # noqa: E402
                           class_weight_source, label_max, npz_names)
from relabel_barrblandskog import MANIFEST, OLD_CLASS, RULE  # noqa: E402

RELABEL_JOB = "k8s/im023/prithvi600m/relabel-b28-job.yaml"
PATHS = {
    "data_dir": "/cephfs/unified_v2_512",
    "frac_dir": "/cephfs/tradslag_fracs",
    "r2_label_dir": "/cephfs/distill/prithvi600m_r2",
    "b28_label_dir": "/cephfs/distill/prithvi600m_r2_b28",
}
RECIPE = (
    "--backbone-name", "prithvi_600m",
    "--enable-markfukt",
    "--enable-multitemporal",
    "--num-temporal-frames", "4",
    "--batch-size", "2",
    "--lr", "1e-4",
    "--weight-decay", "0.35",
    "--unfreeze-layers", "6",
    "--backbone-lr-factor", "0.1",
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
# Not in R4. --lambda-frac is the trainer's default, written out as the TESSERA recipe has it.
FRACTION_HEAD = ("--enable-tradslag-head", "--lambda-frac", "1.0")
# The only flags allowed after --: the local CPU smoke's overrides, spelled out in
# full. A denylist would not hold: train_unified's argparse accepts abbreviations,
# so --checkpoint-d would redirect the run past the refusal below.
SMOKE_EXTRAS = ("--backbone-name", "--patch-size", "--img-size", "--batch-size",
                "--num-workers", "--device")
CHECKPOINT_STATE = ("best_model.pt", "last_checkpoint.pt")


def train_args(variant: str, epochs: int, checkpoint_dir: str, paths: dict[str, str], *,
               fraction_head: bool) -> list[str]:
    classes, label_key = VARIANTS[variant]
    weights = ["--class-stats", str(Path(checkpoint_dir) / B28_STATS)] if variant == "main" else []
    return [*RECIPE,
            *(FRACTION_HEAD if fraction_head else ()),
            "--data-dirs", paths["data_dir"],
            "--frac-dir", paths["frac_dir"],
            "--label-dir", paths[label_key],
            "--num-classes", str(classes),
            "--epochs", str(epochs),
            "--checkpoint-dir", checkpoint_dir,
            *weights]


def preflight(variant: str, paths: dict[str, str]) -> list[str]:
    """Refusals that keep a run from training on the wrong or partial inputs."""
    data, fracs, r2, b28 = (Path(paths[k]) for k in ("data_dir", "frac_dir", "r2_label_dir", "b28_label_dir"))
    classes, label_key = VARIANTS[variant]
    labels = Path(paths[label_key])
    if not npz_names(data):
        return [f"no tiles in {data}"]
    problems = []
    if not npz_names(fracs):
        problems.append(f"no fraction sidecars in {fracs}")
    if not (r2 / "_GATE_OK").is_file():
        problems.append(f"{r2} has no _GATE_OK: the R4 distillation is not complete")
    r2_names = npz_names(r2)
    if variant == "main":
        manifest_path = b28 / MANIFEST
        if not manifest_path.is_file():
            problems.append(f"{manifest_path} missing: run {RELABEL_JOB} first")
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest["rule"] != RULE:
                problems.append(f"labels made by rule {manifest['rule']}, expected {RULE}")
            # Same tile names prove nothing if the labels came from another distillation.
            made = (manifest["inputs"]["distilled"], manifest["outputs"]["distilled"])
            if made != (str(r2), str(b28)):
                problems.append(f"{manifest_path} relabelled {made[0]} into {made[1]}, not {r2} into {b28}")
            b28_names = npz_names(b28)
            missing = sorted(set(r2_names) - set(b28_names))
            unexpected = sorted(set(b28_names) - set(r2_names))
            if missing or unexpected or manifest["distilled_tiles"] != len(r2_names):
                problems.append(f"cohort differs from {r2}: manifest {manifest['distilled_tiles']}, "
                                f"{len(r2_names)} in {r2}, missing {missing[:5]}, "
                                f"unexpected {unexpected[:5]}")
            if manifest["class_28_pixels"] == 0:
                problems.append("relabelled labels hold no class 28")
            if not npz_names(Path(manifest["outputs"]["nmd"])):
                problems.append(f"no relabelled NMD sidecars in {manifest['outputs']['nmd']}")
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


def resume_problems(checkpoint_dir: Path, classes: int, fraction_head: bool) -> list[str]:
    """Why ``checkpoint_dir`` may not be resumed as this run; empty when it may."""
    last, log_path = checkpoint_dir / "last_checkpoint.pt", checkpoint_dir / "training_log.json"
    if not last.is_file():
        return [f"{last} missing: nothing to resume"]
    if not log_path.is_file():
        return [f"{log_path} missing: cannot tell which run {checkpoint_dir} holds"]
    log = json.loads(log_path.read_text())
    config = log.get("config", {})
    problems = []
    if log.get("status") in ("completed", "stopped"):
        problems.append(f"the run in {checkpoint_dir} is {log['status']}, not interrupted")
    if config.get("num_classes") != classes:
        problems.append(f"{checkpoint_dir} trains {config.get('num_classes')} classes, this variant {classes}")
    if bool(config.get("enable_tradslag_head")) != fraction_head:
        problems.append(f"{checkpoint_dir} was started {'with' if config.get('enable_tradslag_head') else 'without'} "
                        f"the fraction head")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--no-fraction-head", action="store_true",
                    help="train without the fraction head, as R4 did (default: with it)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the interrupted run in --checkpoint-dir from its last_checkpoint.pt")
    for key, default in PATHS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", default=default)
    args = ap.parse_args(own)
    paths = {key: getattr(args, key) for key in PATHS}

    refused = [a for a in extra if a.startswith("-") and a.split("=", 1)[0] not in SMOKE_EXTRAS]
    if refused:
        print(f"REFUSED: {refused} after --; only {', '.join(SMOKE_EXTRAS)} may follow it")
        return 1
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.resume:
        problems = resume_problems(checkpoint_dir, VARIANTS[args.variant][0], not args.no_fraction_head)
        if problems:
            print("REFUSED --resume:\n  " + "\n  ".join(problems))
            return 1
    else:
        held = [name for name in CHECKPOINT_STATE if (checkpoint_dir / name).exists()]
        if held:
            print(f"REFUSED: {checkpoint_dir} already holds {held}; pass --resume to continue "
                  f"that run, or move it before rerunning")
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
    print(f"fraction head: {'off, as R4' if args.no_fraction_head else 'on, not in R4'}")
    if args.resume:
        print(f"resuming from {checkpoint_dir / 'last_checkpoint.pt'}")
    command = [sys.executable, str(ROOT / "scripts" / "train_unified.py"),
               *train_args(args.variant, args.epochs, args.checkpoint_dir, paths,
                           fraction_head=not args.no_fraction_head), *extra]
    print("TRAIN:", " ".join(command), flush=True)
    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
