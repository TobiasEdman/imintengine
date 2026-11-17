#!/usr/bin/env python3
"""IM-023 f5: TESSERA retrained on a distillation with five forest types.

Tobias 2026-11-17: barrblandskog (class 28) is refined against the NFI plots
like tallskog, granskog, lövskog and blandskog, instead of being NMD 113's
extent laid over the head's calls (main). The labels come from
k8s/im023/f5/distill-f5-job.yaml: a TESSERA r2 head trained on the
five-type truth (``nfi_truth_f5.py``), forwarded over R4's cohort with forest
set {1, 2, 3, 4, 28} on the relabelled NMD labels.

    f5   29 classes, /cephfs/distill/tessera_r2_f5

Everything else is main's: R4's flags (im023_retrain.RECIPE) and main's class
weights (``b28_class_stats``, class 28 carved out of class 3 in the canonical
NMD labels), so f5 minus main is the effect of how class 28 is labelled. The
existing im023_control and main are the comparison; no control runs here.

``--profile 2080ti`` (no H100 in ICE, Tobias 2026-11-17) changes three
flags: FP32 (Turing has no hardware BF16), batch 2 (FP32 activations of
batch 8 at 504 px exceed 11 GB; the trainer has no gradient accumulation)
and 6 workers. That is a recipe difference from main, recorded in the run.

im023_retrain.py and the Prithvi recipe are pinned by their jobs; this file
imports from them and changes neither.

    python scripts/im023_retrain_f5.py --profile 2080ti --epochs 30 \\
        --checkpoint-dir /cephfs/checkpoints/im023/tessera_f5

A checkpoint directory holding a model or a resume state is refused, because
the trainer would silently continue from it; ``--resume`` continues an
interrupted run of this variant instead.

Arguments after ``--`` are appended to the train_unified.py command for the
local CPU smoke; only ``SMOKE_EXTRAS`` are accepted, and the jobs pass none.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from distill_forest_labels import load_head  # noqa: E402
from im023_retrain import (B28_STATS, LABEL_SAMPLE, RECIPE, b28_class_stats, label_max,  # noqa: E402
                           npz_keys, npz_names)
from im023_retrain_prithvi600m import CHECKPOINT_STATE, resume_problems  # noqa: E402
from relabel_barrblandskog import MANIFEST, OLD_CLASS, RULE  # noqa: E402

DISTILL_JOB = "k8s/im023/f5/distill-f5-job.yaml"
CLASSES = 29
F5_FOREST_CLASSES = (1, 2, 3, 4, 28)
PATHS = {
    "data_dir": "/cephfs/unified_v2_512",
    "frac_dir": "/cephfs/tradslag_fracs",
    "r2_label_dir": "/cephfs/distill/tessera_r2",          # R4's cohort, the reference
    "b28_label_dir": "/cephfs/distill/tessera_r2_b28",     # main's labels, for its class weights
    "f5_label_dir": "/cephfs/distill/tessera_r2_f5",
    "f5_head": "/cephfs/distill/heads/tessera_r2_f5_head.npz",
}
SMOKE_EXTRAS = ("--patch-size", "--img-size", "--batch-size", "--num-workers", "--device")
#: Flag values a profile sets in RECIPE, and flags it adds.
PROFILES = {
    "h100": ({}, ()),
    "2080ti": ({"--batch-size": "2", "--num-workers": "6"}, ("--disable-bf16",)),
}


def recipe(profile: str) -> list[str]:
    values, added = PROFILES[profile]
    flags = list(RECIPE)
    for flag, value in values.items():
        flags[flags.index(flag) + 1] = value
    return [*flags, *added]


def train_args(profile: str, epochs: int, checkpoint_dir: str, paths: dict[str, str]) -> list[str]:
    return [*recipe(profile),
            "--data-dirs", paths["data_dir"],
            "--frac-dir", paths["frac_dir"],
            "--label-dir", paths["f5_label_dir"],
            "--num-classes", str(CLASSES),
            "--epochs", str(epochs),
            "--checkpoint-dir", checkpoint_dir,
            "--class-stats", str(Path(checkpoint_dir) / B28_STATS)]


def preflight(paths: dict[str, str]) -> list[str]:
    """Refusals that keep the run from training on the wrong or partial inputs."""
    data, fracs, r2, b28, f5, head = (Path(paths[k]) for k in (
        "data_dir", "frac_dir", "r2_label_dir", "b28_label_dir", "f5_label_dir", "f5_head"))
    tiles = npz_names(data)
    if not tiles:
        return [f"no tiles in {data}"]
    problems = []
    without = [t for t in tiles[:3] if "tessera" not in npz_keys(data / t)]
    if without:
        problems.append(f"tiles without a tessera key: {without}")
    if not npz_names(fracs):
        problems.append(f"no fraction sidecars in {fracs}")
    if not (r2 / "_GATE_OK").is_file():
        problems.append(f"{r2} has no _GATE_OK: the R4 distillation is not complete")
    if not (f5 / "_GATE_OK").is_file():
        problems.append(f"{f5} has no _GATE_OK: run {DISTILL_JOB} first")
    r2_names, f5_names = npz_names(r2), npz_names(f5)
    if set(f5_names) != set(r2_names):
        problems.append(f"cohort differs from {r2}: {len(f5_names)} in {f5}, {len(r2_names)} in {r2}, "
                        f"missing {sorted(set(r2_names) - set(f5_names))[:5]}, "
                        f"unexpected {sorted(set(f5_names) - set(r2_names))[:5]}")
    if not head.is_file():
        problems.append(f"{head} missing: the labels' head cannot be checked")
    else:
        loaded = load_head(str(head))
        stamp, classes = loaded["head_sha"], sorted(int(c) for c in loaded["classes_"])
        if classes != [0, *F5_FOREST_CLASSES]:
            problems.append(f"{head} has classes {classes}, expected {[0, *F5_FOREST_CLASSES]}")
        for name in f5_names[:LABEL_SAMPLE]:
            with np.load(f5 / name, allow_pickle=False) as z:
                if "head_sha" not in z.files or str(z["head_sha"]) != stamp:
                    problems.append(f"{f5 / name} was not written by {head}")
                    break
    for name in f5_names[:LABEL_SAMPLE]:
        top = label_max(f5 / name)
        if top >= CLASSES:
            problems.append(f"{f5 / name}: label {top} with {CLASSES} classes")
    # main's class weights: b28_class_stats reads the relabel manifest and its NMD labels.
    manifest_path = b28 / MANIFEST
    if not manifest_path.is_file():
        problems.append(f"{manifest_path} missing: main's class weights cannot be derived")
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest["rule"] != RULE:
            problems.append(f"labels made by rule {manifest['rule']}, expected {RULE}")
        if manifest["class_28_pixels"] == 0:
            problems.append("relabelled NMD labels hold no class 28")
    stats = data / "class_stats.json"
    if not stats.is_file() or "pixel_counts" not in json.loads(stats.read_text()):
        problems.append(f"{stats} has no pixel_counts: class 28's weight cannot be derived")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--checkpoint-dir", required=True)
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
        problems = resume_problems(checkpoint_dir, CLASSES, fraction_head=True)
        if problems:
            print("REFUSED --resume:\n  " + "\n  ".join(problems))
            return 1
    else:
        held = [name for name in CHECKPOINT_STATE if (checkpoint_dir / name).exists()]
        if held:
            print(f"REFUSED: {checkpoint_dir} already holds {held}; pass --resume to continue "
                  f"that run, or move it before rerunning")
            return 1
    problems = preflight(paths)
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems))
        return 1
    derived = b28_class_stats(paths["data_dir"], paths["b28_label_dir"])
    target = checkpoint_dir / B28_STATS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(derived, indent=2, ensure_ascii=False))
    info = derived["im023_class_28"]
    print(f"class weights: {target}: main's, class 28 = {info['share']:.2%} of class {OLD_CLASS}")
    print(f"profile: {args.profile} {PROFILES[args.profile]}")
    command = [sys.executable, str(ROOT / "scripts" / "train_unified.py"),
               *train_args(args.profile, args.epochs, args.checkpoint_dir, paths), *extra]
    print("TRAIN:", " ".join(command), flush=True)
    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
