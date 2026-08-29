#!/usr/bin/env python3
"""Generate the label-source ladder manifests (6 backbones × 4 rungs).

See docs/experiments/label_source_ladder.md. Every rung is the same trainer on
the same tiles with the same per-backbone regime; ONLY the label flags change,
so a rung-to-rung delta attributes to exactly one thing:

    rung 1 nmd2018    in-tile 23-class label       (no --label-dir)
    rung 2 nmd2023    28-class NMD2023 sidecar     (--label-dir nmd2023_labels)
    rung 3 nfi        + NFI-distilled forest type  (--label-dir …_distill_…)
    rung 4 tradslag   + Trädslag fraction head     (+ --frac-dir)

Hand-writing 24 near-identical manifests invites exactly the drift this
experiment cannot tolerate (one stray --epochs and a rung delta becomes an
early-stopping artefact). So they are generated from the six existing
per-backbone yamls, which stay the single source of each backbone's regime —
crop size, aux fusion, ΔSAR, model-specific preflights and all.

    python scripts/gen_ladder_manifests.py           # write k8s/ladder/
    python scripts/gen_ladder_manifests.py --check   # verify on disk == generated

--check exits non-zero if any file is missing or stale, so CI can pin the
manifests to this generator.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "k8s" / "ladder"

# Backbone → the yaml that defines its regime. Prithvi-600M uses the v8b
# NMD2023 long job (30 epochs, 504 crop, markfukt off) rather than
# train-prithvi-600m-512-job.yaml (10 epochs); its --warm-start-from is
# stripped below so the ladder cold-starts like every other column.
BASES = {
    "prithvi300m": "k8s/train-prithvi300m-job.yaml",
    "prithvi600m": "k8s/train-v8b-nmd2023-long-job.yaml",
    "croma": "k8s/train-croma-job.yaml",
    "terramind": "k8s/train-terramind-job.yaml",
    "tessera": "k8s/train-tessera-gated-job.yaml",
    "clay": "k8s/train-clay-job.yaml",
}

# The cohort gate. Every rung trains on exactly the tiles that have an NMD2023
# sidecar (~94.5% coverage), so rung 1 — which reads the in-tile 23-class label
# and would otherwise see a superset — cannot mix a cohort change into its
# delta. Rungs 2-4 land on this set anyway via --label-dir; passing it as
# --cohort-dir on rung 1 makes all four identical by construction.
COHORT_DIR = "/cephfs/nmd2023_labels"

# rung → (slug, label-dir or None, num-classes, frac head?)
# Rung 3/4's label dir is per-backbone: each model distils ITS OWN rung-2
# features (user, 2026-08-29 — "that distinguish them more"), so the sidecars
# live in a model-scoped directory rather than one shared pool.
RUNGS = {
    1: ("nmd2018", None, 23, False),
    2: ("nmd2023", COHORT_DIR, 28, False),
    3: ("nfi", "/cephfs/distill/{model}_r2", 28, False),
    4: ("tradslag", "/cephfs/distill/{model}_r2", 28, True),
}

EPOCHS = 30  # fixed across the ladder: see the doc's "Controls"

_FLAG_LINE = r"^(?P<indent>\s*)--{flag}[ =][^\n]*?(?P<cont>\s*\\)?$"

# The base manifests use both YAML label styles — flow (`labels: {a: b}`) and
# block (`labels:\n  a: b`). Both must be rewritten or a job keeps its old
# purpose label and drops out of the ladder's selectors.
_LABELS_FLOW = re.compile(r"^(?P<indent>[ ]*)labels: \{[^}]*\}$", re.M)
_LABELS_BLOCK = re.compile(
    r"^(?P<indent>[ ]*)labels:\n(?:(?P=indent)[ ]{2}\S[^\n]*\n)+", re.M)


def _set_labels(text: str, rung: int, model: str) -> str:
    def flow(m: re.Match) -> str:
        return (f'{m.group("indent")}labels: {{ app: unified-training, '
                f'purpose: ladder, rung: "r{rung}", model: {model} }}')
    text = _LABELS_FLOW.sub(flow, text)
    return _LABELS_BLOCK.sub(lambda m: flow(m) + "\n", text)


def _drop_flag(text: str, flag: str) -> str:
    return re.sub(_FLAG_LINE.format(flag=re.escape(flag)), "", text,
                  flags=re.M).replace("\n\n", "\n")


def _set_flag(text: str, flag: str, value: str) -> str:
    """Rewrite `--flag <value>`, preserving indent and any trailing backslash.

    The replacement is a callable: these lines end in YAML's `\\` continuation,
    which re.sub would otherwise read as a dangling escape.
    """
    pattern = re.compile(_FLAG_LINE.format(flag=re.escape(flag)), re.M)
    m = pattern.search(text)
    if not m:
        return text
    repl = f"{m.group('indent')}--{flag} {value}{m.group('cont') or ''}"
    return pattern.sub(lambda _: repl, text, count=1)


def _ensure_flag_after(text: str, anchor: str, flag: str, value: str) -> str:
    """Insert `--flag value` right after the anchor flag if absent."""
    if re.search(_FLAG_LINE.format(flag=re.escape(flag)), text, re.M):
        return _set_flag(text, flag, value)
    pattern = re.compile(_FLAG_LINE.format(flag=re.escape(anchor)), re.M)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"anchor --{anchor} not found; cannot insert --{flag}")
    cont = m.group("cont") or ""
    line = f"{m.group('indent')}--{flag} {value}{cont}"
    return text[:m.end()] + "\n" + line + text[m.end():]


def render(model: str, rung: int, base_text: str) -> str:
    slug, label_dir, num_classes, with_frac = RUNGS[rung]
    label_dir = label_dir.format(model=model) if label_dir else None
    job = f"ladder-r{rung}-{model}"
    out = base_text

    # Label source — the one axis this experiment varies.
    if label_dir is None:
        out = _drop_flag(out, "label-dir")
    else:
        out = _ensure_flag_after(out, "data-dirs", "label-dir", label_dir)
    # Cohort held constant across all rungs. Rungs 2-4 already land on the
    # sidecar set via --label-dir; rung 1 needs the explicit gate.
    if label_dir is None:
        out = _ensure_flag_after(out, "data-dirs", "cohort-dir", COHORT_DIR)
    if with_frac:
        out = _ensure_flag_after(out, "label-dir", "frac-dir",
                                 "/cephfs/tradslag_fracs")
    else:
        out = _drop_flag(out, "frac-dir")
    out = _set_flag(out, "num-classes", str(num_classes))

    # Controls: cold start, fixed epochs. The base manifests carry prose
    # describing the warm-start we just removed ("Continue from the 0.478
    # checkpoint…"); left in place it would document a control the run does
    # not apply, which is worse than no comment at all.
    out = _drop_flag(out, "warm-start-from")
    out = re.sub(r"^\s*#.*warm-start.*\n", "", out, flags=re.M | re.I)
    out = _set_flag(out, "epochs", str(EPOCHS))

    # Isolate outputs so no ladder run can overwrite a historical checkpoint.
    out = _set_flag(out, "checkpoint-dir",
                    f"/cephfs/checkpoints/ladder/{model}_r{rung}")
    out = re.sub(r"^(\s*)mkdir -p /cephfs/checkpoints/\S+",
                 rf"\1mkdir -p /cephfs/checkpoints/ladder/{model}_r{rung}",
                 out, flags=re.M)
    out = re.sub(r"^(\s*)rm -f /cephfs/checkpoints/\S+/\*\.pt\.tmp",
                 rf"\1rm -f /cephfs/checkpoints/ladder/{model}_r{rung}/*.pt.tmp",
                 out, flags=re.M)

    # Strip vestigial RWO PVC mounts (training-data, training-checkpoints).
    # Ladder checkpoints live on cephfs (RWX); the RWO volumes only supplied
    # the /data weight cache, and an RWO volume held by the dashboard on
    # p02r08srv01 Multi-Attach-blocks any ladder pod scheduled elsewhere
    # (ladder-r1-prithvi600m sat in ContainerCreating exactly this way).
    # Weights fall back to the HF download path the manifests already handle.
    for vol in ("training-data", "training-checkpoints"):
        out = re.sub(
            rf"^\s*- name: {vol}\n(?:\s+(?:mountPath|readOnly|persistentVolumeClaim|claimName):[^\n]*\n)+",
            "", out, flags=re.M)

    # Identity: name + selectable labels (by rung, by model, or both).
    out = re.sub(r"^  name: \S+$", f"  name: {job}", out, count=1, flags=re.M)
    out = _set_labels(out, rung, model)

    header = (
        f"# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        f"# Ladder rung {rung} ({slug}) for {model}. Edit the base manifest\n"
        f"# ({BASES[model]}) or the generator, then regenerate.\n"
        f"# Plan: docs/experiments/label_source_ladder.md\n"
    )
    return header + out.lstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the committed manifests match this generator")
    args = ap.parse_args()

    stale: list[str] = []
    for model, base_rel in BASES.items():
        base_text = (REPO / base_rel).read_text()
        for rung in RUNGS:
            text = render(model, rung, base_text)
            dest = OUT_DIR / f"ladder-r{rung}-{model}-job.yaml"
            if args.check:
                if not dest.exists() or dest.read_text() != text:
                    stale.append(str(dest.relative_to(REPO)))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            print(f"  wrote {dest.relative_to(REPO)}")

    if args.check:
        if stale:
            print(f"STALE ({len(stale)}): " + ", ".join(stale))
            print("Re-run: python scripts/gen_ladder_manifests.py")
            return 1
        print(f"all {len(BASES) * len(RUNGS)} ladder manifests up to date")
        return 0

    print(f"\n{len(BASES) * len(RUNGS)} manifests in {OUT_DIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
