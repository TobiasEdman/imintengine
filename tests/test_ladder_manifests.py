"""The ladder manifests must encode exactly one varying axis.

docs/experiments/label_source_ladder.md rests on a single claim: every rung is
the same run except for its label flags, so a rung-to-rung delta attributes to
one change. A stray --epochs or a leaked --warm-start-from would silently turn
a label-source result into a training-schedule artefact. These tests pin that
claim to the committed manifests, and pin the manifests to their generator.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gen_ladder_manifests import (  # noqa: E402
    BASES,
    COHORT_DIR,
    ENABLE_MARKFUKT,
    EPOCHS,
    OUT_DIR,
    RUNGS,
)

REPO = Path(__file__).resolve().parents[1]


def _flag_args(manifest: dict) -> str:
    """Only real flag lines — comments and preflight greps mention flags too."""
    c = manifest["spec"]["template"]["spec"]["containers"][0]
    body = "\n".join(c.get("args") or c.get("command") or [])
    return " ".join(ln.strip() for ln in body.splitlines()
                    if ln.strip().startswith("--"))


def _manifests():
    for model in BASES:
        for rung in RUNGS:
            path = OUT_DIR / f"ladder-r{rung}-{model}-job.yaml"
            yield model, rung, path


@pytest.mark.parametrize("model,rung,path", list(_manifests()),
                         ids=lambda v: v if isinstance(v, (str, int)) else "")
def test_rung_encodes_its_label_source(model, rung, path):
    assert path.exists(), f"missing manifest {path}"
    doc = yaml.safe_load(path.read_text())
    args = _flag_args(doc)
    _, label_dir, num_classes, with_frac = RUNGS[rung]

    if label_dir is None:
        assert "--label-dir" not in args, "rung 1 must use the in-tile label"
    else:
        assert f"--label-dir {label_dir.format(model=model)}" in args
    assert ("--frac-dir" in args) is with_frac
    assert f"--num-classes {num_classes}" in args


@pytest.mark.parametrize("model,rung,path", list(_manifests()),
                         ids=lambda v: v if isinstance(v, (str, int)) else "")
def test_cohort_is_identical_across_rungs(model, rung, path):
    """All four rungs must train on the same tiles, or a delta is confounded.

    Rungs 2-4 land on the NMD2023 sidecar set via --label-dir. Rung 1 reads
    the in-tile label and would otherwise see the full superset, so it must
    carry the explicit --cohort-dir gate.
    """
    args = _flag_args(yaml.safe_load(path.read_text()))
    if rung == 1:
        assert f"--cohort-dir {COHORT_DIR}" in args, "rung 1 cohort not pinned"
    else:
        assert f"--label-dir" in args


@pytest.mark.parametrize("model", sorted(BASES))
def test_rung3_and_4_self_distil(model):
    """Rung 3/4 read THIS backbone's own distillation, not a shared pool."""
    for rung in (3, 4):
        path = OUT_DIR / f"ladder-r{rung}-{model}-job.yaml"
        args = _flag_args(yaml.safe_load(path.read_text()))
        assert f"--label-dir /cephfs/distill/{model}_r2" in args
        for other in BASES:
            if other != model:
                assert f"/cephfs/distill/{other}_r2" not in args


@pytest.mark.parametrize("model,rung,path", list(_manifests()),
                         ids=lambda v: v if isinstance(v, (str, int)) else "")
def test_controls_are_held_constant(model, rung, path):
    doc = yaml.safe_load(path.read_text())
    args = _flag_args(doc)
    assert "--warm-start-from" not in args, "every rung must cold-start"
    assert f"--epochs {EPOCHS}" in args, "epochs must not vary across the ladder"
    # markfukt is opt-in in the trainer; a manifest that silently drops it
    # trains an 11th-aux-less model and is not comparable to the rest.
    assert ("--enable-markfukt" in args) is ENABLE_MARKFUKT, \
        "soil-moisture aux must be uniform across the ladder"


@pytest.mark.parametrize("model,rung,path", list(_manifests()),
                         ids=lambda v: v if isinstance(v, (str, int)) else "")
def test_identity_and_isolated_outputs(model, rung, path):
    doc = yaml.safe_load(path.read_text())
    meta = doc["metadata"]
    assert meta["name"] == f"ladder-r{rung}-{model}"
    assert meta["labels"]["purpose"] == "ladder"
    assert meta["labels"]["rung"] == f"r{rung}"
    assert meta["labels"]["model"] == model
    # A ladder run must never write into a historical checkpoint dir.
    assert f"/cephfs/checkpoints/ladder/{model}_r{rung}" in _flag_args(doc)


@pytest.mark.parametrize("model,rung,path", list(_manifests()),
                         ids=lambda v: v if isinstance(v, (str, int)) else "")
def test_no_rwo_pvc_mounts(model, rung, path):
    """RWO volumes node-pin the pod behind the dashboard's Multi-Attach lock.

    Ladder jobs must be schedulable on any GPU node: cephfs (RWX) only.
    """
    spec = yaml.safe_load(path.read_text())["spec"]["template"]["spec"]
    claims = {v["persistentVolumeClaim"]["claimName"]
              for v in spec.get("volumes", []) if "persistentVolumeClaim" in v}
    assert claims <= {"training-data-cephfs"}, f"RWO PVC leaked: {claims}"


def test_non_crop_manifests_match_generator():
    """Anchor-independent manifests stay regenerable during crop bootstrap."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/gen_ladder_manifests.py",
            "--non-crop-only",
            "--check",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_cell_of_the_matrix_exists():
    found = {(m, r) for m, r, p in _manifests() if p.exists()}
    assert len(found) == len(BASES) * len(RUNGS) == 28


def test_4f_dependency_lock_is_shared_and_complete():
    """One lock for every 4f job: the base manifest (inherited by the four
    rung jobs) and the generator's pinned distill block must pin the SAME
    package==version set, and nothing in either may be unversioned. Two
    drifting copies would silently split the 'shared lock' the exploratory
    comparison's runtime identity rests on (PR #42 review)."""
    import re

    from scripts.gen_ladder_manifests import _DISTILL_DEPS_PINNED

    base = (REPO / "k8s" / "train-prithvi300m-4f-job.yaml").read_text()

    def pins(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.!+-]*", text))

    base_pins = pins(base)
    distill_pins = pins(_DISTILL_DEPS_PINNED)
    assert base_pins == distill_pins, (
        f"lock drift: only-in-base={sorted(base_pins - distill_pins)} "
        f"only-in-distill={sorted(distill_pins - base_pins)}")
    assert len(base_pins) >= 40, "the full resolved lock, not a top-level list"

    # No unversioned install may sneak in beside the lock (comments exempt).
    for line in base.splitlines():
        code = line.split("#", 1)[0]
        if "pip install" in code and "--no-deps" not in code:
            raise AssertionError(f"unpinned pip path in 4f base: {line.strip()}")
