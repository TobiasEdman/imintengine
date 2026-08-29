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
from scripts.gen_ladder_manifests import BASES, EPOCHS, OUT_DIR, RUNGS  # noqa: E402

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
        assert f"--label-dir {label_dir}" in args
    assert ("--frac-dir" in args) is with_frac
    assert f"--num-classes {num_classes}" in args


@pytest.mark.parametrize("model,rung,path", list(_manifests()),
                         ids=lambda v: v if isinstance(v, (str, int)) else "")
def test_controls_are_held_constant(model, rung, path):
    doc = yaml.safe_load(path.read_text())
    args = _flag_args(doc)
    assert "--warm-start-from" not in args, "every rung must cold-start"
    assert f"--epochs {EPOCHS}" in args, "epochs must not vary across the ladder"


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


def test_manifests_match_generator():
    """Committed manifests are regenerable — no hand-edits."""
    r = subprocess.run([sys.executable, "scripts/gen_ladder_manifests.py", "--check"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_cell_of_the_matrix_exists():
    found = {(m, r) for m, r, p in _manifests() if p.exists()}
    assert len(found) == len(BASES) * len(RUNGS) == 24
