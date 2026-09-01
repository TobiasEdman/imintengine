"""The LUCAS crop-distill stage must be evidence-only and protocol-pinned.

The stage produces the numbers the R5 decision is made on [user-stated
2026-08-31: distillability before any retraining]. Two properties carry
that: (1) the OOF protocol is identical across columns — same split file,
same folds, same head, same truth column — so the numbers compare; (2) the
stage opens NO gate, so the ladder queue cannot auto-train a rung 5 the
decision has not approved. These tests pin both, plus the two generalization
holes found on the way in (from_records dropping the LUCAS columns;
accuracy_suite collapsing crop ids to non-forest).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.gen_ladder_manifests import (  # noqa: E402
    CROP_INDEX, CROP_SPLIT, CROP_TRUTH_COL, DISTILL, OUT_DIR,
)

MODELS = sorted(DISTILL)


def _job_text(path: Path) -> str:
    doc = yaml.safe_load(path.read_text())
    c = doc["spec"]["template"]["spec"]["containers"][0]
    return "\n".join(c.get("command") or c.get("args") or [])


def _crop_path(model: str) -> Path:
    return OUT_DIR / f"crop-distill-{model}-job.yaml"


def test_every_column_has_a_crop_distill_manifest():
    missing = [m for m in MODELS if not _crop_path(m).exists()]
    assert not missing, f"missing crop-distill manifests: {missing}"
    assert (OUT_DIR / "lucas-crop-split-job.yaml").exists()


@pytest.mark.parametrize("model", MODELS)
def test_crop_protocol_is_pinned_and_uniform(model):
    """Folds/head/truth/split identical across columns — or the numbers
    do not compare and the R5 decision rests on noise."""
    text = _job_text(_crop_path(model))
    assert "--folds 5" in text
    assert "--heads mlp" in text
    assert f"--truth-col {CROP_TRUTH_COL}" in text
    assert f"SPLIT={CROP_SPLIT}" in text
    assert f"INDEX={CROP_INDEX}" in text
    assert '--pinned-plots "$SPLIT"' in text


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_follows_the_column_regime(model):
    """img-size and backbone-name are the column's own — clay/croma build
    their backbone at the img-size grid, so a wrong value is not a detail."""
    text = _job_text(_crop_path(model))
    cfg = DISTILL[model]
    assert f"--img-size {cfg['img_size']}" in text
    assert f"--backbone-name {cfg['backbone']}" in text
    assert f"/cephfs/checkpoints/ladder/{model}_r2/best_model.pt" in text


@pytest.mark.parametrize("model", MODELS)
def test_crop_outputs_are_model_scoped(model):
    text = _job_text(_crop_path(model))
    assert f"{model}_r2_crop_features.parquet" in text
    assert f"{model}_r2_crop_distillability.json" in text
    for other in MODELS:
        if other != model:
            assert f"{other}_r2" not in text


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_opens_no_gate(model):
    """THE no-front-run guard: the stage must never WRITE a gate marker —
    it would let the ladder queue auto-submit a rung 5 before the human
    decision. (The manifest may MENTION _GATE_OK in its warning comment.)"""
    writes_gate = [ln for ln in _job_text(_crop_path(model)).splitlines()
                   if "_GATE_OK" in ln and not ln.strip().startswith("#")]
    assert not writes_gate, f"crop-distill writes a gate: {writes_gate}"


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_never_touches_h100_quota(model):
    doc = yaml.safe_load(_crop_path(model).read_text())
    spec = doc["spec"]["template"]["spec"]
    assert spec["nodeSelector"] == {"accelerator": "nvidia-gtx-2080ti"}
    c = spec["containers"][0]
    assert c["resources"]["requests"]["memory"] == "24Gi"


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_mounts_both_pvc_paths(model):
    """The LUCAS index inherits absolute /data/… tile paths from L1; a pod
    with only /cephfs mounted drops every point (the NFI extract died so)."""
    doc = yaml.safe_load(_crop_path(model).read_text())
    c = doc["spec"]["template"]["spec"]["containers"][0]
    mounts = {m["mountPath"] for m in c["volumeMounts"]}
    assert {"/cephfs", "/data"} <= mounts


@pytest.mark.parametrize("model,needle", [
    ("terramind", "terratorch"),
    ("croma", "antofuller/CROMA"),
    ("clay", "Clay-foundation/model"),
])
def test_crop_stage_installs_backbone_deps(model, needle):
    """Same load-time deps as the NFI distill stage — the r2 checkpoints
    cannot even load without them."""
    assert needle in _job_text(_crop_path(model))


def test_split_job_freezes_the_canonical_split():
    text = _job_text(OUT_DIR / "lucas-crop-split-job.yaml")
    assert "build_lucas_crop_split.py" in text
    assert "--lucas-index /cephfs/lucas/lucas_tile_index.parquet" in text
    assert "--data-dir /cephfs/unified_v2_512" in text
    assert "--out-dir /cephfs/distill" in text
    doc = yaml.safe_load((OUT_DIR / "lucas-crop-split-job.yaml").read_text())
    c = doc["spec"]["template"]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in c["resources"]["requests"], \
        "the split is CPU work; a GPU request wastes a 2080ti slot"


def test_output_columns_keep_the_lucas_key_and_truth():
    """from_records(columns=…) DROPS record keys missing from the list —
    the hard-coded NFI list lost point_id and unified_class entirely."""
    from extract_plot_features import output_columns

    cols = output_columns("unified_class", ["f000", "f001"])
    assert "point_id" in cols
    assert "unified_class" in cols
    assert "nfi_forest" not in cols
    assert cols[-2:] == ["f000", "f001"]

    nfi = output_columns(None, ["f000"])
    assert "nfi_forest" in nfi
    assert "unified_class" not in nfi


def test_truth_summary_follows_the_mode():
    """The post-write summary crashed on KeyError('nfi_forest') in crop
    mode — after the parquet was written, failing the whole Job
    (backoffLimit 0). Drive the real writer-schema + summary end to end."""
    import pandas as pd
    from extract_plot_features import output_columns, truth_summary

    feat_cols = ["f000", "f001"]
    records = [
        {"TractID": None, "PlotID": None, "point_id": 7, "Easting": None,
         "Northing": None, "tile_name": "t1", "unified_class": 11,
         "f000": 0.1, "f001": 0.2},
        {"TractID": None, "PlotID": None, "point_id": 8, "Easting": None,
         "Northing": None, "tile_name": "t1", "unified_class": 12,
         "f000": 0.3, "f001": 0.4},
    ]
    crop_df = pd.DataFrame.from_records(
        records, columns=output_columns("unified_class", feat_cols))
    name, dist = truth_summary(crop_df, "unified_class")
    assert name == "unified_class"
    assert dist == {11: 1, 12: 1}

    nfi_df = pd.DataFrame.from_records(
        [{**r, "nfi_forest": 1} for r in records],
        columns=output_columns(None, feat_cols))
    name, dist = truth_summary(nfi_df, None)
    assert name == "nfi_forest"
    assert dist == {1: 2}


def test_generic_suite_scores_in_the_truths_own_space():
    """accuracy_suite collapses ids outside {1..4} to 0 — on crop truth
    every plot lands in one class and overall reads a meaningless 1.0.
    The generic suite must keep crop classes apart."""
    from nfi_head_cv import generic_accuracy_suite
    from validate_against_nfi import accuracy_suite

    truth = np.array([11, 11, 12, 12, 15, 15])
    pred = np.array([11, 12, 12, 12, 15, 11])

    collapsed = accuracy_suite(truth, pred)
    assert collapsed["overall_accuracy_5class"] == 1.0  # the trap, proven

    suite = generic_accuracy_suite(truth, pred)
    assert suite["overall_accuracy"] == round(4 / 6, 4)
    assert 0 < suite["cohen_kappa"] < 1
    assert suite["per_class"]["vete"]["support"] == 2
    assert suite["per_class"]["korn"]["producers_accuracy"] == 1.0


def test_generic_suite_perfect_prediction():
    from nfi_head_cv import generic_accuracy_suite

    y = np.array([11, 12, 13, 21, 21])
    suite = generic_accuracy_suite(y, y.copy())
    assert suite["overall_accuracy"] == 1.0
    assert suite["cohen_kappa"] == 1.0
