"""The inference matrix must be frozen, spread, sha-keyed and evidence-bound.

The matrix is the ladder's visual cross-check: K frozen holdout tiles ×
every completed cell. Its claims rest on (1) the tile panel being frozen
once and geographically spread, (2) predictions being keyed to the exact
checkpoint bytes so a re-trained cell cannot silently show stale paint,
(3) the job following the ladder's manifest conventions (digest image,
pins, evidence records, no gate writes).
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

from scripts.gen_ladder_manifests import OUT_DIR  # noqa: E402

JOB = OUT_DIR / "inference-matrix-job.yaml"


def _job_text() -> str:
    doc = yaml.safe_load(JOB.read_text())
    c = doc["spec"]["template"]["spec"]["containers"][0]
    return "\n".join(c.get("command") or [])


def test_select_spread_is_deterministic_and_spread():
    from ladder_inference_matrix import select_spread

    # A 3×3 grid plus a dense cluster: farthest-point must fan out to the
    # corners before it ever returns to the cluster.
    centers = {f"g{i}{j}": (i * 100.0, j * 100.0)
               for i in range(3) for j in range(3)}
    centers.update({f"c{k}": (105.0 + k, 105.0) for k in range(5)})

    picked = select_spread(centers, 4)
    assert picked == select_spread(centers, 4), "must be deterministic"
    assert picked[0] == "g00", "seed = southernmost, name-tiebroken"
    assert set(picked) <= set(centers)
    corners = {"g00", "g02", "g20", "g22"}
    assert len(set(picked) & corners) >= 3, f"not spread: {picked}"
    assert not any(p.startswith("c") for p in picked), \
        "cluster picked before the grid corners"


def test_select_spread_small_pool_returns_everything():
    from ladder_inference_matrix import select_spread

    centers = {"a": (0.0, 0.0), "b": (1.0, 1.0)}
    assert select_spread(centers, 10) == ["a", "b"]


def test_colorize_matches_unified_palette():
    from imint.training.unified_schema import UNIFIED_COLORS

    from ladder_inference_matrix import colorize

    label = np.array([[0, 11], [21, 22]], dtype=np.uint8)
    rgb = colorize(label)
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 1]) == UNIFIED_COLORS[11]   # vete — gold
    assert tuple(rgb[1, 0]) == UNIFIED_COLORS[21]   # majs
    assert tuple(rgb[1, 1]) == UNIFIED_COLORS[22]   # hygge


def test_summer_rgb_is_uint8_image():
    from ladder_inference_matrix import summer_rgb

    spectral = np.random.default_rng(0).random((24, 16, 16)).astype(np.float32)
    rgb = summer_rgb(spectral)
    assert rgb.shape == (16, 16, 3) and rgb.dtype == np.uint8


def test_job_follows_ladder_conventions():
    doc = yaml.safe_load(JOB.read_text())
    spec = doc["spec"]["template"]["spec"]
    c = spec["containers"][0]
    text = _job_text()

    assert "@sha256:" in c["image"]
    assert spec["nodeSelector"] == {"accelerator": "nvidia-gtx-2080ti"}
    assert {m["mountPath"] for m in c["volumeMounts"]} >= {"/cephfs", "/data"}
    assert "set -euo pipefail" in text
    assert "numpy==" in text and "pip freeze > /cephfs/ops/deps/" in text
    assert "run=$RUN_ID" in text
    # no gate: the matrix is display, it must never unlock a rung
    writes_gate = [ln for ln in text.splitlines()
                   if "_GATE_OK" in ln and not ln.strip().startswith("#")]
    assert not writes_gate


def test_job_installs_every_backbone_loader():
    """One pod loads all six families — the union of loader deps or the
    stricter columns' checkpoints cannot even load."""
    text = _job_text()
    for needle in ("terratorch", "antofuller/CROMA", "Clay-foundation/model"):
        assert needle in text, f"missing loader dep: {needle}"


def test_job_freezes_and_renders_via_the_script():
    text = _job_text()
    assert "scripts/ladder_inference_matrix.py" in text
    assert "--holdout-dir /cephfs/holdout_val_512" in text
    assert "--out-dir /cephfs/ladder_inference" in text
    assert '--git-sha "$HEAD_SHA"' in text


def test_dashboard_serves_and_renders_the_matrix():
    """Wiring pins: the server symlinks the PVC dir, the dashboard fetches
    matrix.json and renders the panel LAST."""
    server = (REPO / "k8s" / "training-dashboard-server.yaml").read_text()
    assert "ln -sfn /cephfs/ladder_inference /www/ladder/inference" in server

    dash = (REPO / "dashboards" / "ladder_dashboard.html").read_text()
    assert "inference/matrix.json" in dash
    assert 'id="infMatrix"' in dash
    panels = [i for i in range(len(dash)) if dash.startswith('<div class="panel">', i)]
    assert dash.find('id="infMatrix"') > panels[-1] - 200, \
        "the inference panel must be the LAST panel"


def test_cell_rerenders_on_checkpoint_change(tmp_path, monkeypatch):
    """Predictions are keyed to checkpoint BYTES: same sha skips, changed
    sha re-renders — a re-trained best_model must never show stale paint."""
    import json

    import ladder_inference_matrix as lim

    calls = {"n": 0}

    class _FakeInfcmp:
        @staticmethod
        def load_model(ckpt, dev, backbone_name=None, img_size=None):
            calls["n"] += 1
            return object(), 7, 0.5, img_size

        @staticmethod
        def run_inference(model, tile_path, dev, img_size=None,
                          aux_channel_names=None):
            return np.zeros((4, 4), dtype=np.uint8)

    class _FakeTorch:
        @staticmethod
        def device(d):
            return type("D", (), {"type": "cpu"})()

        @staticmethod
        def load(*a, **k):
            return {"config": {}}

    monkeypatch.setattr(lim, "_load_infcmp", lambda: _FakeInfcmp)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"weights-v1")
    tiles = [{"name": "t1"}]
    holdout = tmp_path  # unused by the fakes
    out = tmp_path / "out"

    c1 = lim.render_cell("tessera", 2, ckpt, tiles, holdout, out, "cpu")
    assert calls["n"] == 1 and (out / "tessera_r2" / "t1.png").exists()

    c2 = lim.render_cell("tessera", 2, ckpt, tiles, holdout, out, "cpu")
    assert calls["n"] == 1, "unchanged checkpoint must be skipped"
    assert c2["ckpt_sha"] == c1["ckpt_sha"]

    ckpt.write_bytes(b"weights-v2-retrained")
    c3 = lim.render_cell("tessera", 2, ckpt, tiles, holdout, out, "cpu")
    assert calls["n"] == 2, "changed checkpoint must re-render"
    assert c3["ckpt_sha"] != c1["ckpt_sha"]
    cell = json.loads((out / "tessera_r2" / "_cell.json").read_text())
    assert cell["ckpt_sha"] == c3["ckpt_sha"]
