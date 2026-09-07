"""Bit-parity gate for the two-stage cached validation architecture.

Proves the refactor (``infer_tiles.py`` Stage A → cache → ``cache_predict`` →
existing scorers) reproduces the DIRECT fused path, with no GPU or cluster:

  (a) Stage A writes per-tile cache files + a MANIFEST.
  (b) Re-running Stage A is idempotent (already-cached tiles skipped).
  (c) The cache-backed predict_fn returns arrays equal (float16 tolerance) to a
      DIRECT ``run_inference`` / ``run_fraction_inference`` on the same model.
  (d) ``score_against_nfi`` yields an identical dict whether given the direct
      predict_fn or the cached one.

A tiny deterministic fake ``nn.Module`` (dual-head, tessera single-frame path)
plus synthetic tessera tiles keep the fixture GPU-free; the fake stands in for
the real backbone so the *plumbing* — preprocessing reuse, cache round-trip,
scorer contract — is what's under test, exactly where the refactor could break.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


infcmp = _load("_infcmp", SCRIPTS / "inference_comparison.py")
van = _load("validate_against_nfi", SCRIPTS / "validate_against_nfi.py")
infer_tiles = _load("infer_tiles", SCRIPTS / "infer_tiles.py")
cache_predict = _load("cache_predict", SCRIPTS / "cache_predict.py")

from imint.training.unified_dataset import AUX_CHANNEL_NAMES  # noqa: E402

TILE = 16
NUM_CLASSES = 28
EMB_DIM = 128


class _FmSpec:
    family = "tessera"


class FakeDualHead(torch.nn.Module):
    """Deterministic dual-head model over the tessera single-frame path.

    forward(img5d (B,128,H,W), aux=(B,10,H,W), …, return_fractions=bool)
      → logits (B, NUM_CLASSES, H, W), frac_logits (B, 4, H, W)

    Fully deterministic from the input content (fixed conv weights seeded once),
    so the direct path and the cached path are comparing the SAME numbers.
    ``frac_head`` is a real attribute so the frac-inference guard passes.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.classifier = torch.nn.Conv2d(EMB_DIM + 10, NUM_CLASSES, 1)
        self.frac_head = torch.nn.Conv2d(EMB_DIM + 10, 4, 1)
        self.fm_spec = _FmSpec()
        self.eval()

    def forward(self, img5d, aux=None, temporal_coords=None,
                location_coords=None, return_fractions=False):
        x = torch.cat([img5d, aux], dim=1)
        logits = self.classifier(x)
        if return_fractions:
            return logits, self.frac_head(x)
        return logits


def _write_tile(path, seed):
    """Minimal tessera tile: (128,H,W) embedding + the 10 canonical aux maps."""
    rng = np.random.default_rng(seed)
    d = {"tessera": rng.standard_normal((EMB_DIM, TILE, TILE)).astype(np.float32)}
    for ch in AUX_CHANNEL_NAMES:
        d[ch] = rng.standard_normal((TILE, TILE)).astype(np.float32)
    np.savez_compressed(path, **d)


@pytest.fixture()
def tiles(tmp_path):
    data_dir = tmp_path / "tiles"
    data_dir.mkdir()
    for i in range(3):
        _write_tile(data_dir / f"tileX_{i}.npz", seed=100 + i)
    return data_dir


class _FakeInfcmp:
    """infcmp shim: real preprocessing + our fake load_model."""

    def __init__(self, model):
        self._model = model
        self._build_inference_inputs = infcmp._build_inference_inputs
        self.run_inference = infcmp.run_inference
        self.run_fraction_inference = infcmp.run_fraction_inference

    def load_model(self, ckpt_path, device, backbone_name=None, img_size=None):
        return self._model, 0, 0.0, TILE


def _run_stage_a(model, tiles, cache_dir, checkpoint):
    """Run infer_tiles.infer_all with the fake model patched in."""
    import unittest.mock as mock
    with mock.patch.object(infer_tiles, "_load_infcmp",
                           return_value=_FakeInfcmp(model)):
        return infer_tiles.infer_all(
            checkpoint=str(checkpoint), backbone_name="tessera_v1",
            data_dir=str(tiles), tile_list=None, img_size=TILE,
            num_classes=NUM_CLASSES, cache_dir=str(cache_dir),
            batch_size=2, num_workers=0, device="cpu", shard=None,
            log_every=0, produced_at="2026-08-13T00:00:00Z")


@pytest.fixture()
def checkpoint(tmp_path):
    p = tmp_path / "best_model.pt"
    p.write_bytes(b"fake-checkpoint-bytes-deterministic")
    return p


def test_stage_a_writes_cache_and_manifest(tiles, cache_dir_tmp, checkpoint):
    model = FakeDualHead()
    sha, (n_written, n_skipped) = _run_stage_a(
        model, tiles, cache_dir_tmp, checkpoint)
    sha_dir = Path(cache_dir_tmp) / sha
    assert n_written == 3 and n_skipped == 0
    for i in range(3):
        cpath = sha_dir / f"tileX_{i}.npz"
        assert cpath.exists()
        with np.load(cpath) as z:
            assert z["pred"].dtype == np.uint8
            assert z["pred"].shape == (TILE, TILE)
            assert z["probs"].shape == (NUM_CLASSES, TILE, TILE)
            assert z["fracs"].shape == (4, TILE, TILE)
    manifest = json.loads((sha_dir / "MANIFEST.json").read_text())
    assert manifest["ckpt_sha"] == sha
    assert manifest["n_tiles"] == 3
    assert manifest["produced_at"] == "2026-08-13T00:00:00Z"


def test_stage_a_idempotent_skip(tiles, cache_dir_tmp, checkpoint):
    model = FakeDualHead()
    _run_stage_a(model, tiles, cache_dir_tmp, checkpoint)
    _, (n_written2, n_skipped2) = _run_stage_a(
        model, tiles, cache_dir_tmp, checkpoint)
    assert n_written2 == 0 and n_skipped2 == 3


def test_dataset_threads_model_num_frames(tiles):
    """infer_tiles must pass the model's frame count into
    _build_inference_inputs. A single-frame ladder checkpoint (num_frames=1)
    left unthreaded (None) takes the 4-frame branch, and the model's
    single-frame temporal_encoding then rejects the 4x token grid (15376 vs
    3844) — the failure that killed the first ladder-eval wave. Prove the
    dataset forwards num_frames verbatim, independent of family routing."""
    import unittest.mock as mock

    captured = {}

    class _CapInfcmp:
        def _build_inference_inputs(self, tile_path, device, img_size,
                                    aux_names, family="prithvi",
                                    num_frames=None):
            captured["num_frames"] = num_frames
            return {"img5d": torch.zeros(1, 1, 4, 4),
                    "aux": torch.zeros(1, 1, 4, 4), "crop_sz": 4,
                    "temporal_coords": None, "location_coords": None}

    tile = sorted(tiles.glob("*.npz"))[0]
    with mock.patch.object(infer_tiles, "_load_infcmp",
                           return_value=_CapInfcmp()):
        ds = infer_tiles.TileInferenceDataset(
            [str(tile)], "prithvi", TILE, None, num_frames=1)
        _ = ds[0]
    assert captured["num_frames"] == 1, (
        "dataset dropped num_frames — 1-frame checkpoints will crash at "
        "temporal_encoding")


def _direct_probs(model, tile_path, device):
    """DIRECT softmax the fused NFI path produces — via the SAME preprocessing.

    ``run_inference(return_probs=True)`` also reads raw ``spectral`` for optional
    superpixel refinement, a key tessera tiles don't carry; the softmax it
    returns is exactly this. Reuse ``_build_inference_inputs`` (not a
    reimplementation) and take softmax of a plain forward — identical to what the
    validator's predict_fn samples and what Stage A caches.
    """
    inp = infcmp._build_inference_inputs(
        tile_path, device, TILE, None, family="tessera")
    with torch.no_grad():
        logits = model(inp["img5d"], aux=inp["aux"],
                       temporal_coords=inp["temporal_coords"],
                       location_coords=inp["location_coords"])
        return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()


def test_cached_predict_matches_direct(tiles, cache_dir_tmp, checkpoint):
    model = FakeDualHead()
    sha, _ = _run_stage_a(model, tiles, cache_dir_tmp, checkpoint)
    device = torch.device("cpu")

    cached_hard = cache_predict.make_cached_predict_fn(
        cache_dir_tmp, sha, want_fracs=False)
    cached_frac = cache_predict.make_cached_predict_fn(
        cache_dir_tmp, sha, want_fracs=True)

    for i in range(3):
        tp = str(tiles / f"tileX_{i}.npz")
        # DIRECT paths — the same preprocessing + forward the validators use.
        probs_d = _direct_probs(model, tp, device)
        class_d = probs_d.argmax(0).astype(np.int64)
        fracs_d = infcmp.run_fraction_inference(model, tp, device, img_size=TILE)

        class_c, probs_c = cached_hard(tp)
        class_cf, probs_cf, fracs_c = cached_frac(tp)

        # argmax class map is exact (uint8 stored, no lossy step).
        np.testing.assert_array_equal(class_c, class_d)
        np.testing.assert_array_equal(class_cf, class_d)
        # probs/fracs round-trip through float16 storage → float16 tolerance.
        np.testing.assert_allclose(probs_c, probs_d, atol=1e-3)
        np.testing.assert_allclose(fracs_c, fracs_d, atol=1e-3)


def _nfi_index(tiles):
    """Tiny NFI plot index: one plot per tile, forest truth spread over 1..4."""
    def _vol(pine=0, spruce=0, birch=0):
        return {"VolPine": pine, "VolContorta": 0, "VolSpruce": spruce,
                "VolBirch": birch, "VolOtherDec": 0, "Maturityclass": 41}
    rows = []
    specs = [_vol(pine=100), _vol(spruce=100), _vol(birch=100)]
    for i, v in enumerate(specs):
        rows.append({
            "tile_name": f"tileX_{i}",
            "tile_path": str(tiles / f"tileX_{i}.npz"),
            "row": 4 + i, "col": 5 + i, **v,
        })
    return pd.DataFrame(rows)


def test_score_against_nfi_direct_vs_cached(tiles, cache_dir_tmp, checkpoint):
    model = FakeDualHead()
    sha, _ = _run_stage_a(model, tiles, cache_dir_tmp, checkpoint)
    device = torch.device("cpu")
    index_df = _nfi_index(tiles)

    def direct_predict_fn(tile_path):
        probs = _direct_probs(model, tile_path, device)
        return probs.argmax(0).astype(np.int64), probs

    cached_predict_fn = cache_predict.make_cached_predict_fn(
        cache_dir_tmp, sha, want_fracs=False)

    res_direct = van.score_against_nfi(
        index_df.copy(), direct_predict_fn, num_classes=NUM_CLASSES)
    res_cached = van.score_against_nfi(
        index_df.copy(), cached_predict_fn, num_classes=NUM_CLASSES)

    # Class-map argmax is exact → class counts / confusion identical.
    assert res_direct["n_plots"] == res_cached["n_plots"]
    assert res_direct["forest_type_accuracy"] == res_cached["forest_type_accuracy"]
    assert res_direct["confusion_nfi_x_pred"] == res_cached["confusion_nfi_x_pred"]
    assert res_direct["accuracy_suite"] == res_cached["accuracy_suite"]
    # AUROC reads the float16-stored probs → allow tiny numeric drift.
    for cls, d in res_direct["per_class_auroc"].items():
        assert cls in res_cached["per_class_auroc"]
        assert res_cached["per_class_auroc"][cls]["auroc"] == pytest.approx(
            d["auroc"], abs=1e-4)


@pytest.fixture()
def cache_dir_tmp(tmp_path):
    d = tmp_path / "pred_cache"
    d.mkdir()
    return d
