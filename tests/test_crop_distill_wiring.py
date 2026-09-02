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


# --- Codex re-review findings (2026-09-01) ------------------------------


TEMPLATE_MANIFESTS = sorted(
    [OUT_DIR / f"crop-distill-{m}-job.yaml" for m in MODELS]
    + [OUT_DIR / f"distill-{m}-job.yaml" for m in MODELS]
    + [OUT_DIR / "lucas-crop-split-job.yaml",
       OUT_DIR / "distill-pinned-plots-job.yaml"])


@pytest.mark.parametrize("path", TEMPLATE_MANIFESTS, ids=lambda p: p.name)
def test_cpu_job_images_are_digest_pinned(path):
    """Zero-tolerance rule: a mutable tag can be repointed under the
    pipeline's feet. Every generated template job pins by digest."""
    from scripts.gen_ladder_manifests import PYTHON_IMAGE

    doc = yaml.safe_load(path.read_text())
    image = doc["spec"]["template"]["spec"]["containers"][0]["image"]
    assert image == PYTHON_IMAGE
    assert "@sha256:" in image


def test_split_qualifies_on_the_true_six_column_intersection():
    """REQUIRED_KEYS alone (s1_vv_vh) strands tessera: its dataset drops
    embedding-less tiles at init, so a frozen tile without the tessera
    key would abort crop-distill-tessera at the pinned-OOF merge."""
    from build_lucas_crop_split import required_npz_keys, tile_qualifies

    keys = required_npz_keys()
    assert "s1_vv_vh" in keys
    assert "tessera" in keys


def test_missing_tessera_tile_does_not_qualify(tmp_path):
    from build_lucas_crop_split import required_npz_keys, tile_qualifies

    keys = required_npz_keys()
    full = tmp_path / "full.npz"
    np.savez(full, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(4),
             tessera=np.zeros(1))
    no_tess = tmp_path / "no_tess.npz"
    np.savez(no_tess, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(4))
    stale_sar = tmp_path / "stale_sar.npz"
    np.savez(stale_sar, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(0),
             tessera=np.zeros(1))

    assert tile_qualifies(full, keys) is True
    assert tile_qualifies(no_tess, keys) is False
    assert tile_qualifies(stale_sar, keys) is False


def _write_lucas_fixture(tmp_path, n_tiles=10, with_tessera=True,
                         forced_tiles=()):
    """A minimal PVC: n_tiles qualified npz tiles + an index parquet with
    all 11 crop classes × 5 points per tile, inside the crop window.
    ``forced_tiles`` get split='test' (the prior 71-point freeze)."""
    import pandas as pd

    data_dir = tmp_path / "tiles"
    data_dir.mkdir(exist_ok=True)
    rows = []
    for t in range(n_tiles):
        name = f"tile{t:02d}"
        kw = {"s1_vv_vh": np.zeros(1), "s1_enrich_v": np.int32(4)}
        if with_tessera or t != 0:  # tile00 optionally lacks the embedding
            kw["tessera"] = np.zeros(1)
        np.savez(data_dir / f"{name}.npz", **kw)
        pid = 0
        for cls in range(11, 22):
            for k in range(5):
                rows.append({
                    "tile_name": name,
                    "tile_path": str(data_dir / f"{name}.npz"),
                    "row": 10 + pid, "col": 10 + pid,
                    "unified_class": cls,
                    "point_id": t * 1000 + pid,
                    "split": "test" if name in forced_tiles else "train",
                })
                pid += 1
    index = tmp_path / "lucas_tile_index.parquet"
    pd.DataFrame(rows).to_parquet(index)
    return index, data_dir


def _run_split_builder(monkeypatch, index, data_dir, out_dir, *extra):
    import build_lucas_crop_split as blcs

    monkeypatch.setattr(sys, "argv", [
        "build_lucas_crop_split.py",
        "--lucas-index", str(index),
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        *extra,
    ])
    try:
        blcs.main()
    except SystemExit as exc:
        # exit 0 is the frozen no-op path; anything else propagates
        if exc.code not in (None, 0):
            raise


def test_frozen_split_is_immutable(monkeypatch, tmp_path, capsys):
    """Blocker 3: a re-run (e.g. the k8s Job re-applied after its TTL
    removed the Job object) must be a no-op — never a re-freeze. Adding
    a source tile before the re-run must not move the holdout."""
    import json as _json

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir,
                       "--git-sha", "a" * 40)
    frozen_index = (out_dir / "lucas_crop_distill_index.parquet").read_bytes()
    frozen_split = (out_dir / "lucas_crop_split.json").read_bytes()
    manifest = _json.loads(
        (out_dir / "lucas_crop_split.MANIFEST.json").read_text())
    assert manifest["git_sha"] == "a" * 40
    assert set(manifest["artifacts"]) == {
        "lucas_crop_distill_index.parquet", "lucas_crop_split.json"}
    assert (out_dir / ".lucas_crop_split.lock").exists() is False

    # a new tile lands on the PVC; the re-run must ignore it entirely
    np.savez(data_dir / "tile99.npz", s1_vv_vh=np.zeros(1),
             s1_enrich_v=np.int32(4), tessera=np.zeros(1))
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert "refusing to overwrite" in capsys.readouterr().out
    assert (out_dir / "lucas_crop_distill_index.parquet").read_bytes() == frozen_index
    assert (out_dir / "lucas_crop_split.json").read_bytes() == frozen_split

    d = _json.loads(frozen_split)
    assert set(d["required_keys"]) >= {"s1_vv_vh", "tessera"}


def test_corrupt_freeze_is_rejected(monkeypatch, tmp_path):
    """A manifest whose artifacts are missing or hash-mismatched is a
    corrupt freeze: hard refusal with recovery — NEVER accepted as frozen
    (the pre-manifest check accepted a truncated parquet)."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split_p = out_dir / "lucas_crop_split.json"
    original = split_p.read_bytes()
    split_p.write_bytes(original[: len(original) // 2])  # truncation
    with pytest.raises(SystemExit, match="CORRUPT"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split_p.write_bytes(original)
    (out_dir / "lucas_crop_distill_index.parquet").unlink()  # missing artifact
    with pytest.raises(SystemExit, match="CORRUPT"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)


def test_partial_freeze_is_rejected(monkeypatch, tmp_path):
    """Artifacts without the commit marker = interrupted freeze — neither
    side trustworthy; refuse with an explicit recovery path, not rebuild."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    (out_dir / "lucas_crop_split.MANIFEST.json").unlink()
    with pytest.raises(SystemExit, match="PARTIAL"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)


def test_hash_refresh_cannot_hide_key_disagreement(monkeypatch, tmp_path):
    """Codex repro: drop one parquet row AND refresh the marker hash —
    byte-integrity then passes while JSON and parquet disagree on which
    plots exist. Semantic validation must catch it: CORRUPT, and
    --verify must exit non-zero on the same state."""
    import hashlib
    import json as _json
    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    idx_p = out_dir / "lucas_crop_distill_index.parquet"
    man_p = out_dir / "lucas_crop_split.MANIFEST.json"
    df = pd.read_parquet(idx_p)
    df.iloc[:-1].to_parquet(idx_p)  # one row gone
    m = _json.loads(man_p.read_text())
    m["artifacts"]["lucas_crop_distill_index.parquet"] = hashlib.sha256(
        idx_p.read_bytes()).hexdigest()  # refreshed marker hash
    man_p.write_text(_json.dumps(m, indent=1))

    with pytest.raises(SystemExit, match="key sets disagree"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    with pytest.raises(SystemExit, match="INVALID"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")


def test_verify_mode_accepts_a_valid_freeze(monkeypatch, tmp_path, capsys):
    """--verify is the consumers' gate (crop-distill runs it before
    extraction): exit 0 + VALID on a healthy freeze, non-zero otherwise."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")
    assert "freeze VALID" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="INVALID"):
        _run_split_builder(monkeypatch, index, data_dir,
                           tmp_path / "empty", "--verify")


@pytest.mark.parametrize("path", [
    "lucas-crop-split-job.yaml", "crop-distill-tessera-job.yaml"],
    ids=lambda v: v)
def test_job_scripts_are_failure_sensitive(path):
    """Blocker 3 + warning: pipefail (a dead sha256sum piped through cut
    yielded an empty digest logged as OK), 64-hex digest guards, and a
    terminal EXIT record installed before the first fallible command."""
    text = _job_text(OUT_DIR / path)
    lines = text.splitlines()
    assert "set -euo pipefail" in text
    assert "' ERR" not in text, "ERR traps miss explicit exit paths"
    assert 'grep -qE "^[0-9a-f]{64}$"' in text
    trap_at = next(i for i, l in enumerate(lines) if l.strip().startswith("trap"))
    clone_at = next(i for i, l in enumerate(lines) if "git clone" in l)
    assert trap_at < clone_at, "EXIT record must observe clone failures"
    assert "run=$RUN_ID" in text


def test_crop_consumer_verifies_the_freeze_before_extraction():
    """Blocker 2: existence tests admit a publish window or crash-left
    pair; the consumer must run the builder's own --verify gate first."""
    text = _job_text(OUT_DIR / "crop-distill-tessera-job.yaml")
    verify_at = text.index("build_lucas_crop_split.py --verify")
    extract_at = text.index("extract_plot_features.py")
    assert verify_at < extract_at


def test_grouped_folds_never_split_a_tile(tmp_path):
    """Round 4: LUCAS crop points cluster ~1.75/tile; point-level folds
    leak same-tile context train→test. Grouped folds must keep every
    group wholly on one side, while still predicting every point once."""
    from nfi_head_cv import make_folds

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 3)          # 40 tiles × 3 points
    y = rng.integers(11, 14, size=len(groups))
    folds = make_folds(y, 5, groups)
    covered = np.zeros(len(y), dtype=int)
    for tr, te in folds:
        covered[te] += 1
        assert not (set(groups[tr]) & set(groups[te])), "tile straddles folds"
    assert (covered == 1).all(), "every point must be OOF-predicted once"


def test_crop_oof_is_group_folded_and_sha_stamped():
    """The manifests must actually request grouped folds + provenance."""
    text = _job_text(_crop_path("tessera"))
    assert "--group-col tile_name" in text
    assert '--git-sha "$HEAD_SHA"' in text


@pytest.mark.parametrize("path", [
    "lucas-crop-split-job.yaml", "crop-distill-tessera-job.yaml"],
    ids=lambda v: v)
def test_scoring_deps_are_pinned_and_snapshotted(path):
    """Round 4: the deps that determine the numbers (RNG, folds, parquet)
    are version-pinned, and every run snapshots pip freeze to the PVC so
    cross-column drift in the unpinned stack is auditable."""
    text = _job_text(OUT_DIR / path)
    assert "numpy==" in text
    assert "pip freeze > /cephfs/ops/deps/" in text
    if path.startswith("crop-distill"):
        assert "scikit-learn==" in text


def test_schema_reduced_parquet_is_corrupt(monkeypatch, tmp_path):
    """Codex round-4 repro: a parquet reduced to the key columns stays
    hash-consistent after a marker refresh but is unconsumable — the
    verifier must require the full extract schema."""
    import hashlib
    import json as _json
    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    idx_p = out_dir / "lucas_crop_distill_index.parquet"
    man_p = out_dir / "lucas_crop_split.MANIFEST.json"
    pd.read_parquet(idx_p)[["tile_name", "point_id"]].to_parquet(idx_p)
    m = _json.loads(man_p.read_text())
    m["artifacts"]["lucas_crop_distill_index.parquet"] = hashlib.sha256(
        idx_p.read_bytes()).hexdigest()
    man_p.write_text(_json.dumps(m, indent=1))

    with pytest.raises(SystemExit, match="lacks extract columns"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")


def test_manifest_split_holdout_count_mismatch_is_corrupt(monkeypatch, tmp_path):
    import hashlib
    import json as _json

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split_p = out_dir / "lucas_crop_split.json"
    man_p = out_dir / "lucas_crop_split.MANIFEST.json"
    s = _json.loads(split_p.read_text())
    s["n_holdout"] += 1
    split_p.write_text(_json.dumps(s, indent=1))
    m = _json.loads(man_p.read_text())
    m["artifacts"]["lucas_crop_split.json"] = hashlib.sha256(
        split_p.read_bytes()).hexdigest()
    man_p.write_text(_json.dumps(m, indent=1))

    with pytest.raises(SystemExit, match="n_holdout"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")


def test_holdout_fraction_counts_forced_tiles(monkeypatch, tmp_path):
    """Round 4: the 30% target is of ALL qualified tiles — computing it on
    the forced-reduced pool undershot by FRAC × n_forced."""
    import json as _json

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=10,
                                           forced_tiles=("tile09",))
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    # 10 qualified tiles → round(3.0) = 3 holdout INCLUDING the forced one.
    assert len(split["holdout_tiles"]) == 3
    assert "tile09" in split["holdout_tiles"]
    assert split["forced_holdout_tiles_from_prior_split"] == ["tile09"]


def test_freeze_lock_excludes_concurrent_builders(monkeypatch, tmp_path):
    """Two racing builders must never both publish: the loser dies on the
    O_EXCL lock, loudly, before any artifact is written."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / ".lucas_crop_split.lock").write_text("{}")  # holder alive

    with pytest.raises(SystemExit, match="freeze lock"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert not (out_dir / "lucas_crop_split.MANIFEST.json").exists()
    assert not (out_dir / "lucas_crop_distill_index.parquet").exists()


def test_unqualified_tile_is_excluded_from_the_freeze(monkeypatch, tmp_path):
    """End-to-end blocker-2 check: a tile missing the tessera embedding
    must appear on NEITHER side of the frozen split."""
    import json as _json
    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path, with_tessera=False)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    dist = pd.read_parquet(out_dir / "lucas_crop_distill_index.parquet")
    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    assert "tile00" not in set(dist["tile_name"])
    assert "tile00" not in set(split["holdout_tiles"])
