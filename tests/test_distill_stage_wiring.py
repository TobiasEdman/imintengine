"""The distill stage must BUILD its backbone at the resolution it will FEED.

``inference_comparison.load_model`` takes an optional ``img_size``. Clay and
CROMA carry no ``pos_embed`` and omit ``img_size`` from their minimal registry
config, so without the argument the backbone is built at 224 — wrong
``grid_size``, wrong PSP pool count — and is then handed 504 px tiles by
``run_inference``. Prithvi recovers its size from ``pos_embed`` and is
unaffected, which is exactly why this survived the Prithvi-only era: both
distill-stage scripts predate the six-backbone ladder.

The consequence is silent, not a crash. ``extract_plot_features.py`` captures
256-dim features off a wrongly-shaped head, and ``distill_forest_labels.py``
writes dense sidecars from one — so 2 of the ladder's 6 columns would distil
garbage while looking healthy. See docs/experiments/ladder_distill_stage.md.

Asserted at the AST level because exercising the real path needs GPU weights.
What regresses in practice is the keyword going missing under an edit, and
that is precisely what this catches.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every script that loads a checkpoint and then runs inference at a chosen
# --img-size. infer_tiles.py is the reference call shape. The two distill
# scripts originally lacked img_size; render_endgame_frames.py had the
# MIRROR bug (backbone_name but no img_size). Review 2026-08-31 raised the
# bar to BOTH kwargs: img_size because clay/croma carry no pos_embed and
# otherwise get built at a 224 grid; backbone_name because
# checkpoint-only resolution silently defaults to prithvi_300m when the
# saved config lacks the field (pre-2026-08-24 trainer) — working by
# accident of checkpoint recency is not a contract.
SCRIPTS_REQUIRING_IMG_SIZE = [
    "extract_plot_features.py",
    "distill_forest_labels.py",
    "infer_tiles.py",
    "render_endgame_frames.py",
]
REQUIRED_LOAD_MODEL_KWARGS = ("img_size", "backbone_name")


def _load_model_calls(path: Path) -> list[ast.Call]:
    """Every ``…load_model(...)`` call in the file, however it is qualified."""
    tree = ast.parse(path.read_text())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "load_model")
            or (isinstance(node.func, ast.Name) and node.func.id == "load_model")
        )
    ]


@pytest.mark.parametrize("script", SCRIPTS_REQUIRING_IMG_SIZE)
def test_load_model_receives_img_size(script: str) -> None:
    path = ROOT / "scripts" / script
    calls = _load_model_calls(path)
    assert calls, f"{script}: no load_model call found — did the script move?"

    for call in calls:
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        for req in REQUIRED_LOAD_MODEL_KWARGS:
            assert req in kwargs, (
                f"{script}:{call.lineno} calls load_model without {req}. "
                f"img_size: clay/croma get built at a 224 grid and fed "
                f"full-size tiles. backbone_name: a checkpoint lacking the "
                f"field silently resolves to prithvi_300m."
            )


@pytest.mark.parametrize("script", ["extract_plot_features.py",
                                    "distill_forest_labels.py"])
def test_img_size_is_a_cli_flag(script: str) -> None:
    """The value forwarded above must be operator-controlled, not hardcoded.

    Each column of the ladder carries its own crop (496 for Prithvi-300M and
    TerraMind, 504 for the rest), so a literal here would silently apply one
    column's regime to another.
    """
    src = (ROOT / "scripts" / script).read_text()
    assert '"--img-size"' in src or "'--img-size'" in src, (
        f"{script}: --img-size is not an argparse flag; the per-column crop "
        f"cannot be honoured."
    )


# ─── The generated distill manifests — protocol pinning ────────────────────
#
# Distillability is the ladder's ONE cross-backbone claim, and it is valid
# only while every column runs the identical protocol. These tests read the
# GENERATED manifests, so they hold whatever the generator does — a tuned
# flag sneaked into one column's yaml fails here even if the generator was
# bypassed by hand-editing.

MODELS = ["prithvi300m", "prithvi600m", "croma", "terramind", "tessera",
          "clay"]
EXPECTED_IMG = {"prithvi300m": 496, "prithvi600m": 504, "croma": 504,
                "terramind": 496, "tessera": 504, "clay": 504}
# Per-column dense-cohort keys: SAR needs s1_vv_vh; tessera needs its
# precomputed embedding (training drops embedding-less tiles at index
# construction — the dense pass must walk the same 7874-tile cohort).
REQUIRED_DENSE_KEYS = {"croma": "s1_vv_vh", "terramind": "s1_vv_vh",
                       "tessera": "tessera"}


def _distill_manifest(model: str) -> str:
    path = ROOT / "k8s" / "ladder" / f"distill-{model}-job.yaml"
    assert path.exists(), f"missing generated manifest {path.name}"
    return path.read_text()


def test_distill_manifests_are_generated_and_current() -> None:
    import subprocess
    res = subprocess.run(
        ["python3", str(ROOT / "scripts" / "gen_ladder_manifests.py"),
         "--check"], capture_output=True, text=True)
    assert res.returncode == 0, f"stale generated manifests:\n{res.stdout}"


@pytest.mark.parametrize("script", ["extract_plot_features.py",
                                    "distill_forest_labels.py",
                                    "infer_tiles.py"])
def test_distill_scripts_read_aux_from_checkpoint(script: str) -> None:
    """Aux channel sets are PER-COLUMN model properties (terramind's r2
    carries 13: the usual 11 plus delta_vv/delta_vh ΔSAR) — rebuilding
    them from CLI flags is how terramind's extract died on a 13-vs-11
    conv mismatch, and a hardcoded None in infer_tiles killed the first
    ladder-eval wave the same way (11-aux tessera vs canonical 10). Every
    GPU script must read enabled_aux_names from the checkpoint's saved
    config; flags/defaults are only a pre-config fallback."""
    src = (ROOT / "scripts" / script).read_text()
    assert "enabled_aux_names" in src, (
        f"{script}: does not read the checkpoint's enabled_aux_names")


@pytest.mark.parametrize("model", MODELS)
def test_distill_gpu_steps_pin_backbone_name(model: str) -> None:
    """Both GPU steps must pass the column's registry backbone; without it
    load_model falls back to the checkpoint config, which defaults to
    prithvi_300m when the field is absent."""
    expected = {
        "prithvi300m": "prithvi_300m", "prithvi600m": "prithvi_600m",
        "croma": "croma_base", "terramind": "terramind_v1_base",
        "tessera": "tessera_v1", "clay": "clay_v1_5",
    }[model]
    text = _distill_manifest(model)
    for step in ("extract_plot_features.py", "distill_forest_labels.py"):
        seg = text.split(step)[1].split("echo")[0]
        assert f"--backbone-name {expected}" in seg, (
            f"{model}: {step} missing --backbone-name {expected}")


@pytest.mark.parametrize("model", MODELS)
def test_distill_img_size_follows_the_column(model: str) -> None:
    text = _distill_manifest(model)
    occurrences = re.findall(r"--img-size (\d+)", text)
    assert occurrences, f"{model}: no --img-size in manifest"
    assert all(int(v) == EXPECTED_IMG[model] for v in occurrences), (
        f"{model}: --img-size {occurrences} != column regime "
        f"{EXPECTED_IMG[model]}")


@pytest.mark.parametrize("model", MODELS)
def test_distill_protocol_is_pinned_and_uniform(model: str) -> None:
    """One head config, one fold count, one seed, one test-frac, the shared
    pinned plot set — across ALL six columns. A per-column deviation turns
    the controlled comparison into a hyperparameter lottery."""
    text = _distill_manifest(model)
    oof = text.split("nfi_head_cv.py")[1].split("echo")[0]
    assert "--folds 5" in oof and "--heads mlp" in oof, (
        f"{model}: OOF protocol drifted")
    assert "--pinned-plots" in oof, (
        f"{model}: distillability not scored on the pinned plot set")
    head = text.split("train_distill_head.py")[1].split("echo")[0]
    assert "--test-frac 0.2" in head and "--seed 42" in head, (
        f"{model}: deployable-head split drifted")


@pytest.mark.parametrize("model", MODELS)
def test_distill_dense_cohort_filter(model: str) -> None:
    """Each column's dense pass must walk exactly the cohort its training
    saw: SAR columns filter on s1_vv_vh, tessera on its embedding key
    (8 embedding-less tiles failed its first dense pass — correctly
    refused by the gate, but for a cohort nobody trains on). Columns
    without a requirement must carry NO filter — it would silently
    shrink their cohort."""
    text = _distill_manifest(model)
    dense = text.split("distill_forest_labels.py")[1]
    want = REQUIRED_DENSE_KEYS.get(model)
    if want:
        assert f"--require-npz-key {want}" in dense, (
            f"{model}: dense step missing --require-npz-key {want}")
    else:
        assert "--require-npz-key" not in dense, (
            f"{model}: unexpected cohort filter")


@pytest.mark.parametrize("model", MODELS)
def test_distill_never_touches_h100_quota(model: str) -> None:
    """The ladder trainings are packed against the H100 memory quota; the
    distill stage runs on the 2080ti pool by design. A manifest drifting to
    the H100 selector would stall rung-2 submissions."""
    text = _distill_manifest(model)
    assert "nvidia-gtx-2080ti" in text, f"{model}: not on the 2080ti pool"
    assert "nvidia-h100" not in text, f"{model}: targets the H100 pool"


def test_distill_outputs_are_model_scoped() -> None:
    """Each column distils ITS OWN rung-2 (user, 2026-08-29); a shared out
    dir would let one column read another's sidecars."""
    seen = set()
    for model in MODELS:
        text = _distill_manifest(model)
        m = re.search(r"OUT=(/cephfs/distill/\S+)", text)
        assert m, f"{model}: no OUT dir"
        assert m.group(1) == f"/cephfs/distill/{model}_r2"
        seen.add(m.group(1))
    assert len(seen) == len(MODELS)


# ─── The pinned plot set — builder + consumer ──────────────────────────────


def _load_script(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"_distill_{name}", str(ROOT / "scripts" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_npz_key_probe_reads_only_the_zip_directory(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    bps = _load_script("build_pinned_plot_set")
    with_sar = tmp_path / "a.npz"
    np.savez(with_sar, spectral=np.zeros(1), s1_vv_vh=np.zeros(1))
    without = tmp_path / "b.npz"
    np.savez(without, spectral=np.zeros(1))
    corrupt = tmp_path / "c.npz"
    corrupt.write_bytes(b"not a zip")
    assert bps.npz_has_keys(with_sar, ("s1_vv_vh",)) is True
    assert bps.npz_has_keys(without, ("s1_vv_vh",)) is False
    # Unreadable must UNDER-claim (excluded), never raise: a pinned plot
    # that later fails extraction aborts that column's distillability run.
    assert bps.npz_has_keys(corrupt, ("s1_vv_vh",)) is False


def test_pinned_subset_is_canonical_and_fails_loud(tmp_path: Path) -> None:
    """The consumer contract: exact match on the pinned set, canonical
    (tile, Tract, Plot) order — identical fold assignment across columns —
    and a hard SystemExit if extraction dropped ANY pinned plot."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    # The subprocess (same interpreter) trains the real MLP head — CI's
    # test env has no sklearn, which made the happy path exit 1 there
    # while passing locally. Skip cleanly where the dep is absent.
    pytest.importorskip("sklearn")
    import json as _json
    import subprocess
    import sys

    # 40 pinned plots, two classes: sized for the REAL pipeline, not just
    # the subset logic. MLPClassifier(early_stopping=True) performs its own
    # stratified 10% validation split inside each CV fold, which needs ~20
    # samples per fold-train — smaller drafts of this test died inside
    # sklearn, not in the code under test.
    pinned = {"required_keys": ["s1_vv_vh"], "plots": [
        {"tile_name": f"t{i % 5}", "TractID": 1 + i // 5, "PlotID": i}
        for i in range(40)
    ]}
    pin_path = tmp_path / "pinned.json"
    pin_path.write_text(_json.dumps(pinned))

    rng = np.random.default_rng(0)
    rows = []
    plots = pinned["plots"] + [
        {"tile_name": "zz", "TractID": 99, "PlotID": i} for i in range(9)]
    for i, p in enumerate(plots):
        r = {**p, "nfi_forest": i % 2}
        r.update({f"f{k:03d}": float(v)
                  for k, v in enumerate(rng.normal(size=256))})
        rows.append(r)
    feats = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(feats)

    def run(features):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "nfi_head_cv.py"),
             "--features", str(features), "--folds", "2", "--heads", "mlp",
             "--pinned-plots", str(pin_path),
             "--out", str(tmp_path / "out.json")],
            capture_output=True, text=True)

    res = run(feats)
    assert res.returncode == 0, res.stderr
    out = _json.loads((tmp_path / "out.json").read_text())
    assert out["_meta"]["n_plots"] == 40         # extras excluded
    assert out["_meta"]["pinned_plots"]["n_plots"] == 40

    short = tmp_path / "short.parquet"
    pd.DataFrame(rows[1:]).to_parquet(short)      # drop one pinned plot
    res = run(short)
    assert res.returncode != 0, "dropped pinned plot must be fatal"
    assert "pinned" in res.stderr


def test_dense_pass_has_a_cohort_gate() -> None:
    """distill_forest_labels must refuse to finish green with uncovered
    cohort tiles — rungs 3/4 read whatever is in the sidecar dir, so a
    short set silently shrinks the rung-3 training set."""
    src = (ROOT / "scripts" / "distill_forest_labels.py").read_text()
    assert "sidecar gate" in src and "SystemExit" in src


def test_sidecar_provenance_detects_stale_head(tmp_path: Path) -> None:
    """Resume-after-abort with a retrained head must REWRITE, not skip.

    The exists-check originally keyed on file presence alone, so a rerun
    counted an earlier head's sidecars as done and rungs 3/4 trained on a
    silent mix of two heads' labels. sidecar_is_current is the fix: only
    a matching head_sha stamp counts as done; no stamp (pre-provenance
    file) or a different stamp is stale.
    """
    np = pytest.importorskip("numpy")
    dfl = _load_script("distill_forest_labels")

    current = tmp_path / "current.npz"
    np.savez(current, label=np.zeros((2, 2)), head_sha=np.str_("abc123"))
    stale = tmp_path / "stale.npz"
    np.savez(stale, label=np.zeros((2, 2)), head_sha=np.str_("OLDHEAD"))
    unstamped = tmp_path / "unstamped.npz"
    np.savez(unstamped, label=np.zeros((2, 2)))
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a zip")

    assert dfl.sidecar_is_current(str(current), "abc123") is True
    assert dfl.sidecar_is_current(str(stale), "abc123") is False
    assert dfl.sidecar_is_current(str(unstamped), "abc123") is False
    assert dfl.sidecar_is_current(str(corrupt), "abc123") is False


def test_pinned_builder_fails_on_unreadable_tile(tmp_path: Path) -> None:
    """An unreadable tile is a PVC problem, not a cohort property.

    Folding it into the SAR-less remainder would shrink the pinned set
    silently — six columns would then agree on a distillability number
    computed over a degraded population. The builder must abort instead.
    """
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    import subprocess
    import sys

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    np.savez(tiles / "t1.npz", spectral=np.zeros(1), s1_vv_vh=np.zeros(1),
             s1_enrich_v=np.int32(4))
    (tiles / "t2.npz").write_bytes(b"corrupt")

    idx = tmp_path / "index.parquet"
    pd.DataFrame({
        "tile_name": ["t1", "t2"], "TractID": [1, 1], "PlotID": [1, 2],
        "row": [100, 100], "col": [100, 100],   # in-window; crop filter
    }).to_parquet(idx)                          # must not mask the abort


    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_pinned_plot_set.py"),
         "--plot-index", str(idx), "--data-dir", str(tiles),
         "--out", str(tmp_path / "pinned.json")],
        capture_output=True, text=True)
    assert res.returncode != 0, "unreadable tile must abort the builder"
    assert "unreadable" in (res.stdout + res.stderr).lower()
    assert not (tmp_path / "pinned.json").exists(), (
        "no pinned set may be written on a degraded PVC")


@pytest.mark.parametrize("model", MODELS)
def test_distill_mounts_both_pvc_paths(model: str) -> None:
    """The NFI plot index stores absolute /data/ tile paths (written by
    the nfi-* jobs, which mount the PVC there); the ladder convention is
    /cephfs. A distill pod with only one mount drops every plot as "no
    longer in the dataset" — the first submission died exactly so."""
    text = _distill_manifest(model)
    assert "mountPath: /cephfs" in text and "mountPath: /data" in text, (
        f"{model}: distill job must mount the PVC at BOTH /cephfs and /data")


def test_pinned_builder_respects_crop_window(tmp_path: Path) -> None:
    """A pinned plot must survive EVERY column's border crop.

    Each column's extract crops 512 -> its img_size and drops border
    plots; the crop differs per column (496 vs 504), so without this
    filter the pinned set contains plots some columns can never score.
    First cluster submission failed exactly so: 600m covered 935/964
    pinned plots, 300m 904/964, and the exact-match guard refused both.
    The window is [8, 504) — from the tightest img_size (496) in the
    generator's DISTILL table.
    """
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    import json as _json
    import subprocess
    import sys

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    np.savez(tiles / "t1.npz", spectral=np.zeros(1), s1_vv_vh=np.zeros(1),
             s1_enrich_v=np.int32(4))   # version gate: unstamped = excluded

    idx = tmp_path / "index.parquet"
    pd.DataFrame({
        "tile_name": ["t1"] * 4, "TractID": [1] * 4,
        "PlotID": [1, 2, 3, 4],
        # in-window / left-border / bottom-border / exactly-on-edge
        "row": [100, 3, 100, 8], "col": [100, 100, 505, 503],
    }).to_parquet(idx)

    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_pinned_plot_set.py"),
         "--plot-index", str(idx), "--data-dir", str(tiles),
         "--out", str(tmp_path / "pinned.json")],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    got = _json.loads((tmp_path / "pinned.json").read_text())
    kept = {p["PlotID"] for p in got["plots"]}
    assert kept == {1, 4}, (
        f"expected only in-window plots (1) and edge-inclusive (4), "
        f"got {kept} — window must be [8, 504) on both axes")


def test_crop_offset_parity() -> None:
    """The builder inlines crop arithmetic (its CPU pod cannot afford
    validate_against_nfi's import chain). The two implementations must
    never diverge — a drifted offset shifts the pinned window and every
    column silently scores a different plot set."""
    bps = _load_script("build_pinned_plot_set")
    van = _load_script("validate_against_nfi")
    for img in (224, 496, 504, 512, 600):
        assert bps._crop_offset(512, img) == van.crop_offset(512, img), (
            f"crop offset diverged at img_size={img}")


class _Conv:  # noqa: D401 — minimal stand-ins; torch optional in CI
    pass


def test_find_classifier_across_families() -> None:
    """The hook must locate the class-projection conv on every family's
    structure — Prithvi's head.head[2], tessera's self.classifier,
    the UPerNet wrappers' decoder_head.head.head[2] — and refuse to
    guess between ambiguous 1x1 convs (frac/binary heads are 1x1 too;
    hooking the wrong one silently exports the wrong feature). First
    cluster submission of tessera died exactly on the hardcoded path."""
    torch = pytest.importorskip("torch")
    nn = torch.nn
    epf = _load_script("extract_plot_features")

    class Tessera(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Conv2d(128, 28, 1)
            self.frac_head = nn.Conv2d(128, 4, 1)

    class Prithvi(nn.Module):
        def __init__(self):
            super().__init__()
            inner = nn.Sequential(nn.Identity(), nn.Dropout2d(0.1),
                                  nn.Conv2d(256, 28, 1))
            self.head = nn.Module()
            self.head.head = inner

    class UPerWrapped(nn.Module):
        def __init__(self):
            super().__init__()
            seg = nn.Module()
            seg.head = nn.Sequential(nn.Identity(), nn.Dropout2d(0.1),
                                     nn.Conv2d(256, 28, 1))
            self.decoder_head = nn.Module()
            self.decoder_head.head = seg

    assert epf.find_classifier(Tessera()).in_channels == 128
    assert epf.find_classifier(Prithvi()).in_channels == 256
    assert epf.find_classifier(UPerWrapped()).in_channels == 256

    class Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.blob = nn.Sequential(nn.Conv2d(64, 28, 1), nn.Conv2d(64, 4, 1))

    # unique-max fallback: 28 dwarfs 4 → found structurally
    assert epf.find_classifier(Unknown()).out_channels == 28

    class Ambiguous(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv2d(64, 28, 1)
            self.b = nn.Conv2d(96, 28, 1)

    with pytest.raises(TypeError, match="cannot locate"):
        epf.find_classifier(Ambiguous())


@pytest.mark.parametrize("model,needle", [
    ("terramind", "terratorch"),
    ("clay", "Clay-foundation/model"),
    ("croma", "antofuller/CROMA"),
])
def test_distill_installs_backbone_deps(model: str, needle: str) -> None:
    """The r2 checkpoints cannot even LOAD without their loader packages —
    terramind's first submission died on ImportError: terratorch. The
    training manifests carry these installs; the distill jobs must too."""
    assert needle in _distill_manifest(model), (
        f"{model}: distill manifest missing backbone dep '{needle}'")


def test_version_gate_excludes_stale_enrichment(tmp_path: Path) -> None:
    """Key PRESENCE is not qualification: the dataset gates SAR on
    s1_enrich_v == 4, so a tile with old-version s1_vv_vh was never seen
    by SAR training and kills extract/dense at forward time — terramind's
    third submission died on exactly one such tile. The pinned builder
    and the dense filter must both apply the version requirement."""
    np = pytest.importorskip("numpy")
    bps = _load_script("build_pinned_plot_set")

    ok = tmp_path / "ok.npz"
    np.savez(ok, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(4))
    stale = tmp_path / "stale.npz"
    np.savez(stale, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(0))
    unstamped = tmp_path / "unstamped.npz"
    np.savez(unstamped, s1_vv_vh=np.zeros(1))

    assert bps.npz_version_ok(ok, ("s1_vv_vh",)) is True
    assert bps.npz_version_ok(stale, ("s1_vv_vh",)) is False
    assert bps.npz_version_ok(unstamped, ("s1_vv_vh",)) is False
    # keys without a version requirement pass vacuously
    assert bps.npz_version_ok(stale, ("spectral",)) is True
