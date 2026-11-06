import json
from pathlib import Path

import numpy as np
import pytest

from scripts.probe_forest_separability import (
    CLASS_SCHEMES, ROLE, class_labels, digest, metrics, run_probe, sample_pixels, spatial_folds,
)
from scripts.extract_forest_separability import ensure_development_roots, tile_candidates
from scripts.forest_separability_schemes import DEFAULT_CHECKPOINT


def test_raw_codes_cannot_be_confused_with_reduced_training_codes():
    assert class_labels(np.array([113, 115, 116, 117], np.uint16), "binary").tolist() == [0, 1, 1, 1]
    for raw in (np.array([3, 4]), np.array([113.0]), np.array([114]), np.array([[113]])):
        with pytest.raises(ValueError):
            class_labels(raw, "binary")


def test_forest5_keeps_every_dryland_forest_type_apart():
    raw = np.array([111, 112, 113, 114, 115, 116, 117], np.uint16)
    assert class_labels(raw, "forest5").tolist() == [0, 1, 2, 3, 4, 4, 4]
    for outside in (118, 121, 123, 43):  # temporarily non-forest, wetland, mountain birch
        with pytest.raises(ValueError, match="outside"):
            class_labels(np.array([outside], np.uint16), "forest5")
    with pytest.raises(ValueError, match="unknown class scheme"):
        class_labels(raw, "forest4")


def test_sumpskog5_splits_wetland_forest_by_type_only():
    raw = np.array([121, 122, 123, 124, 125, 126, 127], np.uint16)
    assert class_labels(raw, "sumpskog5").tolist() == [0, 1, 2, 3, 4, 4, 4]
    for outside in (111, 113, 128, 23):  # dryland, temporarily non-forest wetland, low mountain
        with pytest.raises(ValueError, match="outside"):
            class_labels(np.array([outside], np.uint16), "sumpskog5")


def test_sampling_uses_valid_interiors_and_caps_classes():
    raw = np.full((140, 140), 113, dtype=np.uint16)
    raw[:, 70:] = 115
    valid = np.ones_like(raw, dtype=bool)
    valid[30:40] = False
    r, c = sample_pixels(raw, valid, per_class=20, seed=42, scheme="binary")
    assert len(r) == 40
    assert valid[r, c].all()
    assert np.all(r % 10 == 0) and np.all(c % 10 == 0)
    assert not np.any(c == 70)  # boundary excluded
    assert np.bincount(class_labels(raw[r, c], "binary")).tolist() == [20, 20]
    with pytest.raises(ValueError):
        sample_pixels(raw, np.zeros_like(valid), per_class=20, seed=42, scheme="binary")


def test_forest5_sampling_requires_and_caps_all_five_classes():
    raw = np.zeros((200, 200), dtype=np.uint16)
    for k, code in enumerate((111, 112, 113, 114, 116)):
        raw[:, k * 40:(k + 1) * 40] = code
    valid = np.ones_like(raw, dtype=bool)
    r, c = sample_pixels(raw, valid, per_class=8, seed=42, scheme="forest5")
    assert np.bincount(class_labels(raw[r, c], "forest5")).tolist() == [8] * 5
    raw[raw == 114] = 113  # a tile without lövblandad barrskog cannot qualify
    with pytest.raises(ValueError, match="fewer than eight"):
        sample_pixels(raw, valid, per_class=8, seed=42, scheme="forest5")


def test_spatial_split_buffers_group_borders_and_scores_each_point_once():
    from scipy.spatial import cKDTree
    xy = np.array([[x, y] for x in (100, 9900, 10100, 19900, 20100, 29900, 40000)
                   for y in (10, 1000)], dtype=float)
    folds = spatial_folds(xy, n_splits=3, block_m=10000, buffer_m=500)
    scored = []
    for train, test in folds:
        assert cKDTree(xy[test]).query(xy[train])[0].min() >= 500
        train_blocks = {tuple(v) for v in np.floor(xy[train] / 10000)}
        test_blocks = {tuple(v) for v in np.floor(xy[test] / 10000)}
        assert not train_blocks & test_blocks
        scored.extend(test.tolist())
    assert sorted(scored) == list(range(len(xy)))
    with pytest.raises(ValueError, match="duplicate"):
        spatial_folds(np.vstack((xy, xy[:1])), n_splits=3, block_m=10000, buffer_m=500)


def make_bundle(tmp_path, scheme="binary", *, model="r4", num_classes=28, prediction=None):
    rng = np.random.default_rng(8)
    k = len(CLASS_SCHEMES[scheme])
    per_group = 12 * k
    y = np.tile(np.repeat(np.arange(k), 12), 8)
    xy = np.array([[400000 + g * 100000 + i * 100, 6500000 + (i % 3) * 100]
                   for g in range(8) for i in range(per_group)], dtype=float)
    signal = (np.eye(k)[y] * 5 + rng.normal(0, 0.1, (len(y), k))).astype(np.float32)
    first_code = np.array([codes[0] for _, codes in CLASS_SCHEMES[scheme]], np.uint16)
    p = tmp_path / "features.npz"
    np.savez(p, raw_nmd=first_code[y], xy=xy, tessera=signal, **{model: signal * 2},
             tile=np.repeat(np.arange(8), per_group),
             base_label=np.full(len(y), 3, np.uint8), distilled_label=(y % 4 + 1).astype(np.uint8),
             **{f"{model}_prediction": (y % 4 + 1 if prediction is None else prediction(y))
                .astype(np.int64)})
    p.with_suffix('.json').write_text(json.dumps({
        'role': ROLE, 'scheme': scheme, 'model_name': model, 'num_classes': num_classes,
        'checkpoint': {'path': (DEFAULT_CHECKPOINT if model == 'r4'
                                else f'/data/checkpoints/im023/{model}/best_model.pt'),
                       'size': 1, 'sha256': 'c' * 64},
        'feature_year': 2022, 'reference_year': 2023, 'bundle_sha256': digest(p)}))
    return p


def test_real_disk_probe_separable_signal_and_majority_baseline(tmp_path):
    path = make_bundle(tmp_path)
    report = run_probe(path, scheme="binary", model_name="r4", n_splits=4)
    assert report['feature_year'] == 2022 and report['reference_year_gap'] == 1
    assert report['n_samples'] == 192 and report['n_tiles'] == 8
    for model in report['models'].values():
        assert model['balanced_accuracy'] == 1
        assert model['per_class']['barrblandskog']['iou'] == 1
        assert model['balanced_accuracy_95pct_block_bootstrap'] == [1, 1]
    assert report['majority_baseline']['balanced_accuracy'] == .5
    json.dumps(report, allow_nan=False)


def test_forest5_probe_reports_full_confusion_and_assigned_classes(tmp_path):
    path = make_bundle(tmp_path, "forest5")
    report = run_probe(path, scheme="forest5", model_name="r4", n_splits=4)
    assert report['scheme'] == 'forest5' and report['n_samples'] == 480
    for model in report['models'].values():
        assert model['balanced_accuracy'] == 1
        assert model['confusion_matrix'] == (np.eye(5, dtype=int) * 96).tolist()
        assert model['balanced_accuracy_95pct_block_bootstrap'] == [1, 1]
    assert report['majority_baseline']['balanced_accuracy'] == pytest.approx(.2)
    assigned = report['assigned_unified_class']
    assert assigned['distilled_label']['barrblandskog'] == {'3': 96}
    assert assigned['distilled_label']['lövskog (115–117)'] == {'1': 96}
    assert assigned['base_label']['tallskog'] == {'3': 96}
    json.dumps(report, allow_nan=False)


def test_receipt_and_role_fail_closed(tmp_path):
    path = make_bundle(tmp_path)
    receipt = path.with_suffix('.json')
    receipt.write_text(json.dumps({'role': 'sealed_holdout', 'bundle_sha256': digest(path)}))
    with pytest.raises(ValueError, match='development'):
        run_probe(path, scheme="binary", model_name="r4")
    receipt.write_text(json.dumps({'role': ROLE, 'scheme': 'binary', 'feature_year': 2022,
                                   'reference_year': 2023, 'bundle_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='receipt'):
        run_probe(path, scheme="binary", model_name="r4")


def test_probe_scheme_must_match_extraction_receipt(tmp_path):
    path = make_bundle(tmp_path)
    with pytest.raises(ValueError, match="scheme 'binary' is not 'forest5'"):
        run_probe(path, scheme="forest5", model_name="r4")
    receipt = json.loads(path.with_suffix('.json').read_text())
    del receipt['scheme']
    path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="scheme None"):
        run_probe(path, scheme="binary", model_name="r4")


def test_root_guard_rejects_evaluation_and_symlinks(tmp_path):
    roots = [tmp_path / n for n in ('unified_v2_512', 'nmd2023_labels', 'tessera_r2')]
    for p in roots:
        p.mkdir()
    ensure_development_roots(*roots)
    with pytest.raises(ValueError):
        ensure_development_roots(tmp_path / 'holdout', *roots[1:])
    link = tmp_path / 'other' / 'unified_v2_512'
    link.parent.mkdir()
    link.symlink_to(roots[0])
    with pytest.raises(ValueError):
        ensure_development_roots(link, *roots[1:])


def test_candidate_year_and_geography_are_selected_before_scores(tmp_path):
    data, labels = tmp_path / 'data', tmp_path / 'labels'
    data.mkdir(); labels.mkdir()
    raw = np.full((512, 512), 113, np.uint16); raw[:, 256:] = 115
    for i in range(8):
        name = f'tile{i}.npz'
        west = 400000 + i * 100000
        np.savez(data / name, tessera=np.empty(0), tessera_year=2022 if i == 0 else 2023,
                 bbox_3006=np.array([west, 6500000, west+5120, 6505120]))
        np.savez(labels / name, nmd_label_raw=raw)
    audit = tmp_path / 'availability.json'
    selection = tile_candidates(data, labels, max_tiles=5, year=2023, scheme="binary",
                                audit_path=audit)
    assert json.loads(audit.read_text())['selected'] == 5
    assert len(selection) == 5 and all('tile0' not in e['path'] for e in selection)
    assert selection == tile_candidates(data, labels, max_tiles=5, year=2023, scheme="binary")
    with pytest.raises(ValueError, match='only 0'):  # no 111/112/114 in these tiles
        tile_candidates(data, labels, max_tiles=5, year=2023, scheme="forest5")


def test_metrics_missing_predictions_have_defined_recall_and_iou():
    m = metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), "binary")
    assert m['balanced_accuracy'] == .5
    assert m['per_class']['lövskog (115–117)']['precision'] is None
    assert m['per_class']['lövskog (115–117)']['recall'] == 0
    assert m['per_class']['barrblandskog']['iou'] == .5


def test_sidecar_replacement_after_inventory_is_rejected(tmp_path):
    from scripts.cache_predict import file_identity
    from scripts.extract_forest_separability import read_sidecar
    from scripts.inference_comparison import TileIdentityError

    path = tmp_path / "sidecar.npz"
    raw = np.array([[113, 115]], dtype=np.uint16)
    np.savez(path, label=np.array([[3, 3]], np.uint8), nmd_label_raw=raw)
    identity = file_identity(path)
    assert np.array_equal(read_sidecar(path, identity)["nmd_label_raw"], raw)
    np.savez(path, label=np.array([[3, 3]], np.uint8),
             nmd_label_raw=np.array([[115, 113]], np.uint16))
    with pytest.raises(TileIdentityError, match="sha256 mismatch"):
        read_sidecar(path, identity)


def test_unavailable_year_preserves_inventory_before_failure(tmp_path):
    data, labels = tmp_path / 'data', tmp_path / 'labels'
    data.mkdir(); labels.mkdir()
    np.savez(data / 'tile.npz', tessera=np.empty(0), tessera_year=2022)
    np.savez(labels / 'tile.npz', label=np.empty(0))
    audit = tmp_path / 'availability.json'
    with pytest.raises(ValueError, match='only 0'):
        tile_candidates(data, labels, max_tiles=5, year=2023, scheme="binary", audit_path=audit)
    assert json.loads(audit.read_text()) == {
        'scanned': 1, 'total': 1, 'tessera_year_counts': {'2022': 1},
        'selected': 0, 'requested_year': 2023, 'complete': True}


def test_authenticated_year_and_geometry_must_match_selection():
    from scripts.extract_forest_separability import validate_selected_metadata

    entry = {'year': 2022, 'bbox_3006': [400000, 6500000, 405120, 6505120]}
    data = {'tessera_year': np.array(2022), 'bbox_3006': np.array(entry['bbox_3006'])}
    validate_selected_metadata(data, entry)
    with pytest.raises(ValueError, match='year'):
        validate_selected_metadata({**data, 'tessera_year': np.array(2018)}, entry)
    with pytest.raises(ValueError, match='geometry'):
        validate_selected_metadata({**data, 'bbox_3006': data['bbox_3006'] + 100000}, entry)


def test_rendered_job_pins_scheme_year_and_source(tmp_path):
    import base64, hashlib, io, tarfile
    from scripts.render_forest_separability_job import render

    render(tmp_path / 'f5', scheme='forest5', feature_year=2023)
    items = json.loads((tmp_path / 'f5' / 'job.json').read_text())['items']
    configmap, job = items
    name = job['metadata']['name']
    assert name.startswith('forest-separability-0127-forest5-2023-') and len(name) <= 63
    env = {e['name']: e['value'] for e in job['spec']['template']['spec']['containers'][0]['env']}
    assert env['PROBE_SCHEME'] == 'forest5' and env['PROBE_FEATURE_YEAR'] == '2023'
    archive = base64.b64decode(configmap['binaryData']['source.tar.gz'])
    assert hashlib.sha256(archive).hexdigest() == env['SOURCE_SHA256']
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        manifest = json.loads(tar.extractfile('SOURCE_MANIFEST.json').read())
    assert manifest['scheme'] == 'forest5' and manifest['feature_year'] == 2023
    assert isinstance(manifest['uncommitted_source'], bool)
    assert 'scripts/probe_forest_separability.py' in manifest['files']
    assert 'scripts/forest_separability_schemes.py' in manifest['files']
    for bad in ({'scheme': 'forest4', 'feature_year': 2022}, {'scheme': 'binary', 'feature_year': 2024}):
        with pytest.raises(ValueError, match='pre-registered'):
            render(tmp_path / 'bad', **bad)


def test_renderer_runs_without_numpy_on_the_operator_machine():
    import subprocess, sys
    renderer = Path(__file__).resolve().parents[1] / 'scripts/render_forest_separability_job.py'
    blocked = ("import sys, runpy; sys.modules['numpy'] = None; sys.argv = ['render', '--help']; "
               f"runpy.run_path({str(renderer)!r}, run_name='__main__')")
    result = subprocess.run([sys.executable, '-I', '-c', blocked], capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--scheme' in result.stdout


def test_sumpskog5_probe_runs_end_to_end_on_disk(tmp_path):
    report = run_probe(make_bundle(tmp_path, "sumpskog5"), scheme="sumpskog5", model_name="r4",
                       n_splits=4)
    assert report['scheme'] == 'sumpskog5' and 'dryland' not in report['reference']
    assert list(report['models']['r4']['per_class'])[2] == 'barrblandskog på våtmark'
    assert report['models']['tessera']['balanced_accuracy'] == 1


# ── the frozen model as a parameter (IM-023 29-class model) ─────────────────


def _main_prediction(y):
    """forest5: half of barrblandskog (index 2) called 28, the rest 3; others 1."""
    half = np.arange(len(y)) % 2 == 0
    return np.where(y == 2, np.where(half, 28, 3), 1)


def test_29_class_probe_reports_the_share_called_barrblandskog(tmp_path):
    path = make_bundle(tmp_path, "forest5", model="main", num_classes=29,
                       prediction=_main_prediction)
    report = run_probe(path, scheme="forest5", model_name="main", n_splits=4)
    assert set(report["models"]) == {"tessera", "main"}
    assert report["frozen_model"] == {"name": "main", "num_classes": 29,
                                      "checkpoint_sha256": "c" * 64}
    assert report["assigned_unified_class"]["main_prediction"]["barrblandskog"] == {
        "28": 48, "3": 48}
    share = report["predicted_class_28_share"]
    assert share["barrblandskog"] == 0.5
    assert share["lövskog (115–117)"] == 0.0 and share["tallskog"] == 0.0
    json.dumps(report, allow_nan=False)


def test_28_class_probe_reports_no_class_28_share(tmp_path):
    report = run_probe(make_bundle(tmp_path, "forest5"), scheme="forest5", model_name="r4",
                       n_splits=4)
    assert report["frozen_model"]["name"] == "r4"
    assert report["predicted_class_28_share"] is None
    assert "r4_prediction" in report["assigned_unified_class"]


def test_probe_model_must_match_the_extraction_receipt(tmp_path):
    path = make_bundle(tmp_path, "forest5", model="main", num_classes=29)
    with pytest.raises(ValueError, match="model 'main' is not 'r4'"):
        run_probe(path, scheme="forest5", model_name="r4")
    receipt = json.loads(path.with_suffix(".json").read_text())
    receipt["num_classes"] = 27
    path.with_suffix(".json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="num_classes"):
        run_probe(path, scheme="forest5", model_name="main")


def test_probe_cli_takes_the_model_name(tmp_path, monkeypatch):
    import sys
    from scripts import probe_forest_separability as probe
    path = make_bundle(tmp_path, "forest5", model="main", num_classes=29,
                       prediction=_main_prediction)
    out = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--features", str(path), "--output", str(out),
                                      "--scheme", "forest5", "--model-name", "main", "--folds", "4"])
    probe.main()
    assert json.loads(out.read_text())["predicted_class_28_share"]["barrblandskog"] == 0.5


MAIN_CHECKPOINT = '/data/checkpoints/im023/tessera_b28/best_model.pt'


@pytest.mark.parametrize("model_classes,requested,native,embed,ok", [
    (28, 28, 504, 128, True), (29, 29, 504, 128, True),
    (28, 29, 504, 128, False), (29, 28, 504, 128, False),
    (29, 29, 512, 128, False), (29, 29, 504, 256, False)])
def test_extract_accepts_28_and_29_class_heads_only(model_classes, requested, native, embed, ok):
    """Other class counts never get this far: check_model refuses them first."""
    from types import SimpleNamespace
    from scripts.extract_forest_separability import validate_architecture
    model = SimpleNamespace(num_classes=model_classes, embed_dim=embed)
    if ok:
        validate_architecture(model, native, requested)
    else:
        with pytest.raises(ValueError):
            validate_architecture(model, native, requested)


@pytest.mark.parametrize("name,checkpoint,num_classes,match", [
    ("main", MAIN_CHECKPOINT, 27, "num_classes"),
    ("r4", MAIN_CHECKPOINT, 28, "is R4"),
    ("main", "/data/checkpoints/ladder/tessera_r4/best_model.pt", 29, "is R4")])
def test_extract_refuses_a_misnamed_model_before_reading_anything(tmp_path, name, checkpoint,
                                                                   num_classes, match):
    from argparse import Namespace
    from scripts import extract_forest_separability as ex
    args = Namespace(model_name=name, checkpoint=Path(checkpoint), num_classes=num_classes,
                     data_dir=tmp_path / 'absent', label_dir=tmp_path, distill_dir=tmp_path,
                     output=tmp_path / 'out', per_class=128, max_tiles=24,
                     feature_year=2022, scheme='forest5')
    with pytest.raises(ValueError, match=match):
        ex.extract(args)
    assert not (tmp_path / 'out').exists()


def test_extract_cli_requires_the_model_arguments(monkeypatch):
    import sys
    from scripts import extract_forest_separability as ex
    seen = []
    monkeypatch.setattr(ex, "extract", seen.append)
    base = ["extract", "--data-dir", "d", "--label-dir", "l", "--distill-dir", "t",
            "--checkpoint", "c.pt", "--output", "o", "--feature-year", "2022",
            "--scheme", "forest5"]
    for missing in ([], ["--model-name", "main"], ["--num-classes", "29"]):
        monkeypatch.setattr(sys, "argv", base + missing)
        with pytest.raises(SystemExit):
            ex.main()
    monkeypatch.setattr(sys, "argv", base + ["--model-name", "main", "--num-classes", "29"])
    ex.main()
    assert (seen[-1].model_name, seen[-1].num_classes) == ("main", 29)
    monkeypatch.setattr(sys, "argv", base + ["--model-name", "main", "--num-classes", "30"])
    with pytest.raises(SystemExit):
        ex.main()


def test_class_constants_match_the_unified_schema():
    """The stdlib-only module mirrors the schema it cannot import without numpy."""
    from imint.schema.unified_schema import (
        NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES, UNIFIED_CLASS_NAMES)
    from scripts.forest_separability_schemes import (
        BARRBLANDSKOG_CLASS, DEFAULT_NUM_CLASSES, NUM_CLASSES_CHOICES)
    assert NUM_CLASSES_CHOICES == (NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES)
    assert DEFAULT_NUM_CLASSES == NUM_CLASSES_BEFORE_BARRBLANDSKOG
    assert UNIFIED_CLASS_NAMES[BARRBLANDSKOG_CLASS] == "barrblandskog"


def test_model_names_and_evidence_tags():
    from scripts.forest_separability_schemes import (
        BUNDLE_KEYS, DEFAULT_CHECKPOINT, check_model, run_tag)
    check_model("main", MAIN_CHECKPOINT, 29)
    check_model("control", '/data/checkpoints/im023/tessera_control/best_model.pt', 28)
    check_model("r4", DEFAULT_CHECKPOINT, 28)
    check_model("r4", "/data/checkpoints/ladder//tessera_r4/./best_model.pt", 28)
    for bad in ("Main", "r4-x", "a" * 13, "", "4r", *BUNDLE_KEYS):
        with pytest.raises(ValueError, match="model name"):
            check_model(bad, MAIN_CHECKPOINT, 29)
    assert {"tile", "xy", "tessera"} <= set(BUNDLE_KEYS)
    with pytest.raises(ValueError, match="num_classes"):
        check_model("main", MAIN_CHECKPOINT, 30)
    for name, checkpoint, classes in (("r4", MAIN_CHECKPOINT, 28),          # R4's name, other weights
                                      ("r4", DEFAULT_CHECKPOINT, 29),        # R4 with 29 classes
                                      ("main", DEFAULT_CHECKPOINT, 28),      # R4's weights renamed
                                      ("main", "/data/checkpoints/ladder//tessera_r4/best_model.pt", 28)):
        with pytest.raises(ValueError, match="is R4"):
            check_model(name, checkpoint, classes)
    assert run_tag("r4", "forest5", 2022) == "forest5-2022"     # R4's existing evidence
    assert run_tag("main", "forest5", 2022) == "main-forest5-2022"


def test_model_arguments_are_all_given_or_none():
    from scripts.forest_separability_schemes import DEFAULT_CHECKPOINT, resolve_model
    assert resolve_model(None, None, None) == ("r4", DEFAULT_CHECKPOINT, 28)
    assert resolve_model("main", MAIN_CHECKPOINT, 29) == ("main", MAIN_CHECKPOINT, 29)
    for partial in (("main", None, None), (None, MAIN_CHECKPOINT, None), (None, None, 29),
                    ("main", MAIN_CHECKPOINT, None)):
        with pytest.raises(ValueError, match="go together"):
            resolve_model(*partial)


def test_schemes_cli_validates_a_run_and_prints_its_tags(capsys):
    from scripts.forest_separability_schemes import main
    main(["--scheme", "forest5", "--", "2022", "2023"])
    assert capsys.readouterr().out.split() == ["forest5-2022", "forest5-2023"]
    main(["--model-name", "main", "--checkpoint", MAIN_CHECKPOINT, "--num-classes", "29",
          "--scheme", "forest5", "--", "2022"])
    assert capsys.readouterr().out.split() == ["main-forest5-2022"]
    for bad in (["--scheme", "forest5", "--", "2022", "--model-name", "main"],  # trailing flag
                ["--scheme", "forest5", "--", "2022", "2024"],
                ["--scheme", "forest5", "--", "2022", "2022"],
                ["--scheme", "forest4", "--", "2022"],
                ["--model-name", "main", "--scheme", "forest5", "--", "2022"]):
        with pytest.raises(SystemExit) as exc:
            main(bad)
        assert exc.value.code == 2


def _rendered(output):
    items = json.loads((output / 'job.json').read_text())['items']
    configmap, job = items
    env = {e['name']: e['value'] for e in job['spec']['template']['spec']['containers'][0]['env']}
    return configmap, job, env


def test_rendered_job_defaults_to_r4(tmp_path):
    from scripts.render_forest_separability_job import render
    render(tmp_path / 'r4', scheme='forest5', feature_year=2022)
    _, job, env = _rendered(tmp_path / 'r4')
    assert job['metadata']['name'].startswith('forest-separability-0127-forest5-2022-')
    assert (env['PROBE_MODEL_NAME'], env['PROBE_CHECKPOINT'], env['PROBE_NUM_CLASSES']) == (
        'r4', '/data/checkpoints/ladder/tessera_r4/best_model.pt', '28')
    assert json.loads((tmp_path / 'r4' / 'identity.json').read_text())['tag'] == 'forest5-2022'


def test_rendered_job_for_the_29_class_model(tmp_path):
    import base64, io, tarfile
    from scripts.render_forest_separability_job import BOOTSTRAP, render
    render(tmp_path / 'main', scheme='forest5', feature_year=2022, model_name='main',
           checkpoint=MAIN_CHECKPOINT, num_classes=29)
    configmap, job, env = _rendered(tmp_path / 'main')
    name = job['metadata']['name']
    assert name.startswith('forest-separability-0127-main-forest5-2022-') and len(name) <= 63
    assert (env['PROBE_MODEL_NAME'], env['PROBE_CHECKPOINT'], env['PROBE_NUM_CLASSES']) == (
        'main', MAIN_CHECKPOINT, '29')
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(configmap['binaryData']['source.tar.gz']))) as tar:
        manifest = json.loads(tar.extractfile('SOURCE_MANIFEST.json').read())
    assert (manifest['model_name'], manifest['checkpoint'], manifest['num_classes']) == (
        'main', MAIN_CHECKPOINT, 29)
    assert json.loads((tmp_path / 'main' / 'identity.json').read_text())['tag'] == 'main-forest5-2022'
    # Both pod steps receive the name; extraction also the checkpoint and class count.
    assert configmap['data']['bootstrap.py'] == BOOTSTRAP
    assert BOOTSTRAP.count("'--model-name',os.environ['PROBE_MODEL_NAME']") == 2
    assert "'--checkpoint',os.environ['PROBE_CHECKPOINT']" in BOOTSTRAP
    assert "'--num-classes',os.environ['PROBE_NUM_CLASSES']" in BOOTSTRAP
    render(tmp_path / 'long', scheme='sumpskog5', feature_year=2023, model_name='a' * 12,
           checkpoint=MAIN_CHECKPOINT, num_classes=29)
    assert len(_rendered(tmp_path / 'long')[1]['metadata']['name']) <= 63


@pytest.mark.parametrize("kwargs,match", [
    ({'model_name': 'main', 'checkpoint': '/data/other/best_model.pt', 'num_classes': 29},
     'checkpoint'),
    ({'model_name': 'main', 'checkpoint': '/data/checkpoints/../secret/best_model.pt',
      'num_classes': 29}, 'checkpoint'),
    ({'model_name': 'main', 'checkpoint': 'data/checkpoints/x/best_model.pt',
      'num_classes': 29}, 'checkpoint'),
    ({'model_name': 'main', 'checkpoint': '/data/checkpoints/x/best_model.pth',
      'num_classes': 29}, 'checkpoint'),
    ({'model_name': 'tessera', 'checkpoint': MAIN_CHECKPOINT, 'num_classes': 29}, 'model name'),
    ({'model_name': 'main', 'checkpoint': MAIN_CHECKPOINT, 'num_classes': 30}, 'num_classes'),
    ({'model_name': 'r4', 'checkpoint': MAIN_CHECKPOINT, 'num_classes': 28}, 'is R4'),
    ({'model_name': 'main'}, 'go together'),
    ({'checkpoint': MAIN_CHECKPOINT}, 'go together'),
])
def test_render_refuses_bad_model_arguments(tmp_path, kwargs, match):
    from scripts.render_forest_separability_job import render
    with pytest.raises(ValueError, match=match):
        render(tmp_path / 'bad', scheme='forest5', feature_year=2022, **kwargs)
    assert not (tmp_path / 'bad').exists()
