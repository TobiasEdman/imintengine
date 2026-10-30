import json
from pathlib import Path

import numpy as np
import pytest

from scripts.probe_forest_separability import (
    CLASS_SCHEMES, ROLE, class_labels, digest, metrics, run_probe, sample_pixels, spatial_folds,
)
from scripts.extract_forest_separability import ensure_development_roots, tile_candidates


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


def make_bundle(tmp_path, scheme="binary"):
    rng = np.random.default_rng(8)
    k = len(CLASS_SCHEMES[scheme])
    per_group = 12 * k
    y = np.tile(np.repeat(np.arange(k), 12), 8)
    xy = np.array([[400000 + g * 100000 + i * 100, 6500000 + (i % 3) * 100]
                   for g in range(8) for i in range(per_group)], dtype=float)
    signal = (np.eye(k)[y] * 5 + rng.normal(0, 0.1, (len(y), k))).astype(np.float32)
    first_code = np.array([codes[0] for _, codes in CLASS_SCHEMES[scheme]], np.uint16)
    p = tmp_path / "features.npz"
    np.savez(p, raw_nmd=first_code[y], xy=xy, tessera=signal, r4=signal * 2,
             tile=np.repeat(np.arange(8), per_group),
             base_label=np.full(len(y), 3, np.uint8), distilled_label=(y % 4 + 1).astype(np.uint8),
             r4_prediction=(y % 4 + 1).astype(np.int64))
    p.with_suffix('.json').write_text(json.dumps({'role': ROLE, 'scheme': scheme, 'feature_year': 2022,
                                                  'reference_year': 2023, 'bundle_sha256': digest(p)}))
    return p


def test_real_disk_probe_separable_signal_and_majority_baseline(tmp_path):
    path = make_bundle(tmp_path)
    report = run_probe(path, scheme="binary", n_splits=4)
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
    report = run_probe(path, scheme="forest5", n_splits=4)
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
        run_probe(path, scheme="binary")
    receipt.write_text(json.dumps({'role': ROLE, 'scheme': 'binary', 'feature_year': 2022,
                                   'reference_year': 2023, 'bundle_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='receipt'):
        run_probe(path, scheme="binary")


def test_probe_scheme_must_match_extraction_receipt(tmp_path):
    path = make_bundle(tmp_path)
    with pytest.raises(ValueError, match="scheme 'binary' is not 'forest5'"):
        run_probe(path, scheme="forest5")
    receipt = json.loads(path.with_suffix('.json').read_text())
    del receipt['scheme']
    path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="scheme None"):
        run_probe(path, scheme="binary")


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
