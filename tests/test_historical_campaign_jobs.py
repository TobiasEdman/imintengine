"""Offline regressions for guarded historical PR67 job bodies.

Only extracted Python runs, on synthetic temporary tiles or stub runners.
The historical shell bodies are never executed past their unconditional guard.
"""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = (
    'k8s/cohort-flag-empty-arrays-job.yaml',
    'k8s/cohort-frame-consistency-job.yaml',
    'k8s/nfi-cover-estimate-job.yaml',
    'k8s/nfi-index-identity-check-job.yaml',
    'k8s/nfi-index-multidir-job.yaml',
    'k8s/nfi-index-rebuild-job.yaml',
    'k8s/nfi2024-aux-audit-job.yaml',
    'k8s/nfi2024-drop-uncovered-job.yaml',
    'k8s/nfi2024-enrich-s1-job.yaml',
    'k8s/nfi2024-fetch-job.yaml',
    'k8s/nfi2024-fix-zero-frames-job.yaml',
    'k8s/nfi2024-key-census-job.yaml',
    'k8s/nfi2024-labels-job.yaml',
    'k8s/nfi2024-manifest-job.yaml',
    'k8s/nfi2024-mark-frame-gaps-job.yaml',
    'k8s/nfi2024-mark-s1-2016-gaps-job.yaml',
    'k8s/nfi2024-retry-job.yaml',
    'k8s/nfi2024-s1-2016-job.yaml',
    'k8s/nfi2024-s1-dual-census-job.yaml',
    'k8s/nfi2024-skg-job.yaml',
    'k8s/nfi2024-vpp-prefetch-job.yaml',
    'k8s/nfi2024-vpp-stamp-job.yaml',
    'k8s/nfi2024-yield-job.yaml',
    'k8s/sen2cor-fallback-loop-job.yaml',
    'k8s/sen2cor-frame2016-nfi2024-job.yaml',
    'k8s/sen2cor-plan-check-job.yaml',
    'k8s/sen2cor-relax-fallbacks-job.yaml',
    'k8s/sen2cor-relax-staged-job.yaml',
    'k8s/tessera-coverage-probe-job.yaml',
    'k8s/tessera-equivalence-job.yaml',
    'k8s/tessera-mark-holes-job.yaml',
    'k8s/tessera-promote-v2-job.yaml',
    'k8s/tessera-staging-job.yaml',
    'k8s/tessera-v2-add-job.yaml',
)


def shell(name):
    doc = yaml.safe_load((ROOT / 'k8s' / name).read_text())
    container = doc['spec']['template']['spec']['containers'][0]
    return next(s for s in container.get('args', []) + container.get('command', []) if '\n' in s)


def body(name):
    return re.search(r"python3? - <<'PYEOF'\n(.*?)\nPYEOF", shell(name), re.S)[1]


@pytest.mark.parametrize('name', HISTORICAL)
def test_historical_jobs_cannot_start(name, tmp_path):
    doc = yaml.safe_load((ROOT / name).read_text())
    assert doc['spec']['suspend'] is True
    script = shell(Path(name).name)
    assert subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True).returncode == 0
    # Execute only the bounded guard prefix plus a local witness. Never execute
    # install/clone/data commands, even if a regression removes the exit.
    prefix = script.split('exit 64\n', 1)[0]
    commands = [line for line in prefix.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    assert len(commands) == 1 and commands[0].startswith('echo "HISTORICAL ONLY:')
    assert 'exit 64\n' in script
    witness = tmp_path / 'ran'
    result = subprocess.run(['bash', '-c', prefix + f'exit 64\ntouch {witness}'], capture_output=True)
    assert result.returncode == 64 and not witness.exists()
    for match in re.finditer(r"python3? (?:- [^\n]*?|-)<<'?(PYEOF|PY)'?\n(.*?)\n\1", script, re.S):
        ast.parse(match[2])


def tile(v2=True, promoted=False):
    data = {'tessera': np.ones((128, 2, 2)), 'tessera_year': np.array(2024),
            'tile_size_px': np.array(2), 'has_tessera': np.array(1),
            'tessera_source': np.array('geotessera-observed')}
    if v2:
        data.update(tessera_v2=np.full((128, 2, 2), 2.), tessera_v2_year=np.array(2024),
                    tessera_v2_source=np.array('geotessera-observed'), tessera_v2_merged=np.array(1))
    if promoted:
        data['tessera_v1'] = np.zeros((128, 2, 2))
    return data


def promote(tmp_path, monkeypatch, data, *, excluded=False, campaign=None, expected=True):
    cohort, stage = tmp_path / 'cohort', tmp_path / 'stage'
    cohort.mkdir(exist_ok=True); stage.mkdir(exist_ok=True)
    if data is not None:
        np.savez(cohort / 'tile.npz', **data)
    if campaign is not None:
        np.savez(stage / 'campaign.npz', **campaign)
    manifest = {'source': 'geotessera-observed', 'cohort': {'tile.npz': 2024} if expected else {},
                'campaign': {'campaign.npz': 2024} if campaign is not None else {},
                'exclusions': {'cohort/tile.npz': 'documented gap'} if excluded else {}}
    expected_path = tmp_path / 'expected.json'
    expected_path.write_text(json.dumps(manifest))
    monkeypatch.setenv('TESSERA_PROMOTION_EXPECTED', str(expected_path))
    tree = ast.parse(body('tessera-promote-v2-job.yaml'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            value = {'COHORT': cohort, 'STAGE': stage, 'REPORT': tmp_path / 'report.json'}.get(node.targets[0].id)
            if value is not None:
                node.value = ast.Constant(str(value))
    exec(compile(ast.fix_missing_locations(tree), '<promotion>', 'exec'), {})
    return json.loads((tmp_path / 'report.json').read_text())


@pytest.mark.parametrize('defect', ['missing', 'empty_membership', 'no_v2', 'wrong_source', 'wrong_year',
                                   'bad_shape', 'nonfinite', 'zero', 'bad_already', 'bad_flag', 'unstamped_campaign'])
def test_promotion_rejects_incomplete_population(defect, tmp_path, monkeypatch):
    data = tile()
    campaign = None
    if defect == 'no_v2': data.pop('tessera_v2')
    if defect == 'wrong_source': data['tessera_v2_source'] = np.array('old')
    if defect == 'wrong_year': data['tessera_v2_year'] = np.array(2023)
    if defect == 'bad_shape': data['tessera_v2'] = np.ones((2, 2))
    if defect == 'nonfinite': data['tessera_v2'][0, 0, 0] = np.nan
    if defect == 'zero': data['tessera_v2'][:] = 0
    if defect in ('bad_already', 'bad_flag'):
        data = tile(v2=False, promoted=True)
        data['tessera_source' if defect == 'bad_already' else 'has_tessera'] = np.array('old' if defect == 'bad_already' else 0)
    if defect == 'unstamped_campaign':
        campaign = tile(v2=False); campaign.pop('tessera_source')
    with pytest.raises(SystemExit, match='FAIL'):
        promote(tmp_path, monkeypatch, None if defect in ('missing', 'empty_membership') else data,
                campaign=campaign, expected=defect != 'empty_membership')
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['status'] == 'failed' and report['errors']
    if data is not None and (tmp_path / 'cohort/tile.npz').exists():
        with np.load(tmp_path / 'cohort/tile.npz') as stored:
            assert ('tessera_v1' in stored) == ('tessera_v1' in data)


def test_promotion_preserves_old_and_validates_resume(tmp_path, monkeypatch):
    data = tile()
    report = promote(tmp_path, monkeypatch, data, campaign=tile(v2=False))
    assert report['states']['verified'] == 2
    with np.load(tmp_path / 'cohort/tile.npz') as archive:
        promoted = dict(archive)
    np.testing.assert_array_equal(promoted['tessera_v1'], data['tessera'])
    np.testing.assert_array_equal(promoted['tessera'], data['tessera_v2'])
    assert str(promoted['tessera_source']) == 'geotessera-observed'
    assert 'tessera_v2' not in promoted
    report = promote(tmp_path, monkeypatch, promoted, campaign=tile(v2=False))
    assert report['states']['already'] == 1 and report['status'] == 'passed'


def test_promotion_allows_only_documented_exclusion(tmp_path, monkeypatch):
    report = promote(tmp_path, monkeypatch, tile(v2=False), excluded=True, campaign=tile(v2=False))
    assert report['states']['excluded'] == 1 and report['states']['verified'] == 1
    assert report['exclusions']['cohort/tile.npz'] == 'documented gap'


def test_promotion_unreadable_fails_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **kw: (_ for _ in ()).throw(OSError('unreadable')))
    with pytest.raises(SystemExit, match='FAIL'):
        promote(tmp_path, monkeypatch, tile())
    assert 'unreadable' in (tmp_path / 'report.json').read_text()


def run_retry(tmp_path, fail=False, succeeds_at=None):
    tree = ast.parse(body('sen2cor-relax-fallbacks-job.yaml'))
    loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'cot')
    class Redirect(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace('/data/debug', str(tmp_path))
            return node
    loop = Redirect().visit(loop)
    calls, gained, stamps = [], set(), []
    def runner(args):
        cot = float(args[args.index('--cot-max') + 1])
        plan = json.loads(Path(args[args.index('--plan') + 1]).read_text())
        sid = plan['scenes'][0]['scene_id']
        calls.append((cot, sid))
        if (cot, sid) == succeeds_at: gained.add('tile')
        return SimpleNamespace(returncode=1 if fail else 0)
    env = {'THRESHOLDS': [8., 10., 12.], 'missing': lambda: [] if gained else ['tile'],
           'fallbacks': {'tile': ['primary', 'a', 'b']}, 'primary': {'tile': 'primary'},
           'attempted': set(), 'plan0': {}, 'dt_from_scene': lambda sid: '',
           'json': json, 'subprocess': SimpleNamespace(run=runner), 'sys': sys, 'STAGE': str(tmp_path),
           'stamp': lambda tiles, cot: stamps.append((tiles, cot)) or len(tiles), 'record': {}}
    code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), '<retry>', 'exec')
    if fail:
        with pytest.raises(SystemExit, match='ABORTING'): exec(code, env)
    else:
        exec(code, env)
    return env, calls, stamps


def test_retry_exhausts_each_threshold_before_relaxing(tmp_path):
    env, calls, _ = run_retry(tmp_path)
    assert calls == [(cot, sid) for cot in (8., 10., 12.) for sid in ('a', 'b')]
    assert len(env['attempted']) == 6


def test_retry_stops_at_lowest_success_and_stamps(tmp_path):
    env, calls, stamps = run_retry(tmp_path, succeeds_at=(10., 'b'))
    assert calls == [(8., 'a'), (8., 'b'), (10., 'a'), (10., 'b')]
    assert env['record'] == {'tile': 10.}
    assert stamps[-1] == (['tile'], 10.)


def test_failed_runner_does_not_consume_attempt(tmp_path):
    env, calls, _ = run_retry(tmp_path, fail=True)
    assert calls == [(8., 'a')] and env['attempted'] == set()


@pytest.mark.parametrize('outcomes', [('nofetch',), ('shape',), ('identical', 'nofetch'),
                                     ('differs', 'shape'), ('identical',), ('exception',)])
def test_equivalence_summary_handles_failures(outcomes, tmp_path, capsys):
    tree = ast.parse(body('tessera-equivalence-job.yaml'))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.With))
    paths = []
    for i, _ in enumerate(outcomes):
        path = tmp_path / f'{i}.npz'; np.savez(path, **tile(v2=False)); paths.append(str(path))
    results = iter(outcomes)
    def fetch(*args):
        result = next(results)
        if result == 'exception': raise RuntimeError('synthetic fetch failure')
        if result == 'nofetch': return None, 0
        if result == 'shape': return np.ones((2, 2)), 1
        return np.ones((128, 2, 2)) * (2 if result == 'differs' else 1), 1
    env = {'tempfile': tempfile, 'GeoTessera': lambda **kw: None, 'picked': paths,
           'np': np, 'resolve_tile_bbox': lambda **kw: [0, 0, 20, 20],
           'TileConfig': lambda **kw: None, 'infer_tile_year': lambda d: 2024,
           '_fetch_tessera_for_tile': fetch}
    exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), '<equivalence>', 'exec'), env)
    assert all(isinstance(row, tuple) and len(row) == 2 for row in env['verdicts'])
    captured = capsys.readouterr().out
    if any(x in ('nofetch', 'shape', 'exception') for x in outcomes):
        assert 'CONCLUSION: inconclusive' in captured
    else:
        assert 'compared=1 failed=0' in captured


def worker_env(tmp_path):
    tree = ast.parse(body('tessera-v2-add-job.yaml'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    env = {'os': os, 'np': np, 'json': json, 'hashlib': hashlib, 'tempfile': tempfile,
           'threading': threading, 'lock': threading.Lock(), 'stats': Counter(),
           'stop': threading.Event(), 'rows': [], 'EVIDENCE': str(evidence),
           'paths': [], 'MIN_FREE_GB': 0., 'SOURCE': 'geotessera-observed', 'gt': None,
           'resolve_tile_bbox': lambda **kw: [0, 0, 20, 20], 'TileConfig': lambda **kw: None,
           '_fetch_tessera_for_tile': lambda *a: (np.full((128, 2, 2), 3.), 1)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<v2-worker>', 'exec'), env)
    env['free_gb'] = lambda: 1000.
    return env, evidence


def test_v2_recovers_evidence_after_tile_write_crash(tmp_path):
    env, evidence = worker_env(tmp_path)
    path = tmp_path / 'tile.npz'; np.savez(path, **tile(v2=False))
    persist = env['persist_comparison']
    env['persist_comparison'] = lambda *a: (_ for _ in ()).throw(RuntimeError('crash after tile write'))
    with pytest.raises(RuntimeError, match='crash'): env['one'](str(path))
    with np.load(path) as archive:
        assert 'tessera_v2' in archive and str(archive['tessera_v2_source']) == 'geotessera-observed'
    assert not list(evidence.glob('*.json'))
    env['persist_comparison'] = persist
    env['_fetch_tessera_for_tile'] = lambda *a: pytest.fail('resume must use stored arrays')
    env['one'](str(path))
    records = list(evidence.glob('*.json')); assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['mean_abs_diff'] == 2. and record['source'] == 'geotessera-observed'
    assert record['path'] == str(path)
    env['one'](str(path))
    assert len(list(evidence.glob('*.json'))) == 1


def test_v2_evidence_survives_later_worker_failure_and_name_collision(tmp_path):
    env, evidence = worker_env(tmp_path)
    for directory in ('cohort', 'campaign'):
        folder = tmp_path / directory; folder.mkdir()
        path = folder / 'same-name.npz'; np.savez(path, **tile(v2=False))
        env['one'](str(path))
    assert len(list(evidence.glob('*.json'))) == 2
    with pytest.raises(KeyError): env['comparison']('bad', {})
    assert len(list(evidence.glob('*.json'))) == 2
    assert len({json.loads(p.read_text())['path'] for p in evidence.glob('*.json')}) == 2


def test_v2_atomic_record_failure_retains_previous_record(tmp_path, monkeypatch):
    env, evidence = worker_env(tmp_path)
    row = env['comparison']('tile', tile())
    env['persist_comparison']('tile', row)
    target = next(evidence.glob('*.json')); original = target.read_bytes()
    monkeypatch.setattr(os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('injected replace failure')))
    with pytest.raises(OSError): env['persist_comparison']('tile', {**row, 'year': 2023})
    assert target.read_bytes() == original and not list(evidence.glob('*.tmp'))


def test_fallback_preserves_history_on_restart(tmp_path):
    script = shell('sen2cor-fallback-loop-job.yaml')
    assert 'rm -f /data/debug/sen2cor_nfi2024_tried.json' not in script
    tree = ast.parse(body('sen2cor-fallback-loop-job.yaml'))
    begin = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'tried')
    # Execute actual persisted-state load, primary seeding, and selection.
    tried_path = tmp_path / 'tried.json'; tried_path.write_text(json.dumps({'tile': ['fallback-a']}))
    env = {'json': json, 'os': os, 'TRIED': str(tried_path),
           'plan0': {'scenes': [{'scene_id': 'primary', 'tile_names': ['tile']}]}}
    exec(compile(ast.Module(body=tree.body[begin:begin+2], type_ignores=[]), '<tried-state>', 'exec'), env)
    loop = next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id == 't' and isinstance(n.iter,ast.Name) and n.iter.id == 'miss')
    env.update(miss=['tile'], fallbacks={'tile': ['primary','fallback-a','fallback-b']}, by_scene={}, no_options=[])
    exec(compile(ast.Module(body=[loop], type_ignores=[]), '<fallback-selection>', 'exec'), env)
    assert env['by_scene'] == {'fallback-b': ['tile']}
    assert json.loads(tried_path.read_text()) == {'tile': ['fallback-a']}


def test_v2_source_uses_observed_client_version():
    tree = ast.parse(body('tessera-v2-add-job.yaml'))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCE')
    seen = []
    env = {'importlib': SimpleNamespace(metadata=SimpleNamespace(
        version=lambda name: seen.append(name) or '9.8.7'))}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<source>', 'exec'), env)
    assert seen == ['geotessera'] and env['SOURCE'] == 'geotessera-9.8.7'
