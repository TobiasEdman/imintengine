"""Offline execution tests for NFI Job bootstrap, isolation and provenance."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts import plan_nfi_topup_manifest as plan

REPO = Path(__file__).resolve().parents[1]


def job_shell(name):
    job = yaml.safe_load((REPO / 'k8s' / name).read_text())
    container = job['spec']['template']['spec']['containers'][0]
    return container.get('args', container['command'][2:])[0]


def embedded_python(name):
    shell = job_shell(name)
    return shell.split("<<'PYEOF'\n", 1)[1].split('\nPYEOF', 1)[0]


COUNT_JOBS = [p.name for p in sorted((REPO / 'k8s').glob('nfi-topup-*-job.yaml'))
              if 'BEFORE=$(' in p.read_text()]


@pytest.mark.parametrize('name', COUNT_JOBS)
@pytest.mark.parametrize('count', [0, 2])
def test_counts_allow_empty_roots_and_count_existing_tiles(name, count, tmp_path):
    root = tmp_path / 'staging with spaces'
    root.mkdir()
    for i in range(count):
        (root / f'tile {i}.npz').touch()
    lines = re.findall(r'^\s*((?:BEFORE|AFTER)=\$\([^\n]+?\))', job_shell(name), re.M)
    assert len(lines) == 2
    shell = 'set -euo pipefail\n' + '\n'.join(lines) + '\necho "$BEFORE $AFTER"'
    result = subprocess.run(['bash', '-c', shell], capture_output=True, text=True,
                            env={**os.environ, **dict.fromkeys(('OUT', 'ROOT', 'DIR'), str(root))})
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [str(count), str(count)]


@pytest.mark.parametrize('year', [2018, 2019, 2021, 2022])
def test_first_fetch_reaches_command_with_fresh_root(year, tmp_path):
    shell = job_shell(f'nfi-topup-fetch-{year}-job.yaml')
    block = shell[shell.index('mkdir -p "$OUT"'):shell.index('set +e')]
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + block + '\necho FETCH_REACHED'],
                            capture_output=True, text=True, env={**os.environ, 'OUT': str(tmp_path / 'new')})
    assert result.returncode == 0 and 'FETCH_REACHED' in result.stdout


@pytest.mark.parametrize('failure', ['missing', 'empty', 'corrupt', 'no_bbox', 'no_year', 'bad_bbox', 'conflict'])
def test_planner_refuses_incomplete_cohort_before_outputs(tmp_path, failure):
    cohort = tmp_path / 'cohort'
    if failure != 'missing':
        cohort.mkdir()
    if failure not in ('missing', 'empty'):
        arrays = dict(bbox_3006=np.array([500000, 6400000, 505120, 6405120]), year=np.array(2019))
        if failure == 'no_bbox': arrays.pop('bbox_3006')
        if failure == 'no_year': arrays.pop('year')
        if failure == 'bad_bbox': arrays['bbox_3006'] = np.array([505120, 6400000, 500000, 6405120])
        if failure == 'conflict': arrays['dates'] = np.array(['2020-06-01', '2020-07-01'])
        np.savez(cohort / 'bad.npz', **arrays)
        # Mixed inventories must fail too, not just an entirely invalid cohort.
        np.savez(cohort / 'good.npz', bbox_3006=np.array([520000, 6400000, 525120, 6405120]), year=2019)
        if failure == 'corrupt': (cohort / 'bad.npz').write_bytes(b'not a zip')
    output = tmp_path / 'output'
    with pytest.raises(SystemExit, match='REFUSING'):
        plan.main(['--data-root', str(tmp_path), '--cohort-dir', str(cohort), '--years', '2019',
                   '--out-dir', str(output), '--summary-json', str(tmp_path / 'summary.json')])
    assert not output.exists() and not (tmp_path / 'summary.json').exists()


def copy_fixture(tmp_path, monkeypatch):
    src, dst, cohort, manifests = [tmp_path / name for name in ('source', 'eval', 'cohort', 'manifests')]
    for directory in (src, dst, cohort, manifests): directory.mkdir()
    np.savez(src / 'tile.npz', bbox_3006=np.array([500000, 6400000, 505120, 6405120]), year=2019,
             has_tessera=0)
    np.savez(cohort / 'train.npz', bbox_3006=np.array([600000, 6400000, 605120, 6405120]), year=2019)
    report = manifests / 'holdout_eval_512_copy.json'
    code = embedded_python('holdout-eval-copy-job.yaml')
    for old, new in {'/data/holdout_val_512': src, '/data/holdout_eval_512': dst,
                     '/data/unified_v2_512': cohort, '/data/manifests/holdout_eval_512_copy.json': report}.items():
        code = code.replace(f'"{old}"', repr(str(new)))
    monkeypatch.setenv('EXPECTED_EXCLUDED', '0')
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**15))
    return src, dst, report, compile(code, 'copy-job', 'exec')


def test_bootstrap_resumes_identical_partial_copy_then_refuses_completion(tmp_path, monkeypatch):
    src, dst, report, code = copy_fixture(tmp_path, monkeypatch)
    shutil.copyfile(src / 'tile.npz', dst / 'tile.npz')
    before = (dst / 'tile.npz').read_bytes()
    shutil.copyfile(src / 'tile.npz', src / 'missing.npz')
    exec(code, {})
    assert (dst / 'missing.npz').read_bytes() == (src / 'missing.npz').read_bytes()
    assert report.exists() and (dst / 'tile.npz').read_bytes() == before
    with pytest.raises(SystemExit, match='manifests already exist'):
        exec(code, {})
    assert (dst / 'tile.npz').read_bytes() == before


def test_bootstrap_refuses_divergent_copy_before_any_write(tmp_path, monkeypatch):
    src, dst, report, code = copy_fixture(tmp_path, monkeypatch)
    np.savez(dst / 'tile.npz', tessera=np.ones((2, 2)), tessera_source='geotessera-0.10.2')
    shutil.copyfile(src / 'tile.npz', src / 'a_missing.npz')
    before = (dst / 'tile.npz').read_bytes()
    source_before = {p.name: p.read_bytes() for p in src.iterdir()}
    with pytest.raises(SystemExit, match='differ from bootstrap'):
        exec(code, {})
    assert (dst / 'tile.npz').read_bytes() == before
    assert not (dst / 'a_missing.npz').exists() and not report.exists()
    assert source_before == {p.name: p.read_bytes() for p in src.iterdir()}


def test_bootstrap_does_not_restore_excluded_tile(tmp_path, monkeypatch):
    src, dst, report, code = copy_fixture(tmp_path, monkeypatch)
    (report.parent / 'holdout_eval_512_forestry.json').write_text('{}')
    with pytest.raises(SystemExit, match='manifests already exist'):
        exec(code, {})
    assert not list(dst.iterdir())


@pytest.mark.parametrize('name', ['holdout-eval-tessera-job.yaml', 'nfi-topup-tessera-job.yaml'])
@pytest.mark.parametrize('runtime_version', ['0.10.2', '0.10.3'])
def test_tessera_pin_runtime_guard_and_stamp_agree(name, runtime_version, monkeypatch):
    shell = job_shell(name)
    assert 'geotessera==0.10.2' in shell
    monkeypatch.setattr(importlib.metadata, 'version', lambda package: runtime_version)
    guard = next(line.strip() for line in shell.splitlines() if line.strip().startswith('python -c "import importlib.metadata'))
    guard_code = guard[len('python -c "'):-1]
    if runtime_version != '0.10.2':
        with pytest.raises(SystemExit): exec(guard_code, {})
    else:
        exec(guard_code, {})
    # Execute only the actual producer validation/stamp assignments, without a
    # network client, tile scan or any rewrite. A different runtime cannot stamp.
    tree = ast.parse(embedded_python(name))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'importlib.metadata': selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('producer_version', 'SRC') for t in node.targets): selected.append(node)
        elif isinstance(node, ast.If) and 'producer_version' in ast.unparse(node.test): selected.append(node)
    ns = {'sys': sys}
    code = compile(ast.Module(body=selected, type_ignores=[]), name, 'exec')
    if runtime_version != '0.10.2':
        with pytest.raises(SystemExit, match='REFUSING'): exec(code, ns)
        assert 'SRC' not in ns
    else:
        exec(code, ns)
        assert ns['SRC'] == f'geotessera-{runtime_version}'


def test_plan_job_requires_reviewed_planner_source():
    path = REPO / 'k8s/nfi-topup-plan-job.yaml'
    container = yaml.safe_load(path.read_text())['spec']['template']['spec']['containers'][0]
    env = {v['name']: v.get('value') for v in container['env']}
    assert hashlib.sha256((REPO / 'scripts/plan_nfi_topup_manifest.py').read_bytes()).hexdigest() == env['PLAN_SCRIPT_SHA256']
    shell = job_shell(path.name)
    check = next(line.strip() for line in shell.splitlines() if '| sha256sum -c -' in line)
    assert shell.index(check) < shell.index('python scripts/plan_nfi_topup_manifest.py')
    # Exercise sha256sum against the reviewed source and a stale digest without
    # invoking the planner or touching any dataset.
    for digest, succeeds in ((env['PLAN_SCRIPT_SHA256'], True), ('0' * 64, False)):
        result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + check], cwd=REPO,
                                capture_output=True, env={**os.environ, 'PLAN_SCRIPT_SHA256': digest})
        assert (result.returncode == 0) == succeeds
