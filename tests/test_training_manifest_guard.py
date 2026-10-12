"""Exercise the manifest policy with decoded YAML and real shell argv witnesses."""
import importlib.util
from pathlib import Path
import subprocess

import pytest
import yaml

CASES = [
    ("if true; then pip install -e .; fi", False),
    ("if true; then pip install -e '.[training]'; fi", True),
    ("if true; then pip install -e .; fi; echo --no-deps", False),
    ("PIP_DISABLE_PIP_VERSION_CHECK=1 pip install -e .", False),
    ("PIP_DISABLE_PIP_VERSION_CHECK=1 pip install -e '.[training]'", True),
    ("if true; then A=1 B='two words' pip install -e . --no-deps; fi", True),
    ("pip install -e .", False),
    ("pip install -e '.'", False),
    ('pip install -e "."', False),
    ("pip install -e .[fetch]", False),
    ("pip install -e '.[fetch]'", False),
    ('pip install -e ".[fetch]"', False),
    ("pip install -e '.[fetch,inference]'", False),
    ("pip install -e '.[nottraining]'", False),
    ("pip install -e .[training]", True),
    ("pip install -e '.[training]'", True),
    ('pip install -e ".[training]"', True),
    ("pip install -e '.[fetch,training]'", True),
    ("pip install -e '.[training,inference]'", True),
    ("pip install -e . --no-deps", True),
    ("pip install --no-deps -e '.'", True),
    ("pip install -e '.[fetch]' '--no-deps'", True),
    ("pip install -e '.' # --no-deps", False),
    ("pip install -e '.'; echo --no-deps", False),
    ("pip install -e '.' && echo --no-deps", False),
    ("pip install -e '.' || pip install --no-deps example", False),
    ("pip install -e '.'\npip install --no-deps example", False),
    ("pip install -e '.' # ignored --no-deps\npip install --no-deps example", False),
    ("pip install -e '.' \\\n --no-deps", True),
    ("pip install \\\n -e '.[fetch]'", False),
    ("pip install -e '.' # --no-deps \\\n echo --no-deps", False),
    ('echo "pip install -e ."', True),
    ('echo "pip install -e .; --no-deps"', True),
    ("pip install --editable='.[fetch]'", False),
    ("pip install --editable '.[training]'", True),
    ("python -m pip install -e '.[fetch]'", False),
    ("python3 -m pip install -e '.[training]'", True),
    ("pip3 install -e '.[fetch]'", False),
    ("pip install -e '.[training]' -e '.[fetch]'", False),
]


@pytest.mark.parametrize('script,allowed', CASES)
def test_manifest_guard_shell_arguments(tmp_path, monkeypatch, script, allowed):
    spec = importlib.util.spec_from_file_location(
        'manifest_guard_under_test', Path(__file__).with_name('test_package_boundaries.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    (tmp_path / 'k8s').mkdir()
    manifest = {'apiVersion': 'v1', 'kind': 'Pod', 'spec': {'containers': [
        {'name': 'trainer', 'command': ['/bin/sh', '-c', script]}]}}
    (tmp_path / 'k8s/train.yaml').write_text(yaml.safe_dump(manifest))
    if allowed:
        guard.test_training_manifests_request_training_dependencies()
    else:
        with pytest.raises(AssertionError, match='omits training dependencies'):
            guard.test_training_manifests_request_training_dependencies()


@pytest.mark.parametrize('script,allowed', CASES)
def test_case_expectation_matches_shell_argv(tmp_path, script, allowed):
    # Only these hand-authored fixtures run. No manifest from the repository is
    # executed. Stub every pip/python entry point; no installer or network runs.
    log = tmp_path / 'argv'
    stub = r'''
pip() { printf '%s\000' __CALL__ "$@" >> "$ARGV_LOG"; }
pip3() { pip "$@"; }
python() { shift 2; pip "$@"; }
python3() { python "$@"; }
'''
    import os
    subprocess.run(['/bin/sh', '-c', stub + '\n' + script], cwd=tmp_path,
                   env={**os.environ, 'ARGV_LOG': str(log)}, check=True,
                   capture_output=True)
    tokens = log.read_bytes().decode().split('\0') if log.exists() else []
    commands = []
    for token in tokens:
        if token == '__CALL__':
            commands.append([])
        elif token:
            commands[-1].append(token)
    actual_allowed = True
    for args in commands:
        for i, token in enumerate(args):
            if token in ('-e', '--editable'):
                requirement = args[i + 1]
            elif token.startswith('--editable='):
                requirement = token.split('=', 1)[1]
            else:
                continue
            extras = requirement[2:-1].split(',') if requirement.startswith('.[') else []
            if 'training' not in extras and '--no-deps' not in args:
                actual_allowed = False
    assert actual_allowed == allowed


@pytest.mark.parametrize('field', ['containers', 'initContainers'])
@pytest.mark.parametrize('kind', ['Pod', 'Job', 'CronJob'])
@pytest.mark.parametrize('command,args,allowed', [
    (['pip', 'install', '-e', '.'], [], False),
    (['pip'], ['install', '-e', '.[fetch]'], False),
    (['python', '-m', 'pip'], ['install', '-e', '.'], False),
    (['pip', 'install', '-e', '.[training]'], [], True),
    (['pip'], ['install', '-e', '.[fetch,training]'], True),
    (['python', '-m', 'pip'], ['install', '-e', '.', '--no-deps'], True),
    (['/bin/sh', '-c'], ['pip install -e .'], False),
    (['/bin/bash', '-lc'], ["pip install -e '.[training]'"], True),
    (['/bin/bash', '-o', 'pipefail', '-c'], ['pip install -e .'], False),
    (['echo'], ["We don't invoke pip here", 'pip install -e .'], True),
    (['sh', 'runner.sh'], ['pip install -e .'], True),
    (['sh', '-c', 'echo ok'], ['pip install -e .'], True),
])
def test_manifest_container_argv_context(tmp_path, monkeypatch, field, kind,
                                         command, args, allowed):
    # First three invalid cases reproduce independent review R1 verbatim.
    spec = importlib.util.spec_from_file_location(
        'manifest_guard_context', Path(__file__).with_name('test_package_boundaries.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    pod_spec = {field: [{'name': 'trainer', 'command': command, 'args': args}]}
    if kind == 'Pod':
        body = pod_spec
    elif kind == 'Job':
        body = {'template': {'spec': pod_spec}}
    else:
        body = {'jobTemplate': {'spec': {'template': {'spec': pod_spec}}}}
    document = {'apiVersion': 'v1', 'kind': kind, 'spec': body}
    (tmp_path / 'k8s').mkdir()
    (tmp_path / 'k8s/train.yaml').write_text(yaml.safe_dump(document))
    if allowed:
        guard.test_training_manifests_request_training_dependencies()
    else:
        with pytest.raises(AssertionError, match='omits training dependencies'):
            guard.test_training_manifests_request_training_dependencies()


@pytest.mark.parametrize('description', [
    "We don't invoke pip here; this is documentation.",
    'pip install -e .',
])
def test_manifest_ignores_documentation_annotations(tmp_path, monkeypatch, description):
    # Both prose/example diagnostics from independent review R1 become acceptance.
    spec = importlib.util.spec_from_file_location(
        'manifest_guard_annotations', Path(__file__).with_name('test_package_boundaries.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    document = {'apiVersion': 'v1', 'kind': 'Pod',
                'metadata': {'annotations': {'description': description}},
                'spec': {'containers': [{'name': 'trainer', 'command': [
                    'sh', '-c', "pip install -e '.[training]'"]}]}}
    (tmp_path / 'k8s').mkdir()
    (tmp_path / 'k8s/train.yaml').write_text(yaml.safe_dump(document))
    guard.test_training_manifests_request_training_dependencies()
