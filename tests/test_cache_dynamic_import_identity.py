"""Paired regression for Bacon's frozen-v1 cached dynamic-import witness."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    for folder in ('imint', 'scripts'):
        (tmp_path / folder).mkdir()
    for name in ('__init__.py', '_cache_identity.py'):
        shutil.copyfile(ROOT / 'imint' / name, tmp_path / 'imint' / name)
    (tmp_path / 'imint/hidden.py').write_text('def value():\n    return 1\n')
    return tmp_path


def run(source, body):
    result = subprocess.run([sys.executable, '-B', '-c', body], cwd=source,
        env={**os.environ, 'PYTHONPATH': str(source)}, capture_output=True,
        text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize('method', ['direct', 'dynamic', 'saved_alias', 'relative'])
@pytest.mark.parametrize('preloaded', [False, True])
def test_cached_dependency_use_is_observed(source, method, preloaded):
    expressions = {
        'direct': 'import imint.hidden\n    return imint.hidden.value()',
        'dynamic': "return importlib.import_module('imint.hidden').value()",
        # Bind the public function BEFORE producer activation. Observing only
        # assignments to importlib.import_module misses this legitimate alias.
        'saved_alias': "return saved_import('imint.hidden').value()",
        'relative': "return importlib.import_module('.hidden', 'imint').value()",
    }
    (source / 'scripts/infer_tiles.py').write_text('''
import importlib
from importlib import import_module as saved_import
from imint._cache_identity import start_capture, producer_revision, producer_sources
start_capture(producer=True)
def consume():
    ''' + expressions[method] + '''
value = consume()
try:
    pin = producer_revision()
    assert consume() == value
    producer_revision(expected=pin)
    RESULT = {'accepted': True, 'value': value,
              'tracked': 'imint/hidden.py' in producer_sources()}
except RuntimeError as exc:
    RESULT = {'accepted': False, 'error': str(exc), 'value': value}
''')
    preload = '''
import imint.hidden
from pathlib import Path
Path('imint/hidden.py').write_text('def value():\\n    return 2\\n')
''' if preloaded else ''
    result = run(source, preload + '''
from scripts import infer_tiles
import json
print(json.dumps(infer_tiles.RESULT))
''')
    if preloaded:
        assert result['value'] == 1  # Actual stale loaded code, not new disk bytes.
        assert not result['accepted'], result
        assert 'before capture' in result['error']
    else:
        assert result == {'accepted': True, 'value': 1, 'tracked': True}


def test_unrelated_dynamic_import_and_import_errors_keep_semantics(source):
    (source / 'scripts/infer_tiles.py').write_text('''
import importlib
from imint._cache_identity import start_capture, producer_revision
start_capture(producer=True)
def missing():
    return importlib.import_module('imint.no_such_module')
PIN = producer_revision()
''')
    result = run(source, '''
import json, importlib
from scripts import infer_tiles
# Unrelated cached imports from an unobserved caller must not join the producer.
module = importlib.import_module('imint.hidden')
assert module.value() == 1
infer_tiles.producer_revision(expected=infer_tiles.PIN)
try:
    infer_tiles.missing()
except ModuleNotFoundError as exc:
    assert exc.name == 'imint.no_such_module'
else:
    raise AssertionError('normal import failure swallowed')
infer_tiles.producer_revision(expected=infer_tiles.PIN)
print(json.dumps({'unchanged': True}))
''')
    assert result == {'unchanged': True}
