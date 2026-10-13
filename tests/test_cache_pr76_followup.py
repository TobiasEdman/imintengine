"""PR76 cache-policy witnesses; synthetic files and CPU fixtures only."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def probe(code, **env):
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1', **env},
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_plain_import_never_installs_cache_observer():
    probe('''
import sys
calls = []
sys.addaudithook = calls.append
import imint
assert calls == [], 'ordinary library import installed process-wide observer'
assert 'imint._cache_identity' not in sys.modules
print('plain library import is inert')
''')


def test_unused_external_encoder_does_not_change_producer_identity(tmp_path):
    code = 'from scripts import infer_tiles; print(infer_tiles.producer_revision())'
    before = probe(code)
    (tmp_path / 'use_croma.py').write_text('UNUSED_ENCODER = 1\n')
    after = probe(code, PYTHONPATH=os.pathsep.join((str(ROOT), str(tmp_path))))
    assert before == after


def test_unrelated_file_does_not_change_producer_identity():
    path = ROOT / 'imint/pr76-unused-fixture.json'
    code = 'from scripts import infer_tiles; print(infer_tiles.producer_revision())'
    before = probe(code)
    try:
        path.write_text('{"unrelated": true}')
        assert before == probe(code)
    finally:
        path.unlink(missing_ok=True)


def test_unrelated_distribution_does_not_enter_producer_identity():
    code = '''
import json, importlib.metadata as metadata
import torch
from scripts import infer_tiles
from imint._cache_identity import producer_environment
REAL = metadata.distributions
class Unrelated:
    metadata = {'Name': 'unrelated-pr76-doc-tool'}
    version = '1.0'
metadata.distributions = lambda: [*REAL(), Unrelated()]
environment = producer_environment()
assert 'unrelated-pr76-doc-tool' not in json.dumps(environment)
print('unrelated distribution excluded')
'''
    probe(code)


import shutil
import textwrap
import pytest


@pytest.fixture
def mini(tmp_path):
    root = tmp_path / 'mini'
    (root / 'imint').mkdir(parents=True)
    (root / 'scripts').mkdir()
    for name in ('__init__.py', '_cache_identity.py'):
        shutil.copyfile(ROOT / 'imint' / name, root / 'imint' / name)
    (root / 'imint/hidden.py').write_text('VALUE = 1\n')
    return root


def mini_probe(root, entry, driver):
    (root / 'scripts/infer_tiles.py').write_text(textwrap.dedent(entry))
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(driver)], cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('local', [False, True])
def test_preloaded_dependency_cannot_escape_through_cached_import(mini, local):
    # Bacon's cached-import witness, including an import local to a function.
    entry = '''
from imint._cache_identity import start_capture, producer_revision
start_capture(producer=True)
'''
    entry += ('def use():\n    import imint.hidden\n    return imint.hidden.VALUE\nuse()\n'
              if local else 'import imint.hidden\n')
    mini_probe(mini, entry, '''
import imint.hidden
from pathlib import Path
Path('imint/hidden.py').write_text('VALUE = 2\\n')
from scripts import infer_tiles
try:
    infer_tiles.producer_revision()
except RuntimeError as e:
    assert 'before capture' in str(e), str(e)
else:
    raise AssertionError('preloaded producer dependency escaped')
''')


@pytest.mark.parametrize('mutate', [False, True])
def test_lazy_dependency_is_fixed_before_pin_or_refused(mini, mutate):
    mini_probe(mini, '''
from imint._cache_identity import start_capture, producer_revision
start_capture(producer=True)
def prepare():
    import imint.hidden
prepare()  # Known lazy closure must be established before publication.
PIN = producer_revision()
def use():
    import imint.hidden
    producer_revision(expected=PIN)
    return imint.hidden.VALUE
''', f'''
from scripts import infer_tiles
assert infer_tiles.use() == 1
from pathlib import Path
if {mutate!r}:
    Path('imint/hidden.py').write_text('VALUE = 2\\n')
    try:
        infer_tiles.use()
    except RuntimeError as e:
        assert 'changed' in str(e)
    else:
        raise AssertionError('lazy source change accepted')
else:
    assert infer_tiles.use() == 1
''')


def test_late_unprepared_dependency_fails_closed(mini):
    mini_probe(mini, '''
from imint._cache_identity import start_capture, producer_revision
start_capture(producer=True)
PIN = producer_revision()
def use():
    import imint.hidden
    producer_revision(expected=PIN)
''', '''
from scripts import infer_tiles
try:
    infer_tiles.use()
except RuntimeError as e:
    assert 'dependencies differ' in str(e)
else:
    raise AssertionError('unprepared dependency silently adopted')
''')


def test_canonical_entry_alias_keeps_dependency_globals(mini):
    mini_probe(mini, '''
from imint._cache_identity import start_capture, producer_revision
start_capture(producer=True)
import imint.hidden
''', '''
import importlib.util
spec = importlib.util.spec_from_file_location('unregistered_alias', 'scripts/infer_tiles.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pin = module.producer_revision()
from pathlib import Path
Path('imint/hidden.py').write_text('VALUE = 2\\n')
try:
    module.producer_revision(expected=pin)
except RuntimeError:
    pass
else:
    raise AssertionError('alias dependency escaped')
''')


def test_cli_reports_sparse_byte_budget_and_stops_without_choice(tmp_path, capsys, monkeypatch):
    from scripts import score_against_truth as score
    from types import SimpleNamespace
    path = tmp_path / 'sparse.npz'
    with path.open('wb') as stream:
        stream.truncate(512 * 1024**3)
    args = SimpleNamespace(input_validation=None)
    monkeypatch.setattr(score, 'make_cached_predict_fn', lambda *a, **k: pytest.fail('cache read before choice'))
    with pytest.raises(SystemExit, match='Choose --input-validation'):
        score._validation_options(args, [path, path])
    text = capsys.readouterr().out
    assert '1 distinct candidate tiles' in text
    assert str(512 * 1024**3) + ' source bytes' in text


@pytest.mark.parametrize('mode,kwargs,error', [
    ('immutable', {}, 'immutable requires'),
    ('content', {'assume_immutable_inputs': True}, 'does not accept'),
])
def test_cli_refuses_incomplete_or_mixed_trust_choice(tmp_path, mode, kwargs, error):
    from scripts import score_against_truth as score
    from types import SimpleNamespace
    args = SimpleNamespace(input_validation=mode, input_manifest=None,
        input_manifest_sha256=None, immutable_input_root=None, assume_immutable_inputs=False)
    vars(args).update(kwargs)
    with pytest.raises(SystemExit, match=error):
        score._validation_options(args, [])


def test_stage_b_byte_counts_and_explicit_immutable_receipt(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from scripts import cache_predict as cache
    from imint._cache_identity import source_revision
    tile = tmp_path / 'source.npz'
    tile.write_bytes(b'original source bytes')
    source = cache.file_identity(tile)
    checkpoint = hashlib.sha256(b'checkpoint').hexdigest()
    directory = tmp_path / checkpoint[:16]
    directory.mkdir()
    sources = {'recorded/gpu/producer.py': 'recorded-source-digest'}
    inference = dict(checkpoint={'sha256': checkpoint}, img_size=2, num_classes=2,
                     preprocessing=source_revision(sources))
    pin = cache.contract_pin(inference)
    (directory / 'MANIFEST.json').write_text(json.dumps(dict(cache_version=2,
        inference=inference, producer_sources=sources)))
    np.savez(directory / tile.name, pred=np.zeros((2, 2), np.uint8),
        probs=np.full((2, 2, 2), .5, np.float16), fracs=np.zeros((4, 2, 2), np.float16),
        metadata=np.array(json.dumps(dict(version=2, source=source, inference=inference))))
    real = cache.file_identity
    hashed = []
    def measured(path):
        identity = real(path)
        hashed.append(identity['size'])
        return identity
    monkeypatch.setattr(cache, 'file_identity', measured)
    reader = cache.make_cached_predict_fn(tmp_path, checkpoint[:16], want_fracs=False, producer_pin=pin)
    reader(tile); reader(tile)
    assert hashed == [len(b'original source bytes')] * 2
    receipt, digest = cache.write_input_manifest(directory, [source])
    options = dict(producer_pin=pin, input_manifest=receipt, input_manifest_sha256=digest,
                   immutable_input_root=tmp_path, assume_immutable_inputs=True)
    fast = cache.make_cached_predict_fn(tmp_path, checkpoint[:16], want_fracs=False, **options)
    fast(tile)
    assert len(hashed) == 2  # No source payload hash under the explicit operator assumption.
    with pytest.raises(cache.CacheValidationError, match='requires'):
        cache.make_cached_predict_fn(tmp_path, checkpoint[:16], want_fracs=False,
                                     **{**options, 'assume_immutable_inputs': False})
    with pytest.raises(cache.CacheValidationError, match='sha256 mismatch'):
        cache.make_cached_predict_fn(tmp_path, checkpoint[:16], want_fracs=False,
                                     **{**options, 'input_manifest_sha256': '0' * 64})
    tile.write_bytes(b'modified source bytes')
    with pytest.raises(cache.CacheValidationError, match='source.sha256'):
        reader(tile)


@pytest.mark.parametrize('method', ['spawn', 'fork'])
def test_worker_preprocessing_matches_parent_contract(tmp_path, method):
    # Fresh producer entry import in both parent and worker; real NPZ reads and
    # preprocessing, no model construction or weights.
    script = tmp_path / 'worker_probe.py'
    script.write_text('''
from scripts import infer_tiles as infer
from imint._cache_identity import producer_revision, producer_sources
from scripts import inference_comparison as ic
import multiprocessing as mp
import numpy as np
import sys
from pathlib import Path

def consume(dataset):
    return dataset[0]['crop_sz']

if __name__ == '__main__':
    path = Path(sys.argv[2])
    np.savez(path, tessera=np.zeros((128, 4, 4), np.float32))
    pin = producer_revision(helpers=ic)
    ds = infer.TileInferenceDataset([path], 'tessera', 4, None,
        identities=[infer.file_identity(path)], producer=pin,
        producer_sources=producer_sources())
    with mp.get_context(sys.argv[1]).Pool(1) as pool:
        assert pool.apply(consume, (ds,)) == 4
''')
    result = subprocess.run([sys.executable, str(script), method, str(tmp_path / 'tile.npz')],
        cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
                      'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reader_import_never_discovers_producer_packages():
    probe("""
from unittest.mock import patch
import importlib.util
with patch.object(importlib.util, 'find_spec', side_effect=AssertionError('producer discovery')):
    from scripts import cache_predict
from imint import _cache_identity
assert not _cache_identity._ACTIVE
print('reader import leaves producer capture inactive')
""")


@pytest.mark.parametrize('relative', [False, True])
def test_used_model_resource_is_bound_for_relative_and_absolute_opens(mini, relative):
    (mini / 'imint/fm').mkdir()
    resource = mini / 'imint/fm/config.json'
    resource.write_text('{"value": 1}')
    mini_probe(mini, f"""
from imint._cache_identity import start_capture, producer_revision, producer_sources
from pathlib import Path
start_capture(producer=True)
path = Path('imint/fm/config.json')
if not {relative!r}:
    path = path.absolute()
with path.open() as stream:
    value = stream.read()
PIN = producer_revision()
assert 'imint/fm/config.json' in producer_sources()
""", """
from scripts import infer_tiles
from pathlib import Path
Path('imint/fm/config.json').write_text('{"value": 2}')
try:
    infer_tiles.producer_revision(expected=infer_tiles.PIN)
except RuntimeError as e:
    assert 'changed' in str(e)
else:
    raise AssertionError('used model config mutation escaped')
""")
