"""Repair scripts must preserve NPZ payloads outside their intended edits."""
import importlib.util
from pathlib import Path
import socket
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected network/fetch operation')

    for attr in ('connect', 'connect_ex', 'bind'):
        monkeypatch.setattr(socket.socket, attr, forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    # Stub before import: never load credentials or the real fetch stack.
    env = types.ModuleType('imint.config.env')
    env.load_env = lambda: None
    fetch = types.ModuleType('imint.training.tile_fetch')
    fetch.N_BANDS = 6
    fetch._fetch_single_scene = forbidden
    fetch.bbox_3006_to_wgs84 = lambda bbox: bbox
    monkeypatch.setitem(sys.modules, env.__name__, env)
    monkeypatch.setitem(sys.modules, fetch.__name__, fetch)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    path = ROOT / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tile(tmp_path, layout):
    arrays = {
        'dates': np.array(['2018-09-01', '2018-05-01', '2018-07-01', '2018-08-01']),
        'doy': np.array([244, 121, 182, 213], dtype=np.int16),
        'temporal_mask': np.ones(4, dtype=np.uint8),
        'n_parcels': np.array(2),
        'labels': np.array([[1, 2], [3, 4]], dtype=np.uint8),
        'aux': np.arange(12, dtype=np.float32).reshape(3, 2, 2),
        'metadata': np.array('synthetic-preserved'),
    }
    stack = np.arange(96, dtype=np.float32).reshape(24, 2, 2) + 1
    if layout in ('spectral', 'both'):
        arrays['spectral'] = stack.copy()
    if layout in ('image', 'both'):
        arrays['image'] = stack.copy()
    path = tmp_path / 'tile.npz'
    np.savez_compressed(path, **arrays)
    return path, arrays


def call(module, path, operation, dry_run=False):
    if operation == 'blank':
        return module.blank_one(path.name, path.parent, dry_run)
    return module.patch_one({
        'name': path.name, 'bbox': [0, 0, 20, 20],
        'fetch_from': '2017-09-01', 'fetch_to': '2017-10-31',
        'current_d0_date': '2018-09-01',
    }, path.parent, dry_run)


def prepare(operation, monkeypatch):
    name = 'blank_failed_crop_tiles' if operation == 'blank' else 'patch_autumn_frame'
    module = load_script(name, monkeypatch)
    if operation == 'patch':
        module._fetch_single_scene = lambda *a, **kw: (
            np.full((6, 2, 2), 0.25, dtype=np.float32), '2017-09-20')
    return module


@pytest.mark.parametrize('operation', ['blank', 'patch'])
@pytest.mark.parametrize('layout', ['spectral', 'image', 'both'])
def test_repair_roundtrip_preserves_other_payloads(tmp_path, monkeypatch, operation, layout):
    module = prepare(operation, monkeypatch)
    path, expected = tile(tmp_path, layout)
    result = call(module, path, operation)
    assert result['status'] == ('blanked' if operation == 'blank' else 'ok')
    for key in ('spectral', 'image'):
        if key in expected:
            expected[key][:6] = 0 if operation == 'blank' else 0.25
    expected['dates'][0] = '' if operation == 'blank' else '2017-09-20'
    if operation == 'blank':
        expected['doy'][0] = 0
        expected['temporal_mask'][0] = 0
    else:
        expected['doy'][0] = 263
        expected['temporal_mask'][0] = 1
    # Real compressed write/rename and reload, not a mocked persistence layer.
    with np.load(path, allow_pickle=False) as actual:
        assert set(actual.files) == set(expected)
        for name, value in expected.items():
            assert actual[name].dtype == value.dtype
            np.testing.assert_array_equal(actual[name], value, err_msg=name)
    assert not path.with_suffix('.tmp.npz').exists()
    before = path.read_bytes()
    assert call(module, path, operation)['status'] == ('skip' if operation == 'blank' else 'already_patched')
    assert path.read_bytes() == before


@pytest.mark.parametrize('operation', ['blank', 'patch'])
@pytest.mark.parametrize('layout', ['spectral', 'image'])
def test_repair_dry_run_preserves_file_bytes(tmp_path, monkeypatch, operation, layout):
    module = prepare(operation, monkeypatch)
    path, _ = tile(tmp_path, layout)
    before = path.read_bytes()
    assert call(module, path, operation, True)['status'] == ('dry_blanked' if operation == 'blank' else 'dry_ok')
    assert path.read_bytes() == before
    assert not path.with_suffix('.tmp.npz').exists()


@pytest.mark.parametrize('operation', ['blank', 'patch'])
def test_repair_failed_temp_write_preserves_original(tmp_path, monkeypatch, operation):
    module = prepare(operation, monkeypatch)
    path, _ = tile(tmp_path, 'spectral')
    before = path.read_bytes()

    def failed_write(target, **arrays):
        assert Path(target) == path.with_suffix('.tmp.npz')
        Path(target).write_bytes(b'synthetic-incomplete-write')
        raise OSError('synthetic disk error')

    monkeypatch.setattr(module.np, 'savez_compressed', failed_write)
    assert call(module, path, operation)['status'] == 'write_error'
    assert path.read_bytes() == before
    assert not path.with_suffix('.tmp.npz').exists()
    with np.load(path, allow_pickle=False) as actual:
        assert actual['dates'][0] == '2018-09-01'


def test_blank_non_crop_keeps_original(tmp_path, monkeypatch):
    module = prepare('blank', monkeypatch)
    path, arrays = tile(tmp_path, 'spectral')
    arrays['n_parcels'] = np.array(0)
    np.savez_compressed(path, **arrays)
    before = path.read_bytes()
    assert call(module, path, 'blank')['status'] == 'non_crop_kept'
    assert path.read_bytes() == before
