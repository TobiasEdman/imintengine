"""Offline source-boundary tests; no upstream/R4 qualification is claimed."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from imint.inference.lclu import EMBEDDING, Grid, InputUnavailable, write_json
from imint.inference.lclu_sources import RepositorySources
from imint.training.copernicus_dem import read_dem_window
from scripts.cache_predict import file_identity
from scripts import prepare_lclu as prep


def raster(path, values, transform):
    with rasterio.open(path, 'w', driver='GTiff', crs='EPSG:3006', transform=transform,
                       width=values.shape[1], height=values.shape[0], count=1,
                       dtype=values.dtype, nodata=-9999) as dst:
        dst.write(values, 1)
    return path


def test_dem_preserves_real_zero_and_rejects_missing_or_partial_sources(tmp_path):
    transform = from_origin(647000, 6640000, 10, 10)
    path = raster(tmp_path/'dem.tif', np.zeros((20, 20), np.float32), transform)
    bounds = (647000, 6639800, 647200, 6640000)
    result = read_dem_window([path], *bounds, size_px=20)
    assert result.shape == (20, 20) and np.all(result == 0) and result.dtype == np.float32
    with pytest.raises(ValueError, match='cover'):
        read_dem_window([path], 647000, 6639700, 647300, 6640000, size_px=30)
    with pytest.raises(rasterio.errors.RasterioIOError):
        read_dem_window([tmp_path/'absent.tif'], *bounds, size_px=20)
    with pytest.raises(ValueError, match='cover'):
        read_dem_window([], *bounds, size_px=20)
    values = np.arange(400, dtype=np.float32).reshape(20, 20)
    raster(path, values, transform)
    np.testing.assert_array_equal(read_dem_window([path], *bounds, size_px=20), values)


def test_source_identity_receipt_reuses_hash_and_detects_same_size_change(tmp_path, monkeypatch):
    path = tmp_path/'large-source.tif'
    path.write_bytes(b'abcd')
    record = file_identity(path)
    receipts = tmp_path/'receipts'
    calls = []
    original = prep.verify_file
    def counted(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(prep, 'verify_file', counted)
    assert prep.verify_source_file(record, tmp_path, receipts) == path
    assert prep.verify_source_file(record, tmp_path, receipts) == path
    assert len(calls) == 1
    previous = path.stat()
    path.write_bytes(b'wxyz')
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match='identity mismatch'):
        prep.verify_source_file(record, tmp_path, receipts)
    assert len(calls) == 2  # ctime/identity invalidates even restored mtime


def test_embedding_cache_requires_exact_readable_metadata(tmp_path):
    prep.validate_embedding_cache(tmp_path)
    (tmp_path/'cached.npy').write_bytes(b'cached')
    with pytest.raises(FileNotFoundError): prep.validate_embedding_cache(tmp_path)
    metadata = tmp_path/'tessera_metadata.json'
    metadata.write_text('not json')
    with pytest.raises(ValueError): prep.validate_embedding_cache(tmp_path)
    expected = {'dataset_version':'1.0', 'dataset_version_path':'v1',
                'dataset_variant':'vultr', 'dataset_path':'v1',
                'embeddings_subdir':'global_0.1_degree_representation'}
    write_json(metadata, expected)
    prep.validate_embedding_cache(tmp_path)
    write_json(metadata, {**expected, 'dataset_variant':'dclimate'})
    with pytest.raises(ValueError, match='identity'): prep.validate_embedding_cache(tmp_path)


def test_canonical_metadata_names_and_all_vrts_are_pinned(tmp_path):
    index, manifest = tmp_path/'index.json', tmp_path/'MANIFEST.json'
    write_json(index, {})
    write_json(manifest, {})
    config = {'vpp_index':file_identity(index), 'skg_manifest':file_identity(manifest), 'skg_vrts':[]}
    names = ['markfukt/markfukt.vrt', 'tradhojd/tradhojd_10m.vrt'] + [
        f'grunddata/{v}_omdrev1.vrt' for v in ('volym', 'grundyta', 'medeldiameter')]
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.write_text('pinned mapping')
        config['skg_vrts'].append(file_identity(path))
    assert prep.verify_aux_metadata(config, tmp_path) == (index, manifest)
    config['skg_vrts'].pop()
    with pytest.raises(ValueError, match='Unpinned'): prep.verify_aux_metadata(config, tmp_path)
    copied = tmp_path/'copy.json'
    copied.write_bytes(index.read_bytes())
    config['vpp_index'] = file_identity(copied)
    with pytest.raises(ValueError, match='exact index'): prep.verify_aux_metadata(config, tmp_path)


def test_embedding_budget_exact_year_and_reprojection_validity(tmp_path, monkeypatch):
    grid = Grid.from_lonlat(17.64, 59.86)
    arr = np.full((512, 512, 128), .25, np.float32)
    arr[200:210, 220:230] = 0
    ref = raster(tmp_path/'reference.tif', np.ones((512,512), np.int16), grid.transform())
    source = tmp_path/'embedding.npy'
    calls = []
    registry = SimpleNamespace(load_blocks_for_region=lambda **kw: [(2024, 17.65, 59.85)],
                               fetch_landmask=lambda **kw: ref)
    def fetch(**kw):
        calls.append(kw)
        source.write_bytes(b'bytes')
        return arr, rasterio.crs.CRS.from_epsg(3006), grid.transform()
    gt = SimpleNamespace(registry=registry, fetch_embedding=fetch)
    monkeypatch.setattr(prep, 'embedding_files', lambda *_: {source:5})
    with pytest.raises(InputUnavailable, match='byte budget'):
        prep.assemble_embedding(gt, grid, 2024, max_blocks=4, max_bytes=4)
    assert not calls
    merged, valid, receipt = prep.assemble_embedding(gt, grid, 2024, max_blocks=4, max_bytes=5)
    assert calls == [{'lon':17.65, 'lat':59.85, 'year':2024, 'refresh':False}]
    assert merged.shape == (128,512,512) and merged.dtype == np.float16
    assert valid.sum() == 512**2 - 100 and not valid[200:210,220:230].any()
    assert merged[0,100,100] == .25 and receipt['uncached_source_bytes'] == 5
    registry.load_blocks_for_region = lambda **kw: [(2023,17.65,59.85)]
    with pytest.raises(InputUnavailable, match='exact-year'):
        prep.assemble_embedding(gt, grid, 2024, max_blocks=4, max_bytes=5)
    registry.load_blocks_for_region = lambda **kw: [(2024,17.65,59.85)] * 5
    with pytest.raises(InputUnavailable, match='block budget'):
        prep.assemble_embedding(gt, grid, 2024, max_blocks=4, max_bytes=5)
    assert len(calls) == 1


def test_preparation_timeout_preserves_private_log_and_qualifies_whole_window(tmp_path, monkeypatch):
    grid = Grid.from_lonlat(17.64, 59.86)
    b = grid.bounds()
    config = tmp_path/'sources.json'
    write_json(config, {'version':1, 'embedding':EMBEDDING, 'qualified_regions':[
        {'country':'SE', 'verification':'operator-check-1', 'years':[2024],
         'bounds_3006':[b[k] for k in ('west','south','east','north')]}]})
    source = RepositorySources({'config':file_identity(config), 'python':'python3.12',
                                'timeout_seconds':7}, tmp_path, tmp_path/'cache')
    assert source.years(grid) == [2024]
    assert source.years(Grid(grid.east+10,grid.north)) == []
    def timeout(command, **kwargs):
        assert kwargs['timeout'] == 7 and command[-1].startswith(str(source.cache))
        kwargs['stdout'].write('private upstream diagnostic')
        kwargs['stdout'].flush()
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='operator log'): source.prepare(grid, 2024)
    logs = list((source.cache/'failures').glob('*.log'))
    assert len(logs) == 1 and logs[0].read_text() == 'private upstream diagnostic'
    assert not list(source.cache.glob('.preparing-*'))
    assert list(source.cache.iterdir()) == [source.cache/'failures']


def test_preparation_preserves_real_virtual_environment(tmp_path):
    import venv
    environment = tmp_path/'runtime'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment/'bin/python'
    assert executable.is_symlink()
    config = tmp_path/'sources.json'
    write_json(config, {'version':1, 'embedding':EMBEDDING, 'qualified_regions':[]})
    source = RepositorySources({'config':file_identity(config), 'python':str(executable),
                                'timeout_seconds':7}, tmp_path, tmp_path/'cache')
    prefix = subprocess.check_output([source.python, '-c', 'import sys; print(sys.prefix)'], text=True)
    assert Path(prefix.strip()) == environment


def test_county_mask_is_pinned_independently_of_omdrev_aux_selection(tmp_path):
    relative = 'grunddata/omdrev2/volym_03.tif'
    path = tmp_path/relative
    path.parent.mkdir(parents=True)
    transform = from_origin(647000, 6640000, 10, 10)
    raster(path, np.ones((20,20), np.int16), transform)
    bounds = [647000, 6639800, 647200, 6640000]
    manifest = tmp_path/'MANIFEST.json'
    write_json(manifest, {'derived':{relative:{'bounds':bounds, 'sha256':file_identity(path)['sha256'],
                                              'from':'source.zip'}},
                         'files':{'source.zip':{'sha256':'a'*64}}})
    pinned = prep.verify_county_masks(manifest, tuple(bounds), tmp_path/'receipts')
    assert set(pinned) == {relative}  # no auxiliary omdrev counts involved
    assert pinned[relative]['source_sha256'] == 'a'*64
    raster(path, np.zeros((20,20), np.int16), transform)
    with pytest.raises(ValueError, match='identity mismatch'):
        prep.verify_county_masks(manifest, tuple(bounds), tmp_path/'receipts')
