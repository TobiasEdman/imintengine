"""Synthetic input tests; no claim of verification of the private R4 artifact."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image

from imint.inference.lclu import (
    AUX_NAMES, EMBEDDING, BundleCatalog, Grid, LcluService, TesseraR4,
    InputUnavailable, export_map, validate_bundle, write_json,
)
from scripts.cache_predict import file_identity


@pytest.fixture
def bundle(tmp_path):
    grid = Grid.from_lonlat(17.64, 59.86)
    arrays = {'tessera': np.full((128, 512, 512), .125, np.float16),
              'valid_embedding': np.ones((512, 512), bool),
              'year': np.int32(2024), 'tessera_year': np.int32(2024), 'vpp_year': np.int32(2024),
              'crs': np.array('EPSG:3006'), 'transform': np.array(list(grid.transform())[:6]),
              'easting': grid.east, 'northing': grid.north}
    for name in AUX_NAMES:
        arrays[name] = np.full((512, 512), 1, np.float32)
    arrays['height'][10:12, 20:22] = np.nan
    arrays['vpp_sosd'][:] = 24130
    arrays['vpp_eosd'][:] = 24260
    arrays['valid_embedding'][250:260, 250:260] = False
    arrays['tessera'][:, 250:260, 250:260] = 0
    path = tmp_path / 'public-synthetic.npz'
    np.savez_compressed(path, **arrays)
    entry = {'year': 2024, 'center_3006': [grid.east, grid.north], 'public_area': True,
             'input': file_identity(path), 'embedding': EMBEDDING,
             'auxiliary_provenance': {'vpp': {'year': 2024, 'season': 1, 'index': {'sha256': 'a'*64, 'size': 1}, 'files': []}}}
    return grid, arrays, entry, path


class SyntheticModel:
    contract = {'synthetic': True, 'checkpoint': {'sha256': 'a' * 64, 'size': 1}}
    calls = 0

    def predict(self, path, record):
        self.calls += 1
        pred = np.zeros((504, 504), np.uint8)
        pred[100:300, 100:300] = 2
        return pred


def service(bundle, tmp_path):
    _, _, entry, _ = bundle
    return LcluService(SyntheticModel(), BundleCatalog([entry], tmp_path), tmp_path / 'outputs')


def test_geometry_module_is_bound_to_serving_identity():
    from imint.inference.lclu import ROOT, implementation_identity
    assert implementation_identity()['imint/tile_grid.py'] == file_identity(ROOT / 'imint/tile_grid.py')['sha256']


def test_geometry_native_crop_and_geotiff_roundtrip(tmp_path):
    grid = Grid.from_lonlat(17.64, 59.86)
    assert (grid.east, grid.north) == (647850, 6638770)
    assert grid.transform(504).c - grid.transform().c == 40
    assert grid.transform(504).f - grid.transform().f == -40
    b = grid.bounds(504)
    assert b['east'] - b['west'] == b['north'] - b['south'] == 5040
    pred = np.zeros((504, 504), np.uint8)
    pred[:, :252] = 7
    valid = np.ones((504, 504), bool)
    valid[:70, :80] = False
    result = export_map(pred, valid, grid, tmp_path)
    with rasterio.open(tmp_path / 'classification.tif') as src:
        assert src.shape == (504, 504) and src.crs.to_epsg() == 3006
        assert src.bounds == (b['west'], b['south'], b['east'], b['north'])
        assert src.transform == grid.transform(504) and src.nodata == 255
        assert set(np.unique(src.read(1))) == {0, 7, 255}
        assert np.all(src.read_masks(1)[:70, :80] == 0)
        assert np.all(src.read_masks(1)[100:, 300:] == 255)
    rgba = np.array(Image.open(tmp_path / 'overlay.png'))
    # Reprojection never invents interpolated class colors; 0 is opaque black.
    assert set(map(tuple, np.unique(rgba.reshape(-1, 4), axis=0))) == {
        (0, 0, 0, 255), (194, 156, 213, 255), (0, 0, 0, 0)}
    assert result['valid_pixels'] == 504**2 - 70*80
    assert result['display_resampling'] == 'nearest'


def test_bundle_input_parity_with_canonical_builder(bundle):
    import torch
    from scripts.inference_comparison import _build_inference_inputs, _close_inference_inputs
    from imint.training.unified_dataset import normalize_aux_channel
    grid, arrays, entry, path = bundle
    validate_bundle(path, entry['input'], grid, 2024)
    inp = _build_inference_inputs(path, torch.device('cpu'), 504, AUX_NAMES, family='tessera')
    try:
        assert (inp['y0'], inp['x0'], inp['crop_sz']) == (4, 4, 504)
        np.testing.assert_array_equal(inp['img5d'].numpy()[0], arrays['tessera'][:, 4:508, 4:508].astype(np.float32))
        expected = np.stack([normalize_aux_channel(n, arrays[n][4:508, 4:508]) for n in AUX_NAMES])
        np.testing.assert_array_equal(inp['aux'].numpy()[0], expected)
        assert inp['temporal_coords'] is None and inp['location_coords'] is None
    finally:
        _close_inference_inputs(inp)


@pytest.mark.parametrize('mutation', ['year', 'grid', 'shape', 'aux', 'invalid_mask'])
def test_bundle_rejects_misleading_input(bundle, mutation):
    grid, arrays, _, path = bundle
    if mutation == 'year': arrays['vpp_year'] = np.int32(2023)
    if mutation == 'grid': arrays['transform'][2] += 10
    if mutation == 'shape': arrays['tessera'] = arrays['tessera'][:, :504, :504]
    if mutation == 'aux': del arrays['height']
    if mutation == 'invalid_mask': arrays['valid_embedding'][:] = True
    np.savez_compressed(path, **arrays)
    with pytest.raises((ValueError, KeyError)):
        validate_bundle(path, file_identity(path), grid, 2024)


def test_persistent_cache_and_atomic_failure(bundle, tmp_path, monkeypatch):
    grid, _, _, _ = bundle
    app = service(bundle, tmp_path)
    phases = []
    first = app.run(grid, 2024, phases.append)
    assert phases == ['preparing', 'classifying', 'exporting']
    assert not first['cache_hit'] and first['valid_pixels'] == 504**2 - 100
    restarted = service(bundle, tmp_path)
    second = restarted.run(grid, 2024, lambda _: None)
    assert second['cache_hit'] and restarted.model.calls == 0
    assert first['cache_key'] == second['cache_key']
    assert not list(app.output_root.glob('.pending-*'))
    directory = app.output_root / first['cache_key']
    provenance = json.loads((directory / 'provenance.json').read_text())
    assert 'path' not in provenance['input'] and provenance['year'] == 2024
    (directory / 'overlay.png').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='integrity'):
        restarted.run(grid, 2024, lambda _: None)
    fresh = LcluService(SyntheticModel(), app.sources, tmp_path / 'failed')
    def fail(*args): raise RuntimeError('export failed')
    monkeypatch.setattr('imint.inference.lclu.export_map', fail)
    with pytest.raises(RuntimeError, match='export failed'):
        fresh.run(grid, 2024, lambda _: None)
    assert not list(fresh.output_root.iterdir())


def test_reject_year_grid_source_and_changed_file(bundle, tmp_path):
    grid, _, entry, path = bundle
    app = service(bundle, tmp_path)
    assert app.sources.years(grid) == [2024]
    for year in (2017, 2025, 2023):
        with pytest.raises(InputUnavailable): app.key(grid, year)
    with pytest.raises(InputUnavailable): app.key(Grid(grid.east + 10, grid.north), 2024)
    changed = {**entry, 'embedding': {**EMBEDDING, 'variant': 'dclimate'}}
    with pytest.raises(ValueError): BundleCatalog([changed], tmp_path)
    path.write_bytes(b'changed after qualification')
    with pytest.raises(ValueError, match='identity mismatch'): app.run(grid, 2024, lambda _: None)


def test_full_runtime_budget_keeps_verified_cache_readable(bundle, tmp_path, monkeypatch):
    from imint.inference.lclu_storage import StorageLimitError, MAP_BYTES
    grid, _, _, _ = bundle
    app = service(bundle, tmp_path)
    app.storage.max_bytes = MAP_BYTES - 1
    with pytest.raises(StorageLimitError):
        app.run(grid, 2024, lambda _: None)
    assert app.model.calls == 0 and not list(app.output_root.iterdir())
    app.storage.max_bytes = 4 * 2**30
    first = app.run(grid, 2024, lambda _: None)
    restarted = service(bundle, tmp_path)
    restarted.storage.max_bytes = 1
    monkeypatch.setattr('imint.inference.lclu_storage.shutil.disk_usage',
                        lambda _: type('Disk', (), {'free': 0})())
    cached = restarted.run(grid, 2024, lambda _: None)
    assert cached['cache_hit'] and cached['cache_key'] == first['cache_key']
    assert restarted.model.calls == 0


def test_downloaded_provenance_excludes_nested_operator_paths(bundle, tmp_path):
    grid, _, entry, _ = bundle
    secret = '/private/operator/heldout/secret.tif'
    entry['auxiliary_provenance']['source'] = {'path': secret}
    entry['auxiliary_provenance']['vpp']['files'] = [
        {'sha256': 'c'*64, 'size': 10, 'file': secret, 'nested': {'path': secret}}]
    entry['embedding_provenance'] = {'registry': {'sha256': 'b'*64, 'size': 11, 'path': secret},
                                   'nested': {'path': secret}}
    app = service(bundle, tmp_path)
    result = app.run(grid, 2024, lambda _: None)
    published = (app.output_root / result['cache_key'] / 'provenance.json').read_text()
    assert secret not in published and 'heldout' not in published and 'nested' not in published
    receipt = json.loads(published)
    assert receipt['embedding_provenance']['registry'] == {'sha256': 'b'*64, 'size': 11}
    entry['auxiliary_provenance']['vpp']['files'][0]['sha256'] = secret
    with pytest.raises(ValueError, match='SHA256'): BundleCatalog([entry], tmp_path)


def test_synthetic_checkpoint_strict_loader_and_real_forward(bundle, tmp_path):
    """Real architecture/loader/forward, RANDOM weights: not private R4 verification."""
    import torch
    from imint.fm.registry import build_backbone
    from imint.fm.tessera_seg import TesseraSegmentationModel
    from scripts.inference_comparison import run_inference
    torch.set_num_threads(2)
    torch.manual_seed(19)
    encoder, _ = build_backbone('tessera_v1', img_size=504, pretrained=False)
    original = TesseraSegmentationModel(encoder, num_classes=28, n_aux_channels=11,
                                       enable_tradslag_head=True, aux_fusion='gated').eval()
    checkpoint = tmp_path / 'SYNTHETIC-r4-shape.pt'
    torch.save({'model_state_dict': original.state_dict(),
                'config': {'backbone_name': 'tessera_v1', 'num_classes': 28,
                           'n_aux_channels': 11, 'num_temporal_frames': 1, 'enabled_aux_names': AUX_NAMES}}, checkpoint)
    record = file_identity(checkpoint)
    resident = TesseraR4(record, tmp_path, 'cpu')
    grid, _, entry, path = bundle
    actual = resident.predict(path, entry['input'])
    expected = run_inference(resident.model, str(path), torch.device('cpu'), img_size=504,
                             aux_channel_names=AUX_NAMES)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (504, 504) and actual.min() >= 0 and actual.max() < 28
    record['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity mismatch'): TesseraR4(record, tmp_path, 'cpu')


def test_more_native_warmups_than_capacity_keep_new_and_saved_maps_usable(bundle, tmp_path, monkeypatch):
    """8 real 75-MiB native packages cycle through a 512-MiB runtime budget."""
    from imint.inference.lclu import public_identity
    from imint.inference.lclu_sources import RepositorySources
    from imint.inference.lclu_storage import StorageBudget

    grid, arrays, original_entry, _ = bundle
    config = tmp_path/'sources.json'
    bounds = grid.bounds()
    write_json(config, {'version':1, 'embedding':EMBEDDING, 'qualified_regions':[
        {'country':'SE', 'verification':'synthetic-capacity-test', 'years':list(range(2017,2025)),
         'bounds_3006':[bounds[n] for n in ('west','south','east','north')]}]})
    settings = {'config':file_identity(config), 'python':'unused', 'timeout_seconds':10}
    root = tmp_path/'serving'
    guard = StorageBudget(root, max_bytes=512*2**20, min_free_bytes=1)
    source = RepositorySources(settings, tmp_path, root, storage=guard)
    produced = []

    def native_package(request_grid, year, target):
        target.mkdir()
        data = {**arrays, 'year':np.int32(year), 'tessera_year':np.int32(year), 'vpp_year':np.int32(year),
                'vpp_sosd':np.full((512,512), (year-2000)*1000+130, np.float32),
                'vpp_eosd':np.full((512,512), (year-2000)*1000+260, np.float32)}
        # Uncompressed native NPZ deliberately exercises real retained bytes,
        # not tiny all-zero compressed fixtures or a mocked byte counter.
        path = target/'input.npz'
        np.savez(path, **data)
        assert 75*2**20 < path.stat().st_size < 80*2**20
        entry = {**original_entry, 'year':year, 'input':file_identity(path),
                 'source_config':public_identity(settings['config']),
                 'auxiliary_provenance':{'vpp':{'year':year, 'season':1,
                    'index':{'sha256':'a'*64,'size':1}, 'files':[]}}}
        entry['input']['path'] = 'input.npz'
        write_json(target/'entry.json', entry)
        produced.append(target)

    monkeypatch.setattr(source, '_prepare', native_package)
    app = LcluService(SyntheticModel(), source, root, storage=guard)
    results = []
    for year in range(2017,2025):
        source.check_preparation(grid, year)
        source.prepare(grid, year)
        result = app.run(grid, year, lambda _: None)
        assert not result['cache_hit'] and result['valid_pixels'] == 504**2-100
        results.append(result)
        assert guard.used_bytes() <= guard.max_bytes
    assert len(produced) == 8 and app.model.calls == 8
    assert not produced[0].exists()  # actually evicted, not just ignored
    assert 0 < sum(p.exists() for p in produced) < 8
    assert guard.used_bytes() == sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

    # The persistent result sidecars rebuild the lookup after a process restart.
    restarted_sources = RepositorySources(settings, tmp_path, root)
    restarted = LcluService(SyntheticModel(), restarted_sources, root)
    monkeypatch.setattr(restarted_sources, 'prepared', lambda *_: pytest.fail('Reprepared a saved map'))
    restarted.storage.max_bytes = 1  # even with no space for a fresh input
    saved = restarted.run(grid, 2017, lambda _: None)
    assert saved['cache_hit'] and saved['cache_key'] == results[0]['cache_key']
    assert restarted.model.calls == 0
    assert (restarted.output_root/saved['cache_key']/'classification.tif').is_file()
    # The new shortcut cannot bypass immutable source-configuration checks.
    config.write_text(config.read_text()+' ')
    with pytest.raises(ValueError, match='configuration changed'):
        restarted.run(grid, 2017, lambda _: None)


def test_request_cache_binds_all_preparation_producer_code(bundle, tmp_path, monkeypatch):
    from imint.inference import lclu
    grid, _, _, _ = bundle
    app = service(bundle, tmp_path)
    original = app.run(grid, 2024, lambda _: None)
    producer = lclu.preparation_identity()
    assert app.code.items() >= producer.items()
    actual = lclu.file_identity
    def changed_file(path):
        identity = actual(path)
        if path == lclu.ROOT/'imint/training/skg_grunddata.py':
            return {**identity, 'sha256':'b'*64}
        return identity
    monkeypatch.setattr(lclu, 'file_identity', changed_file)
    with pytest.raises(RuntimeError, match='Serving code changed'):
        app.run(grid, 2024, lambda _: None)
    restarted = service(bundle, tmp_path)
    changed = restarted.run(grid, 2024, lambda _: None)
    assert not changed['cache_hit'] and changed['cache_key'] != original['cache_key']


def test_persistent_lookup_does_not_accept_rebound_result_metadata(bundle, tmp_path):
    grid, _, _, _ = bundle
    app = service(bundle, tmp_path)
    result = app.run(grid, 2024, lambda _: None)
    metadata = app.output_root/result['cache_key']/'result.json'
    original = json.loads(metadata.read_text())
    rebound = 'a'*64
    write_json(metadata, {**original, 'request_key':rebound})
    restarted = service(bundle, tmp_path)
    with pytest.raises(RuntimeError, match='request identity'):
        restarted.cached_request(rebound)


def test_inference_holds_repository_input_lease_during_competing_eviction(bundle, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    import threading
    from imint.inference.lclu import public_identity
    from imint.inference.lclu_sources import RepositorySources
    from imint.inference.lclu_storage import StorageBudget, PREPARATION_BYTES

    grid, _, entry, input_path = bundle
    bounds = grid.bounds()
    config = tmp_path/'sources.json'
    write_json(config, {'version':1, 'embedding':EMBEDDING, 'qualified_regions':[
        {'country':'SE', 'verification':'lease-integration-test', 'years':[2024],
         'bounds_3006':[bounds[n] for n in ('west','south','east','north')]}]})
    settings = {'config':file_identity(config), 'python':'unused', 'timeout_seconds':10}
    root = tmp_path/'serving'
    guard = StorageBudget(root, max_bytes=512*2**20, min_free_bytes=1)
    source = RepositorySources(settings, tmp_path, root, storage=guard)

    def prepare(request_grid, year, target):
        target.mkdir()
        shutil.copyfile(input_path, target/'input.npz')
        write_json(target/'entry.json', {**entry,
            'input':{**entry['input'], 'path':'input.npz'},
            'source_config':public_identity(settings['config'])})

    monkeypatch.setattr(source, '_prepare', prepare)
    prepared, _ = source.prepare(grid, 2024)
    others = [root/'lclu-inputs'/f'{i:064x}' for i in range(5)]
    chunk = b'x'*(60*2**20)
    for target in others[:4]:
        with guard.reserve(len(chunk), paths=(target,)):
            target.mkdir()
            (target/'input.npz').write_bytes(chunk)
    entered, release = threading.Event(), threading.Event()

    class ReadingModel(SyntheticModel):
        def predict(self, path, record):
            assert path == prepared
            entered.set()
            assert release.wait(10)
            assert file_identity(path)['sha256'] == record['sha256']
            return super().predict(path, record)

    app = LcluService(ReadingModel(), source, root, storage=guard)
    with ThreadPoolExecutor(2) as pool:
        prediction = pool.submit(app.run, grid, 2024, lambda _: None)
        try:
            assert entered.wait(10)
            with guard.reserve(PREPARATION_BYTES, paths=(others[4],)):
                others[4].mkdir()
                (others[4]/'input.npz').write_bytes(chunk)
            assert prepared.is_file() and any(not p.exists() for p in others[:4])
        finally:
            release.set()
        result = prediction.result()
    assert result['valid_pixels'] == 504**2-100 and not result['cache_hit']
    assert guard.used_bytes() == sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
