"""A growing shared HR-VPP index must not invalidate a staged annual map."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from imint.data.wekeo_vpp import fetch_vpp_tiles_local
from imint.inference.lclu import verify_file, write_json
from scripts.cache_predict import file_identity
from scripts.prepare_lclu import verify_source_file
from scripts.stage_lclu_demo import select_vpp, snapshot_vpp


@pytest.fixture
def vpp_cache(tmp_path):
    source = tmp_path / 'cache'
    source.mkdir()
    bounds = (670000, 6580000, 675120, 6585120)
    footprint = list(transform_bounds('EPSG:3006', 'EPSG:4326', *bounds))
    values = {'SOSD': 120, 'EOSD': 280, 'LENGTH': 160, 'MAXV': 12000, 'MINV': 3000}
    index = {}
    for metric, value in values.items():
        name = f'VPP_2024_S2_T33VXF-010m_V101_s1_{metric}.tif'
        with rasterio.open(source / name, 'w', driver='GTiff', width=32, height=32,
                           count=1, dtype='int16', crs='EPSG:3006', nodata=-32768,
                           transform=from_bounds(*bounds, 32, 32)) as dst:
            dst.write(np.full((32, 32), value, dtype=np.int16), 1)
        index[name] = {'metric': metric, 'year': 2024, 'season': 1,
                       'tileId': 'T33VXF', 'bounds_4326': footprint}
    write_json(source / 'index.json', index)
    return source / 'index.json', bounds, footprint, values


def test_snapshot_reads_same_pixels_after_shared_index_changes(vpp_cache, tmp_path):
    index_path, bounds, footprint, values = vpp_cache
    selected = select_vpp(json.loads(index_path.read_text()), 2024, footprint)
    frozen = snapshot_vpp(index_path, selected, tmp_path / 'snapshots')
    pin = file_identity(frozen)
    before = fetch_vpp_tiles_local(*bounds, size_px=32, vpp_cog_dir=frozen.parent, year=2024)

    # A new prefetch may rewrite or replace the global index; serving retains
    # the exact-year metadata it qualified against.
    write_json(index_path, {'another-area-and-year.tif': {'year': 2025}})
    assert verify_file(pin, tmp_path) == frozen
    assert snapshot_vpp(index_path, selected, tmp_path / 'snapshots') == frozen
    assert file_identity(frozen) == pin
    after = fetch_vpp_tiles_local(*bounds, size_px=32, vpp_cog_dir=frozen.parent, year=2024)
    for metric, value in values.items():
        band = metric.lower()
        np.testing.assert_array_equal(after[band], before[band])
        np.testing.assert_allclose(after[band], value * (0.0001 if metric in ('MAXV', 'MINV') else 1))
    for name in selected:
        assert (frozen.parent / name).is_symlink()
        assert (frozen.parent / name).samefile(index_path.parent / name)
    assert len(list((tmp_path / 'snapshots').glob('*/index.json'))) == 1
    with pytest.raises(ValueError, match='2025'):
        select_vpp(json.loads(frozen.read_text()), 2025, footprint)


def test_changed_raster_still_fails_identity_check(vpp_cache, tmp_path):
    index_path, _, _, _ = vpp_cache
    selected = json.loads(index_path.read_text())
    frozen = snapshot_vpp(index_path, selected, tmp_path / 'snapshots')
    name = next(iter(selected))
    pin = {**file_identity(index_path.parent / name), 'path': name}
    receipts = tmp_path / 'receipts'
    verify_source_file(pin, frozen.parent, receipts)
    # Same path and size, changed pixels: cache-growth tolerance must not
    # weaken the original byte-identity contract.
    with rasterio.open(index_path.parent / name, 'r+') as dst:
        dst.write(np.full((32, 32), 99, dtype=np.int16), 1)
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_source_file(pin, frozen.parent, receipts)


def test_snapshot_rejects_changed_metadata_and_wrong_link(vpp_cache, tmp_path):
    index_path, _, _, _ = vpp_cache
    selected = json.loads(index_path.read_text())
    snapshots = tmp_path / 'snapshots'
    frozen = snapshot_vpp(index_path, selected, snapshots)
    write_json(frozen, {})
    with pytest.raises(ValueError, match='metadata changed'):
        snapshot_vpp(index_path, selected, snapshots)
    write_json(frozen, selected)
    name = next(iter(selected))
    link = frozen.parent / name
    link.unlink()
    link.write_bytes(b'unexpected copy')
    with pytest.raises(ValueError, match='link differs'):
        snapshot_vpp(index_path, selected, snapshots)


@pytest.mark.parametrize('name', ['../escape.tif', '/absolute.tif', 'index.json'])
def test_snapshot_rejects_non_raster_basenames(tmp_path, name):
    with pytest.raises(ValueError, match='basenames'):
        snapshot_vpp(tmp_path / 'index.json', {name: {}}, tmp_path / 'snapshots')


def test_missing_selected_file_is_not_published(vpp_cache, tmp_path):
    index_path, _, _, _ = vpp_cache
    selected = json.loads(index_path.read_text())
    (index_path.parent / next(iter(selected))).unlink()
    with pytest.raises(FileNotFoundError):
        snapshot_vpp(index_path, selected, tmp_path / 'snapshots')
    assert not (tmp_path / 'snapshots').exists()


@pytest.mark.parametrize('year', [2024, 2025])
def test_stage_verifies_snapshot_targets_before_dem_preparation(vpp_cache, tmp_path, monkeypatch, year):
    """Exercise stage's real index/hash path; the identity reader rejects symlinks.

    Only HR-VPP 2024 is cached: 2025 stages on it through VPP_YEAR_SUBSTITUTIONS,
    and 2023, which has no table entry, is refused instead of falling back.
    """
    from scripts import stage_lclu_demo as staging

    index_path, bounds, _, _ = vpp_cache
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'vpp_wekeo').symlink_to(index_path.parent, target_is_directory=True)
    forestry = data / 'skg_open'
    for name in ('grunddata/volym_omdrev1.vrt', 'grunddata/grundyta_omdrev1.vrt',
                 'grunddata/medeldiameter_omdrev1.vrt', 'markfukt/markfukt.vrt',
                 'tradhojd/tradhojd_10m.vrt'):
        path = forestry / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('source mapping fixture')
    write_json(forestry / 'MANIFEST.json', {'derived': {}})
    scope = tmp_path / 'scope.json'
    west, south, east, north = bounds
    write_json(scope, {'version': 1, 'bounds_3006': [west + 20, south + 20, east - 20, north - 20],
                      'blocks': [{'year': y, 'grid_size': 1, 'scales_size': 1,
                                  'variant': 'vultr', 'version': '1.0'} for y in (2023, year)]})
    verified = []
    original_verify = staging.verify_source_file
    def verify(record, base, receipts):
        result = original_verify(record, base, receipts)
        verified.append(result)
        return result
    monkeypatch.setattr(staging, 'verify_source_file', verify)
    monkeypatch.setattr(staging, '_get_cog_urls', lambda *_: ['https://example.org/dem.tif'])
    class ReachedDem(Exception):
        pass
    def stop_at_dem(*_args, **_kwargs):
        raise ReachedDem
    monkeypatch.setattr(staging, 'fetch', stop_at_dem)
    with pytest.raises(ValueError, match='Exact-year 2023 HR-VPP'):
        staging.stage(scope, 2023, data, tmp_path / 'staged-2023')
    with pytest.raises(ReachedDem):
        staging.stage(scope, year, data, tmp_path / 'staged')
    assert len(verified) == 5
    assert all(path.is_file() and not path.is_symlink() for path in verified)
    frozen, = (tmp_path / 'staged/vpp').glob('*/index.json')
    assert {meta['year'] for meta in json.loads(frozen.read_text()).values()} == {2024}
    assert len(json.loads(frozen.read_text())) == 5
