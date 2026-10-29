"""Compact COG parity across a seam, nodata edges and exact-year admission."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform
from pyproj import Transformer

from imint.inference.lclu import EMBEDDING, Grid, InputUnavailable, write_json
from imint.inference.lclu_cog import assemble_cog_embedding, convert_block
from imint.inference.lclu_sources import RepositorySources
from scripts.cache_predict import file_identity
from scripts import prepare_lclu as prep


def test_compact_cog_matches_original_pipeline_at_seam_and_missing_edges(tmp_path, monkeypatch):
    grid = Grid.from_lonlat(18, 59.3)
    blocks, arrays, refs, files = [], {}, {}, {}
    for i, lon in enumerate((17.95, 18.05)):
        # Two overlapping native blocks, with a fractional offset from the
        # output grid so this exercises interpolation, not just integer copies.
        tx = from_origin(grid.east - 3000 + 2980*i + .3, grid.north + 3000.7, 10, 10)
        q = np.full((600, 302, 128), 20 + i, np.int8)
        q[100:120, 100:120] = 0
        q[:, -1, :] = 0
        scales = np.linspace(.001, .02, 600*302, dtype=np.float32).reshape(600, 302)
        qp, sp, rp = [tmp_path/f'{i}-{name}' for name in ('q.npy', 's.npy', 'ref.tif')]
        np.save(qp, q)
        np.save(sp, scales)
        with rasterio.open(rp, 'w', driver='GTiff', width=302, height=600,
                           count=1, dtype='uint8', crs='EPSG:3006', transform=tx) as dst:
            dst.write(np.ones((600, 302), np.uint8), 1)
        record = convert_block(qp, sp, rp, tmp_path/'cogs', 2025, lon, 59.3)
        blocks.append(record)
        arrays[lon] = (q.astype(np.float32)*scales[:, :, None], 'EPSG:3006', tx)
        refs[lon] = rp
        files.update({p: p.stat().st_size for p in (qp, sp, rp)})
    gt = SimpleNamespace(
        registry=SimpleNamespace(load_blocks_for_region=lambda **kw: [(2025, lon, 59.3) for lon in arrays],
                                 fetch_landmask=lambda **kw: refs[kw['lon']]),
        fetch_embedding=lambda **kw: arrays[kw['lon']])
    monkeypatch.setattr(prep, 'embedding_files', lambda *_: files)
    expected, expected_valid, _ = prep.assemble_embedding(gt, grid, 2025, max_blocks=4, max_bytes=1)
    index = {'version': 1, 'embedding': EMBEDDING, 'blocks': blocks}
    result, valid, provenance = assemble_cog_embedding(index, tmp_path, grid, 2025, tmp_path/'receipts')
    assert valid.sum() > 200_000 and not valid.all()
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(valid, expected_valid)
    assert provenance['blocks'] == 2 and provenance['uncached_source_bytes'] == 0
    assert len(provenance['files']) == 4
    with pytest.raises(InputUnavailable, match='exact-year'):
        assemble_cog_embedding(index, tmp_path, grid, 2024, tmp_path/'receipts')
    blocks[0]['year'] = 2024
    with pytest.raises(ValueError, match='contract'):
        assemble_cog_embedding(index, tmp_path, grid, 2024, tmp_path/'receipts')


def test_municipality_holes_and_buffer_do_not_expand_click_admission(tmp_path):
    grid = Grid.from_lonlat(18, 59.3)
    polygon = box(grid.east-1000, grid.north-1000, grid.east+1000, grid.north+1000)
    polygon = polygon.difference(box(grid.east+200, grid.north+200, grid.east+400, grid.north+400))
    geometry = mapping(transform(Transformer.from_crs(3006, 4326, always_xy=True).transform, polygon))
    path = tmp_path/'sources.json'
    write_json(path, {'version': 1, 'embedding': EMBEDDING, 'qualified_regions': [
        {'country': 'SE', 'name': 'Testkommun', 'verification': 'unit-test', 'years': [2025],
         'geometry': geometry, 'bounds_3006': [grid.east-5000, grid.north-5000,
                                              grid.east+5000, grid.north+5000]}]})
    sources = RepositorySources({'config': file_identity(path), 'python': 'unused',
                                 'timeout_seconds': 10}, tmp_path, tmp_path/'cache')
    assert sources.years(grid) == [2025]
    assert sources.years(Grid(grid.east+1500, grid.north)) == []
    assert sources.years(Grid(grid.east+300, grid.north+300)) == []
    assert sources.areas()[0]['properties'] == {'name': 'Testkommun', 'years': [2025]}
    point = sources.locations()[0]
    assert sources.years(Grid.from_lonlat(point['lon'], point['lat'])) == [2025]
    with pytest.raises(InputUnavailable):
        sources.identity(grid, 2024)
    with pytest.raises(InputUnavailable):
        sources.identity(Grid(grid.east+1500, grid.north), 2025)
