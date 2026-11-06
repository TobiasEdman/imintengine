"""Fire-tab TESSERA years (task 0151): layout, stitch, year plan, manifest."""
import json
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from imint.data.wekeo_vpp import _VPP_PRODUCT_TYPES
from imint.inference import lclu
from imint.inference.lclu_legend import MAP_COLORS
from scripts import render_fire_tessera_years as fire

ROOT = Path(__file__).resolve().parents[2]
SCOPE = json.loads((ROOT / 'config/lclu/fire_karbole.json').read_text())
IMAGE = 'python@sha256:' + '0' * 64


def test_committed_scope_covers_the_fire_tab_grid_with_two_untied_windows():
    bounds, grids, assignment = fire.check_scope(SCOPE)
    assert bounds == (521870, 6865550, 524540, 6871140)
    assert SCOPE['aoi']['transform'] == [10, 0, 521870, 0, -10, 6871140]
    assert assignment.shape == (559, 267)  # the fire tab's 267x559 overlays
    # The split falls on the pixel edge N 6868350: rows 0-278 north, 279-558 south.
    assert (assignment[:279] == 0).all() and (assignment[279:] == 1).all()
    height, width = assignment.shape
    x = bounds[0] + 10 * (np.arange(width) + 0.5)
    y = bounds[3] - 10 * (np.arange(height) + 0.5)
    for i, grid in enumerate(grids):
        b = grid.bounds(lclu.OUTPUT_SIZE)
        rows, cols = np.nonzero(assignment == i)
        assert (b['west'] < x[cols]).all() and (x[cols] < b['east']).all()
        assert (b['south'] < y[rows]).all() and (y[rows] < b['north']).all()
        context = np.minimum.reduce([x[cols] - b['west'], b['east'] - x[cols],
                                     y[rows] - b['south'], b['north'] - y[rows]])
        assert context.min() >= 1110  # metres of model context to the window edge
        # Each window's centre is an admitted point of the qualified region.
        assert bounds[0] < grid.east < bounds[2] and bounds[1] < grid.north < bounds[3]
    assert {r['year'] for r in SCOPE['blocks']} == set(SCOPE['years']) == set(lclu.YEARS)


def test_assignment_rejects_ties_gaps_and_off_lattice_centres():
    # Symmetric centres put the split on row 279's pixel centre.
    with pytest.raises(ValueError, match='Ambiguous'):
        fire.assign_windows(fire.FIRE_AOI_3006,
                            fire.window_grids([(523200, 6869740), (523200, 6866950)]))
    # One 5040 m window cannot cover the 5590 m tall AOI.
    with pytest.raises(ValueError, match='outside'):
        fire.assign_windows(fire.FIRE_AOI_3006, fire.window_grids([(523200, 6868340)]))
    with pytest.raises(ValueError, match='lattice'):
        fire.window_grids([(523205, 6869740)])


def test_stitch_puts_each_window_pixel_at_its_coordinate_from_the_nearest_window():
    bounds, grids, assignment = fire.check_scope(SCOPE)
    maps = [np.full((504, 504), 20 + i, np.uint8) for i in range(len(grids))]

    def mark(i, x, y, value):
        b = grids[i].bounds(lclu.OUTPUT_SIZE)
        maps[i][(b['north'] - y) // 10, (x - b['west']) // 10] = value

    north, south, overlap = (522005, 6870005), (524005, 6866005), (523005, 6868005)
    mark(0, *north, 7)
    mark(1, *south, 8)
    mark(0, *overlap, 9)  # the overlap point is nearer the south window's centre
    mark(1, *overlap, 6)
    out = fire.stitch(bounds, grids, maps, assignment)

    def at(x, y):
        return out[(bounds[3] - y) // 10, (x - bounds[0]) // 10]

    assert (at(*north), at(*south), at(*overlap)) == (7, 8, 6)
    assert sorted(np.unique(out).tolist()) == [6, 7, 8, 20, 21]
    assert (out[:279] != 21).all() and (out[279:] != 20).all()


def test_reads_the_service_export_and_colours_like_its_overlay(tmp_path):
    from PIL import Image

    grid = fire.window_grids([SCOPE['windows_center_3006'][0]])[0]
    prediction = np.full((504, 504), 2, np.int64)
    prediction[:, 252:] = 10
    valid = np.ones((504, 504), bool)
    valid[:, :8] = False
    lclu.export_map(prediction, valid, grid, tmp_path)
    classes = fire.read_window(tmp_path / 'classification.tif', grid)
    assert np.array_equal(classes, np.where(valid, prediction, lclu.NODATA))
    with pytest.raises(ValueError, match='requested grid'):
        fire.read_window(tmp_path / 'classification.tif', lclu.Grid(grid.east + 10, grid.north))
    overlay = np.asarray(Image.open(tmp_path / 'overlay.png')).reshape(-1, 4)
    palette = fire.display_palette()
    assert {tuple(c) for c in np.unique(overlay, axis=0)} == {
        tuple(palette[2]), tuple(palette[10]), tuple(palette[lclu.NODATA])}


def test_year_outputs_are_the_aoi_grid_in_legend_colours_with_transparent_nodata(tmp_path):
    import rasterio
    from PIL import Image

    assert set(MAP_COLORS) == set(range(28))
    classes = np.full((559, 267), lclu.NODATA, np.uint8)
    classes[0, :28] = np.arange(28)
    outputs = fire.write_year(tmp_path / '2025', classes, fire.FIRE_AOI_3006)
    assert set(outputs) == {'classification.tif', 'classification.png'}
    with rasterio.open(tmp_path / '2025/classification.tif') as src:
        assert src.crs.to_epsg() == 3006 and src.nodata == lclu.NODATA
        assert src.transform == Affine(10, 0, 521870, 0, -10, 6871140)
        assert np.array_equal(src.read(1), classes)
    rgba = np.asarray(Image.open(tmp_path / '2025/classification.png'))
    assert rgba.shape == (559, 267, 4)
    for key, colour in MAP_COLORS.items():
        assert tuple(rgba[0, key]) == (*colour, 255)
    assert (rgba[1:, :, 3] == 0).all()
    with pytest.raises(ValueError, match='legend'):
        fire.colour(np.array([[200]], np.uint8))


def test_year_plan_uses_each_years_hrvpp_except_2025_on_2024(tmp_path):
    plan = {p['year']: p for p in fire.year_plan(SCOPE, list(lclu.YEARS), tmp_path)}
    assert list(plan) == list(range(2017, 2026))
    for year in range(2017, 2025):
        assert plan[year]['vpp_year'] == year and not plan[year]['vpp_year_substituted']
    assert plan[2025]['vpp_year'] == 2024 and plan[2025]['vpp_year_substituted']
    assert plan[2025]['vpp_index'] == plan[2024]['vpp_index'] == tmp_path / 'vpp_wekeo/index.json'
    # The four 33VWJ years absent from the shared cache on 2026-11-06.
    assert {y for y, p in plan.items() if p['vpp_cache'].startswith('vpp_wekeo_fire/')} == {
        2017, 2019, 2020, 2023}


def test_year_plan_never_falls_back_to_another_year(tmp_path):
    with pytest.raises(lclu.InputUnavailable):
        fire.year_plan(SCOPE, [2016], tmp_path)
    without_2024 = {**SCOPE, 'vpp_index_by_year': {
        k: v for k, v in SCOPE['vpp_index_by_year'].items() if k != '2024'}}
    with pytest.raises(lclu.InputUnavailable, match='HR-VPP 2024'):
        fire.year_plan(without_2024, [2025], tmp_path)
    with pytest.raises(ValueError, match='unique'):
        fire.year_plan(SCOPE, [2018, 2018], tmp_path)


def _vpp_cache(directory: Path, year: int, metrics=tuple(_VPP_PRODUCT_TYPES)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = {}
    for metric in metrics:
        name = f'VPP_{year}_S2_T33VWJ-010m_V101_s1_{metric}.tif'
        (directory / name).write_bytes(b'cog')
        # The real 33VWJ footprint from /data/vpp_wekeo/index.json.
        index[name] = {'bounds_4326': [14.9996, 61.2307, 17.1112, 62.2318], 'metric': metric,
                       'season': 1, 'tileId': 'T33VWJ', 'year': year}
    (directory / 'index.json').write_text(json.dumps(index))


def test_preflight_checks_every_years_hrvpp_before_any_download(tmp_path):
    _vpp_cache(tmp_path / 'vpp_wekeo_fire', 2019)
    _vpp_cache(tmp_path / 'vpp_wekeo', 2024)
    plan = fire.year_plan(SCOPE, [2019, 2025], tmp_path)
    fire.preflight(SCOPE, plan)
    (tmp_path / 'vpp_wekeo/VPP_2024_S2_T33VWJ-010m_V101_s1_MAXV.tif').unlink()
    with pytest.raises(lclu.InputUnavailable, match='2025: HR-VPP 2024 files absent'):
        fire.preflight(SCOPE, plan)
    _vpp_cache(tmp_path / 'vpp_wekeo_fire', 2019, metrics=('SOSD', 'EOSD'))
    with pytest.raises(ValueError, match='HR-VPP'):
        fire.preflight(SCOPE, fire.year_plan(SCOPE, [2019], tmp_path))
    with pytest.raises(FileNotFoundError):  # the gap-fill Job has not run
        fire.preflight(SCOPE, fire.year_plan(SCOPE, [2017], tmp_path / 'empty'))


def _identity(char: str) -> dict:
    return {'sha256': char * 64, 'size': 1}


def _record(year: int, vpp_year: int, vpp_hash: str = 'c') -> dict:
    return {'tessera_year': year, 'vpp_year': vpp_year, 'vpp_year_substituted': vpp_year != year,
            'vpp_cache': 'vpp_wekeo/index.json', 'sources_config': _identity('a'),
            'vpp_index': _identity('b'),
            'vpp_files': [{'file': f'VPP_{vpp_year}_S2_T33VWJ-010m_V101_s1_SOSD.tif',
                           **_identity(vpp_hash)}],
            'windows': [{'center_3006': c, 'input': _identity('d')}
                        for c in SCOPE['windows_center_3006']],
            'outputs': {'classification.png': _identity('e')}}


def _manifest(**changes):
    arguments = dict(
        image=IMAGE, git_sha='1' * 40, run_args={'run_id': 'r'},
        scope={'path': 'config/lclu/fire_karbole.json', 'identity': _identity('f')},
        checkpoint=_identity('9'), inference={'model': 'r4', 'runtime': {}},
        runtime_lock={'path': 'docker/lclu/requirements.lock', **_identity('8')},
        legend_file=_identity('7'), bounds=fire.FIRE_AOI_3006,
        grids=fire.window_grids(SCOPE['windows_center_3006']),
        years={2024: _record(2024, 2024), 2025: _record(2025, 2024)},
        produced_at='2026-11-06T00:00:00+00:00')
    return fire.build_manifest(**{**arguments, **changes})


def test_manifest_pins_inputs_and_labels_2025_as_hrvpp_2024():
    manifest = _manifest()
    assert {'image', 'image_digest', 'git_sha', 'run_args', 'input_data_hash',
            'produced_at'} <= manifest.keys()
    assert manifest['image_digest'] == 'sha256:' + '0' * 64
    assert manifest['model']['checkpoint'] == _identity('9')
    assert manifest['aoi']['shape'] == [559, 267]
    assert manifest['years']['2024']['vpp_files'][0]['sha256'] == 'c' * 64
    assert 'note' not in manifest['years']['2024']
    assert manifest['years']['2025']['vpp_year'] == 2024
    assert manifest['years']['2025']['vpp_year_substituted'] is True
    assert 'HR-VPP 2024' in manifest['years']['2025']['note']
    assert [s['year'] for s in manifest['vpp_year_substitutions']] == [2025]
    assert 'HR-VPP 2025 är inte publicerad' in manifest['historical_note']
    json.dumps(manifest, allow_nan=False)
    # The input hash binds every year's HR-VPP file identities.
    changed = _manifest(years={2024: _record(2024, 2024, 'x'), 2025: _record(2025, 2024)})
    assert changed['input_data_hash'] != manifest['input_data_hash']


def test_manifest_refuses_unlisted_hrvpp_years_and_mutable_images():
    with pytest.raises(ValueError, match='VPP_YEAR_SUBSTITUTIONS'):
        _manifest(years={2024: _record(2024, 2023)})
    with pytest.raises(ValueError, match='VPP_YEAR_SUBSTITUTIONS'):
        _manifest(years={2025: _record(2025, 2025)})
    with pytest.raises(ValueError, match='digest'):
        _manifest(image='python:3.12')
    with pytest.raises(ValueError, match='commit'):
        _manifest(git_sha='main')


def test_render_stitches_service_windows_into_years_and_a_manifest(tmp_path, monkeypatch):
    """render()'s glue, with stage() and LcluService replaced by fakes honouring
    their contracts; the real exporter writes each window's GeoTIFF."""
    from scripts import stage_lclu_demo
    from scripts.cache_predict import file_identity

    data, out = tmp_path / 'data', tmp_path / 'out'
    _vpp_cache(data / 'vpp_wekeo', 2024)
    checkpoint = tmp_path / 'r4.pt'
    checkpoint.write_bytes(b'weights')
    staged = []

    def stage(scope_path, year, data_root, root, *, shared_root, vpp_index):
        staged.append((year, vpp_index))
        root.mkdir(parents=True)
        index = json.loads(vpp_index.read_text())
        sources = root / f'sources-{year}.json'
        lclu.write_json(sources, {'vpp_index': file_identity(vpp_index), 'vpp_files': {
            name: lclu.public_identity(file_identity(vpp_index.parent / name)) for name in index}})
        lclu.write_json(root / 'serving.json', {
            'version': 1, 'device': 'cpu', 'checkpoint': file_identity(checkpoint),
            'preparation': {'config': file_identity(sources), 'python': 'python',
                            'timeout_seconds': 300}})

    class Service:
        identity = {'model': 'r4', 'runtime': {'numpy': np.__version__}}

        def __init__(self, root):
            self.root = root

        @classmethod
        def from_manifest(cls, path, output_root):
            manifest = json.loads(path.read_text())
            assert len(manifest['preparations']) == 2 and manifest['checkpoint']['size'] == 7
            return cls(output_root)

        def run(self, grid, year, phase):
            phase('classifying')
            key = f'{year}-{grid.north}'
            directory = self.root / 'lclu' / key
            directory.mkdir(parents=True)
            north = grid.north == SCOPE['windows_center_3006'][0][1]
            lclu.export_map(np.full((504, 504), year % 28 if north else 27, np.int64),
                            np.ones((504, 504), bool), grid, directory)
            vpp = [lclu.public_identity(file_identity(p))
                   for p in sorted((data / 'vpp_wekeo').glob('*.tif'))]
            lclu.write_json(directory / 'provenance.json', {
                'input': _identity('d'), 'auxiliary_provenance': {'vpp': {'files': vpp}}})
            return {'cache_key': key, 'vpp_year': lclu.vpp_year_for(year),
                    'vpp_year_substituted': lclu.vpp_year_for(year) != year,
                    'files': {name: lclu.public_identity(file_identity(directory / name))
                              for name in ('classification.tif', 'provenance.json')}}

    monkeypatch.setattr(stage_lclu_demo, 'stage', stage)
    monkeypatch.setattr(lclu, 'LcluService', Service)
    run = fire.render(ROOT / 'config/lclu/fire_karbole.json', [2025, 2024], data, out, 'r1',
                      image=IMAGE, git_sha='1' * 40, run_args={'run_id': 'r1'})
    assert staged == [(2024, data / 'vpp_wekeo/index.json'), (2025, data / 'vpp_wekeo/index.json')]
    manifest = json.loads((run / 'MANIFEST.json').read_text())
    assert manifest['years']['2025']['vpp_year'] == 2024 and 'note' in manifest['years']['2025']
    assert manifest['model']['checkpoint'] == lclu.public_identity(file_identity(checkpoint))
    assert len(manifest['years']['2024']['vpp_files']) == 5
    import rasterio
    for year in (2024, 2025):
        with rasterio.open(run / str(year) / 'classification.tif') as src:
            stitched = src.read(1)
        assert (stitched[:279] == year % 28).all() and (stitched[279:] == 27).all()
        assert (run / str(year) / 'window-1-provenance.json').is_file()
        assert manifest['years'][str(year)]['outputs']['classification.png'] == \
            lclu.public_identity(file_identity(run / str(year) / 'classification.png'))
    with pytest.raises(FileExistsError):  # a run directory is never reused
        fire.render(ROOT / 'config/lclu/fire_karbole.json', [2024], data, out, 'r1',
                    image=IMAGE, git_sha='1' * 40, run_args={})
