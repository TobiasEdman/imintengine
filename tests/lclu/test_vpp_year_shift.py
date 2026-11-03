"""HR-VPP year-shift control on synthetic areas.

The real service path runs (RegionalSources, RepositorySources.prepared and
its child wrapper, LcluService identity and request keys, StorageBudget), and
so does the real HR-VPP child entry point (``vpp-child``), in-process. Only
the model, the preparation child's arrays and prepare_aux's arrays are
synthetic. No R4 weights, no source data, no claim about map quality.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import traceback

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from imint.inference.lclu import (
    AUX_NAMES, EMBEDDING, Grid, InputUnavailable, LcluService, NODATA, validate_bundle, write_json,
)
from imint.inference.lclu_sources import RegionalSources
from imint.inference.lclu_storage import PREPARATION_BYTES, StorageBudget
from scripts import compare_vpp_year_shift as shift
from scripts.cache_predict import contract_pin, file_identity

ROOT = Path(__file__).resolve().parents[2]
X0, Y0 = 130 * 5040, 1300 * 5040
# Two synthetic areas. B has no qualified 2018 source, and prepare_aux finds
# no usable 2022 phenology at B's location (only the HR-VPP child sees this).
AREAS = {'Åsele test': (box(X0 + 3003, Y0 + 1007, X0 + 13004, Y0 + 8996), range(2018, 2025)),
         'Bjuv test': (box(X0 + 13004, Y0 + 1507, X0 + 18006, Y0 + 4493), range(2019, 2025))}
UNUSABLE = {('Bjuv test', 2022)}
THRESHOLD = 130          # the synthetic model's spring-start day of year
HOLE = slice(250, 256)   # invalid TESSERA pixels in every synthetic input


def synthetic_aux(grid: Grid, year: int) -> dict:
    """Static layers depend on the grid only; HR-VPP on the year (spring start moves 2 days a year)."""
    aux = {name: np.full((512, 512), 1 + (grid.east // 10 + i) % 7, np.float32)
           for i, name in enumerate(AUX_NAMES)}
    columns = np.broadcast_to(np.arange(512) // 64, (512, 512))
    aux['vpp_sosd'] = ((year % 100) * 1000 + 120 + 2 * (year - 2018) + columns).astype(np.float32)
    aux['vpp_eosd'] = np.full((512, 512), (year % 100) * 1000 + 260 + year - 2018, np.float32)
    aux['vpp_length'] = np.full((512, 512), 140 + year - 2018, np.float32)
    aux['vpp_maxv'] = np.full((512, 512), 0.8 + 0.01 * (year - 2018), np.float32)
    aux['vpp_minv'] = np.full((512, 512), 0.04 + 0.001 * (year - 2018), np.float32)
    return aux


def vpp_files(year: int) -> list[dict]:
    return [{'file': f'VPP_{year}_{metric}.tif', 'sha256': hashlib.sha256(f'{year}{metric}'.encode()).hexdigest(),
             'size': 1} for metric in ('SOSD', 'EOSD', 'LENGTH', 'MAXV', 'MINV')]


def fake_prepare_aux(config, base, grid, year):
    """prepare_aux's HR-VPP outcomes on synthetic data; a per-year snapshot holds only its year."""
    region = config['qualified_regions'][0]
    if year not in region['years']:
        raise InputUnavailable('Exact-year HR-VPP source coverage is incomplete')
    if (region['name'], year) in UNUSABLE:
        raise InputUnavailable('No usable exact-year phenology in this window')
    return synthetic_aux(grid, year), {'vpp': {'year': year, 'season': 1, 'files': vpp_files(year)}}


class Boundary:
    """Stands in for both child processes: same commands, exit status and logs.

    ``scripts/prepare_lclu.py`` writes a synthetic service input; the HR-VPP
    child runs the script's real ``vpp-child`` entry point over a synthetic
    prepare_aux.
    """

    def __init__(self):
        self.prepared, self.vpp = [], []

    def __call__(self, command, *, timeout, log_path):
        try:
            if command[1].endswith('scripts/prepare_lclu.py'):
                self.prepare(dict(zip(command[2::2], command[3::2])))
            else:
                assert command[1] == str(shift.SCRIPT) and command[2] == 'vpp-child'
                args = dict(zip(command[3::2], command[4::2]))
                region = json.loads(Path(args['--config']).read_text())['qualified_regions'][0]
                self.vpp.append({'area': region['name'], 'config_years': region['years'],
                                 'center': (int(args['--east']), int(args['--north'])),
                                 'year': int(args['--vpp-year'])})
                assert shift.main(command[2:]) == 0
            log_path.write_text('done\n')
        except Exception:
            log_path.write_text(traceback.format_exc())
            raise subprocess.CalledProcessError(1, command) from None

    def prepare(self, args: dict) -> None:
        grid, year = Grid(int(args['--east']), int(args['--north'])), int(args['--year'])
        region = json.loads(Path(args['--config']).read_text())['qualified_regions'][0]
        assert region['years'] == [year]          # the service routes the exact year's source
        self.prepared.append({'area': region['name'], 'center': (grid.east, grid.north), 'year': year})
        output = Path(args['--output'])
        arrays = {'tessera': np.full((128, 512, 512), .125, np.float16),
                  'valid_embedding': np.ones((512, 512), bool),
                  'year': np.int32(year), 'tessera_year': np.int32(year), 'vpp_year': np.int32(year),
                  'crs': np.array('EPSG:3006'), 'transform': np.array(list(grid.transform())[:6]),
                  'easting': np.int32(grid.east), 'northing': np.int32(grid.north),
                  **synthetic_aux(grid, year)}
        arrays['valid_embedding'][HOLE, HOLE] = False
        np.savez_compressed(output / 'input.npz', **arrays)
        record = {**file_identity(output / 'input.npz'), 'path': 'input.npz'}
        validate_bundle(output / 'input.npz', record, grid, year)
        write_json(output / 'entry.json', {
            'year': year, 'center_3006': [grid.east, grid.north], 'public_area': True,
            'input': record, 'embedding': EMBEDDING,
            'source_config': {k: file_identity(Path(args['--config']))[k] for k in ('sha256', 'size')},
            'auxiliary_provenance': {'vpp': {'year': year, 'season': 1, 'index': {'sha256': 'a' * 64, 'size': 1},
                                             'files': vpp_files(year)}}})


class VppModel:
    """Forest above, crop below; class + 1 where spring starts on or after THRESHOLD."""
    contract = {'synthetic': True, 'checkpoint': {'sha256': 'c' * 64, 'size': 1}}

    def __init__(self):
        self.calls = []

    def predict(self, path, record):
        path = Path(path)
        assert file_identity(path)['sha256'] == record['sha256']
        scratch = next(p for p in path.parents if p.name.startswith('shard-'))
        inputs = scratch / 'lclu-inputs'
        assert sum(bool(shift.HEX64.fullmatch(p.name)) for p in inputs.iterdir()) == 1
        with np.load(path, allow_pickle=False) as data:
            digests = {name: shift.array_sha256(data[name]) for name in data.files}
            spring = np.mod(data['vpp_sosd'][4:508, 4:508], 1000)
            t = data['transform']
            years = {k: int(data[k]) for k in ('year', 'tessera_year', 'vpp_year')}
        self.calls.append({'center': (int(t[2]) + 2560, int(t[5]) - 2560), **years, 'digests': digests})
        rows = np.broadcast_to(np.arange(504)[:, None], (504, 504))
        base = np.where(rows < 252, 1, 11)
        return np.where(spring >= THRESHOLD, base + 1, base).astype(np.int64)


def wgs(geometry):
    return transform(Transformer.from_crs(3006, 4326, always_xy=True).transform, geometry)


def write_live(live: Path) -> Path:
    """A live release: one staged-COG sources config per area and year, and its serving manifest."""
    preparations = []
    for name, (polygon, years) in AREAS.items():
        w, s, e, n = polygon.bounds
        for year in years:
            path = live / 'regions' / shift.slug(name) / str(year) / f'sources-{year}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, {'version': 1, 'embedding': EMBEDDING,
                              'cog_index': {'path': f'cog-index-{year}.json', 'sha256': '0' * 64, 'size': 1},
                              'qualified_regions': [{
                                  'country': 'SE', 'name': name, 'verification': 'synthetic', 'years': [year],
                                  'bounds_3006': [w - 3700, s - 3700, e + 3700, n + 3700],
                                  'geometry': mapping(wgs(polygon))}]})
            preparations.append({'config': file_identity(path), 'python': 'unused', 'timeout_seconds': 10})
    manifest = live / 'demo' / 'serving.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {'version': 1, 'device': 'cpu',
                          'checkpoint': {'path': 'r4.pt', 'sha256': 'c' * 64, 'size': 1},
                          'preparations': preparations})
    return manifest


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob('*'))}


def with_arg(args: list[str], name: str, value: str) -> list[str]:
    i = args.index(name)
    return [*args[:i + 1], value, *args[i + 2:]]


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    """A live release with two areas, and a finished two-shard run against it."""
    base = tmp_path_factory.mktemp('vpp-shift')
    live = base / 'live'
    manifest = write_live(live)
    (live / 'results').mkdir()
    storage = StorageBudget(base / 'independent', max_bytes=2**30, min_free_bytes=1)
    sources = RegionalSources(json.loads(manifest.read_text())['preparations'], manifest.parent,
                              base / 'independent', storage=storage)
    independent = LcluService(VppModel(), sources, base / 'independent', storage=storage)
    pin = contract_pin(independent.identity)
    boundary, model = Boundary(), VppModel()
    args = ['run', '--run', 't1', '--manifest', str(manifest), '--expect-inference-pin', pin,
            '--output-root', str(base / 'experiments'), '--live-root', str(live)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr('imint.inference.lclu_sources.run_preparation', boundary)
        mp.setattr('scripts.prepare_lclu.prepare_aux', fake_prepare_aux)
        mp.setenv('IMINT_LCLU_STORAGE_MIN_FREE_BYTES', '1')
        mp.delenv('IMINT_API_OUTPUT_DIR', raising=False)
        before = snapshot(live)
        for index in (0, 1):
            assert shift.main([*args, '--shard', str(index), '--num-shards', '2'], model=model) == 0
        assert snapshot(live) == before
    root = base / 'experiments' / 't1'
    experiment = json.loads((root / 'experiment.json').read_text())
    return {'base': base, 'live': live, 'manifest': manifest, 'args': args, 'pin': pin, 'root': root,
            'boundary': boundary, 'model': model, 'experiment': experiment,
            'independent': independent,
            'records': {p['id']: json.loads((root / 'pairs' / f"{p['id']}.json").read_text())
                        for p in experiment['pairs']}}


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.setenv('IMINT_LCLU_STORAGE_MIN_FREE_BYTES', '1')
    monkeypatch.delenv('IMINT_API_OUTPUT_DIR', raising=False)
    monkeypatch.setattr('scripts.prepare_lclu.prepare_aux', fake_prepare_aux)
    child = Boundary()
    monkeypatch.setattr('imint.inference.lclu_sources.run_preparation', child)
    return child


def copy_run(setup, tmp_path) -> tuple[Path, list[str]]:
    """A private copy of the finished run, and CLI args pointing at it."""
    shutil.copytree(setup['root'], tmp_path / 'experiments' / 't1')
    args = with_arg(setup['args'], '--output-root', str(tmp_path / 'experiments'))
    return tmp_path / 'experiments' / 't1', args


# ── Plan ────────────────────────────────────────────────────────────────────

def test_pairs_are_the_catalogue_locations_times_2019_to_2024(setup):
    experiment, independent = setup['experiment'], setup['independent']
    locations = independent.sources.locations()
    assert [a['id'] for a in experiment['areas']] == ['asele-test', 'bjuv-test']
    for area, point in zip(experiment['areas'], locations, strict=True):
        grid = Grid.from_lonlat(point['lon'], point['lat'])        # what the page's click requests
        assert area['center_3006'] == [grid.east, grid.north]
    assert experiment['years'] == list(range(2019, 2025))
    assert len(experiment['pairs']) == 2 * 6
    assert all(p['vpp_years'] == [p['year'], p['year'] - 1] for p in experiment['pairs'])


@pytest.mark.parametrize('count', [1, 2, 4, 5])
def test_shards_partition_the_pairs(setup, count):
    pairs = setup['experiment']['pairs']
    parts = [shift.shard(pairs, i, count) for i in range(count)]
    assert sorted(p['id'] for part in parts for p in part) == sorted(p['id'] for p in pairs)
    with pytest.raises(ValueError, match='Shard index'):
        shift.shard(pairs, count, count)


def test_plan_reports_admission_without_writing(setup, capsys):
    before = snapshot(setup['live'])
    assert shift.main(['plan', '--manifest', str(setup['manifest']), '--num-shards', '4']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['pairs'] == 12 and report['admitted'] == 11
    assert report['not_admitted'] == ['bjuv-test-2019'] and report['shards'] == [3, 3, 3, 3]
    assert snapshot(setup['live']) == before


# ── The shift itself ────────────────────────────────────────────────────────

def test_only_hrvpp_differs_and_it_comes_from_year_y_minus_1(setup):
    """Hash the inputs the model actually read: HR-VPP and its year differ, nothing else."""
    vpp = {(*c['center'], c['year'], c['vpp_year']): c['digests'] for c in setup['model'].calls}
    compared = [p for p in setup['experiment']['pairs'] if setup['records'][p['id']]['status'] == 'compared']
    assert len(compared) == 10 and len(setup['model'].calls) == 3 * 10
    for pair in compared:
        grid, year = Grid(*pair['center_3006']), pair['year']
        normal, shifted = vpp[(grid.east, grid.north, year, year)], vpp[(grid.east, grid.north, year, year - 1)]
        differing = {name for name in normal if normal[name] != shifted[name]}
        assert differing == set(shift.VPP_NAMES) | {'vpp_year'}
        for name in shift.VPP_NAMES:
            assert normal[name] == shift.array_sha256(synthetic_aux(grid, year)[name])
            assert shifted[name] == shift.array_sha256(synthetic_aux(grid, year - 1)[name])
        record = setup['records'][pair['id']]
        assert record['inputs']['differing_arrays'] == sorted(differing)
        assert record['inputs']['array_sha256'] == {'normal': normal, 'shifted': shifted}
        assert record['inputs']['years']['shifted'] == {'year': year, 'tessera_year': year, 'vpp_year': year - 1}
        assert record['vpp_sources']['previous']['year'] == year - 1
        assert record['repeat_control'] == {'changed_pixels': 0}
    # The y-1 layers come from the y-1 qualified source of the same location, never another year.
    for read in setup['boundary'].vpp:
        assert read['config_years'] == [read['year']]


def test_the_normal_map_is_the_live_services_request(setup):
    pair = next(p for p in setup['experiment']['pairs'] if p['id'] == 'asele-test-2021')
    grid = Grid(*pair['center_3006'])
    assert setup['records'][pair['id']]['request_key'] == setup['independent'].key(grid, 2021)


@pytest.mark.parametrize('change, message', [
    ('tessera', 'differ outside HR-VPP'), ('static', 'differ outside HR-VPP'),
    ('same-vpp', 'identical for both years'), ('year', 'Unexpected input years')])
def test_check_shift_refuses_anything_but_a_pure_hrvpp_change(tmp_path, change, message):
    grid = Grid(X0 + 5000, Y0 + 5000)
    arrays = {'tessera': np.full((128, 8, 8), .125, np.float16), 'year': np.int32(2021),
              'tessera_year': np.int32(2021), 'vpp_year': np.int32(2021),
              **{k: v[:8, :8] for k, v in synthetic_aux(grid, 2021).items()}}
    np.savez(tmp_path / 'normal.npz', **arrays)
    vpp = {k: v[:8, :8] for k, v in synthetic_aux(grid, 2020).items() if k in shift.VPP_NAMES}
    shift.write_shifted(tmp_path / 'normal.npz', vpp, 2020, tmp_path / 'shifted.npz')
    assert shift.check_shift(tmp_path / 'normal.npz', tmp_path / 'shifted.npz', 2021, 2020)['differing_arrays'] == \
        sorted(shift.SHIFTED_ARRAYS)
    with np.load(tmp_path / 'shifted.npz') as data:
        changed = {k: data[k] for k in data.files}
    if change == 'tessera':
        changed['tessera'] = changed['tessera'] + np.float16(1)
    elif change == 'static':
        changed['dem'] = changed['dem'] + 1
    elif change == 'same-vpp':
        changed.update({k: arrays[k] for k in shift.VPP_NAMES})
    np.savez(tmp_path / 'shifted.npz', **changed)
    expected_previous = 2019 if change == 'year' else 2020     # y-2 layers labelled y-1 are refused
    with pytest.raises(ValueError, match=message):
        shift.check_shift(tmp_path / 'normal.npz', tmp_path / 'shifted.npz', 2021, expected_previous)


# ── Metrics ─────────────────────────────────────────────────────────────────

def test_metrics_on_a_known_toy_pair():
    normal = np.full((504, 504), NODATA, np.uint8)
    normal[:100], normal[100:200], normal[200:300] = 1, 11, 9      # forest, crop, other
    shifted = normal.copy()
    shifted[:10] = 2              # 5,040 forest pixels: tallskog -> granskog
    shifted[100:105] = 12         # 2,520 crop pixels: vete -> korn
    shifted[105] = 1              # 504 crop pixels: vete -> tallskog
    m = shift.compare_maps(normal, shifted)
    assert (m['valid_pixels'], m['changed_pixels']) == (151200, 8064)
    assert m['changed_share'] == pytest.approx(8064 / 151200)
    c = np.array(m['confusion'])
    assert (c[1, 2], c[1, 1], c[11, 12], c[11, 1], c[11, 11], c[9, 9]) == (5040, 45360, 2520, 504, 47376, 50400)
    assert c.sum() == 151200 and np.count_nonzero(c) == 6
    g = m['groups']
    assert (g['forest']['pixels'], g['forest']['changed']) == (50400, 5040)
    assert g['forest']['change_rate'] == pytest.approx(0.1)
    assert g['forest']['share_of_changed'] == pytest.approx(5040 / 8064)
    assert (g['crop']['pixels'], g['crop']['changed']) == (50400, 3024)
    assert g['crop']['change_rate'] == pytest.approx(0.06)
    assert g['crop']['share_of_changed'] == pytest.approx(3024 / 8064)
    assert (g['other']['changed'], g['other']['share_of_changed']) == (0, 0)
    assert m['top_transitions'] == [{'from': 1, 'to': 2, 'pixels': 5040}, {'from': 11, 'to': 12, 'pixels': 2520},
                                    {'from': 11, 'to': 1, 'pixels': 504}]
    unchanged = shift.compare_maps(normal, normal)
    assert unchanged['changed_pixels'] == 0 and unchanged['groups']['forest']['share_of_changed'] is None
    other_mask = shifted.copy()
    other_mask[400, 400] = 3
    with pytest.raises(ValueError, match='one validity mask'):
        shift.compare_maps(normal, other_mask)


def test_hrvpp_input_difference_on_a_known_toy_pair():
    valid = np.ones((504, 504), bool)
    valid[:4] = False
    normal = {name: np.zeros((512, 512), np.float32) for name in shift.VPP_NAMES}
    shifted = {name: np.zeros((512, 512), np.float32) for name in shift.VPP_NAMES}
    normal['vpp_sosd'][:], shifted['vpp_sosd'][:] = 24130, 23120          # day 130 vs day 120
    shifted['vpp_sosd'][8:10, 4:508] = 0                                  # y-1 nodata, 2 valid rows
    normal['vpp_maxv'][:], shifted['vpp_maxv'][:] = 0.9, 0.8
    d = shift.vpp_difference(normal, shifted, valid)
    sosd, maxv = d['vpp_sosd'], d['vpp_maxv']
    pixels = 500 * 504
    assert sosd['pixels'] == pixels and sosd['raw_unit'] == 'day of year'
    assert sosd['pixels_nodata_only_in_previous'] == 2 * 504 and sosd['pixels_nodata_only_in_y'] == 0
    assert sosd['pixels_with_data_both'] == pixels - 2 * 504
    assert sosd['mean_abs_diff_raw'] == pytest.approx(10) and sosd['mean_diff_raw'] == pytest.approx(10)
    # Scaled: training standard deviations (DOY std 31.11); nodata normalizes to the mean (DOY 129.40).
    expected = ((pixels - 1008) * 10 + 1008 * abs(130 - 129.40)) / pixels / 31.11
    assert sosd['mean_abs_diff_scaled'] == pytest.approx(expected, rel=1e-5)
    assert maxv['mean_abs_diff_raw'] == pytest.approx(0.1, rel=1e-5)
    assert maxv['mean_abs_diff_scaled'] == pytest.approx(0.1 / 0.57, rel=1e-5)
    assert d['vpp_length']['pixels_with_data_both'] == 0 and d['vpp_length']['mean_abs_diff_raw'] is None
    assert d['vpp_length']['mean_abs_diff_scaled'] == 0


def test_as_map_masks_nodata_and_rejects_foreign_classes():
    valid = np.ones((504, 504), bool)
    valid[0, 0] = False
    classes = shift.as_map(np.ones((504, 504), np.int64), valid)
    assert classes.dtype == np.uint8 and classes[0, 0] == NODATA and (classes[valid] == 1).all()
    with pytest.raises(ValueError, match='Invalid model classification'):
        shift.as_map(np.full((504, 504), 28, np.int64), valid)


def test_records_match_their_persisted_maps(setup):
    for pair in setup['experiment']['pairs']:
        record = setup['records'][pair['id']]
        if record['status'] != 'compared':
            assert not (setup['root'] / 'pairs' / f"{pair['id']}.npz").exists()
            continue
        with np.load(setup['root'] / 'pairs' / f"{pair['id']}.npz") as maps:
            assert shift.compare_maps(maps['normal'], maps['shifted']) == record['metrics']
    # Known synthetic result: in 2021, columns 256-383 of the input start spring two days later.
    m = setup['records']['asele-test-2021']['metrics']
    assert (m['valid_pixels'], m['changed_pixels']) == (504 * 504 - 36, 128 * 504)
    assert m['groups']['forest']['share_of_changed'] == m['groups']['crop']['share_of_changed'] == 0.5


# ── Missing y-1 HR-VPP ──────────────────────────────────────────────────────

def test_missing_previous_year_is_reported_not_filled(setup):
    records, boundary = setup['records'], setup['boundary']
    assert {pid: (r['missing'], r['reason']) for pid, r in records.items() if r['status'] == 'missing'} == {
        'bjuv-test-2019': ('previous_vpp', 'No qualified HR-VPP source for 2018 at this location'),
        'bjuv-test-2023': ('previous_vpp', 'HR-VPP 2022: No usable exact-year phenology in this window')}
    bjuv = Grid(*records['bjuv-test-2019']['pair']['center_3006'])
    # Exactly y-1 is read for each pair: nothing for a missing 2018 source, once for the
    # unusable 2022, and no other year in their place.
    reads = {area: sorted(r['year'] for r in boundary.vpp if r['area'] == area) for area in AREAS}
    assert reads == {'Åsele test': [2018, 2019, 2020, 2021, 2022, 2023],
                     'Bjuv test': [2019, 2020, 2021, 2022, 2023]}
    assert {r['center'] for r in boundary.vpp if r['area'] == 'Bjuv test'} == {(bjuv.east, bjuv.north)}
    # The year-y map of a missing pair is never prepared or classified.
    assert ('Bjuv test', 2019) not in [(p['area'], p['year']) for p in boundary.prepared]
    assert ('Bjuv test', 2023) not in [(p['area'], p['year']) for p in boundary.prepared]
    assert not [c for c in setup['model'].calls if c['center'] == (bjuv.east, bjuv.north)
                and c['year'] in (2019, 2023)]
    assert all(c['vpp_year'] in (c['year'], c['year'] - 1) for c in setup['model'].calls)


def test_vpp_missing_messages_are_prepare_aux_raises():
    from scripts.prepare_lclu import prepare_aux
    source = inspect.getsource(prepare_aux)
    for message in shift.VPP_MISSING:
        assert f"raise InputUnavailable('{message}')" in source


def test_vpp_child_writes_layers_or_the_missing_reason_only(tmp_path, monkeypatch):
    config = tmp_path / 'sources-2020.json'
    write_json(config, {'qualified_regions': [{'name': 'Åsele test', 'years': [2020]}]})
    sha = file_identity(config)['sha256']
    grid = Grid(X0 + 5000, Y0 + 5000)
    monkeypatch.setattr('scripts.prepare_lclu.prepare_aux', fake_prepare_aux)
    args = ['vpp-child', '--config', str(config), '--config-sha256', sha, '--east', str(grid.east),
            '--north', str(grid.north), '--output']
    (tmp_path / 'ok').mkdir()
    assert shift.main([*args, str(tmp_path / 'ok'), '--vpp-year', '2020']) == 0
    read = shift.read_vpp(tmp_path / 'ok', 2020)
    assert read['provenance']['year'] == 2020
    assert all(np.array_equal(read['arrays'][n], synthetic_aux(grid, 2020)[n]) for n in shift.VPP_NAMES)
    with pytest.raises(ValueError, match='another year'):
        shift.read_vpp(tmp_path / 'ok', 2019)
    (tmp_path / 'missing').mkdir()
    assert shift.main([*args, str(tmp_path / 'missing'), '--vpp-year', '2019']) == 0
    assert shift.read_vpp(tmp_path / 'missing', 2019) == {'missing': 'Exact-year HR-VPP source coverage is incomplete'}
    assert sorted(p.name for p in (tmp_path / 'missing').iterdir()) == ['missing.json']

    def other(*_):
        raise InputUnavailable('No Swedish forestry coverage for this location')
    monkeypatch.setattr('scripts.prepare_lclu.prepare_aux', other)
    (tmp_path / 'other').mkdir()
    with pytest.raises(InputUnavailable, match='forestry'):
        shift.main([*args, str(tmp_path / 'other'), '--vpp-year', '2020'])
    with pytest.raises(ValueError, match='differs from the serving manifest'):
        shift.main([*with_arg(args, '--config-sha256', '0' * 64), str(tmp_path / 'other'), '--vpp-year', '2020'])


# ── Storage and live roots ─────────────────────────────────────────────────

def test_inputs_are_deleted_after_every_pair(setup):
    for scratch in (setup['root'] / 'scratch').iterdir():
        assert not (scratch / 'vpp-shift').exists()
        inputs = scratch / 'lclu-inputs'
        assert not [p for p in inputs.rglob('*') if p.is_file()], 'a prepared input or log was kept'
        assert not [p for p in (scratch / 'lclu').iterdir()]


def test_output_never_lands_in_a_live_root(setup, tmp_path, boundary, monkeypatch):
    live, manifest = setup['live'], setup['manifest']
    for output in (live / 'experiments', live, live.parent, manifest.parent / 'x'):
        with pytest.raises(ValueError, match='overlaps the live service root'):
            shift.main(with_arg(setup['args'], '--output-root', str(output)), model=VppModel())
    # The manifest's directory is protected even when not named as a live root.
    other = with_arg(setup['args'], '--live-root', str(tmp_path / 'elsewhere'))
    with pytest.raises(ValueError, match='overlaps'):
        shift.main(with_arg(other, '--output-root', str(manifest.parent / 'experiments')), model=VppModel())
    monkeypatch.setenv('IMINT_API_OUTPUT_DIR', str(tmp_path / 'api'))
    with pytest.raises(ValueError, match='overlaps'):
        shift.main(with_arg(other, '--output-root', str(tmp_path / 'api' / 'x')), model=VppModel())
    with pytest.raises(ValueError, match='overlaps'):
        shift.main(['summarize', '--run', 't1', '--output-root', str(live), '--live-root', str(live)])
    assert not (live / 'experiments').exists() and not (manifest.parent / 'experiments').exists()
    assert not (tmp_path / 'api').exists() and boundary.prepared == boundary.vpp == []


def test_inference_pin_mismatch_refuses_before_any_pair(setup, tmp_path, boundary):
    args = with_arg(setup['args'], '--output-root', str(tmp_path / 'experiments'))
    with pytest.raises(ValueError, match='is not the expected'):
        shift.main(with_arg(args, '--expect-inference-pin', 'f' * 64), model=VppModel())
    assert boundary.prepared == boundary.vpp == []
    assert not (tmp_path / 'experiments' / 't1' / 'experiment.json').exists()


def test_resume_skips_verified_and_reruns_corrupt_or_missing(setup, tmp_path, boundary):
    root, args = copy_run(setup, tmp_path)
    model = VppModel()
    assert shift.main([*args, '--shard', '0', '--num-shards', '1'], model=model) == 0
    assert model.calls == [] and boundary.prepared == boundary.vpp == []      # missing pairs are final
    corrupt = root / 'pairs' / 'asele-test-2022.npz'
    data = corrupt.read_bytes()
    corrupt.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    (root / 'pairs' / 'bjuv-test-2020.json').unlink()
    model = VppModel()
    assert shift.main([*args, '--shard', '0', '--num-shards', '1'], model=model) == 0
    assert sorted({(c['center'], c['year']) for c in model.calls}) == sorted(
        (tuple(setup['records'][pid]['pair']['center_3006']), year)
        for pid, year in (('asele-test-2022', 2022), ('bjuv-test-2020', 2020)))
    for pid in ('asele-test-2022', 'bjuv-test-2020'):
        record = json.loads((root / 'pairs' / f'{pid}.json').read_text())
        assert record['metrics'] == setup['records'][pid]['metrics']


def test_a_failing_pair_is_recorded_and_fails_the_shard(setup, tmp_path, boundary, monkeypatch):
    root, args = copy_run(setup, tmp_path)
    (root / 'pairs' / 'asele-test-2024.json').unlink()

    def broken(config, base, grid, year):
        raise OSError(5, 'Input/output error')
    monkeypatch.setattr('scripts.prepare_lclu.prepare_aux', broken)
    assert shift.main([*args, '--shard', '0', '--num-shards', '1'], model=VppModel()) == 1
    failure = json.loads((root / 'failed' / 'asele-test-2024.json').read_text())
    assert failure['error'] == 'RuntimeError' and 'HR-VPP 2023 read failed' in failure['message']
    assert 'Input/output error' in (root / 'operator' / 'preparation-logs' /
                                    'asele-test-2024.previous-vpp.log').read_text()
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['complete'] is False and summary['pending'] == ['asele-test-2024']
    assert not (root / 'scratch' / 'shard-0-of-1' / 'vpp-shift').exists()


def test_a_changed_manifest_refuses_the_existing_run(setup, tmp_path, boundary):
    root, args = copy_run(setup, tmp_path)
    experiment = json.loads((root / 'experiment.json').read_text())
    write_json(root / 'experiment.json', {**experiment, 'years': [2019]})
    with pytest.raises(ValueError, match='new release name'):
        shift.main([*args, '--shard', '0', '--num-shards', '1'], model=VppModel())


# ── Summary ─────────────────────────────────────────────────────────────────

def test_summary_table_and_aggregate(setup):
    summary = json.loads((setup['root'] / 'summary.json').read_text())
    records = setup['records']
    assert summary['complete'] is True
    assert summary['pairs'] == {'planned': 12, 'compared': 10, 'missing': 2, 'pending': 0}
    table = summary['changed_share_table']
    assert table['Bjuv test']['2019'] == table['Bjuv test']['2023'] == 'missing'
    assert table['Åsele test']['2021'] == records['asele-test-2021']['metrics']['changed_share']
    shares = [r['metrics']['changed_share'] for r in records.values() if r['status'] == 'compared']
    aggregate = summary['aggregate']
    assert aggregate['changed_share']['median'] == statistics.median(shares)
    assert aggregate['changed_share']['max'] == max(shares)
    assert records[aggregate['changed_share']['max_pair']]['metrics']['changed_share'] == max(shares)
    # Spring starts later in y, so the y map holds the + 1 classes the y-1 map loses.
    moving = aggregate['classes_moving_most']
    assert {c['class'] for c in moving} == {2, 12}
    assert {c['main_destination']['class'] for c in moving} == {1, 11}
    assert summary['repeat_control']['max_changed_pixels'] == 0
    assert aggregate['vpp_input_difference']['vpp_sosd']['mean_abs_diff_raw']['median'] == pytest.approx(2)
    markdown = (setup['root'] / 'summary.md').read_text()
    assert '| Area | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |' in markdown
    assert re.search(r'\| Bjuv test \| missing \| [0-9.]+ % \|', markdown)
    assert 'not accuracy' in markdown and 'Pending' not in markdown


def test_summarize_rebuilds_from_records_only(setup, tmp_path):
    root, _ = copy_run(setup, tmp_path)
    (root / 'summary.json').unlink()
    (root / 'summary.md').unlink()
    assert shift.main(['summarize', '--run', 't1', '--output-root', str(tmp_path / 'experiments'),
                       '--live-root', str(setup['live'])]) == 0
    assert json.loads((root / 'summary.json').read_text())['pairs'] == {
        'planned': 12, 'compared': 10, 'missing': 2, 'pending': 0}


def test_nothing_on_the_path_reaches_cdse():
    code = '''
import importlib, sys
importlib.import_module('scripts.compare_vpp_year_shift')
importlib.import_module('scripts.prepare_lclu')
assert 'imint.data.cdse_vpp' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, env={**os.environ, 'PYTHONPATH': str(ROOT)})


# ── Job ─────────────────────────────────────────────────────────────────────

def _job() -> dict:
    yaml = pytest.importorskip('yaml')
    (job,) = yaml.safe_load_all((ROOT / 'k8s/lclu-vpp-year-shift-job.yaml').read_text())
    return job


def _coverage_jobs() -> dict:
    yaml = pytest.importorskip('yaml')
    return {doc['metadata']['name']: doc
            for doc in yaml.safe_load_all((ROOT / 'k8s/lclu-coverage-2024-job.yaml').read_text())}


def _value(script: str, name: str) -> str:
    return re.search(rf'^\s*{name}=(\S+)$', script, re.M)[1]


def test_job_runs_the_shards_it_declares_on_r10s_runtime():
    job = _job()
    spec, pod = job['spec'], job['spec']['template']['spec']
    (container,) = pod['containers']
    script = container['command'][-1]
    assert job['metadata']['name'] == 'lclu-vpp-year-shift' and job['kind'] == 'Job'
    assert spec['completionMode'] == 'Indexed' and spec['completions'] == spec['parallelism'] == 4
    assert spec['backoffLimit'] == 0 and int(_value(script, 'SHARDS')) == spec['completions']
    assert container['resources']['requests'] == container['resources']['limits'] == {'cpu': '4', 'memory': '8Gi'}
    coverage = _coverage_jobs()['lclu-coverage-2024']['spec']['template']['spec']
    coverage_script = coverage['containers'][0]['command'][-1]
    base = re.search(r'^FROM (\S+)$', (ROOT / 'docker/lclu/Dockerfile').read_text(), re.M)[1]
    assert container['image'] == coverage['containers'][0]['image'] == base and '@sha256:' in base
    for name in ('PIN', 'LIVE'):
        assert _value(script, name) == _value(coverage_script, name)
    assert _value(script, 'LIVE') == '/work/releases/lclu-stockholm-0124-r10'
    assert 'scripts/compare_vpp_year_shift.py run' in script
    assert '--shard "$JOB_COMPLETION_INDEX" --num-shards "$SHARDS"' in script
    assert '--expect-inference-pin "$PIN"' in script and '--manifest "$LIVE/demo/serving.json"' in script
    assert '--output-root /work/experiments/vpp-year-shift' in script and script.count('--live-root') == 3
    for line in ('LOCK=$(sha256sum docker/lclu/requirements.lock', 'test -f "/work/runtime/$LOCK/READY"'):
        assert line in script and line in coverage_script
    assert 'apt-get' not in script and 'pip install' not in script
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)
    (init,) = pod['initContainers']
    (coverage_init,) = coverage['initContainers']
    source = init['command'][-1]
    sha = _value(source, 'SHA')
    assert source.replace(f'SHA={sha}', 'SHA=X') == coverage_init['command'][-1].replace(
        f"SHA={_value(coverage_init['command'][-1], 'SHA')}", 'SHA=X')
    assert {k: v for k, v in init.items() if k != 'command'} == {
        k: v for k, v in coverage_init.items() if k != 'command'}
    for c in pod['containers'] + pod['initContainers']:
        assert c['securityContext'] == {'allowPrivilegeEscalation': False, 'capabilities': {'drop': ['ALL']}}
    mounts = {m['name']: m for m in container['volumeMounts']}
    assert mounts['source']['readOnly'] and mounts['data']['readOnly']
    claims = {v['name']: v.get('persistentVolumeClaim') for v in pod['volumes']}
    assert claims['work'] == {'claimName': 'lclu-stockholm-0124'} and claims['data']['readOnly']


def test_job_refuses_to_run_before_its_source_is_pinned():
    source = _job()['spec']['template']['spec']['initContainers'][0]['command'][-1]
    sha = _value(source, 'SHA')
    if sha != '__SOURCE_COMMIT__':
        assert re.fullmatch(r'[0-9a-f]{40}', sha)
        git = ['git', '-C', str(ROOT)]
        if subprocess.run([*git, 'cat-file', '-e', sha], capture_output=True).returncode:
            pytest.skip(f'{sha} not in this clone')
        assert subprocess.run([*git, 'merge-base', '--is-ancestor', sha, 'HEAD']).returncode == 0
        changed = subprocess.run([*git, 'diff', '--name-only', sha, 'HEAD', '--',
                                  'scripts/compare_vpp_year_shift.py', 'scripts/produce_lclu_coverage.py'],
                                 capture_output=True, text=True, check=True).stdout.split()
        assert changed == [], f'bump SHA in k8s/lclu-vpp-year-shift-job.yaml; changed: {changed}'
        return
    # The guard is the script's first check, before any directory or network access.
    done = subprocess.run(['sh', '-c', source], capture_output=True, text=True)
    assert done.returncode == 1 and 'FATAL: pin SHA to a commit' in done.stdout


def test_job_reserve_keeps_the_live_services_reserve_free():
    spec = importlib.util.spec_from_file_location('render_job', ROOT / 'docker/lclu/render_job.py')
    render_job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render_job)
    live = render_job.resources('python@sha256:' + '0' * 64, 2024)['items'][-1]
    env = {e['name']: e['value'] for e in live['spec']['template']['spec']['containers'][0]['env']}
    job = _job()
    pods = job['spec']['completions']
    need = int(env['IMINT_LCLU_STORAGE_MIN_FREE_BYTES']) + pods * 3 * PREPARATION_BYTES
    mine = {e['name']: e['value'] for e in job['spec']['template']['spec']['containers'][0]['env']}
    assert int(mine['IMINT_LCLU_STORAGE_MIN_FREE_BYTES']) >= max(need, 6 * 2**30)


def test_identity_files_equal_the_live_release_source():
    """The Job's PIN is r10's: every file of the inference identity is unchanged since r10's source."""
    from imint.inference.lclu import implementation_identity
    source = _coverage_jobs()['lclu-coverage-2024']['spec']['template']['spec']['initContainers'][0]
    sha = _value(source['command'][-1], 'SHA')
    git = ['git', '-C', str(ROOT)]
    if subprocess.run([*git, 'cat-file', '-e', sha], capture_output=True).returncode:
        pytest.skip(f'{sha} not in this clone')
    for relative, digest in implementation_identity().items():
        blob = subprocess.run([*git, 'show', f'{sha}:{relative}'], capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == digest, f'{relative} differs from r10; the PIN would not match'
    deleted = subprocess.run([*git, 'diff', '--name-only', '--diff-filter=D', sha, '--', 'imint/training', 'imint/fm'],
                             capture_output=True, text=True, check=True).stdout.split()
    assert deleted == []
