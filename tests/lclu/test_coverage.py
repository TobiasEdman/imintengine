"""Heltäckande markslagskarta 2024: batch producer on synthetic windows.

The real service path runs (RegionalSources, RepositorySources._prepare and
its failure wrapping, LcluService.run, export_map, StorageBudget); only the
model and the preparation child process (the data boundary) are synthetic.
No R4 weights, no source data, no claim about map quality.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from imint.inference.lclu import (
    AUX_NAMES, EMBEDDING, Grid, InputUnavailable, LcluService, NODATA, public_identity,
    validate_bundle, write_json,
)
from imint.inference.lclu_legend import MAP_COLORS
from imint.inference.lclu_sources import RegionalSources
from imint.inference.lclu_storage import MAP_BYTES, PREPARATION_BYTES, StorageBudget, StorageLimitError
from scripts import produce_lclu_coverage as cov
from scripts.cache_predict import contract_pin, file_identity

ROOT = Path(__file__).resolve().parents[2]
# Two lattice cells of 5,040 m from the EPSG:3006 origin near Stockholm.
X0, Y0 = 130 * cov.CELL, 1300 * cov.CELL
# Edges deliberately off the 10 m lattice, so no centre or corner lies on one.
AREA_A = box(X0 + 3003, Y0 + 1007, X0 + 13004, Y0 + 8996)
AREA_B = box(X0 + 13004, Y0 + 1507, X0 + 18006, Y0 + 4493)   # touches A's east edge
# Lattice cell (2, 1) is "open water": no valid TESSERA pixel after the county mask.
WATER = (X0 + 2 * cov.CELL + cov.HALF, Y0 + cov.CELL + cov.HALF)
# Real plan counts (docs/plans/lclu_2024_coverage.md): windows intersecting each polygon.
PLAN_COUNTS = {'0380': 119, '0581': 120, '0680': 109, '1880': 95, '0580': 86, '1980': 67,
               '1480': 62, '1283': 32, '1280': 24, '0180': 21, 'kiruna_tatort': 4}


def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A seamless synthetic class map of world position, irregular class boundaries."""
    return (1 + (np.floor(x / 130 + 2.3 * np.sin(y / 410))
                 + np.floor(y / 170 + 1.7 * np.cos(x / 530))) % 6).astype(np.uint8)


def world(west: float, north: float, height: int, width: int):
    x = west + 10 * np.arange(width)[None, :] + 5
    y = north - 10 * np.arange(height)[:, None] - 5
    return np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width))


class FieldModel:
    """Predicts the world field; asserts inputs are deleted between windows."""
    contract = {'synthetic': True, 'checkpoint': {'sha256': 'c' * 64, 'size': 1}}

    def __init__(self):
        self.calls = []

    def predict(self, path, record):
        inputs = path.parent.parent
        assert inputs.name == 'lclu-inputs'
        assert sum(bool(cov.HEX64.fullmatch(p.name)) for p in inputs.iterdir()) == 1
        with np.load(path, allow_pickle=False) as data:
            t = data['transform']
        self.calls.append((int(t[2]) + 2560, int(t[5]) - 2560))
        return field(*world(t[2] + 40, t[5] - 40, 504, 504)).astype(np.int64)


class Child:
    """Stands in for the scripts/prepare_lclu.py child: same package, exit status and log.

    It ends with the child's own final check (validate_bundle), so a window
    without valid pixels fails exactly as the real child does.
    """

    def __init__(self, invalid=(), transient=()):
        self.invalid, self.transient, self.calls = set(invalid), set(transient), []

    def __call__(self, command, *, timeout, log_path):
        args = dict(zip(command[2::2], command[3::2]))
        grid, year = Grid(int(args['--east']), int(args['--north'])), int(args['--year'])
        output = Path(args['--output'])
        self.calls.append((grid.east, grid.north))
        try:
            if (grid.east, grid.north) in self.transient:
                raise OSError(5, 'Input/output error')
            arrays = {'tessera': np.full((128, 512, 512), .125, np.float16),
                      'valid_embedding': np.ones((512, 512), bool),
                      'year': np.int32(year), 'tessera_year': np.int32(year), 'vpp_year': np.int32(year),
                      'crs': np.array('EPSG:3006'), 'transform': np.array(list(grid.transform())[:6]),
                      'easting': np.int32(grid.east), 'northing': np.int32(grid.north)}
            for name in AUX_NAMES:
                arrays[name] = np.ones((512, 512), np.float32)
            arrays['vpp_sosd'][:] = 24130
            arrays['vpp_eosd'][:] = 24260
            arrays['valid_embedding'][250:256, 250:256] = False    # model nodata inside each window
            if (grid.east, grid.north) in self.invalid:
                arrays['valid_embedding'][:] = False
            np.savez_compressed(output / 'input.npz', **arrays)
            record = {**file_identity(output / 'input.npz'), 'path': 'input.npz'}
            validate_bundle(output / 'input.npz', record, grid, year)
            write_json(output / 'entry.json', {
                'year': year, 'center_3006': [grid.east, grid.north], 'public_area': True,
                'input': record, 'embedding': EMBEDDING,
                'source_config': public_identity(file_identity(Path(args['--config']))),
                'auxiliary_provenance': {'vpp': {'year': year, 'season': 1,
                                                 'index': {'sha256': 'a' * 64, 'size': 1}, 'files': []}}})
            log_path.write_text('prepared\n')
        except Exception:
            log_path.write_text(traceback.format_exc())
            raise subprocess.CalledProcessError(1, command) from None


def wgs(geometry):
    return transform(Transformer.from_crs(3006, 4326, always_xy=True).transform, geometry)


def write_live(live: Path, configs: dict) -> Path:
    """A live release: one staged-COG sources config per area and its serving manifest."""
    preparations = []
    for code, config in configs.items():
        path = live / 'regions' / code / 'sources-2024.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, config)
        preparations.append({'config': file_identity(path), 'python': 'unused', 'timeout_seconds': 10})
    manifest = live / 'demo' / 'serving.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {'version': 1, 'device': 'cpu',
                          'checkpoint': {'path': 'r4.pt', 'sha256': 'c' * 64, 'size': 1},
                          'preparations': preparations})
    return manifest


def live_service(manifest: Path, root: Path) -> LcluService:
    """An independent service on the same manifest, as the live API would build it."""
    storage = StorageBudget(root, max_bytes=2**30, min_free_bytes=1)
    sources = RegionalSources(json.loads(manifest.read_text())['preparations'], manifest.parent, root,
                              storage=storage)
    return LcluService(FieldModel(), sources, root, storage=storage)


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*'))}


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    """A two-area campaign, a live release serving it, and a finished coverage release."""
    base = tmp_path_factory.mktemp('coverage')
    areas, blocks, configs = [], [], {}
    for code, polygon in (('a', AREA_A), ('b', AREA_B)):
        w, s, e, n = polygon.bounds
        bounds = [w - 3700, s - 3700, e + 3700, n + 3700]
        feature = {'type': 'Feature', 'properties': {'code': code, 'name': f'Synthetic {code}'},
                   'geometry': mapping(wgs(polygon))}
        block = {'version': '1.0', 'variant': 'vultr', 'year': 2024, 'lon': 18.05 if code == 'a' else 18.15,
                 'lat': 59.15, 'grid_size': 200, 'scales_size': 200}
        blocks.append(block)
        areas.append({'id': code, 'bounds_3006': bounds, 'buffer_m': 3700, 'municipality': feature,
                      'block_keys': [f"2024_{block['lon']:.2f}_{block['lat']:.2f}"]})
        configs[code] = {'version': 1, 'embedding': EMBEDDING,
                         'cog_index': {'path': 'cog-index-2024.json', 'sha256': '0' * 64, 'size': 1},
                         'qualified_regions': [{
                             'country': 'SE', 'name': f'Synthetic {code}', 'verification': 'synthetic',
                             'years': [2024], 'bounds_3006': bounds, 'geometry': feature['geometry']}]}
    campaign = base / 'campaign.json'
    write_json(campaign, {'version': 1, 'years': [2024], 'areas': areas, 'blocks': blocks})
    live = base / 'live'
    manifest = write_live(live, configs)
    (live / 'results').mkdir()
    (live / 'results' / '.api-worker.lock').write_text('')
    pin = contract_pin(live_service(manifest, base / 'independent').identity)
    child = Child(invalid={WATER})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr('imint.inference.lclu_sources.run_preparation', child)
        mp.setenv('IMINT_LCLU_STORAGE_MIN_FREE_BYTES', '1')
        mp.delenv('IMINT_API_OUTPUT_DIR', raising=False)
        before = snapshot(live)
        model = FieldModel()
        args = ['--config', str(campaign), '--manifest', str(manifest), '--live-root', str(live),
                '--output-root', str(base / 'coverage'), '--release', 't1', '--run-id', '1',
                '--expect-inference-pin', pin]
        assert cov.main([*args, '--shard', '0', '--num-shards', '2'], model=model) == 0
        assert cov.main([*args, '--shard', '1', '--num-shards', '2'], model=model) == 0
        assert cov.main([*args, '--num-shards', '2', '--mosaic']) == 0
        assert snapshot(live) == before
    return {'base': base, 'campaign': campaign, 'manifest': manifest, 'live': live, 'args': args,
            'configs': configs, 'pin': pin, 'model': model, 'child': child,
            'release': base / 'coverage' / '2024' / 't1', 'plan': cov.build_plan(campaign)}


def copy_release(setup, tmp_path) -> tuple[Path, list[str]]:
    """A private copy of the finished release, and CLI args pointing at it."""
    output = tmp_path / 'coverage'
    shutil.copytree(setup['release'], output / '2024' / 't1')
    args = [a if a != str(setup['base'] / 'coverage') else str(output) for a in setup['args']]
    return output / '2024' / 't1', args


def with_arg(args: list[str], name: str, value: str) -> list[str]:
    i = args.index(name)
    return [*args[:i + 1], value, *args[i + 2:]]


@pytest.fixture
def boundary(monkeypatch):
    """Install a synthetic preparation child for one test."""
    monkeypatch.setenv('IMINT_LCLU_STORAGE_MIN_FREE_BYTES', '1')
    monkeypatch.delenv('IMINT_API_OUTPUT_DIR', raising=False)

    def install(child):
        monkeypatch.setattr('imint.inference.lclu_sources.run_preparation', child)
        return child
    return install


# ── Plan ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def real_plan():
    return cov.build_plan(ROOT / 'config/lclu/municipalities.json')


def test_real_lattice_matches_the_plan_counts(real_plan):
    s = cov.summary(real_plan)
    assert {k: v['lattice'] for k, v in s['areas'].items()} == PLAN_COUNTS
    assert sum(PLAN_COUNTS.values()) == 739
    # 13 windows straddle Linköping/Norrköping; each runs once.
    shared = [w for w in real_plan['windows'] if w['kind'] == 'lattice' and len(w['areas']) > 1]
    assert len(shared) == 13 and {tuple(w['areas']) for w in shared} == {('0580', '0581')}
    assert s['total']['lattice'] == 739 - 13 == 726
    # The service admits only centres inside a polygon: fills cover what remains.
    assert (s['total']['admitted'], s['total']['refused']) == (490, 236)
    assert s['areas']['kiruna_tatort']['admitted'] == 0 and s['areas']['kiruna_tatort']['fill'] > 0
    assert all(v['unreachable_pixels'] == 0 for v in s['areas'].values())
    assert s['total']['runnable'] == 490 + s['total']['fill']


def test_real_lattice_is_national_and_windows_are_service_grids(real_plan):
    for w in real_plan['windows']:
        grid = Grid(w['east'], w['north'])
        b = grid.bounds(cov.OUTPUT_SIZE)
        assert (b['east'] - b['west'], b['north'] - b['south']) == (cov.CELL, cov.CELL)
        assert (b['west'], b['north']) == (w['east'] - cov.HALF, w['north'] + cov.HALF)
        if w['kind'] == 'lattice':
            assert b['west'] % cov.CELL == 0 and b['south'] % cov.CELL == 0
        else:
            assert w['east'] % 10 == 0 and w['north'] % 10 == 0


@pytest.mark.parametrize('count', [1, 3, 8])
def test_shards_partition_the_runnable_windows(real_plan, count):
    shards = [cov.shard(real_plan, i, count) for i in range(count)]
    ids = [w['id'] for part in shards for w in part]
    assert len(ids) == len(set(ids))
    assert set(ids) == {w['id'] for w in cov.runnable(real_plan)}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    with pytest.raises(ValueError):
        cov.shard(real_plan, count, count)


def test_window_slices_one_helper_for_every_frame():
    east, north = X0 + cov.HALF, Y0 + cov.HALF          # window spans [X0, X0+5040) x (Y0, Y0+5040]
    frame, local = cov.window_slices(east, north, X0 + 100, Y0 + 5000, 10, 600)
    assert frame == (slice(0, 10), slice(0, 494)) and local == (slice(4, 14), slice(10, 504))
    frame, local = cov.window_slices(east, north, X0 - 30, Y0 + 5040, 600, 20)
    assert frame == (slice(0, 504), slice(3, 20)) and local == (slice(0, 504), slice(0, 17))
    assert cov.window_slices(east, north, X0 + cov.CELL, Y0 + cov.CELL, 5, 5) is None
    assert cov.window_slices(east, north, X0, Y0, 5, 5) is None          # frame wholly south


def test_synthetic_plan_admission_and_fills(setup):
    plan = setup['plan']
    lattice = {w['id']: w for w in plan['windows'] if w['kind'] == 'lattice'}
    assert len(lattice) == 7
    shared = lattice['L0132-1300']                 # cell (2, 0): A and B, centred in A
    assert shared['areas'] == ['a', 'b'] and shared['region'] == 'a'
    refused = sorted(w['id'] for w in lattice.values() if w['region'] is None)
    assert refused == ['L0130-1300', 'L0130-1301']  # centre x = 2,520 m < A's west edge
    fills = [w for w in plan['windows'] if w['kind'] == 'fill']
    assert len(fills) == 2 and all(w['region'] == 'a' for w in fills)
    assert all(AREA_A.contains(box(w['east'] - 1, w['north'] - 1, w['east'] + 1, w['north'] + 1))
               for w in fills)


# ── Production ──────────────────────────────────────────────────────────────

def test_every_runnable_window_ran_once_and_inputs_were_deleted(setup):
    release, plan = setup['release'], setup['plan']
    produced = [w for w in cov.runnable(plan) if (w['east'], w['north']) != WATER]
    assert sorted(setup['model'].calls) == sorted((w['east'], w['north']) for w in produced)
    for w in cov.runnable(plan):
        record = cov.verify_sidecar(release, w, setup['pin'])
        assert record is not None and record['year'] == 2024
    leftovers = [p for p in (release / 'scratch').rglob('*')
                 if cov.HEX64.fullmatch(p.name) or p.name.startswith(('.preparing-', '.pending-'))]
    assert leftovers == []
    assert sorted(p.name for p in (release / 'scratch').iterdir()) == ['shard-0-of-2', 'shard-1-of-2']


def test_window_without_valid_pixels_is_unavailable_not_failed(setup, tmp_path):
    release, plan = setup['release'], setup['plan']
    water = next(w for w in plan['windows'] if (w['east'], w['north']) == WATER)
    record = cov.verify_sidecar(release, water, setup['pin'])
    assert record['status'] == 'unavailable' and record['deterministic'] is True
    assert record['cause'] == cov.UNAVAILABLE_CAUSE == (
        'imint.inference.lclu.InputUnavailable: TESSERA saknar giltiga pixlar för platsen och året.')
    assert record['request_key'] == live_service(setup['manifest'], tmp_path / 'live').key(
        Grid(*WATER), 2024)
    assert [len(a['logs']) for a in record['attempts']] == [1, 1]      # prepared twice, same cause
    assert setup['child'].calls.count(WATER) == 2
    assert not (release / 'windows' / water['id']).exists()
    assert not list((release / 'failed').iterdir())
    markers = [json.loads(p.read_text()) for p in sorted((release / 'done').iterdir())]
    assert [m['unavailable'] for m in markers].count([water['id']]) == 1
    assert all(m['failed'] == [] and m['aborted'] is None for m in markers)


def test_only_the_specific_cause_is_deterministic(tmp_path):
    def wrapped(text):
        log = tmp_path / f'{len(list(tmp_path.iterdir()))}.log'
        log.write_text(text)
        try:
            raise RuntimeError('Annual preparation failed') from subprocess.CalledProcessError(1, ['x'])
        except RuntimeError as exc:
            return exc, [log]

    assert cov.deterministic_cause(InputUnavailable(cov.NO_VALID_PIXELS), []) == cov.UNAVAILABLE_CAUSE
    assert cov.deterministic_cause(InputUnavailable('Missing exact-year COG blocks'), []) is None
    assert cov.deterministic_cause(*wrapped(f'Traceback ...\n{cov.UNAVAILABLE_CAUSE}\n')) == cov.UNAVAILABLE_CAUSE
    assert cov.deterministic_cause(*wrapped('Traceback ...\nOSError: [Errno 5] Input/output error\n')) is None
    exc, logs = wrapped(f'{cov.UNAVAILABLE_CAUSE}\n')
    assert cov.deterministic_cause(exc, logs + logs) is None                 # ambiguous logs
    assert cov.deterministic_cause(RuntimeError(cov.UNAVAILABLE_CAUSE), logs) is None   # no child exit


def test_window_carries_the_live_services_request_key(setup, tmp_path):
    """An independent service on the same manifest computes the same key: on-demand provenance."""
    live = live_service(setup['manifest'], tmp_path / 'other')
    for w in cov.runnable(setup['plan']):
        record = json.loads((setup['release'] / 'windows' / f"{w['id']}.json").read_text())
        assert record['request_key'] == live.key(Grid(w['east'], w['north']), 2024)


def test_inference_pin_mismatch_refuses_before_any_window(setup, tmp_path, boundary):
    child = boundary(Child())
    model = FieldModel()
    args = with_arg(with_arg(setup['args'], '--expect-inference-pin', '0' * 64),
                    '--output-root', str(tmp_path / 'coverage'))
    with pytest.raises(ValueError, match='is not the expected'):
        cov.main(args, model=model)
    assert child.calls == [] and model.calls == []
    assert not list((tmp_path / 'coverage' / '2024' / 't1' / 'windows').glob('*'))


def test_plan_admission_must_match_the_manifest_sources(setup, tmp_path, boundary):
    child = boundary(Child())
    configs = json.loads(json.dumps(setup['configs']))
    configs['b']['qualified_regions'][0]['years'] = [2023]     # b no longer offers 2024
    manifest = write_live(tmp_path / 'live', configs)
    args = with_arg(with_arg(setup['args'], '--manifest', str(manifest)), '--output-root',
                    str(tmp_path / 'coverage'))
    args = with_arg(args, '--live-root', str(tmp_path / 'live'))
    with pytest.raises(ValueError, match=r"1 windows: plan admission differs .*'L0133-1300'"):
        cov.main(args, model=FieldModel())
    assert child.calls == []
    with pytest.raises(ValueError, match='plan admission differs'):
        cov.check_manifest(manifest, setup['plan'], 2024)
    cov.check_manifest(setup['manifest'], setup['plan'], 2024)


@pytest.mark.parametrize('change', ['remote_tiles', 'no_cog_index'])
def test_only_staged_cog_sources_are_accepted(setup, tmp_path, boundary, change):
    child = boundary(Child())
    configs = json.loads(json.dumps(setup['configs']))
    if change == 'remote_tiles':
        configs['a']['remote_tiles'] = 'https://data.source.coop/tessera/tessera/npy/v1'
    else:
        del configs['a']['cog_index']
    manifest = write_live(tmp_path / 'live', configs)
    args = with_arg(with_arg(setup['args'], '--manifest', str(manifest)), '--output-root',
                    str(tmp_path / 'coverage'))
    args = with_arg(args, '--live-root', str(tmp_path / 'live'))
    with pytest.raises(ValueError, match='Only staged cog_index sources'):
        cov.main(args, model=FieldModel())
    assert child.calls == []


def test_output_never_lands_under_a_live_root(setup, tmp_path):
    live = setup['live']
    for output, roots in ((live / 'coverage', [live]), (live, [live]), (live.parent, [live]),
                          (tmp_path, [tmp_path / 'deeper' / 'live'])):
        with pytest.raises(ValueError, match='overlaps the live service root'):
            cov.check_roots(output, roots)
    cov.check_roots(tmp_path / 'coverage', [live])
    with pytest.raises(ValueError, match='overlaps'):
        cov.main(with_arg(setup['args'], '--output-root', str(live / 'coverage')), model=FieldModel())
    assert not (live / 'coverage').exists()


def test_free_space_below_the_batch_reserve_refuses(setup, tmp_path, boundary, monkeypatch):
    child = boundary(Child())
    monkeypatch.setenv('IMINT_LCLU_STORAGE_MIN_FREE_BYTES', str(6 * 2**30))
    free = 6 * 2**30 + PREPARATION_BYTES + MAP_BYTES - 1
    monkeypatch.setattr('scripts.produce_lclu_coverage.shutil.disk_usage',
                        lambda _: type('Disk', (), {'free': free})())
    with pytest.raises(StorageLimitError, match='batch reserve'):
        cov.main(with_arg(setup['args'], '--output-root', str(tmp_path / 'coverage')), model=FieldModel())
    assert child.calls == []


def test_scratch_and_mosaic_locks_fail_fast(setup, tmp_path, boundary):
    boundary(Child())
    release, args = copy_release(setup, tmp_path)
    scratch = release / 'scratch' / 'shard-0-of-1'
    scratch.mkdir()
    with cov.hold_lock(scratch / '.lock'):
        with pytest.raises(RuntimeError, match='held by another process'):
            cov.main(args, model=FieldModel())
    with cov.hold_lock(release / 'mosaic' / '.lock'):
        with pytest.raises(RuntimeError, match='held by another process'):
            cov.main([*args, '--num-shards', '2', '--mosaic'])


def test_resume_skips_verified_and_reruns_corrupt_or_missing(setup, tmp_path, boundary, monkeypatch):
    child = boundary(Child(invalid={WATER}))
    release, args = copy_release(setup, tmp_path)
    windows = [w for w in cov.runnable(setup['plan']) if (w['east'], w['north']) != WATER]
    corrupt, missing, fill = windows[0], windows[1], next(w for w in windows if w['kind'] == 'fill')
    tif = release / 'windows' / corrupt['id'] / 'classification.tif'
    data = tif.read_bytes()
    tif.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))       # one flipped byte
    (release / 'windows' / f"{missing['id']}.json").unlink()
    sidecar = release / 'windows' / f"{fill['id']}.json"
    write_json(sidecar, {**json.loads(sidecar.read_text()), 'inference_pin': '0' * 64})   # stale identity
    model = FieldModel()
    assert cov.main(args, model=model) == 0
    assert sorted(model.calls) == sorted((w['east'], w['north']) for w in (corrupt, missing, fill))
    assert WATER not in child.calls                              # a verified unavailable window is final
    assert all(cov.verify_sidecar(release, w, setup['pin']) for w in cov.runnable(setup['plan']))
    rerun = FieldModel()
    assert cov.main(args, model=rerun) == 0 and rerun.calls == []
    # A verified sidecar under another request key (e.g. changed sources) is re-run.
    key = cov.Producer.request_key
    monkeypatch.setattr(cov.Producer, 'request_key',
                        lambda self, w: 'f' * 64 if w['id'] == corrupt['id'] else key(self, w))
    changed = FieldModel()
    assert cov.main(args, model=changed) == 0
    assert changed.calls == [(corrupt['east'], corrupt['north'])]


def test_transient_failure_stays_a_failure_and_blocks_the_mosaic(setup, tmp_path, boundary):
    release, args = copy_release(setup, tmp_path)
    window = cov.runnable(setup['plan'])[0]
    (release / 'windows' / f"{window['id']}.json").unlink()
    child = boundary(Child(transient={(window['east'], window['north'])}))
    args = with_arg(args, '--run-id', '2')
    assert cov.main(args, model=FieldModel()) == 1
    assert child.calls == [(window['east'], window['north'])]        # never retried as "unavailable"
    failure = json.loads((release / 'failed' / f"{window['id']}.json").read_text())
    assert failure['error'] == 'RuntimeError'
    marker = json.loads(cov.marker_path(release, 0, 1).read_text())
    assert (marker['run'], marker['failed'], marker['aborted']) == ('2', [window['id']], None)
    with pytest.raises(RuntimeError, match='recorded failed windows'):
        cov.main([*args, '--num-shards', '1', '--mosaic'])


def test_consecutive_failures_of_one_type_abort_the_shard(setup, tmp_path, boundary):
    plan = setup['plan']
    child = boundary(Child(transient={(w['east'], w['north']) for w in cov.runnable(plan)}))
    args = with_arg(setup['args'], '--output-root', str(tmp_path / 'coverage'))
    assert cov.main(args, model=FieldModel()) == 1
    assert len(child.calls) == cov.FAILURE_STREAK
    release = tmp_path / 'coverage' / '2024' / 't1'
    marker = json.loads(cov.marker_path(release, 0, 1).read_text())
    assert marker['aborted']['error'] == 'RuntimeError' and marker['aborted']['consecutive'] == 3
    assert marker['aborted']['not_attempted'] == len(cov.runnable(plan)) - 3
    assert len(list((release / 'failed').iterdir())) == 3


def test_a_changed_plan_refuses_the_existing_release(setup, tmp_path):
    _, args = copy_release(setup, tmp_path)
    with pytest.raises(ValueError, match='new release name'):
        cov.main([*args, '--areas', 'a'], model=FieldModel())


# ── Shard markers ───────────────────────────────────────────────────────────

def test_mosaic_waits_for_this_runs_markers_not_for_sidecars(setup, tmp_path):
    release, args = copy_release(setup, tmp_path)
    identity = json.loads((release / 'release.json').read_text())
    markers = cov.check_shards(release, identity, 2, '1')
    assert [m['shard'] for m in markers] == [0, 1]
    # Every sidecar exists, but run 2's shards have not finished: no mosaic.
    assert cov.check_shards(release, identity, 2, '2') is None
    with pytest.raises(RuntimeError, match='still incomplete'):
        cov.main([*with_arg(args, '--run-id', '2'), '--num-shards', '2', '--mosaic'], poll=0)
    # A marker set of the wrong size is not this run's either.
    assert cov.check_shards(release, identity, 3, '1') is None
    # This run's markers, but a stale sidecar from an earlier identity: refused.
    window = cov.runnable(setup['plan'])[0]
    sidecar = release / 'windows' / f"{window['id']}.json"
    write_json(sidecar, {**json.loads(sidecar.read_text()), 'inference_pin': '1' * 64})
    with pytest.raises(RuntimeError, match='lack a verified sidecar'):
        cov.main([*args, '--num-shards', '2', '--mosaic'])


def test_mosaic_refuses_another_inference_identity(setup, tmp_path):
    _, args = copy_release(setup, tmp_path)
    with pytest.raises(ValueError, match='Release inference identity'):
        cov.main([*with_arg(args, '--expect-inference-pin', 'e' * 64), '--num-shards', '2', '--mosaic'])


# ── Mosaic ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('area, polygon', [('a', AREA_A), ('b', AREA_B)])
def test_mosaic_cog_georeferencing_and_known_pixels(setup, area, polygon):
    release = setup['release']
    path = release / 'mosaic' / f'{area}-2024.tif'
    w, s, e, n = polygon.bounds
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 3006 and src.res == (10.0, 10.0) and src.nodata == NODATA
        assert src.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG'
        assert (src.transform.c, src.transform.f) == (math.floor(w / 10) * 10, math.ceil(n / 10) * 10)
        assert all(src.colormap(1)[k][:3] == MAP_COLORS[k] for k in MAP_COLORS)
        data = src.read(1)
        x, y = world(src.transform.c, src.transform.f, *src.shape)
        inside = cov.inside_mask(polygon, int(src.transform.c), int(src.transform.f), *src.shape)
        classified = data != NODATA
        assert not (classified & ~inside).any()                    # clipped to the municipality
        assert np.array_equal(data[classified], field(x, y)[classified])
        for level in src.overviews(1):
            assert set(np.unique(src.read(1, out_shape=(src.height // level, src.width // level)))) <= \
                set(range(28)) | {NODATA}
        if area == 'a':
            # A fill window's pixel (refused cell) and a lattice window's pixel, by coordinate.
            for px, py in ((X0 + 3505, Y0 + 2005), (X0 + 9995, Y0 + 7005)):
                assert data[src.index(px, py)] == field(np.array(px), np.array(py))
            # A model-nodata block: input pixels 250:256 of the lattice window at cell (1, 0).
            row, col = src.index(X0 + 5040 + 10 * 247 + 5, Y0 + 5040 - 10 * 247 - 5)
            assert data[row, col] == NODATA and inside[row, col]
            # The unavailable window's cell is nodata inside the municipality.
            water = cov.inside_mask(polygon, X0 + 2 * cov.CELL, Y0 + 2 * cov.CELL, 504, 504)
            assert data[src.index(X0 + 2 * cov.CELL + 1005, Y0 + cov.CELL + 1005)] == NODATA
        else:
            # B's pixels west of x = 15,120 m come from A's shared lattice window.
            assert data[src.index(X0 + 14005, Y0 + 3005)] == field(np.array(X0 + 14005), np.array(Y0 + 3005))
    manifest = json.loads(path.with_suffix('.json').read_text())
    coverage = manifest['coverage']
    assert manifest['label'] == 'Heltäckande markslagskarta 2024'
    assert manifest['cog'] == public_identity(file_identity(path))
    assert 'seams' not in manifest and 'seam' not in json.dumps(manifest)
    assert coverage['uncovered_pixels'] == 0 and coverage['inside_pixels'] == int(inside.sum())
    assert (coverage['classified_pixels'] + coverage['nodata_pixels'] + coverage['unavailable_pixels']
            == int(inside.sum()))
    assert coverage['nodata_pixels'] + coverage['unavailable_pixels'] == int((inside & ~classified).sum())
    assert manifest['checkpoint'] == {'sha256': 'c' * 64, 'size': 1}
    listed = {x['id']: x for x in manifest['windows']}
    assert listed['L0132-1300']['status'] == 'produced'
    if area == 'a':
        assert listed['L0130-1300']['status'] == 'refused'
        assert listed['L0132-1301']['status'] == 'unavailable'
        assert coverage['unavailable_pixels'] == int(water.sum()) > 0
        assert coverage['nodata_pixels'] > 0
    else:
        assert coverage['unavailable_pixels'] == 0
    produced = [x for x in manifest['windows'] if x['status'] == 'produced']
    assert all(re.fullmatch(r'[0-9a-f]{64}', x['classification']['sha256']) for x in produced)
    # Seam statistics are operator-only; a seamless field shows no seam.
    seams = json.loads((release / 'operator' / f'{area}-2024-seams.json').read_text())
    assert seams['cog'] == manifest['cog'] and 0.5 < seams['seams']['seam_to_interior'] < 2


def test_seam_metric_separates_a_seam_from_none():
    plan = {'windows': [{'id': f'L{c:04d}-1300', 'kind': 'lattice', 'areas': ['a'], 'region': 'a',
                         'east': c * cov.CELL + cov.HALF, 'north': Y0 + cov.HALF} for c in (130, 131)]}
    owner = cov.Ownership(plan)
    west, north = X0, Y0 + cov.CELL
    who = owner.owners(west, north, 504, 1008)
    assert (who[:, :504] == 1).all() and (who[:, 504:] == 2).all()
    distance = owner.edge_distance(who, west, north)
    assert distance[0, 0] == 0 and distance[252, 252] == 251 and distance[10, 503] == 0
    classes = field(*world(west, north, 504, 1008))
    seamless, seam = cov.SeamCounter(3), cov.SeamCounter(3)
    seamless.add(classes, who, distance)
    shifted = np.where(who == 2, classes % 6 + 1, classes).astype(np.uint8)   # right window relabelled
    seam.add(shifted, who, distance)
    assert 0.5 < seamless.report()['seam_to_interior'] < 2
    assert seam.report()['seam_to_interior'] > 5
    assert seam.report()['interior'] == seamless.report()['interior']


def test_mosaic_refuses_a_release_with_missing_windows(setup, tmp_path):
    release, args = copy_release(setup, tmp_path)
    window = cov.runnable(setup['plan'])[-1]
    (release / 'windows' / f"{window['id']}.json").unlink()
    with pytest.raises(RuntimeError, match='lack a verified sidecar'):
        cov.main([*args, '--num-shards', '2', '--mosaic'])


# ── Job ─────────────────────────────────────────────────────────────────────

def _jobs() -> dict:
    yaml = pytest.importorskip('yaml')
    return {doc['metadata']['name']: doc
            for doc in yaml.safe_load_all((ROOT / 'k8s/lclu-coverage-2024-job.yaml').read_text())}


def _value(script: str, name: str) -> str:
    return re.search(rf'^\s*{name}=(\S+)$', script, re.M)[1]


def test_jobs_run_the_shards_they_declare_and_share_one_configuration():
    jobs = _jobs()
    assert set(jobs) == {'lclu-coverage-2024', 'lclu-coverage-2024-mosaic'}
    produce, mosaic = (jobs[name]['spec'] for name in ('lclu-coverage-2024', 'lclu-coverage-2024-mosaic'))
    scripts = {name: job['spec']['template']['spec']['containers'][0]['command'][-1]
               for name, job in jobs.items()}
    assert produce['completionMode'] == 'Indexed' and produce['completions'] == produce['parallelism']
    assert int(_value(scripts['lclu-coverage-2024'], 'SHARDS')) == produce['completions']
    for name in ('SHARDS', 'RELEASE', 'RUN', 'PIN', 'LIVE'):
        assert _value(scripts['lclu-coverage-2024'], name) == _value(scripts['lclu-coverage-2024-mosaic'], name)
    assert '--shard "$JOB_COMPLETION_INDEX" --num-shards "$SHARDS"' in scripts['lclu-coverage-2024']
    assert '--mosaic' in scripts['lclu-coverage-2024-mosaic']
    for script in scripts.values():
        assert '--run-id "$RUN"' in script and '--expect-inference-pin "$PIN"' in script
        assert script.count('--live-root') == 3 and '--output-root /work/coverage' in script
        assert '[[ "$PIN" =~ ^[0-9a-f]{64}$ ]]' in script and 'apt-get' not in script
        subprocess.run(['bash', '-n'], input=script, text=True, check=True)
    init = [job['spec']['template']['spec']['initContainers'] for job in jobs.values()]
    assert init[0] == init[1]
    (fetch,) = init[0]
    source = fetch['command'][-1]
    assert re.fullmatch(r'alpine/git@sha256:[0-9a-f]{64}', fetch['image'])
    assert '--filter=blob:none' in source and 'git sparse-checkout set' in source
    assert 'test "$(git rev-parse HEAD)" = "$SHA"' in source and "grep -Eq '^[0-9a-f]{40}$'" in source
    assert 'apt-get' not in source
    subprocess.run(['sh', '-n'], input=source, text=True, check=True)
    base = re.search(r'^FROM (\S+)$', (ROOT / 'docker/lclu/Dockerfile').read_text(), re.M)[1]
    for job, (cpu, memory) in ((jobs['lclu-coverage-2024'], ('4', '8Gi')),
                               (jobs['lclu-coverage-2024-mosaic'], ('1', '4Gi'))):
        pod = job['spec']['template']['spec']
        (container,) = pod['containers']
        assert container['image'] == base and '@sha256:' in base
        assert container['resources']['requests'] == container['resources']['limits'] == {
            'cpu': cpu, 'memory': memory}
        assert fetch['resources']['requests'] == fetch['resources']['limits']
        for c in pod['containers'] + pod['initContainers']:
            assert c['securityContext'] == {'allowPrivilegeEscalation': False, 'capabilities': {'drop': ['ALL']}}


def test_batch_reserve_keeps_the_live_services_reserve_free():
    spec = importlib.util.spec_from_file_location('render_job', ROOT / 'docker/lclu/render_job.py')
    render_job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render_job)
    live = render_job.resources('python@sha256:' + '0' * 64, 2024)['items'][-1]
    env = {e['name']: e['value'] for e in live['spec']['template']['spec']['containers'][0]['env']}
    live_reserve = int(env['IMINT_LCLU_STORAGE_MIN_FREE_BYTES'])
    jobs = _jobs()
    pods = jobs['lclu-coverage-2024']['spec']['completions']
    need = live_reserve + pods * (PREPARATION_BYTES + MAP_BYTES) + 2**30   # + mosaic headroom
    for job in jobs.values():
        env = {e['name']: e['value'] for e in job['spec']['template']['spec']['containers'][0]['env']}
        assert int(env['IMINT_LCLU_STORAGE_MIN_FREE_BYTES']) >= max(need, 6 * 2**30)


def test_job_source_pin_matches_the_producer():
    source = _jobs()['lclu-coverage-2024']['spec']['template']['spec']['initContainers'][0]['command'][-1]
    sha = _value(source, 'SHA')
    if sha == '__SOURCE_COMMIT__':
        pytest.skip('source commit not yet pinned; the Job refuses to run until it is')
    assert re.fullmatch(r'[0-9a-f]{40}', sha)
    git = ['git', '-C', str(ROOT)]
    if subprocess.run([*git, 'cat-file', '-e', sha], capture_output=True).returncode:
        pytest.skip(f'{sha} not in this clone')
    assert subprocess.run([*git, 'merge-base', '--is-ancestor', sha, 'HEAD']).returncode == 0
    changed = subprocess.run([*git, 'diff', '--name-only', sha, 'HEAD', '--',
                              'scripts/produce_lclu_coverage.py'],
                             capture_output=True, text=True, check=True).stdout.split()
    assert changed == [], f'bump SHA in k8s/lclu-coverage-2024-job.yaml; changed: {changed}'


def test_cdse_vpp_is_unreachable_from_the_producer_and_preparation():
    """Why the Jobs set no VPP_SOURCE: nothing on this path imports the CDSE VPP router."""
    code = '''
import importlib, sys
for name in ('scripts.produce_lclu_coverage', 'imint.inference.lclu', 'imint.inference.lclu_sources',
             'scripts.inference_comparison', 'scripts.prepare_lclu', 'imint.data.wekeo_vpp',
             'imint.training.skg_open', 'imint.training.skg_grunddata', 'imint.training.slu_markfukt',
             'imint.training.copernicus_dem', 'imint.inference.lclu_cog', 'scripts.enrich_tiles_tessera'):
    importlib.import_module(name)
assert 'imint.data.cdse_vpp' not in sys.modules, 'cdse_vpp became reachable; set VPP_SOURCE=wekeo'
'''
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONPATH': str(ROOT)})
