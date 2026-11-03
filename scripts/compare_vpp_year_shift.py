#!/usr/bin/env python3
"""HR-VPP year-shift control: how much does an annual map change with last year's HR-VPP?

HR-VPP 2025 is not published, so a 2025 map would have to use HR-VPP 2024.
This measures that effect on the years where both exist. For each demo area of
the live release, at its catalogue location (``GET /v1/lclu`` ``locations``,
the point the page selects), and each year y in 2019-2024, the served R4
classifies the same window twice:

* normal: the input the service prepares for (grid, y), through
  ``RegionalSources.prepared``: TESSERA y, HR-VPP y and the static layers;
* shifted: the same input with only the five HR-VPP layers replaced by year
  y-1, and ``vpp_year`` set to y-1.

The y-1 layers come from ``prepare_lclu.prepare_aux``, unchanged, under the
y-1 qualified source configuration of the same grid. They are the layers the
y-1 map of that window gets. Before classifying, each pair hashes every array
of both inputs. Only the HR-VPP layers and ``vpp_year`` may differ, and the
HR-VPP layers and source files must differ. The normal input is also
classified twice as a repeat control. This is a map-to-map consistency check
on demo areas. It does not score anything against field truth.

Why the override lives here and not in ``prepare_lclu.py``:

* ``prepare()`` passes the request year to ``prepare_aux``, which selects
  index entries with ``meta['year'] == year`` and reads
  ``fetch_vpp_tiles_local(year=year)``; ``prepare()`` stamps
  ``vpp_year = year``. ``validate_bundle`` and ``public_provenance`` reject
  any other VPP year, so the shifted input cannot go through
  ``LcluService.run``. It is classified by the same ``model.predict`` call
  and validity mask that ``LcluService._run_prepared`` uses.
* Each served (area, year) configuration pins a per-year HR-VPP index
  snapshot and file hashes (``stage_lclu_demo.snapshot_vpp``). A year flag
  inside the y configuration would find no y-1 files.
* ``prepare_lclu.py`` is hashed into ``preparation_identity`` and
  ``implementation_identity``; any edit changes the live inference pin.

Pair states, one record each (``pairs/<area>-<year>.json``): ``compared``,
or ``missing`` (no qualified source for y at the location, or no y-1 HR-VPP:
no qualified y-1 source, or ``prepare_aux`` finds no complete or usable y-1
HR-VPP). Another year is never used. A pair with a verified record is
skipped on resume; any other outcome is a failure, retried on resume. All
prepared, shifted and HR-VPP inputs are deleted after every pair. Each shard
rewrites ``summary.json`` and ``summary.md`` from all records when it ends.

    python scripts/compare_vpp_year_shift.py plan --manifest serving.json --num-shards 4
    python scripts/compare_vpp_year_shift.py run --run r1 --shard 0 --num-shards 4 \\
        --expect-inference-pin <hex> --manifest /work/releases/lclu-stockholm-0124-r10/demo/serving.json \\
        --output-root /work/experiments/vpp-year-shift \\
        --live-root /work/releases/lclu-stockholm-0124-r10 --live-root /work/campaign --live-root /work/runtime
    python scripts/compare_vpp_year_shift.py summarize --run r1 \\
        --output-root /work/experiments/vpp-year-shift --live-root ...
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from imint.inference.lclu import (
    AUX_NAMES, CROP, NODATA, OUTPUT_SIZE, SOURCE_SIZE, Grid, InputUnavailable, public_identity,
    validate_bundle, write_json,
)
from imint.schema.unified_schema import FOREST_CLASSES, NUM_CLASSES_BEFORE_BARRBLANDSKOG as NUM_CLASSES
from scripts.cache_predict import atomic_write, contract_pin, file_identity
from scripts.produce_lclu_coverage import (
    HEX64, NAME, check_roots, check_sources, hold_lock, preparations, publish_once,
)

YEARS = tuple(range(2019, 2025))
SHIFT = 1
VPP_NAMES = tuple(name for name in AUX_NAMES if name.startswith('vpp_'))
SHIFTED_ARRAYS = frozenset({*VPP_NAMES, 'vpp_year'})
# prepare_lclu.prepare_aux's two outcomes for a year without usable HR-VPP at a
# grid; tests/lclu/test_vpp_year_shift.py ties them to the real code.
VPP_MISSING = ('Exact-year HR-VPP source coverage is incomplete',
               'No usable exact-year phenology in this window')
INNER = (slice(CROP, CROP + OUTPUT_SIZE),) * 2      # the 504 px map inside the 512 px input
GROUPS = {'forest': tuple(c for c in FOREST_CLASSES if c < NUM_CLASSES),
          'crop': tuple(range(11, 22))}             # LPIS crop classes 11-21 (unified schema v5)
GROUPS['other'] = tuple(c for c in range(NUM_CLASSES) if c not in GROUPS['forest'] + GROUPS['crop'])
FAILURE_STREAK = 3                                  # consecutive failures of one type that stop a shard
LABEL = 'HR-VPP year-shift control: map-to-map consistency on demo areas, not accuracy'
SCRIPT = Path(__file__).resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Areas and pairs ────────────────────────────────────────────────────────

def slug(name: str) -> str:
    ascii_name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '-', ascii_name.lower()).strip('-')


def demo_areas(sources) -> list[dict]:
    """Every catalogue area at the location the page selects for it, as the service snaps it."""
    areas = []
    for area, point in zip(sources.areas(), sources.locations(), strict=True):
        grid = Grid.from_lonlat(point['lon'], point['lat'])
        areas.append({'id': slug(area['properties']['name']), 'name': area['properties']['name'],
                      'lon': point['lon'], 'lat': point['lat'], 'center_3006': [grid.east, grid.north],
                      'years_served': sources.years(grid)})
    ids = [a['id'] for a in areas]
    if not areas or len(set(ids)) != len(ids) or not all(NAME.fullmatch(i) for i in ids):
        raise ValueError(f'Demo areas need distinct names: {ids}')
    return areas


def plan_pairs(areas: list[dict]) -> list[dict]:
    return [{'id': f"{a['id']}-{year}", 'area': a['id'], 'name': a['name'], 'year': year,
             'vpp_years': [year, year - SHIFT], 'center_3006': a['center_3006']}
            for a in areas for year in YEARS]


def check_shard(index: int, count: int) -> None:
    if not (type(count) is int and count >= 1 and type(index) is int and 0 <= index < count):
        raise ValueError('Shard index must satisfy 0 <= shard < num_shards')


def shard(pairs: list[dict], index: int, count: int) -> list[dict]:
    check_shard(index, count)
    return pairs[index::count]


# ── Inputs ─────────────────────────────────────────────────────────────────

def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f'{value.dtype.str}{value.shape}'.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def input_digests(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as data:
        return {name: array_sha256(data[name]) for name in sorted(data.files)}


def write_shifted(normal: Path, vpp: dict[str, np.ndarray], vpp_year: int, output: Path) -> None:
    """The normal input with only the HR-VPP layers and their year replaced."""
    with np.load(normal, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if set(vpp) != set(VPP_NAMES):
        raise ValueError('Exactly the R4 HR-VPP layers are replaced')
    for name, value in vpp.items():
        if value.shape != arrays[name].shape or value.dtype != arrays[name].dtype:
            raise ValueError(f'HR-VPP layer {name} does not match the normal input')
    arrays.update(vpp, vpp_year=np.int32(vpp_year))
    atomic_write(output, lambda stream: np.savez_compressed(stream, **arrays))


def check_shift(normal: Path, shifted: Path, year: int, vpp_year: int) -> dict:
    """Hash every array: the two inputs may differ only in HR-VPP, which must differ."""
    a, b = input_digests(normal), input_digests(shifted)
    if a.keys() != b.keys():
        raise ValueError('The shifted input carries other arrays than the normal input')
    differing = sorted(name for name in a if a[name] != b[name])
    outside = sorted(set(differing) - SHIFTED_ARRAYS)
    if outside:
        raise ValueError(f'Inputs differ outside HR-VPP: {outside}')
    if not set(differing) & set(VPP_NAMES):
        raise ValueError('HR-VPP layers are identical for both years')
    with np.load(normal, allow_pickle=False) as n, np.load(shifted, allow_pickle=False) as s:
        years = {'normal': {k: int(n[k]) for k in ('year', 'tessera_year', 'vpp_year')},
                 'shifted': {k: int(s[k]) for k in ('year', 'tessera_year', 'vpp_year')}}
    if (years['normal'] != {'year': year, 'tessera_year': year, 'vpp_year': year}
            or years['shifted'] != {'year': year, 'tessera_year': year, 'vpp_year': vpp_year}):
        raise ValueError(f'Unexpected input years {years}')
    return {'years': years, 'differing_arrays': differing,
            'array_sha256': {'normal': a, 'shifted': b}}


def read_vpp(directory: Path, vpp_year: int) -> dict:
    """The child's HR-VPP layers, or the reason the year has none at this grid."""
    marker = directory / 'missing.json'
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record['vpp_year'] != vpp_year or record['reason'] not in VPP_MISSING:
            raise ValueError('Unexpected HR-VPP missing record')
        return {'missing': record['reason']}
    meta = json.loads((directory / 'vpp.json').read_text())
    if meta['vpp_year'] != vpp_year or meta['provenance']['year'] != vpp_year:
        raise ValueError('HR-VPP layers of another year')
    path = directory / 'vpp.npz'
    if public_identity(file_identity(path)) != meta['npz']:
        raise ValueError('HR-VPP layers changed after the child wrote them')
    with np.load(path, allow_pickle=False) as data:
        if sorted(data.files) != sorted(VPP_NAMES):
            raise ValueError('Expected exactly the R4 HR-VPP layers')
        arrays = {name: data[name] for name in VPP_NAMES}
    for name, value in arrays.items():
        if (value.shape != (SOURCE_SIZE, SOURCE_SIZE) or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError(f'Invalid HR-VPP layer {name}')
    return {'arrays': arrays, 'provenance': meta['provenance']}


def vpp_child(config_path: Path, sha256: str, grid: Grid, vpp_year: int, output: Path) -> None:
    """Child process: year ``vpp_year``'s HR-VPP layers as prepare_aux reads them under this config.

    prepare_aux sets SKG_OPEN_DIR, which is why it runs in a child, as in
    preparation. Its static layers are discarded; the shifted input keeps the
    normal input's.
    """
    from scripts.prepare_lclu import prepare_aux

    identity = public_identity(file_identity(config_path))
    if identity['sha256'] != sha256:
        raise ValueError('Source configuration differs from the serving manifest')
    try:
        aux, provenance = prepare_aux(json.loads(config_path.read_text()), config_path.parent,
                                      grid, vpp_year)
    except InputUnavailable as exc:
        if str(exc) not in VPP_MISSING:
            raise
        write_json(output / 'missing.json', {'vpp_year': vpp_year, 'reason': str(exc)})
        return
    if public_identity(file_identity(config_path)) != identity:
        raise ValueError('Source configuration changed while reading HR-VPP')
    if provenance['vpp']['year'] != vpp_year:
        raise ValueError('prepare_aux returned HR-VPP of another year')
    path = output / 'vpp.npz'
    atomic_write(path, lambda stream: np.savez_compressed(stream, **{n: aux[n] for n in VPP_NAMES}))
    write_json(output / 'vpp.json', {'vpp_year': vpp_year, 'npz': public_identity(file_identity(path)),
                                     'provenance': provenance['vpp']})


# ── Metrics ────────────────────────────────────────────────────────────────

def as_map(prediction: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The classes export_map writes: model classes on valid pixels, NODATA elsewhere."""
    if (prediction.shape != (OUTPUT_SIZE, OUTPUT_SIZE) or valid.shape != prediction.shape
            or not np.issubdtype(prediction.dtype, np.integer)
            or np.any((prediction < 0) | (prediction >= NUM_CLASSES))):
        raise ValueError('Invalid model classification output')
    return np.where(valid, prediction, NODATA).astype(np.uint8)


def transitions(confusion: np.ndarray, limit: int = 10) -> list[dict]:
    """The largest off-diagonal cells, largest first (ties by class id)."""
    cells = sorted(((int(count), i, j) for (i, j), count in np.ndenumerate(confusion)
                    if i != j and count), key=lambda t: (-t[0], t[1], t[2]))
    return [{'from': i, 'to': j, 'pixels': count} for count, i, j in cells[:limit]]


def group_change(confusion: np.ndarray) -> dict:
    """Change inside forest, crop and other classes, grouped by the class in the y-VPP map."""
    changed = int(confusion.sum() - np.trace(confusion))
    out = {}
    for name, classes in GROUPS.items():
        pixels = int(confusion[list(classes)].sum())
        moved = pixels - int(sum(confusion[c, c] for c in classes))
        out[name] = {'classes': list(classes), 'pixels': pixels, 'changed': moved,
                     'change_rate': moved / pixels if pixels else None,
                     'share_of_changed': moved / changed if changed else None}
    return out


def compare_maps(normal: np.ndarray, shifted: np.ndarray) -> dict:
    """Class change from the y-VPP map (rows) to the y-1-VPP map (columns) on valid pixels."""
    valid = normal != NODATA
    if normal.shape != shifted.shape or not np.array_equal(valid, shifted != NODATA):
        raise ValueError('The two maps must share one grid and one validity mask')
    confusion = np.bincount(normal[valid].astype(np.int64) * NUM_CLASSES + shifted[valid],
                            minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    total = int(confusion.sum())
    changed = total - int(np.trace(confusion))
    return {'valid_pixels': total, 'changed_pixels': changed,
            'changed_share': changed / total if total else None,
            'groups': group_change(confusion), 'top_transitions': transitions(confusion),
            'confusion': confusion.tolist()}


def vpp_difference(normal: dict, shifted: dict, valid: np.ndarray) -> dict:
    """Per HR-VPP band, the input change between y and y-1 over the map's valid pixels.

    ``mean_abs_diff_scaled`` is in units of the band's training standard
    deviation: both years pass through ``normalize_aux_channel``, the
    transform of the model input. The raw differences (y minus y-1) cover
    pixels where both years have data (HR-VPP nodata is 0): day of year for the
    two date bands (stored as YYDDD), stored units otherwise.
    """
    from imint.training.unified_dataset import AUX_YYDDD_DATE_CHANNELS, normalize_aux_channel

    out = {}
    for name in VPP_NAMES:
        a, b = normal[name][INNER][valid], shifted[name][INNER][valid]
        scaled = np.abs(normalize_aux_channel(name, a) - normalize_aux_channel(name, b))
        date = name in AUX_YYDDD_DATE_CHANNELS
        both = (a != 0) & (b != 0)
        raw_a, raw_b = (np.mod(a, 1000), np.mod(b, 1000)) if date else (a, b)
        diff = (raw_a - raw_b)[both].astype(np.float64)
        out[name] = {'mean_abs_diff_scaled': float(scaled.astype(np.float64).mean()) if a.size else None,
                     'raw_unit': 'day of year' if date else 'stored value',
                     'mean_abs_diff_raw': float(np.abs(diff).mean()) if diff.size else None,
                     'mean_diff_raw': float(diff.mean()) if diff.size else None,
                     'pixels': int(a.size), 'pixels_with_data_both': int(both.sum()),
                     'pixels_nodata_only_in_y': int(((a == 0) & (b != 0)).sum()),
                     'pixels_nodata_only_in_previous': int(((a != 0) & (b == 0)).sum())}
    return out


# ── Records ────────────────────────────────────────────────────────────────

def missing(kind: str, reason: str) -> dict:
    """A pair that cannot run as specified: no qualified y, or no y-1 HR-VPP. Never another year."""
    return {'status': 'missing', 'missing': kind, 'reason': reason}


def experiment_identity(root: Path) -> str:
    return public_identity(file_identity(root / 'experiment.json'))['sha256']


def verify_pair(root: Path, pair: dict, experiment: str, pin: str) -> dict | None:
    """The pair's record if it verifies (and, when compared, its class maps); else None."""
    try:
        record = json.loads((root / 'pairs' / f"{pair['id']}.json").read_text())
        if (record['version'] != 1 or record['experiment'] != experiment
                or record['inference_pin'] != pin or record['pair'] != pair):
            return None
        if record['status'] == 'missing':
            return record if record['missing'] in ('year', 'previous_vpp') and record['reason'] else None
        if record['status'] != 'compared':
            return None
        if public_identity(file_identity(root / 'pairs' / f"{pair['id']}.npz")) != record['classes']:
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextmanager
def exclusive(path: Path):
    """A blocking advisory lock; shards rewrite the summary one at a time."""
    with path.open('a') as stream:     # never unlinked: a new inode would bypass the lock
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _stats(values: list[tuple[float, str]]) -> dict | None:
    if not values:
        return None
    numbers = [v for v, _ in values]
    top = max(values, key=lambda t: t[0])
    return {'pairs': len(numbers), 'median': statistics.median(numbers),
            'mean': statistics.fmean(numbers), 'min': min(numbers), 'max': top[0], 'max_pair': top[1]}


def build_summary(experiment: dict, records: dict[str, dict | None]) -> dict:
    """Area x year table and the aggregate over every compared pair."""
    from imint.inference.lclu_legend import MAP_NAMES

    pairs = experiment['pairs']
    compared = {p['id']: records[p['id']] for p in pairs
                if records.get(p['id']) and records[p['id']]['status'] == 'compared'}
    absent = [{'pair': p['id'], 'missing': records[p['id']]['missing'], 'reason': records[p['id']]['reason']}
              for p in pairs if records.get(p['id']) and records[p['id']]['status'] == 'missing']
    pending = [p['id'] for p in pairs if not records.get(p['id'])]
    table = {}
    for p in pairs:
        record = records.get(p['id'])
        cell = (None if record is None else 'missing' if record['status'] == 'missing'
                else record['metrics']['changed_share'])
        table.setdefault(p['name'], {})[str(p['year'])] = cell
    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for record in compared.values():
        pooled += np.array(record['metrics']['confusion'], np.int64)
    changed = int(pooled.sum() - np.trace(pooled))
    classes = []
    for c in range(NUM_CLASSES):
        pixels, moved = int(pooled[c].sum()), int(pooled[c].sum() - pooled[c, c])
        if moved:
            off = pooled[c].copy()
            off[c] = 0
            to = int(np.argmax(off))
            classes.append({'class': c, 'name': MAP_NAMES[c], 'pixels': pixels, 'changed': moved,
                            'change_rate': moved / pixels, 'share_of_changed': moved / changed,
                            'main_destination': {'class': to, 'name': MAP_NAMES[to], 'pixels': int(off[to])}})
    classes.sort(key=lambda c: (-c['changed'], c['class']))
    top = [{**t, 'from_name': MAP_NAMES[t['from']], 'to_name': MAP_NAMES[t['to']],
            'share_of_changed': t['pixels'] / changed} for t in transitions(pooled)]
    groups = {name: _stats([(r['metrics']['groups'][name]['change_rate'], pid) for pid, r in compared.items()
                            if r['metrics']['groups'][name]['change_rate'] is not None])
              for name in GROUPS}
    vpp = {name: {field: _stats([(r['vpp_difference'][name][field], pid) for pid, r in compared.items()
                                 if r['vpp_difference'][name][field] is not None])
                  for field in ('mean_abs_diff_scaled', 'mean_abs_diff_raw')}
           for name in VPP_NAMES}
    repeat = [r['repeat_control']['changed_pixels'] for r in compared.values()]
    return {
        'version': 1, 'label': experiment['label'], 'run': experiment['run'],
        'inference_pin': experiment['inference_pin'], 'years': experiment['years'], 'shift': SHIFT,
        'complete': not pending,
        'pairs': {'planned': len(pairs), 'compared': len(compared), 'missing': len(absent),
                  'pending': len(pending)},
        'changed_share_table': table,
        'aggregate': {
            'changed_share': _stats([(r['metrics']['changed_share'], pid) for pid, r in compared.items()
                                     if r['metrics']['changed_share'] is not None]),
            'group_change_rate': groups,
            'pooled_groups': group_change(pooled) if compared else None,
            'classes_moving_most': classes[:10],
            'top_transitions': top,
            'vpp_input_difference': vpp,
        },
        'repeat_control': {'pairs': len(repeat), 'max_changed_pixels': max(repeat) if repeat else None,
                           'note': 'normal input classified twice; nonzero means the model is not deterministic'},
        'missing': absent, 'pending': pending,
        'definitions': {
            'changed_share': 'valid pixels whose class differs between the y-VPP and y-1-VPP maps / valid pixels',
            'groups': 'by class in the y-VPP map; forest = schema FOREST_CLASSES, crop = LPIS classes 11-21',
            'confusion': 'rows: class with HR-VPP y; columns: class with HR-VPP y-1',
            'mean_abs_diff_scaled': 'mean |normalized y - normalized y-1|, in training standard deviations',
        },
    }


def _percent(value) -> str:
    return '-' if value is None else value if isinstance(value, str) else f'{100 * value:.2f} %'


def summary_markdown(summary: dict) -> str:
    years = [str(y) for y in summary['years']]
    lines = [f"# HR-VPP year shift: map change with HR-VPP y-1 instead of y ({summary['run']})", '',
             (f"Inference pin `{summary['inference_pin']}`. Same window, TESSERA y, static layers and "
              'R4 in both maps. Map-to-map consistency on demo areas, not accuracy.'), '',
             'Compared {compared} of {planned} pairs; {missing} missing, {pending} pending.'.format(
                 **summary['pairs']), '',
             'Share of valid pixels whose class differs:', '',
             '| Area | ' + ' | '.join(years) + ' |', '|---|' + '---:|' * len(years)]
    for area, row in summary['changed_share_table'].items():
        lines.append(f'| {area} | ' + ' | '.join(_percent(row.get(y)) for y in years) + ' |')
    aggregate = summary['aggregate']
    lines.append('')
    if aggregate['changed_share']:
        s = aggregate['changed_share']
        lines.append(f"Median {_percent(s['median'])}, mean {_percent(s['mean'])}, "
                     f"max {_percent(s['max'])} ({s['max_pair']}).")
        for name in ('forest', 'crop'):
            g = aggregate['group_change_rate'][name]
            if g:
                lines.append(f"Change rate inside {name} classes: median {_percent(g['median'])}, "
                             f"max {_percent(g['max'])} ({g['max_pair']}).")
        pooled = aggregate['pooled_groups']
        lines.append(f"Of all changed pixels, {_percent(pooled['forest']['share_of_changed'])} are forest "
                     f"and {_percent(pooled['crop']['share_of_changed'])} crop in the y-VPP map.")
        lines += ['', 'Classes that move most (all compared pairs pooled):', '']
        for c in aggregate['classes_moving_most'][:5]:
            lines.append(f"- {c['name']}: {_percent(c['change_rate'])} of its pixels, "
                         f"mostly to {c['main_destination']['name']}")
        lines.append('')
        lines.append(f"Repeat control: at most {summary['repeat_control']['max_changed_pixels']} "
                     'changed pixels when the normal input is classified twice.')
    if summary['missing']:
        lines += ['', 'Missing: ' + '; '.join(f"{m['pair']} ({m['reason']})" for m in summary['missing'])]
    if summary['pending']:
        lines += ['', 'Pending: ' + ', '.join(summary['pending'])]
    return '\n'.join(lines) + '\n'


def summarize(root: Path) -> dict:
    """Rewrite summary.json and summary.md from every verified pair record."""
    experiment = json.loads((root / 'experiment.json').read_text())
    sha = experiment_identity(root)
    with exclusive(root / 'summary.lock'):
        records = {p['id']: verify_pair(root, p, sha, experiment['inference_pin']) for p in experiment['pairs']}
        summary = build_summary(experiment, records)
        write_json(root / 'summary.json', summary)
        text = summary_markdown(summary).encode()
        atomic_write(root / 'summary.md', lambda stream: stream.write(text))
    return summary


# ── One shard ──────────────────────────────────────────────────────────────

class Shift:
    """One shard of one run: an exclusive scratch root, the identity and source checks, its pairs."""

    def __init__(self, manifest: Path, root: Path, scratch: Path, model, *, run: str, expect_pin: str):
        from imint.inference.lclu_storage import PREPARATION_BYTES, StorageLimitError

        self.manifest_path = manifest.resolve()
        self.preparations = preparations(json.loads(self.manifest_path.read_text()))
        self.root, self.scratch, self.model = root, scratch, model
        self.work = scratch / 'vpp-shift'
        self.pin = None
        scratch.mkdir(parents=True, exist_ok=True)
        self._lock = hold_lock(scratch / '.lock')
        try:
            for name in ('pairs', 'failed', 'done', 'operator'):
                (root / name).mkdir(parents=True, exist_ok=True)
            self.clean()
            service = self.service()
            self.identity = service.identity
            self.pin = contract_pin(self.identity)
            if self.pin != expect_pin:
                raise ValueError(f'Inference identity {self.pin} is not the expected {expect_pin}; '
                                 'the source checkout or runtime differs from the live release')
            # One prepared input, one shifted input and one HR-VPP read at a time.
            free = shutil.disk_usage(scratch).free
            need = service.storage.min_free_bytes + 3 * PREPARATION_BYTES
            if free < need:
                raise StorageLimitError(f'{free} bytes free; the batch reserve needs {need}')
            check_sources(service.sources)
            self.areas = demo_areas(service.sources)
            self.pairs = plan_pairs(self.areas)
            publish_once(root / 'experiment.json', {
                'version': 1, 'label': LABEL, 'run': run, 'years': list(YEARS), 'shift': SHIFT,
                'serving_manifest': public_identity(file_identity(self.manifest_path)),
                'inference_pin': self.pin, 'inference': self.identity,
                'areas': self.areas, 'pairs': self.pairs,
                'design': ('normal: the service-prepared input for (grid, y). shifted: that input with '
                           f'only {list(VPP_NAMES)} replaced by year y-1 as prepare_aux reads them '
                           'under the y-1 qualified source, and vpp_year = y-1.')})
            self.experiment = experiment_identity(root)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._lock.close()

    def __enter__(self) -> 'Shift':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def service(self):
        """A fresh budget, sources and service; a deleted input never stays in an old ledger."""
        from imint.inference.lclu import LcluService
        from imint.inference.lclu_sources import RegionalSources
        from imint.inference.lclu_storage import StorageBudget

        storage = StorageBudget.from_environment(self.scratch)
        sources = RegionalSources(self.preparations, self.manifest_path.parent, self.scratch, storage=storage)
        service = LcluService(self.model, sources, self.scratch, storage=storage)
        if self.pin is not None and contract_pin(service.identity) != self.pin:
            raise RuntimeError('Serving code changed during the run')
        return service

    def clean(self) -> None:
        """Delete every prepared, shifted and HR-VPP input and staging leftover (under the lock)."""
        from imint.inference.lclu_storage import StorageBudget

        StorageBudget.cleanup_staging(self.scratch)
        inputs = self.scratch / 'lclu-inputs'
        for path in inputs.glob('*') if inputs.is_dir() else ():
            if HEX64.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        if self.work.exists():
            shutil.rmtree(self.work)

    def _keep_log(self, log: Path, name: str) -> Path:
        target = self.root / 'operator' / 'preparation-logs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(log, target)
        return target

    def collect_logs(self, pair: dict) -> None:
        """Move the service preparation's failure logs to the operator directory."""
        failures = self.scratch / 'lclu-inputs' / 'failures'
        for n, log in enumerate(sorted(failures.glob('*.log')) if failures.is_dir() else ()):
            self._keep_log(log, f"{pair['id']}.{n}.log")

    def previous_vpp(self, storage, source, grid: Grid, vpp_year: int, pair: dict) -> dict:
        """Year y-1's HR-VPP layers through the qualified y-1 source, in a bounded child."""
        from imint.inference import lclu_sources
        from imint.inference.lclu_storage import PREPARATION_BYTES

        output = self.work / 'previous-vpp'
        log = output / 'vpp.log'
        command = [source.python, str(SCRIPT), 'vpp-child', '--config', str(source.config_path),
                   '--config-sha256', source.settings['config']['sha256'],
                   '--east', str(grid.east), '--north', str(grid.north),
                   '--vpp-year', str(vpp_year), '--output', str(output)]
        with storage.reserve(PREPARATION_BYTES, paths=(output,)):
            output.mkdir(parents=True)
            try:
                lclu_sources.run_preparation(command, timeout=source.timeout, log_path=log)
            except Exception as exc:
                kept = self._keep_log(log, f"{pair['id']}.previous-vpp.log") if log.is_file() else None
                raise RuntimeError(f'HR-VPP {vpp_year} read failed; operator log: {kept}') from exc
        return read_vpp(output, vpp_year)

    def compare(self, pair: dict) -> dict:
        """Classify the pair's normal input twice and its shifted input once; or report it missing."""
        from imint.inference.lclu_storage import PREPARATION_BYTES

        start = time.perf_counter()
        grid, year, previous_year = Grid(*pair['center_3006']), pair['year'], pair['year'] - SHIFT
        service = self.service()
        sources = service.sources
        if year not in sources.years(grid):
            return missing('year', f'No qualified source for {year} at this location')
        try:
            # The service's own routing for (grid, y-1); never another year.
            previous = sources._source(grid, previous_year)
            previous_source = previous.identity(grid, previous_year)
        except InputUnavailable:
            return missing('previous_vpp',
                           f'No qualified HR-VPP source for {previous_year} at this location')
        vpp = self.previous_vpp(service.storage, previous, grid, previous_year, pair)
        if 'missing' in vpp:
            return missing('previous_vpp', f"HR-VPP {previous_year}: {vpp['missing']}")
        read = time.perf_counter()
        request_key = service.key(grid, year)
        with sources.prepared(grid, year) as (path, entry):
            validate_bundle(path, entry['input'], grid, year)
            prepared = time.perf_counter()
            normal_files = {f['sha256'] for f in entry['auxiliary_provenance']['vpp']['files']}
            previous_files = {f['sha256'] for f in vpp['provenance']['files']}
            if not normal_files or not previous_files or normal_files & previous_files:
                raise ValueError('HR-VPP source files of the two years are not distinct')
            shifted = self.work / 'shifted' / 'input.npz'
            with service.storage.reserve(PREPARATION_BYTES, paths=(shifted.parent,)):
                shifted.parent.mkdir(parents=True)
                write_shifted(path, vpp['arrays'], previous_year, shifted)
            shift = check_shift(path, shifted, year, previous_year)
            shifted_record = {**file_identity(shifted), 'path': shifted.name}
            with np.load(path, allow_pickle=False) as data:
                valid = data['valid_embedding'][INNER].copy()
                normal_vpp = {name: data[name] for name in VPP_NAMES}
            built = time.perf_counter()
            normal = as_map(self.model.predict(path, entry['input']), valid)
            repeat = as_map(self.model.predict(path, entry['input']), valid)
            shifted_map = as_map(self.model.predict(shifted, shifted_record), valid)
            classified = time.perf_counter()
            difference = vpp_difference(normal_vpp, vpp['arrays'], valid)
        classes = self.root / 'pairs' / f"{pair['id']}.npz"
        atomic_write(classes, lambda stream: np.savez_compressed(stream, normal=normal, shifted=shifted_map))
        return {'status': 'compared', 'request_key': request_key,
                'sources': {'normal': sources.identity(grid, year), 'previous_vpp': previous_source},
                'vpp_sources': {'normal': entry['auxiliary_provenance']['vpp'], 'previous': vpp['provenance']},
                'inputs': {'normal': public_identity(entry['input']),
                           'shifted': public_identity(shifted_record), **shift},
                'metrics': compare_maps(normal, shifted_map),
                'repeat_control': {'changed_pixels': int((normal != repeat).sum())},
                'vpp_difference': difference,
                'classes': public_identity(file_identity(classes)),
                'timing_s': {'previous_vpp': read - start, 'prepare': prepared - read,
                             'shifted_input': built - prepared, 'classify_three': classified - built,
                             'total': time.perf_counter() - start}}

    def run(self, pairs: list[dict]) -> dict:
        counts = {'skipped': 0, 'compared': 0, 'missing': 0, 'failed': 0}
        failed, aborted = [], None
        streak_type, streak = None, 0
        for n, pair in enumerate(pairs, 1):
            try:
                if verify_pair(self.root, pair, self.experiment, self.pin) is not None:
                    counts['skipped'] += 1
                else:
                    try:
                        record = self.compare(pair)
                    finally:
                        self.collect_logs(pair)
                        self.clean()
                    classes = self.root / 'pairs' / f"{pair['id']}.npz"
                    if record['status'] == 'missing':
                        classes.unlink(missing_ok=True)
                    write_json(self.root / 'pairs' / f"{pair['id']}.json",
                               {'version': 1, 'experiment': self.experiment, 'inference_pin': self.pin,
                                'pair': pair, **record, 'produced_at': now()})
                    (self.root / 'failed' / f"{pair['id']}.json").unlink(missing_ok=True)
                    counts[record['status']] += 1
                    detail = (f"{100 * record['metrics']['changed_share']:.2f} % changed"
                              if record['status'] == 'compared' else record['reason'])
                    print(f"[{n}/{len(pairs)}] {pair['id']} {record['status']}: {detail}", flush=True)
                streak_type, streak = None, 0
            except Exception as exc:   # one pair's failure must not stop the shard
                counts['failed'] += 1
                failed.append(pair['id'])
                write_json(self.root / 'failed' / f"{pair['id']}.json",
                           {'pair': pair, 'error': type(exc).__name__, 'message': str(exc)[:2000], 'at': now()})
                print(f"[{n}/{len(pairs)}] {pair['id']} FAILED {type(exc).__name__}: {exc}", flush=True)
                streak = streak + 1 if type(exc).__name__ == streak_type else 1
                streak_type = type(exc).__name__
                if streak >= FAILURE_STREAK:
                    aborted = {'error': streak_type, 'consecutive': streak, 'last_pair': pair['id'],
                               'not_attempted': len(pairs) - n}
                    print(f'ABORTED after {streak} consecutive {streak_type} failures', flush=True)
                    break
        return {'counts': counts, 'failed': failed, 'aborted': aborted}


# ── CLI ────────────────────────────────────────────────────────────────────

def run_root(output_root: Path, run: str) -> Path:
    if not NAME.fullmatch(run):
        raise ValueError('Run names are lowercase letters, digits and hyphens')
    return output_root.resolve() / run


def live_roots(args) -> list[Path]:
    live = list(args.live_root)
    if os.environ.get('IMINT_API_OUTPUT_DIR'):
        live.append(Path(os.environ['IMINT_API_OUTPUT_DIR']))
    if getattr(args, 'manifest', None):
        live.append(args.manifest.resolve().parent)
    return live


def plan(manifest: Path, count: int) -> dict:
    """Areas, pairs and each pair's admission for y and y-1; reads only, writes nothing."""
    from imint.inference.lclu_sources import RegionalSources
    from imint.inference.lclu_storage import StorageBudget

    check_shard(0, count)
    manifest = manifest.resolve()
    with tempfile.TemporaryDirectory(prefix='vpp-year-shift-plan-') as cache:
        storage = StorageBudget(Path(cache), max_bytes=1, min_free_bytes=1)
        sources = RegionalSources(preparations(json.loads(manifest.read_text())), manifest.parent,
                                  Path(cache), storage=storage)
        check_sources(sources)
        areas = demo_areas(sources)
    pairs = plan_pairs(areas)
    served = {a['id']: set(a['years_served']) for a in areas}
    return {'label': LABEL, 'years': list(YEARS), 'shift': SHIFT, 'areas': areas, 'pairs': len(pairs),
            'admitted': sum(p['year'] in served[p['area']] and p['year'] - SHIFT in served[p['area']]
                            for p in pairs),
            'not_admitted': [p['id'] for p in pairs if not {p['year'], p['year'] - SHIFT} <= served[p['area']]],
            'shards': [len(shard(pairs, i, count)) for i in range(count)]}


def main(argv: list[str] | None = None, *, model=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    dry = commands.add_parser('plan', help='List areas, pairs and admission; run nothing, write nothing')
    dry.add_argument('--manifest', type=Path, required=True)
    dry.add_argument('--num-shards', type=int, default=1)
    run = commands.add_parser('run', help='Compare one shard of the pairs, then rewrite the summary')
    run.add_argument('--manifest', type=Path, required=True, help="The live release's serving.json")
    run.add_argument('--expect-inference-pin', required=True,
                     help="contract_pin of a live result's provenance.json 'inference'")
    run.add_argument('--shard', type=int, default=0)
    run.add_argument('--num-shards', type=int, default=1)
    summary = commands.add_parser('summarize', help='Rewrite summary.json and summary.md from the records')
    for command in (run, summary):
        command.add_argument('--run', required=True)
        command.add_argument('--output-root', type=Path, default=Path('/work/experiments/vpp-year-shift'))
        command.add_argument('--live-root', type=Path, action='append', required=True,
                             help='Live service directory the output must stay out of (repeat)')
    child = commands.add_parser('vpp-child', help='Internal: one HR-VPP read in a bounded child process')
    child.add_argument('--config', type=Path, required=True)
    child.add_argument('--config-sha256', required=True)
    child.add_argument('--east', type=int, required=True)
    child.add_argument('--north', type=int, required=True)
    child.add_argument('--vpp-year', type=int, required=True)
    child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == 'vpp-child':
        if args.east % 10 or args.north % 10:
            parser.error('Centre must be snapped to the EPSG:3006 10 m grid')
        vpp_child(args.config.resolve(), args.config_sha256, Grid(args.east, args.north),
                  args.vpp_year, args.output.resolve())
        return 0
    if args.command == 'plan':
        print(json.dumps(plan(args.manifest, args.num_shards), ensure_ascii=False))
        return 0
    check_roots(args.output_root, live_roots(args))
    root = run_root(args.output_root, args.run)
    if args.command == 'summarize':
        if not (root / 'experiment.json').is_file():
            parser.error(f'{root} has no experiment.json')
        print(json.dumps(summarize(root)['pairs']))
        return 0
    if not HEX64.fullmatch(args.expect_inference_pin):
        parser.error('The inference pin is 64 hex digits')
    check_shard(args.shard, args.num_shards)       # validate before writing anything
    if model is None:
        from imint.inference.lclu import TesseraR4
        manifest = json.loads(args.manifest.read_text())
        model = TesseraR4(manifest['checkpoint'], args.manifest.resolve().parent,
                          manifest.get('device', 'cpu'))
    scratch = root / 'scratch' / f'shard-{args.shard}-of-{args.num_shards}'
    with Shift(args.manifest, root, scratch, model, run=args.run,
               expect_pin=args.expect_inference_pin) as experiment:
        mine = shard(experiment.pairs, args.shard, args.num_shards)
        marker = root / 'done' / f'shard-{args.shard}-of-{args.num_shards}.json'
        marker.unlink(missing_ok=True)
        print(f'shard {args.shard}/{args.num_shards}: {len(mine)} of {len(experiment.pairs)} pairs, '
              f'{len(experiment.areas)} areas, run {root}', flush=True)
        outcome = experiment.run(mine)
        write_json(marker, {'version': 1, 'run': args.run, 'shard': args.shard,
                            'num_shards': args.num_shards, 'inference_pin': experiment.pin,
                            'experiment': experiment.experiment, 'pairs': len(mine), **outcome,
                            'finished_at': now()})
    totals = summarize(root)['pairs']
    print(json.dumps({'shard': args.shard, **outcome['counts'], 'aborted': outcome['aborted'],
                      'run': totals}), flush=True)
    return 1 if outcome['failed'] or outcome['aborted'] else 0


if __name__ == '__main__':
    sys.exit(main())
