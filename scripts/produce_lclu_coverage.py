#!/usr/bin/env python3
"""Heltäckande markslagskarta 2024: the annual-map service run over whole areas.

Every window is one on-demand map. The producer opens the live release's
serving manifest (same R4 checkpoint, same qualified sources) and calls
``LcluService.run`` for the window's grid, so each window carries the request
key and provenance the live service computes for that grid and year. The
inference identity must equal ``--expect-inference-pin`` (taken from a live
result's provenance) before any window runs. The producer never shares the
live service's storage: own root, own budget, own prepared inputs, no API
lock. Each prepared input is deleted once its window is exported.

Windows (EPSG:3006, 10 m, 504 x 504 px):

* ``lattice``: the national 5,040 m lattice (origin 0,0), every cell that
  intersects a municipality polygon (config/lclu/municipalities.json).
* The service admits a grid only when its centre lies inside exactly one
  qualified polygon and its 512 px input inside that region's bounds
  (``RepositorySources.years``). A lattice cell centred outside every polygon
  is refused. ``fill`` windows, centred on 10 m corners inside the polygon,
  then cover the municipality pixels of refused cells; a fill is exactly the
  map a click at its centre gives. Before producing, the plan's admission is
  checked window by window against the manifest's own sources.

Ownership in the mosaic is fixed by the plan: a lattice window owns its cell;
a fill owns the pixels of refused cells that no earlier fill covers.

Window states, one sidecar each (``windows/<id>.json``): ``produced`` (five
verified artifacts) or ``unavailable`` (preparation found no valid TESSERA
pixel, the one deterministic cause, reproduced on a second attempt; its pixels
stay nodata and are counted). A window with a verified sidecar under the same
request key is skipped. Anything else is a failure, retried on resume; three
consecutive failures of one type stop the shard. Each shard writes
``done/shard-i-of-N.json`` for its run; the mosaic waits for all N markers of
its run.

    python scripts/produce_lclu_coverage.py --dry-run --num-shards 8 [--manifest serving.json]
    python scripts/produce_lclu_coverage.py --release r1 --run-id 1 --shard 0 --num-shards 8 \\
        --expect-inference-pin <hex> \\
        --manifest /work/releases/lclu-stockholm-0124-r9/demo/serving.json \\
        --live-root /work/releases/lclu-stockholm-0124-r9 --live-root /work/campaign
    python scripts/produce_lclu_coverage.py --release r1 --run-id 1 --num-shards 8 --mosaic \\
        --wait-seconds 3600 --expect-inference-pin <hex> --manifest ... --live-root ...
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from imint.inference.lclu import (
    InputUnavailable, NODATA, OUTPUT_SIZE, SOURCE_SIZE, public_identity, write_json,
)
from scripts.cache_predict import contract_pin, file_identity

YEAR = 2024
GSD = 10
CELL = OUTPUT_SIZE * GSD          # 5,040 m lattice cell = one 504 px output window
HALF = CELL // 2
INPUT_HALF = SOURCE_SIZE * GSD // 2
STRIP = 512                       # mosaic rows per pass; one GTiff tile row
ARTIFACTS = ('classification.tif', 'overlay.png', 'legend.json', 'provenance.json', 'result.json')
HEX64 = re.compile(r'[0-9a-f]{64}')
NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,62}')
FAILURE_STREAK = 3                # consecutive failures of one type that stop a shard
# validate_bundle's message (imint/inference/lclu.py) when no valid TESSERA
# pixel is left after the county mask. The preparation child dies on it, so
# the parent only sees its operator log; this exact last line is the one
# deterministic cause. tests/lclu/test_coverage.py ties it to the real code.
NO_VALID_PIXELS = 'TESSERA saknar giltiga pixlar för platsen och året.'
UNAVAILABLE_CAUSE = f'{InputUnavailable.__module__}.{InputUnavailable.__qualname__}: {NO_VALID_PIXELS}'


def label(year: int) -> str:
    return f'Heltäckande markslagskarta {year}'


def window_slices(east: int, north: int, west: int, top: int, height: int, width: int):
    """The part of the window centred at (east, north) inside a 10 m frame.

    Returns ``((rows, cols), (window_rows, window_cols))`` — the same pixels as
    frame slices and as slices of the 504 px window — or None when disjoint.
    """
    r0, c0 = (top - north - HALF) // GSD, (east - HALF - west) // GSD
    rows = slice(max(r0, 0), min(r0 + OUTPUT_SIZE, height))
    cols = slice(max(c0, 0), min(c0 + OUTPUT_SIZE, width))
    if rows.start >= rows.stop or cols.start >= cols.stop:
        return None
    return (rows, cols), (slice(rows.start - r0, rows.stop - r0), slice(cols.start - c0, cols.stop - c0))


# ── Window plan ────────────────────────────────────────────────────────────

def load_areas(config: Path) -> list[dict]:
    """Every campaign area, its polygon in EPSG:3006 exactly as the service projects it."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    from scripts.stage_lclu_campaign import load_campaign

    project = Transformer.from_crs(4326, 3006, always_xy=True).transform
    return [{'id': a['id'], 'name': a['municipality']['properties']['name'],
             'polygon': transform(project, shape(a['municipality']['geometry'])),
             'bounds': a['bounds_3006']} for a in load_campaign(config)['areas']]


def admitting_area(areas: list[dict], east: int, north: int) -> str | None:
    """The one area whose region admits this centre, mirroring RepositorySources.years.

    Planning needs no manifest; ``check_admission`` then proves the plan
    against the manifest's own sources before anything is produced.
    """
    from shapely.geometry import Point

    point = Point(east, north)
    owners = [a['id'] for a in areas if a['polygon'].covers(point)
              and a['bounds'][0] <= east - INPUT_HALF and a['bounds'][1] <= north - INPUT_HALF
              and a['bounds'][2] >= east + INPUT_HALF and a['bounds'][3] >= north + INPUT_HALF]
    return owners[0] if len(owners) == 1 else None   # two would be an ambiguous region


def inside_mask(polygon, west: int, north: int, height: int, width: int) -> np.ndarray:
    """Pixels of a 10 m frame whose centre lies in the polygon (the clip rule everywhere)."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from shapely.geometry import box

    part = polygon.intersection(box(west, north - GSD * height, west + GSD * width, north))
    if part.is_empty:
        return np.zeros((height, width), bool)
    return geometry_mask([part], out_shape=(height, width), invert=True,
                         transform=from_origin(west, north, GSD, GSD))


def _box_sums(frame: np.ndarray, half: int) -> np.ndarray:
    """For every corner of ``frame``, the count of True pixels in the 2*half box centred on it."""
    integral = np.zeros((frame.shape[0] + 1, frame.shape[1] + 1), np.int64)
    integral[1:, 1:] = frame.cumsum(0).cumsum(1)
    rows = np.arange(frame.shape[0] + 1)
    cols = np.arange(frame.shape[1] + 1)
    r0, r1 = np.clip(rows - half, 0, frame.shape[0]), np.clip(rows + half, 0, frame.shape[0])
    c0, c1 = np.clip(cols - half, 0, frame.shape[1]), np.clip(cols + half, 0, frame.shape[1])
    return (integral[np.ix_(r1, c1)] - integral[np.ix_(r0, c1)]
            - integral[np.ix_(r1, c0)] + integral[np.ix_(r0, c0)])


def _cover(fills: list[dict], west: int, north: int, height: int, width: int) -> np.ndarray:
    covered = np.zeros((height, width), bool)
    for f in fills:
        if (part := window_slices(f['east'], f['north'], west, north, height, width)) is not None:
            covered[part[0]] = True
    return covered


def _fills_for_cell(area: dict, areas: list[dict], col: int, row: int,
                    fills: list[dict]) -> tuple[list[dict], int]:
    """Greedy maximum-coverage fills for one refused cell's pixels of one area.

    Candidates are 10 m corners inside the area polygon that the service
    admits for this area. Ties resolve to the north-west-most corner, so the
    plan is deterministic. Pixels no admissible corner can reach are counted,
    never silently dropped.
    """
    west, north = col * CELL, (row + 1) * CELL
    todo = inside_mask(area['polygon'], west, north, OUTPUT_SIZE, OUTPUT_SIZE)
    todo &= ~_cover(fills, west, north, OUTPUT_SIZE, OUTPUT_SIZE)
    if not todo.any():
        return [], 0
    # A fill reaching this cell is centred within half a window of it.
    size = 2 * OUTPUT_SIZE
    ext_west, ext_north = west - HALF, north + HALF
    corners = inside_mask(area['polygon'], ext_west - GSD // 2, ext_north + GSD // 2, size + 1, size + 1)
    added = []
    while todo.any():
        frame = np.zeros((size, size), bool)
        frame[OUTPUT_SIZE // 2:OUTPUT_SIZE // 2 + OUTPUT_SIZE,
              OUTPUT_SIZE // 2:OUTPUT_SIZE // 2 + OUTPUT_SIZE] = todo
        score = np.where(corners, _box_sums(frame, OUTPUT_SIZE // 2), -1)
        while True:
            i, j = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[i, j] <= 0:
                return added, int(todo.sum())
            east, north_c = ext_west + GSD * int(j), ext_north - GSD * int(i)
            if admitting_area(areas, east, north_c) == area['id']:
                break
            score[i, j] = -1     # rasterized as inside, but on the edge for the exact test
        fill = {'east': east, 'north': north_c}
        added.append(fill)
        todo &= ~_cover([fill], west, north, OUTPUT_SIZE, OUTPUT_SIZE)
    return added, 0


def build_plan(config: Path, selected: list[str] | None = None, year: int = YEAR) -> dict:
    """Lattice windows, their admission, and fills, for the selected areas (all by default)."""
    from shapely.geometry import box
    from shapely.prepared import prep

    areas = load_areas(config)
    known = [a['id'] for a in areas]
    chosen = known if not selected else selected
    if len(set(chosen)) != len(chosen) or not set(chosen) <= set(known):
        raise ValueError(f'Unknown or repeated area; choose from {known}')
    targets = [a for a in areas if a['id'] in chosen]
    prepared = {a['id']: prep(a['polygon']) for a in targets}

    def members(west: int, south: int) -> list[str]:
        cell = box(west, south, west + CELL, south + CELL)
        return [a['id'] for a in targets if prepared[a['id']].intersects(cell)]

    cells = set()
    for area in targets:
        w, s, e, n = area['polygon'].bounds
        cells |= {(col, row) for col in range(math.floor(w / CELL), math.floor(e / CELL) + 1)
                  for row in range(math.floor(s / CELL), math.floor(n / CELL) + 1)
                  if prepared[area['id']].intersects(box(col * CELL, row * CELL,
                                                         (col + 1) * CELL, (row + 1) * CELL))}
    windows = []
    for col, row in sorted(cells, key=lambda c: (-c[1], c[0])):
        east, north = col * CELL + HALF, row * CELL + HALF
        windows.append({'id': f'L{col:04d}-{row:04d}', 'kind': 'lattice', 'east': east, 'north': north,
                        'areas': members(col * CELL, row * CELL),
                        'region': admitting_area(areas, east, north)})
    fills, unreachable = [], {a['id']: 0 for a in targets}
    for window in windows:
        if window['region'] is not None:
            continue
        col, row = (window['east'] - HALF) // CELL, (window['north'] - HALF) // CELL
        for area in targets:
            if area['id'] in window['areas']:
                added, lost = _fills_for_cell(area, areas, col, row, fills)
                fills += [{**f, 'region': area['id']} for f in added]
                unreachable[area['id']] += lost
    for f in fills:
        windows.append({'id': f"F{f['east']:07d}-{f['north']:07d}", 'kind': 'fill',
                        'east': f['east'], 'north': f['north'],
                        'areas': members(f['east'] - HALF, f['north'] - HALF), 'region': f['region']})
    if len({w['id'] for w in windows}) != len(windows):
        raise ValueError('Duplicate window identifiers')
    return {'version': 1, 'label': label(year), 'year': year,
            'config': public_identity(file_identity(config)), 'areas': chosen,
            'lattice': {'crs': 'EPSG:3006', 'origin': [0, 0], 'cell_m': CELL, 'window_px': OUTPUT_SIZE},
            'admission': ('centre covered by exactly one qualified municipality polygon and the '
                          '512 px input inside its bounds_3006 (RepositorySources.years)'),
            'windows': windows, 'unreachable_pixels': unreachable}


def runnable(plan: dict) -> list[dict]:
    return [w for w in plan['windows'] if w['region'] is not None]


def shard(plan: dict, index: int, count: int) -> list[dict]:
    if not (type(count) is int and count >= 1 and type(index) is int and 0 <= index < count):
        raise ValueError('Shard index must satisfy 0 <= shard < num_shards')
    return runnable(plan)[index::count]


def summary(plan: dict, num_shards: int = 1) -> dict:
    per_area = {}
    for area in plan['areas']:
        lattice = [w for w in plan['windows'] if w['kind'] == 'lattice' and area in w['areas']]
        per_area[area] = {'lattice': len(lattice),
                          'admitted': sum(w['region'] is not None for w in lattice),
                          'refused': sum(w['region'] is None for w in lattice),
                          'fill': sum(w['kind'] == 'fill' and w['region'] == area for w in plan['windows']),
                          'unreachable_pixels': plan['unreachable_pixels'][area]}
    lattice = [w for w in plan['windows'] if w['kind'] == 'lattice']
    return {'areas': per_area,
            'total': {'lattice': len(lattice), 'admitted': sum(w['region'] is not None for w in lattice),
                      'refused': sum(w['region'] is None for w in lattice),
                      'fill': sum(w['kind'] == 'fill' for w in plan['windows']),
                      'runnable': len(runnable(plan))},
            'shards': [len(shard(plan, i, num_shards)) for i in range(num_shards)]}


# ── Serving manifest checks ────────────────────────────────────────────────

def preparations(manifest: dict) -> list[dict]:
    if manifest['version'] != 1 or 'preparations' not in manifest or (
            {'preparation', 'inputs'} & manifest.keys()):
        raise ValueError('Coverage runs need the regional serving manifest of a release')
    return manifest['preparations']


def check_sources(sources) -> None:
    """Staged compact-COG sources only.

    A ``remote_tiles`` source trims its band cache to budget during
    preparation (``enforce_band_budget``), which could delete the live
    service's bands; any other source type is outside what was qualified here.
    """
    wrong = [public_identity(s.settings['config'])['sha256'][:12] for s in sources.sources
             if 'cog_index' not in s.config or 'remote_tiles' in s.config]
    if wrong:
        raise ValueError(f'Only staged cog_index sources without remote_tiles are allowed; '
                         f'source configs {wrong}')


def check_admission(sources, plan: dict, year: int) -> None:
    """Each window is runnable in the plan exactly when the manifest's sources admit it."""
    from imint.inference.lclu import Grid

    wrong = [w['id'] for w in plan['windows']
             if (year in sources.years(Grid(w['east'], w['north']))) != (w['region'] is not None)]
    if wrong:
        raise ValueError(f'{len(wrong)} windows: plan admission differs from the serving manifest: '
                         f'{wrong[:20]}')


def check_manifest(manifest_path: Path, plan: dict, year: int) -> None:
    """Dry-run check of a manifest against the plan, with a throwaway cache."""
    from imint.inference.lclu_sources import RegionalSources
    from imint.inference.lclu_storage import StorageBudget

    manifest_path = manifest_path.resolve()
    with tempfile.TemporaryDirectory(prefix='lclu-coverage-check-') as cache:
        storage = StorageBudget(Path(cache), max_bytes=1, min_free_bytes=1)
        sources = RegionalSources(preparations(json.loads(manifest_path.read_text())),
                                  manifest_path.parent, Path(cache), storage=storage)
        check_sources(sources)
        check_admission(sources, plan, year)


# ── Release directory ──────────────────────────────────────────────────────

def check_roots(output: Path, live_roots: list[Path]) -> None:
    """The coverage root and every live service root must be disjoint trees."""
    output = output.resolve()
    for live in live_roots:
        live = live.resolve()
        if output == live or live in output.parents or output in live.parents:
            raise ValueError(f'Coverage output {output} overlaps the live service root {live}')


def hold_lock(path: Path):
    """An exclusive advisory lock until the returned file is closed; fails fast if taken."""
    stream = path.open('a')       # never unlinked: a new inode would bypass the lock
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f'{path} is held by another process') from None
    return stream


def publish_once(path: Path, value: dict) -> None:
    """Create a release record atomically, or require the existing one to be identical."""
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)       # fails if a concurrent shard won the race
        except FileExistsError:
            pass
        finally:
            os.unlink(temporary)
    if path.read_bytes() != data:
        raise ValueError(f'{path} records a different plan or identity; use a new release name')


def release_dir(output_root: Path, year: int, release: str) -> Path:
    if not NAME.fullmatch(release):
        raise ValueError('Release names are lowercase letters, digits and hyphens')
    return output_root.resolve() / str(year) / release


def verify_sidecar(release: Path, window: dict, pin: str) -> dict | None:
    """The window's sidecar if it verifies (and, when produced, all five artifacts); else None."""
    sidecar = release / 'windows' / f"{window['id']}.json"
    directory = release / 'windows' / window['id']
    try:
        record = json.loads(sidecar.read_text())
        if (record['version'] != 1 or record['window'] != window or record['inference_pin'] != pin
                or not HEX64.fullmatch(record['request_key'])):
            return None
        if record['status'] == 'unavailable':
            ok = (record['deterministic'] is True and record['cause'] == UNAVAILABLE_CAUSE
                  and len(record['attempts']) == 2
                  and all(a['cause'] == UNAVAILABLE_CAUSE for a in record['attempts']))
            return record if ok else None
        if record['status'] != 'produced' or set(record['files']) != set(ARTIFACTS):
            return None
        for name, identity in record['files'].items():
            if public_identity(file_identity(directory / name)) != identity:
                return None
        provenance = json.loads((directory / 'provenance.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        if (provenance['request_key'] != record['request_key'] or result['cache_key'] != record['cache_key']
                or provenance['geometry']['center'] != [window['east'], window['north']]):
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def discard(release: Path, window: dict) -> None:
    (release / 'windows' / f"{window['id']}.json").unlink(missing_ok=True)
    directory = release / 'windows' / window['id']
    if directory.exists():
        shutil.rmtree(directory)


def deterministic_cause(exc: BaseException, logs: list[Path]) -> str | None:
    """UNAVAILABLE_CAUSE when the failure is 'no valid TESSERA pixel', else None.

    In-process the type and message carry it. Through the preparation child,
    ``RepositorySources._prepare`` wraps the child's exit in a RuntimeError,
    so only the child's own operator log can show it, as its last line.
    """
    for error in (exc, exc.__cause__):
        if isinstance(error, InputUnavailable) and str(error) == NO_VALID_PIXELS:
            return UNAVAILABLE_CAUSE
    if (isinstance(exc, RuntimeError) and isinstance(exc.__cause__, subprocess.CalledProcessError)
            and len(logs) == 1):
        lines = [line for line in logs[0].read_text(encoding='utf-8', errors='replace').splitlines()
                 if line.strip()]
        if lines and lines[-1] == UNAVAILABLE_CAUSE:
            return UNAVAILABLE_CAUSE
    return None


def marker_path(release: Path, index: int, count: int) -> Path:
    return release / 'done' / f'shard-{index}-of-{count}.json'


# ── Producing windows ──────────────────────────────────────────────────────

class Producer:
    """One shard: an exclusive scratch root, its identity and admission checks, its windows."""

    def __init__(self, manifest: Path, release: Path, scratch: Path, model, plan: dict, *,
                 expect_pin: str, year: int = YEAR):
        from imint.inference.lclu_storage import MAP_BYTES, PREPARATION_BYTES, StorageLimitError

        self.manifest_path = manifest.resolve()
        self.preparations = preparations(json.loads(self.manifest_path.read_text()))
        self.release, self.scratch, self.model, self.year = release, scratch, model, year
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._lock = hold_lock(self.scratch / '.lock')
        try:
            for name in ('windows', 'failed', 'done', 'operator'):
                (release / name).mkdir(parents=True, exist_ok=True)
            self.clean_inputs()
            # Identity and request keys come from a service on the same manifest.
            self.keys = self.service()
            self.identity = self.keys.identity
            self.pin = contract_pin(self.identity)
            if self.pin != expect_pin:
                raise ValueError(f'Inference identity {self.pin} is not the expected {expect_pin}; '
                                 'the source checkout or runtime differs from the live release')
            free = shutil.disk_usage(self.scratch).free
            need = self.keys.storage.min_free_bytes + PREPARATION_BYTES + MAP_BYTES
            if free < need:
                raise StorageLimitError(f'{free} bytes free; the batch reserve needs {need}')
            check_sources(self.keys.sources)
            check_admission(self.keys.sources, plan, year)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._lock.close()

    def __enter__(self) -> 'Producer':
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
        return LcluService(self.model, sources, self.scratch, storage=storage)

    def clean_inputs(self) -> None:
        """Delete every prepared input and staging leftover (under the scratch lock)."""
        from imint.inference.lclu_storage import StorageBudget

        StorageBudget.cleanup_staging(self.scratch)
        inputs = self.scratch / 'lclu-inputs'
        for path in inputs.glob('*') if inputs.is_dir() else ():
            if HEX64.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)

    def request_key(self, window: dict) -> str:
        from imint.inference.lclu import Grid
        return self.keys.key(Grid(window['east'], window['north']), self.year)

    def _collect_logs(self, window: dict, attempt: int) -> list[Path]:
        """Move this attempt's preparation logs to the release's operator directory."""
        failures = self.scratch / 'lclu-inputs' / 'failures'
        moved = []
        for n, log in enumerate(sorted(failures.glob('*.log')) if failures.is_dir() else ()):
            target = self.release / 'operator' / 'preparation-logs' / f"{window['id']}.{attempt}.{n}.log"
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(log, target)
            moved.append(target)
        return moved

    def _produce(self, window: dict) -> dict:
        from imint.inference.lclu import Grid

        start = time.perf_counter()
        grid = Grid(window['east'], window['north'])
        service = self.service()
        result = service.run(grid, self.year, lambda phase: print(
            f"{window['id']} {phase} {time.perf_counter() - start:.1f}s", flush=True))
        target = self.release / 'windows' / window['id']
        discard(self.release, window)
        os.rename(service.output_root / result['cache_key'], target)
        record = {'version': 1, 'status': 'produced', 'release': self.release.name, 'year': self.year,
                  'inference_pin': self.pin, 'window': window,
                  'request_key': result['request_key'], 'cache_key': result['cache_key'],
                  'source': service.sources.identity(grid, self.year), 'cache_hit': result['cache_hit'],
                  'files': {name: public_identity(file_identity(target / name)) for name in ARTIFACTS},
                  'timing_ms': result['timing_ms'],
                  'produced_at': datetime.now(timezone.utc).isoformat()}
        write_json(self.release / 'windows' / f"{window['id']}.json", record)
        return record

    def run(self, window: dict, key: str) -> dict:
        """Produce the window, or record it unavailable after the deterministic cause twice."""
        attempts = []
        for attempt in (1, 2):
            try:
                record = self._produce(window)
                break
            except Exception as exc:
                logs = self._collect_logs(window, attempt)
                if deterministic_cause(exc, logs) is None:
                    raise
                attempts.append({'cause': UNAVAILABLE_CAUSE,
                                 'logs': [public_identity(file_identity(p)) for p in logs]})
                print(f"{window['id']} no valid TESSERA pixel (attempt {attempt})", flush=True)
            finally:
                self.clean_inputs()
        else:
            discard(self.release, window)
            record = {'version': 1, 'status': 'unavailable', 'release': self.release.name,
                      'year': self.year, 'inference_pin': self.pin, 'window': window, 'request_key': key,
                      'reason': 'no valid TESSERA pixel in the window after the county mask',
                      'cause': UNAVAILABLE_CAUSE, 'deterministic': True, 'attempts': attempts,
                      'produced_at': datetime.now(timezone.utc).isoformat()}
            write_json(self.release / 'windows' / f"{window['id']}.json", record)
        (self.release / 'failed' / f"{window['id']}.json").unlink(missing_ok=True)
        return record

    def produce(self, windows: list[dict]) -> dict:
        counts = {'skipped': 0, 'produced': 0, 'unavailable': 0, 'failed': 0}
        failed, unavailable, aborted = [], [], None
        streak_type, streak = None, 0
        for n, window in enumerate(windows, 1):
            try:
                key = self.request_key(window)
                record = verify_sidecar(self.release, window, self.pin)
                if record is not None and record['request_key'] == key:
                    counts['skipped'] += 1
                else:
                    record = self.run(window, key)
                    counts[record['status']] += 1
                    print(f"[{n}/{len(windows)}] {window['id']} {record['status']}", flush=True)
                if record['status'] == 'unavailable':
                    unavailable.append(window['id'])
                streak_type, streak = None, 0
            except Exception as exc:   # one window's failure must not stop the shard
                counts['failed'] += 1
                failed.append(window['id'])
                write_json(self.release / 'failed' / f"{window['id']}.json",
                           {'window': window, 'error': type(exc).__name__, 'message': str(exc)[:2000],
                            'at': datetime.now(timezone.utc).isoformat()})
                print(f"[{n}/{len(windows)}] {window['id']} FAILED {type(exc).__name__}: {exc}", flush=True)
                streak = streak + 1 if type(exc).__name__ == streak_type else 1
                streak_type = type(exc).__name__
                if streak >= FAILURE_STREAK:
                    aborted = {'error': streak_type, 'consecutive': streak, 'last_window': window['id'],
                               'not_attempted': len(windows) - n}
                    print(f'ABORTED after {streak} consecutive {streak_type} failures', flush=True)
                    break
        return {'counts': counts, 'failed': failed, 'unavailable': unavailable, 'aborted': aborted}


def open_release(args, plan: dict, model) -> Producer:
    release = release_dir(args.output_root, args.year, args.release)
    release.mkdir(parents=True, exist_ok=True)
    publish_once(release / 'plan.json', plan)
    scratch = release / 'scratch' / f'shard-{args.shard}-of-{args.num_shards}'
    producer = Producer(args.manifest, release, scratch, model, plan,
                        expect_pin=args.expect_inference_pin, year=args.year)
    try:
        publish_once(release / 'release.json', {
            'version': 1, 'label': label(args.year), 'release': args.release, 'year': args.year,
            'plan': public_identity(file_identity(release / 'plan.json')),
            'serving_manifest': public_identity(file_identity(producer.manifest_path)),
            'inference_pin': producer.pin, 'inference': producer.identity})
    except BaseException:
        producer.close()
        raise
    return producer


# ── Mosaic ─────────────────────────────────────────────────────────────────

class Ownership:
    """Which plan window supplies each mosaic pixel: its lattice window, else the first fill."""

    def __init__(self, plan: dict):
        self.windows = plan['windows']
        self.lattice = {((w['east'] - HALF) // CELL, (w['north'] - HALF) // CELL): i + 1
                        for i, w in enumerate(self.windows)
                        if w['kind'] == 'lattice' and w['region'] is not None}
        self.fills = [(i + 1, w) for i, w in enumerate(self.windows) if w['kind'] == 'fill']
        self.west = np.array([0] + [w['east'] - HALF for w in self.windows], np.int64)
        self.north = np.array([0] + [w['north'] + HALF for w in self.windows], np.int64)

    def owners(self, west: int, north: int, height: int, width: int) -> np.ndarray:
        cols = (west + GSD * np.arange(width)) // CELL
        rows = (north - GSD * (np.arange(height) + 1)) // CELL
        ucols, ci = np.unique(cols, return_inverse=True)
        urows, ri = np.unique(rows, return_inverse=True)
        table = np.array([[self.lattice.get((c, r), 0) for c in ucols] for r in urows], np.int32)
        owner = table[np.ix_(ri, ci)]
        for k, w in self.fills:
            if (part := window_slices(w['east'], w['north'], west, north, height, width)) is not None:
                view = owner[part[0]]
                view[view == 0] = k
        return owner

    def edge_distance(self, owner: np.ndarray, west: int, north: int) -> np.ndarray:
        """Pixels from each pixel to the nearest edge of its own window (-1 where unowned)."""
        height, width = owner.shape
        col = (west + GSD * np.arange(width)[None, :] - self.west[owner]) // GSD
        row = (self.north[owner] - north + GSD * np.arange(height)[:, None]) // GSD
        distance = np.minimum(np.minimum(col, OUTPUT_SIZE - 1 - col), np.minimum(row, OUTPUT_SIZE - 1 - row))
        return np.where(owner > 0, distance, -1)


class SeamCounter:
    """Class disagreement of 4-neighbour pairs: across window seams, near window edges, interior."""

    def __init__(self, band: int):
        self.band = band
        self.counts = {name: [0, 0] for name in ('seam', 'near_edge', 'interior')}
        self._last = None

    def _pairs(self, a, b, oa, ob, da, db) -> None:
        valid = (a != NODATA) & (b != NODATA)
        differ = a != b
        seam = valid & (oa != ob)
        same = valid & (oa == ob)
        near = same & ((da < self.band) | (db < self.band))
        for name, mask in (('seam', seam), ('near_edge', near), ('interior', same & ~near)):
            self.counts[name][0] += int(mask.sum())
            self.counts[name][1] += int((mask & differ).sum())

    def add(self, classes: np.ndarray, owner: np.ndarray, distance: np.ndarray) -> None:
        self._pairs(classes[:, :-1], classes[:, 1:], owner[:, :-1], owner[:, 1:],
                    distance[:, :-1], distance[:, 1:])
        self._pairs(classes[:-1], classes[1:], owner[:-1], owner[1:], distance[:-1], distance[1:])
        if self._last is not None:
            self._pairs(self._last[0], classes[:1], self._last[1], owner[:1], self._last[2], distance[:1])
        self._last = (classes[-1:].copy(), owner[-1:].copy(), distance[-1:].copy())

    def report(self) -> dict:
        out = {'band_px': self.band, 'pairs': '4-neighbour pixel pairs, both classified'}
        for name, (pairs, differ) in self.counts.items():
            out[name] = {'pairs': pairs, 'disagreeing': differ,
                         'rate': differ / pairs if pairs else None}
        seam, interior = out['seam']['rate'], out['interior']['rate']
        out['seam_to_interior'] = seam / interior if seam is not None and interior else None
        return out


def _colormap() -> dict:
    from imint.inference.lclu_legend import MAP_COLORS
    return {**{k: (*v, 255) for k, v in MAP_COLORS.items()}, NODATA: (0, 0, 0, 0)}


def _partial(directory: Path, stem: str) -> Path:
    fd, name = tempfile.mkstemp(prefix=f'.{stem}.', suffix='.partial.tif', dir=directory)
    os.close(fd)
    return Path(name)


def build_mosaic(release: Path, plan: dict, area: dict, records: dict, band: int, reserve: int) -> dict:
    """Write one area's COG at native 10 m in EPSG:3006; return its coverage and seam counts."""
    import rasterio
    import rasterio.shutil
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    from imint.inference.lclu_storage import StorageLimitError

    area_id = area['id']
    w, s, e, n = area['polygon'].bounds
    west, north = math.floor(w / GSD) * GSD, math.ceil(n / GSD) * GSD
    width, height = math.ceil(e / GSD) - west // GSD, north // GSD - math.floor(s / GSD)
    out = release / 'mosaic'
    # Upper bound: the uncompressed partial GTiff and the COG side by side.
    if shutil.disk_usage(out).free < reserve + 2 * width * height:
        raise StorageLimitError(f'{area_id}: free space below the batch reserve plus the mosaic')
    transform = from_origin(west, north, GSD, GSD)
    ownership = Ownership(plan)
    status = {i + 1: records[wdw['id']]['status'] for i, wdw in enumerate(plan['windows'])
              if wdw['id'] in records}
    unavailable = np.zeros(len(plan['windows']) + 1, bool)
    unavailable[np.array([k for k, state in status.items() if state == 'unavailable'], np.intp)] = True
    members = [(i + 1, wdw) for i, wdw in enumerate(plan['windows'])
               if area_id in wdw['areas'] and status.get(i + 1) == 'produced']
    final = out / f'{area_id}-{plan["year"]}.tif'
    partial, cog = _partial(out, final.stem), _partial(out, final.stem + '-cog')
    profile = {'driver': 'GTiff', 'width': width, 'height': height, 'count': 1, 'dtype': 'uint8',
               'crs': 'EPSG:3006', 'transform': transform, 'nodata': NODATA, 'tiled': True,
               'blockxsize': STRIP, 'blockysize': STRIP, 'compress': 'deflate'}
    seams = SeamCounter(band)
    coverage = {'inside_pixels': 0, 'classified_pixels': 0, 'nodata_pixels': 0,
                'unavailable_pixels': 0, 'uncovered_pixels': 0}
    try:
        with rasterio.open(partial, 'w', **profile) as dst:
            for r0 in range(0, height, STRIP):
                h = min(STRIP, height - r0)
                top = north - GSD * r0
                classes = np.full((h, width), NODATA, np.uint8)
                owner = ownership.owners(west, top, h, width)
                inside = inside_mask(area['polygon'], west, top, h, width)
                for k, wdw in members:
                    part = window_slices(wdw['east'], wdw['north'], west, top, h, width)
                    if part is None:
                        continue
                    frame, local = part
                    take = (owner[frame] == k) & inside[frame]
                    if not take.any():
                        continue
                    path = release / 'windows' / wdw['id'] / 'classification.tif'
                    with rasterio.open(path) as src:
                        expected = from_origin(wdw['east'] - HALF, wdw['north'] + HALF, GSD, GSD)
                        if (src.crs.to_epsg() != 3006 or src.transform != expected
                                or src.shape != (OUTPUT_SIZE, OUTPUT_SIZE)):
                            raise ValueError(f"{wdw['id']}: window raster is off its planned grid")
                        values = src.read(1, window=Window.from_slices(*local))
                    classes[frame][take] = values[take]
                owner = np.where(inside, owner, 0)
                missing = inside & unavailable[owner]
                coverage['inside_pixels'] += int(inside.sum())
                coverage['classified_pixels'] += int((classes != NODATA).sum())
                coverage['unavailable_pixels'] += int(missing.sum())
                coverage['nodata_pixels'] += int((inside & (owner > 0) & ~missing & (classes == NODATA)).sum())
                coverage['uncovered_pixels'] += int((inside & (owner == 0)).sum())
                seams.add(classes, owner, ownership.edge_distance(owner, west, top))
                dst.write(classes, 1, window=Window(0, r0, width, h))
            dst.write_colormap(1, _colormap())
        rasterio.shutil.copy(partial, cog, driver='COG', COMPRESS='DEFLATE', BLOCKSIZE=STRIP,
                             RESAMPLING='NEAREST', OVERVIEWS='AUTO')
        with rasterio.open(cog) as src, rasterio.open(partial) as ref:
            if (src.crs.to_epsg() != 3006 or src.transform != transform or src.shape != (height, width)
                    or src.nodata != NODATA or src.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG'):
                raise ValueError('Mosaic COG geometry/layout verification failed')
            for r0 in range(0, height, STRIP):
                window = Window(0, r0, width, min(STRIP, height - r0))
                if not np.array_equal(src.read(1, window=window), ref.read(1, window=window)):
                    raise ValueError('Mosaic COG pixels differ from the assembled mosaic')
        os.replace(cog, final)
    finally:
        partial.unlink(missing_ok=True)
        cog.unlink(missing_ok=True)
    return {'cog': public_identity(file_identity(final)), 'transform': list(transform)[:6],
            'shape': [height, width], 'coverage': coverage, 'seams': seams.report()}


def check_shards(release: Path, identity: dict, count: int, run_id: str) -> list[dict] | None:
    """Every shard's marker from this run, or None while one is missing.

    Raises as soon as any marker of this run records a failure, even while
    siblings are missing: a failed pod can end the Job before they write
    theirs. A marker from an earlier run counts as missing, so a resumed
    release never builds on the previous run's completion.
    """
    markers, missing = [], False
    for index in range(count):
        try:
            marker = json.loads(marker_path(release, index, count).read_text())
        except (OSError, ValueError):
            missing = True
            continue
        if marker.get('run') != run_id:
            missing = True
            continue
        if (marker['inference_pin'] != identity['inference_pin'] or marker['plan'] != identity['plan']
                or (marker['shard'], marker['num_shards']) != (index, count)):
            raise RuntimeError(f'Shard {index} marker belongs to another plan or identity')
        markers.append(marker)
    bad = [m['shard'] for m in markers if m['failed'] or m['aborted']]
    if bad:
        raise RuntimeError(f'Shards {bad} of run {run_id} recorded failed windows; resume them first')
    return None if missing else markers


def wait_for_release(release: Path, seconds: float, poll: float = 30) -> None:
    """Wait for the producers' plan.json and release.json; the mosaic Job may start first.

    Both are published atomically, so existence means complete; mosaic()
    validates their plan and identity afterwards.
    """
    deadline = time.monotonic() + seconds
    while not ((release / 'release.json').is_file() and (release / 'plan.json').is_file()):
        if time.monotonic() >= deadline:
            raise RuntimeError(f'{release} has no release metadata after {seconds:.0f} s')
        print(f'waiting for release metadata in {release}', flush=True)
        time.sleep(poll)


def wait_for_shards(release: Path, identity: dict, count: int, run_id: str,
                    seconds: int, poll: float = 30) -> list[dict]:
    deadline = time.monotonic() + seconds
    while (markers := check_shards(release, identity, count, run_id)) is None:
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Shard markers of run {run_id} still incomplete after {seconds} s')
        print(f'waiting for {count} shard markers of run {run_id}', flush=True)
        time.sleep(poll)
    return markers


def mosaic(release: Path, plan: dict, config: Path, band: int, *, count: int, run_id: str,
           expect_pin: str, wait_seconds: int = 0, poll: float = 30) -> list[dict]:
    """Per-area COG and coverage manifest once every shard of this run finished cleanly."""
    from imint.inference.lclu_legend import legend
    from imint.inference.lclu_storage import StorageBudget

    if public_identity(file_identity(config)) != plan['config']:
        raise ValueError('Campaign configuration differs from the release plan')
    deadline = time.monotonic() + wait_seconds      # one bound for metadata and markers
    wait_for_release(release, wait_seconds, poll)
    publish_once(release / 'plan.json', plan)       # verifies; never rewrites
    identity = json.loads((release / 'release.json').read_text())
    if identity['plan'] != public_identity(file_identity(release / 'plan.json')):
        raise ValueError('release.json does not bind this plan')
    if identity['inference_pin'] != expect_pin:
        raise ValueError(f"Release inference identity {identity['inference_pin']} is not {expect_pin}")
    out = release / 'mosaic'
    out.mkdir(exist_ok=True)
    (release / 'operator').mkdir(exist_ok=True)
    with hold_lock(out / '.lock'):
        for stale in out.glob('.*.partial.tif'):
            stale.unlink()
        wait_for_shards(release, identity, count, run_id, max(0.0, deadline - time.monotonic()), poll)
        records, missing = {}, []
        for window in runnable(plan):
            record = verify_sidecar(release, window, identity['inference_pin'])
            if record is None:
                missing.append(window['id'])
            else:
                records[window['id']] = record
        if missing:
            raise RuntimeError(f'{len(missing)} windows lack a verified sidecar, e.g. {missing[:10]}')
        reserve = StorageBudget.from_environment(out).min_free_bytes
        areas = {a['id']: a for a in load_areas(config)}
        manifests = []
        for area_id in plan['areas']:
            product = build_mosaic(release, plan, areas[area_id], records, band, reserve)
            windows = []
            for w in plan['windows']:
                if area_id not in w['areas']:
                    continue
                entry = {'id': w['id'], 'kind': w['kind'], 'center_3006': [w['east'], w['north']]}
                record = records.get(w['id'])
                if record is None:
                    windows.append({**entry, 'status': 'refused'})
                elif record['status'] == 'unavailable':
                    windows.append({**entry, 'status': 'unavailable', 'reason': record['reason'],
                                    'request_key': record['request_key']})
                else:
                    provenance = json.loads((release / 'windows' / w['id'] / 'provenance.json').read_text())
                    windows.append({**entry, 'status': 'produced', 'request_key': record['request_key'],
                                    'cache_key': record['cache_key'],
                                    'classification': record['files']['classification.tif'],
                                    'provenance': record['files']['provenance.json'],
                                    'input': provenance['input'], 'source': record['source']})
            sources = sorted({json.dumps(w['source'], sort_keys=True) for w in windows if 'source' in w})
            inference = identity['inference']
            manifest = {
                'version': 1, 'label': plan['label'], 'release': identity['release'], 'year': plan['year'],
                'area': {'id': area_id, 'name': areas[area_id]['name']}, 'crs': 'EPSG:3006',
                'resolution_m': GSD, 'nodata': NODATA, 'clip': 'municipality polygon, pixel-centre rule',
                'cog': product['cog'], 'transform': product['transform'], 'shape': product['shape'],
                'checkpoint': inference['model']['checkpoint'], 'embedding': inference['embedding'],
                'inference_pin': identity['inference_pin'], 'serving_manifest': identity['serving_manifest'],
                'sources': [json.loads(s) for s in sources], 'config': plan['config'],
                'plan': identity['plan'], 'legend': legend(),
                'coverage': {**product['coverage'], **summary(plan)['areas'][area_id]},
                'windows': windows, 'produced_at': datetime.now(timezone.utc).isoformat()}
            write_json(out / f"{area_id}-{plan['year']}.json", manifest)
            # Seam statistics describe model behaviour: operator-only, never in the public manifest.
            write_json(release / 'operator' / f"{area_id}-{plan['year']}-seams.json",
                       {'release': identity['release'], 'area': area_id, 'cog': product['cog'],
                        'seams': product['seams']})
            manifests.append(manifest)
            print(f"{area_id}: {product['shape'][1]}x{product['shape'][0]} px, coverage {product['coverage']}",
                  flush=True)
    return manifests


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None, *, model=None, poll: float = 30) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', type=Path, default=root / 'config/lclu/municipalities.json')
    parser.add_argument('--areas', nargs='+', help='Area ids (default: all in the config)')
    parser.add_argument('--year', type=int, choices=(YEAR,), default=YEAR)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--output-root', type=Path, default=Path('/work/coverage'))
    parser.add_argument('--release')
    parser.add_argument('--run-id', help='Names this run; the mosaic waits for its shard markers')
    parser.add_argument('--manifest', type=Path, help="The live release's serving.json")
    parser.add_argument('--expect-inference-pin',
                        help="contract_pin of a live result's provenance.json 'inference'")
    parser.add_argument('--live-root', type=Path, action='append', default=[],
                        help='Live service directory the coverage output must stay out of (repeat)')
    parser.add_argument('--dry-run', action='store_true', help='List the plan; run nothing, write nothing')
    parser.add_argument('--list', action='store_true', help='With --dry-run: one line per window')
    parser.add_argument('--mosaic', action='store_true', help='Build per-area COGs from verified windows')
    parser.add_argument('--wait-seconds', type=int, default=0,
                        help='With --mosaic: wait this long for all shard markers of the run')
    parser.add_argument('--seam-band', type=int, default=3)
    args = parser.parse_args(argv)

    plan = build_plan(args.config, args.areas, args.year)
    if args.dry_run:
        report = {'label': plan['label'], **summary(plan, args.num_shards)}
        if args.manifest:
            check_manifest(args.manifest, plan, args.year)
            report['serving_manifest'] = 'source types and admission match the plan'
        print(json.dumps(report, ensure_ascii=False))
        if args.list:
            mine = {w['id'] for w in shard(plan, args.shard, args.num_shards)}
            for w in plan['windows']:
                state = 'refused' if w['region'] is None else ('this-shard' if w['id'] in mine else 'other-shard')
                print(f"{w['id']} {w['kind']} {w['east']} {w['north']} {','.join(w['areas'])} {state}")
        return 0
    if not (args.release and args.manifest and args.live_root and args.run_id and args.expect_inference_pin):
        parser.error('--release, --run-id, --expect-inference-pin, --manifest and at least one '
                     '--live-root are required to run')
    if not NAME.fullmatch(args.run_id) or not HEX64.fullmatch(args.expect_inference_pin):
        parser.error('--run-id is lowercase letters, digits and hyphens; the pin is 64 hex digits')
    live = list(args.live_root)
    if os.environ.get('IMINT_API_OUTPUT_DIR'):
        live.append(Path(os.environ['IMINT_API_OUTPUT_DIR']))
    live.append(args.manifest.resolve().parent)
    check_roots(args.output_root, live)
    release = release_dir(args.output_root, args.year, args.release)
    if args.mosaic:
        mosaic(release, plan, args.config, args.seam_band, count=args.num_shards, run_id=args.run_id,
               expect_pin=args.expect_inference_pin, wait_seconds=args.wait_seconds, poll=poll)
        return 0
    windows = shard(plan, args.shard, args.num_shards)    # validate before writing anything
    if model is None:
        from imint.inference.lclu import TesseraR4
        manifest = json.loads(args.manifest.read_text())
        model = TesseraR4(manifest['checkpoint'], args.manifest.resolve().parent,
                          manifest.get('device', 'cpu'))
    with open_release(args, plan, model) as producer:
        marker = marker_path(producer.release, args.shard, args.num_shards)
        marker.unlink(missing_ok=True)
        print(f'shard {args.shard}/{args.num_shards}: {len(windows)} windows, release {release}', flush=True)
        outcome = producer.produce(windows)
        write_json(marker, {'version': 1, 'release': args.release, 'run': args.run_id,
                            'shard': args.shard, 'num_shards': args.num_shards,
                            'inference_pin': producer.pin,
                            'plan': public_identity(file_identity(producer.release / 'plan.json')),
                            'windows': len(windows), **outcome,
                            'finished_at': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'shard': args.shard, **outcome['counts'], 'aborted': outcome['aborted']}), flush=True)
    return 1 if outcome['failed'] or outcome['aborted'] else 0


if __name__ == '__main__':
    sys.exit(main())
