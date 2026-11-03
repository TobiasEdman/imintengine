"""Bounded on-demand preparation adapter for qualified Swedish source regions.

GeoTessera 0.10.2 needs Python >=3.12; preparation can use an explicit separate
interpreter while the resident model/API keeps its existing environment.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import selectors
import subprocess
import tempfile
import threading
import time
import weakref

from pyproj import Transformer
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform

from imint.inference.lclu import (
    EMBEDDING, YEARS, Grid, InputUnavailable, ROOT, public_identity,
    validate_bundle, verify_file, preparation_identity,
)
from scripts.cache_predict import contract_pin, file_identity
from imint.inference.lclu_storage import (
    PREPARATION_BYTES, PREPARATION_LOG_BYTES, StorageBudget,
)


_TRUNCATED = b'[Earlier preparation output truncated]\n'


def run_preparation(command: list[str], *, timeout: int, log_path: Path) -> None:
    """Drain a child while retaining at most 1 MiB of diagnostics, including errors.

    Neither disk nor memory grows with stdout/stderr volume. The deadline also
    covers draining the pipe, and the child is reaped before its allowance is
    released, including on timeout.
    """
    tail = bytearray()
    truncated = False
    keep = PREPARATION_LOG_BYTES - len(_TRUNCATED)
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as child:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        tail.extend(chunk)
                        if len(tail) > keep:
                            del tail[:-keep]
                            truncated = True
            child.wait(timeout=max(0, deadline - time.monotonic()))
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
            log_path.write_bytes((_TRUNCATED if truncated else b'') + tail)


class RegionalSources:
    """Route each exact-year request to one independently qualified inventory."""

    def __init__(self, settings: list[dict], base: Path, cache: Path, *, storage=None):
        if not isinstance(settings, list) or not settings:
            raise ValueError('At least one regional preparation inventory is required')
        self.storage = storage or StorageBudget.from_environment(cache)
        self.sources = [RepositorySources(item, base, cache, storage=self.storage) for item in settings]

    def years(self, grid: Grid) -> list[int]:
        return sorted({year for source in self.sources for year in source.years(grid)})

    def _source(self, grid: Grid, year: int):
        matches = [source for source in self.sources if year in source.years(grid)]
        if not matches:
            raise InputUnavailable('Verifierad källtäckning saknas för platsen och året.')
        if len(matches) != 1:
            raise ValueError('Overlapping regional preparation inventories')
        return matches[0]

    def identity(self, grid: Grid, year: int) -> dict:
        return self._source(grid, year).identity(grid, year)

    def prepare(self, grid: Grid, year: int) -> tuple[Path, dict]:
        return self._source(grid, year).prepare(grid, year)

    def prepared(self, grid: Grid, year: int):
        return self._source(grid, year).prepared(grid, year)

    def check_preparation(self, grid: Grid, year: int) -> None:
        self._source(grid, year).check_preparation(grid, year)

    def prefetch(self, bounds: tuple, year: int) -> dict:
        """Warm every inventory that reads remote tiles for this box and year."""
        done = []
        for source in self.sources:
            try:
                report = source.prefetch(bounds, year)
            except InputUnavailable:
                continue
            if report['state'] == 'prefetched':
                done.append(report)
        return {'state': 'prefetched' if done else 'not-needed', 'sources': done}

    def locations(self) -> list[dict]:
        tx = Transformer.from_crs(3006, 4326, always_xy=True)
        project = Transformer.from_crs(4326, 3006, always_xy=True).transform
        points = [transform(project, shape(area['geometry'])).representative_point()
                  for area in self.areas()]
        return [dict(zip(('lon', 'lat'), tx.transform(p.x, p.y))) for p in points]

    def areas(self) -> list[dict]:
        grouped = {}
        for source in self.sources:
            for area in source.areas():
                key = json.dumps({'name': area['properties']['name'],
                                  'geometry': area['geometry']}, sort_keys=True)
                if key not in grouped:
                    grouped[key] = area
                else:
                    grouped[key]['properties']['years'] = sorted(set(
                        grouped[key]['properties']['years'] + area['properties']['years']))
        return list(grouped.values())


class RepositorySources:
    def __init__(self, settings: dict, base: Path, cache: Path, *, storage=None):
        self.settings = settings
        self.config_path = verify_file(settings['config'], base)
        self.config = json.loads(self.config_path.read_text())
        if self.config['embedding'] != EMBEDDING or self.config['version'] != 1:
            raise ValueError('Unsupported preparation source contract')
        self.timeout = settings['timeout_seconds']
        if type(self.timeout) is not int or not 1 <= self.timeout <= 900:
            raise ValueError('Preparation timeout must be 1–900 seconds')
        # Resolving a venv's python symlink selects the base interpreter and
        # loses that environment's packages. Preserve the invoked executable.
        self.python = os.path.abspath(base / settings['python'])
        self.cache = cache.resolve() / 'lclu-inputs'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.storage = storage or StorageBudget.from_environment(cache)
        # One preparation per key at a time: a warm-up and a map request for
        # the same grid and year must not both stage and rename the target.
        # The API holds one worker process (its own flock), so a process lock
        # suffices; a lock lives only while some caller holds a reference.
        self._locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._locks_guard = threading.Lock()
        # Only operator-qualified regions are advertised. A registry row alone
        # is never sufficient to enable a year in the product.
        self.regions = self.config['qualified_regions']
        self.polygons = []
        for region in self.regions:
            if (region.get('country') != 'SE' or not region.get('verification')
                    or any(type(y) is not int or y not in YEARS for y in region['years'])):
                raise ValueError('Regions require Swedish coverage qualification and exact years')
            polygon = None
            if 'geometry' in region:
                polygon = shape(region['geometry'])
                if polygon.geom_type not in ('Polygon', 'MultiPolygon') or polygon.is_empty or not polygon.is_valid:
                    raise ValueError('Region must have a valid nonempty municipality polygon')
                polygon = transform(Transformer.from_crs(4326, 3006, always_xy=True).transform, polygon)
            self.polygons.append(polygon)

    def years(self, grid: Grid) -> list[int]:
        b = grid.bounds()
        return sorted({year for region, polygon in zip(self.regions, self.polygons)
                       if (polygon is None or polygon.covers(Point(grid.east, grid.north)))
                       and region['bounds_3006'][0] <= b['west']
                       and region['bounds_3006'][1] <= b['south']
                       and region['bounds_3006'][2] >= b['east']
                       and region['bounds_3006'][3] >= b['north']
                       for year in region['years']})

    def identity(self, grid: Grid, year: int) -> dict:
        if year not in self.years(grid):
            raise InputUnavailable('Verifierad källtäckning saknas för platsen och året.')
        if public_identity(file_identity(self.config_path)) != public_identity(self.settings['config']):
            raise ValueError('Preparation configuration changed; restart with a new manifest')
        return {'source_config': public_identity(self.settings['config'])}

    def locations(self) -> list[dict]:
        tx = Transformer.from_crs(3006, 4326, always_xy=True)
        points = [tx.transform(p.representative_point().x, p.representative_point().y)
                  for p in self.polygons if p is not None]
        return [{'lon': lon, 'lat': lat} for lon, lat in points]

    def areas(self) -> list[dict]:
        return [{'type': 'Feature', 'geometry': mapping(shape(r['geometry'])),
                 'properties': {'name': str(r.get('name', 'Demo')), 'years': r['years']}}
                for r in self.regions if 'geometry' in r]

    def prefetch(self, bounds: tuple, year: int) -> dict:
        """Warm the remote-tile band cache for a view or municipality box (EPSG:3006).

        Staged sources have nothing to fetch. The box is clipped to the
        qualified regions offering the year, so a request never reaches beyond
        what the service would serve.
        """
        if 'remote_tiles' not in self.config:
            return {'state': 'not-needed'}
        west, south, east, north = bounds
        boxes = {(max(west, r['bounds_3006'][0]), max(south, r['bounds_3006'][1]),
                  min(east, r['bounds_3006'][2]), min(north, r['bounds_3006'][3]))
                 for r in self.regions if year in r['years']}
        boxes = sorted(b for b in boxes if b[0] < b[2] and b[1] < b[3])
        if not boxes:
            raise InputUnavailable('Verifierad källtäckning saknas för området och året.')
        if public_identity(file_identity(self.config_path)) != public_identity(self.settings['config']):
            raise ValueError('Preparation configuration changed; restart with a new manifest')
        reports = []
        for box in boxes:
            command = [self.python, str(ROOT / 'scripts/prefetch_lclu.py'),
                       '--config', str(self.config_path), '--bounds', *map(str, box),
                       '--year', str(year)]
            done = subprocess.run(command, cwd=ROOT, timeout=self.timeout,
                                  capture_output=True, text=True)
            if done.returncode:
                # Operator-side only, like preparation failures.
                logs = self.cache / 'failures'
                logs.mkdir(exist_ok=True)
                log = logs / f'prefetch-{year}-{int(box[0])}-{int(box[1])}.log'
                log.write_text(done.stdout + done.stderr)
                raise RuntimeError(f'Annual prefetch failed; operator log: {log}')
            reports.append(json.loads(done.stdout.strip().splitlines()[-1]))
        return {'state': 'prefetched', 'boxes': reports}

    def _lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.Lock()
            return lock

    def _target(self, grid: Grid, year: int) -> Path:
        identity = self.identity(grid, year)
        producer = preparation_identity()
        key = contract_pin({'source': identity, 'grid': grid.describe(), 'year': year,
                            'producer': producer})
        return self.cache / key

    def check_preparation(self, grid: Grid, year: int) -> None:
        if not self._target(grid, year).is_dir():
            self.storage.check(PREPARATION_BYTES)

    def prepare(self, grid: Grid, year: int) -> tuple[Path, dict]:
        """Warm an input; callers reading it later must use the prepared lease."""
        with self.prepared(grid, year) as result:
            return result

    @contextmanager
    def prepared(self, grid: Grid, year: int):
        target = self._target(grid, year)
        with self.storage.lease(target):
            with self._lock(target.name):
                if not target.exists():
                    log = self.cache / 'failures' / f'{target.name}.log'
                    with self.storage.reserve(PREPARATION_BYTES, paths=(target, log)):
                        self._prepare(grid, year, target)
                        log.unlink(missing_ok=True)
            entry = json.loads((target / 'entry.json').read_text())
            if (entry['year'] != year or entry['center_3006'] != [grid.east, grid.north]
                    or entry['embedding'] != EMBEDDING
                    or entry['source_config'] != public_identity(self.settings['config'])):
                raise ValueError('Prepared input identity mismatch')
            path = verify_file(entry['input'], target)
            yield path, entry

    def _prepare(self, grid: Grid, year: int, target: Path) -> None:
        staging = Path(tempfile.mkdtemp(prefix='.preparing-', dir=self.cache))
        try:
            command = [self.python, str(ROOT / 'scripts/prepare_lclu.py'),
                       '--config', str(self.config_path), '--east', str(grid.east),
                       '--north', str(grid.north), '--year', str(year),
                       '--output', str(staging)]
            # Logs stay operator-side; no private paths/provider errors in HTTP.
            run_preparation(command, timeout=self.timeout,
                            log_path=staging / 'preparation.log')
            entry = json.loads((staging / 'entry.json').read_text())
            validate_bundle(verify_file(entry['input'], staging), entry['input'], grid, year)
            os.rename(staging, target)
        except Exception as exc:
            if (staging / 'preparation.log').is_file():
                logs = self.cache / 'failures'
                logs.mkdir(exist_ok=True)
                log_path = logs / f'{target.name}.log'
                os.replace(staging / 'preparation.log', log_path)
                raise RuntimeError(f'Annual preparation failed; operator log: {log_path}') from exc
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
