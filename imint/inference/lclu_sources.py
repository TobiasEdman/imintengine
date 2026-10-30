"""Bounded on-demand preparation adapter for qualified Swedish source regions.

GeoTessera 0.10.2 needs Python >=3.12; preparation can use an explicit separate
interpreter while the resident model/API keeps its existing environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from pyproj import Transformer
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform

from imint.inference.lclu import (
    EMBEDDING, YEARS, Grid, InputUnavailable, ROOT, public_identity,
    validate_bundle, verify_file,
)
from scripts.cache_predict import contract_pin, file_identity


class RegionalSources:
    """Route each exact-year request to one independently qualified inventory."""

    def __init__(self, settings: list[dict], base: Path, cache: Path):
        if not isinstance(settings, list) or not settings:
            raise ValueError('At least one regional preparation inventory is required')
        self.sources = [RepositorySources(item, base, cache) for item in settings]

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

    def locations(self) -> list[dict]:
        return [point for source in self.sources for point in source.locations()]

    def areas(self) -> list[dict]:
        return [area for source in self.sources for area in source.areas()]


class RepositorySources:
    def __init__(self, settings: dict, base: Path, cache: Path):
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
        self.cache = cache / 'lclu-inputs'
        self.cache.mkdir(parents=True, exist_ok=True)
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

    def prepare(self, grid: Grid, year: int) -> tuple[Path, dict]:
        identity = self.identity(grid, year)
        if public_identity(file_identity(self.config_path)) != public_identity(self.settings['config']):
            raise ValueError('Preparation configuration changed; restart with a new manifest')
        producer = {str(p.relative_to(ROOT)): file_identity(p)['sha256'] for p in [
            ROOT / 'scripts/prepare_lclu.py', ROOT / 'scripts/enrich_tiles_tessera.py',
            ROOT / 'imint/inference/lclu.py', ROOT / 'imint/inference/lclu_sources.py',
            ROOT / 'imint/inference/lclu_cog.py',
            *sorted((ROOT / 'imint/training').glob('*.py')),
            ROOT / 'imint/data/wekeo_vpp.py']}
        key = contract_pin({'source': identity, 'grid': grid.describe(), 'year': year,
                            'producer': producer})
        target = self.cache / key
        if not target.exists():
            staging = Path(tempfile.mkdtemp(prefix='.preparing-', dir=self.cache))
            try:
                command = [self.python, str(ROOT / 'scripts/prepare_lclu.py'),
                           '--config', str(self.config_path), '--east', str(grid.east),
                           '--north', str(grid.north), '--year', str(year),
                           '--output', str(staging)]
                # Logs stay operator-side; no private paths/provider errors in HTTP.
                with (staging / 'preparation.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, timeout=self.timeout, check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
                entry = json.loads((staging / 'entry.json').read_text())
                validate_bundle(verify_file(entry['input'], staging), entry['input'], grid, year)
                os.rename(staging, target)
            except Exception as exc:
                if (staging / 'preparation.log').is_file():
                    logs = self.cache / 'failures'
                    logs.mkdir(exist_ok=True)
                    log_path = logs / f'{staging.name}.log'
                    os.replace(staging / 'preparation.log', log_path)
                    raise RuntimeError(f'Annual preparation failed; operator log: {log_path}') from exc
                raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        entry = json.loads((target / 'entry.json').read_text())
        if (entry['year'] != year or entry['center_3006'] != [grid.east, grid.north]
                or entry['embedding'] != EMBEDDING
                or entry['source_config'] != public_identity(self.settings['config'])):
            raise ValueError('Prepared input identity mismatch')
        path = verify_file(entry['input'], target)
        return path, entry
