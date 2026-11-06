#!/usr/bin/env python3
"""Read-only acceptance audit of every requested LCLU area/year.

Checks saved qualification inputs/results and the optional live catalogue.
Does not download data, run inference, change manifests, or certify accuracy.
Prints JSON; redirect it to a local evidence file. An incomplete audit exits 1.

A qualified year whose HR-VPP year comes from lclu.VPP_YEAR_SUBSTITUTIONS is
reported as ``qualified_substituted_vpp``, never as plain ``qualified``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from pyproj import Transformer
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box, shape
from shapely.ops import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.inference.lclu import (
    EMBEDDING, Grid, public_identity, public_provenance, validate_bundle, verify_file,
    vpp_substitutions, vpp_year_for,
)
from scripts.cache_predict import contract_pin, file_identity
from scripts.inference_comparison import _file_snapshot
from scripts.prepare_lclu import verify_aux_metadata
from scripts.stage_lclu_campaign import block_key, load_campaign
from scripts.stage_lclu_demo import select_vpp

R4_SHA256 = '1270bb2af88074840020f3b481ed5e98a72f0b7ceb51befac2b8fa4675385668'
GAPS = {'missing_vpp', 'missing_tessera', 'missing_static_aux'}
SUBSTITUTED = 'qualified_substituted_vpp'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def pinned(path, identity):
    require(public_identity(file_identity(path)) == public_identity(identity),
            f'Identity mismatch: {path.name}')
    return path


def cells_by_key(cells, expected):
    indexed = {(c['area_id'], c['year']): c for c in cells}
    require(len(indexed) == len(cells), 'Duplicate area/year cells')
    require(set(indexed) == expected, 'Coverage does not include exactly the requested area/years')
    return indexed


class SourceVerifier:
    """Read existing stat-bound SHA receipts, otherwise hash once; never write."""

    def __init__(self):
        self.checked = {}
        self.counts = Counter()

    def verify(self, record, base, receipts):
        path = (base / record['path']).resolve()
        identity = public_identity(record)
        snapshot = list(_file_snapshot(path.stat()))
        require(path.is_file() and path.stat().st_size == identity['size'],
                f'Source size/type mismatch: {path.name}')
        key = contract_pin({'path': str(path), **identity})
        if self.checked.get(key) == snapshot:
            return path
        expected = {'identity': identity, 'stat': snapshot}
        try:
            receipt = read(receipts / f'{key}.json')
        except (ValueError, OSError):
            receipt = None
        if receipt == expected:
            self.counts['existing_sha256_receipts'] += 1
        else:
            verify_file(record, base)
            self.counts['sha256_rehashed'] += 1
        require(list(_file_snapshot(path.stat())) == snapshot,
                f'Source changed during audit: {path.name}')
        self.checked[key] = snapshot
        return path


def identities(records):
    return sorted((public_identity(r)['sha256'], r['size']) for r in records)


def verify_sources(config, base, entry, area, plan, grid, year, sources):
    checking = base / f'sources-{year}-checking.json'
    original = read(pinned(checking, entry['source_config']))
    require(original['qualified_regions'] == [] and
            {**original, 'qualified_regions': config['qualified_regions']} == config,
            'Serving sources differ from the qualified preparation configuration')
    receipts = (base / config['identity_cache']).resolve()
    index_path, skg_path = verify_aux_metadata(config, base)
    index = read(index_path)
    vpp_year = vpp_year_for(year)
    wgs = transform_bounds('EPSG:3006', 'EPSG:4326', *area['bounds_3006'], densify_pts=41)
    selected = select_vpp(index, vpp_year, wgs)
    require(set(selected) == set(config['vpp_files']), 'Serving VPP inventory differs from the area')
    for name, record in config['vpp_files'].items():
        sources.verify({**record, 'path': name}, index_path.parent, receipts)
    aux = entry['auxiliary_provenance']
    vpp = aux['vpp']
    require(vpp['year'] == vpp_year and vpp['season'] == 1 and
            vpp['index'] == public_identity(config['vpp_index']),
            'Qualification VPP identity differs from serving sources')
    bounds = grid.bounds()
    window = tuple(bounds[k] for k in ('west', 'south', 'east', 'north'))
    window_wgs = transform_bounds('EPSG:3006', 'EPSG:4326', *window)
    window_vpp = select_vpp(index, vpp_year, window_wgs)
    require(sorted(vpp['files'], key=lambda r: r['file']) ==
            [{'file': name, **public_identity(config['vpp_files'][name])} for name in sorted(window_vpp)],
            'Qualification VPP rasters differ from the window sources')

    forest = aux['forestry']
    manifest = read(skg_path)
    require(forest['mirror_manifest_sha256'] == config['skg_manifest']['sha256']
            and forest['rasters'] and forest['counties'], 'Qualification forestry identity is incomplete')
    for relative, record in forest['rasters'].items():
        derived = manifest['derived'][relative]
        require(record['sha256'] == derived['sha256'] and
                record['source_sha256'] == manifest['files'][derived['from']]['sha256'],
                'Qualification forestry rasters differ from the mirror')
    required = set(forest['rasters']) | {name for name, meta in manifest['derived'].items()
                if 'bounds' in meta and box(*meta['bounds']).intersects(box(*area['bounds_3006']))}
    for name in required:
        path = skg_path.parent / name
        sources.verify({'path': name, 'sha256': manifest['derived'][name]['sha256'],
                        'size': path.stat().st_size}, skg_path.parent, receipts)
    require(config['dem_sources'], 'No serving DEM sources')
    for record in config['dem_sources']:
        sources.verify(record, base, receipts)
    require(identities(aux['dem']['files']) == identities(config['dem_sources']),
            'Qualification DEM identity differs from serving sources')

    cog_path = verify_file(config['cog_index'], base)
    cog = read(cog_path)
    require(cog['version'] == 1 and cog['embedding'] == EMBEDDING, 'Wrong COG contract')
    wanted = {block_key(b): b for b in plan['blocks']
              if b['year'] == year and block_key(b) in area['block_keys']}
    actual = {block_key(b): b for b in cog['blocks']}
    require(set(actual) == set(wanted) and len(actual) == len(cog['blocks']),
            'Serving TESSERA blocks differ from the requested area/year')
    observed = []
    west, south, east, north = window_wgs
    window_blocks = 0
    for key, block in actual.items():
        row = wanted[key]
        require([r['size'] for r in block['original'][:2]] == [row['grid_size'], row['scales_size']],
                'COG original sizes differ from the pinned registry')
        for field in ('vectors', 'scales'):
            sources.verify(block[field], cog_path.parent, receipts)
        if (block['lon'] + .05 >= west and block['lon'] - .05 <= east and
                block['lat'] + .05 >= south and block['lat'] - .05 <= north):
            window_blocks += 1
            observed.extend(block[field] for field in ('vectors', 'scales'))
    emb = entry['embedding_provenance']
    require(emb['cog_index'] == public_identity(config['cog_index']) and
            emb['storage'] == 'compact-cog' and 1 <= window_blocks <= 4 and
            emb['blocks'] == window_blocks and identities(emb['files']) == identities(observed),
            'Qualification TESSERA identity differs from serving sources')


def verify_campaign(campaign: Path, work: Path, api: str | None = None) -> dict:
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'cells': [], 'errors': [], 'scope': 'Qualification evidence and catalogue; not accuracy or browser QA'}
    try:
        plan = load_campaign(campaign)
        areas = {a['id']: a for a in plan['areas']}
        expected = {(a, y) for a in areas for y in plan['years']}
        serving = read(work/'serving.json')
        pinned(campaign, serving['campaign'])
        require(serving['checkpoint']['sha256'] == R4_SHA256, 'Unapproved R4 checkpoint')
        verify_file(serving['checkpoint'], work)
        preflight = read(work/'preflight.json')
        require(preflight['years'] == sorted(plan['years']), 'Not all inventoried years were requested')
        before = cells_by_key(preflight['cells'], expected)
        coverage = cells_by_key(serving['coverage'], expected)
        require(serving['version'] == 1 and serving['device'] == 'cpu', 'Unsupported serving manifest')
        configs = {}
        sources = SourceVerifier()
        for preparation in serving['preparations']:
            require(type(preparation['timeout_seconds']) is int and 1 <= preparation['timeout_seconds'] <= 900,
                    'Invalid preparation timeout')
            python = work / preparation['python']
            require(python.is_file() and os.access(python, os.X_OK), 'Preparation interpreter is unavailable')
            path = verify_file(preparation['config'], work)
            require(path not in configs, 'Duplicate serving configuration')
            configs[path] = read(path)
        used_configs, static_by_area = set(), {}
        for key, cell in coverage.items():
            area_id, year = key
            vpp_year = vpp_year_for(year)
            substituted = vpp_year != year
            detail = {'area_id': area_id, 'year': year, 'vpp_year': vpp_year,
                      'status': SUBSTITUTED if cell['status'] == 'qualified' and substituted
                      else cell['status']}
            try:
                initial = before[key]
                require(all(c.get('vpp_year') == vpp_year and c.get('vpp_year_substituted') is substituted
                            for c in (initial, cell)),
                        'Cell HR-VPP year differs from the explicit substitution table')
                if initial['status'] in GAPS:
                    require(cell['status'] == initial['status'], 'A source gap was advertised as qualified')
                    require(cell.get('vpp_missing_metrics') == initial.get('vpp_missing_metrics')
                            and cell.get('vpp_missing_files') == initial.get('vpp_missing_files'),
                            'Gap explanation differs from preflight')
                    if cell['status'] == 'missing_vpp':
                        require(cell.get('vpp_missing_metrics') or cell.get('vpp_missing_files'),
                                'Missing VPP must name the absent metrics/files')
                else:
                    require(initial['status'] == 'source_ready' and cell['status'] == 'qualified',
                            'Source-ready cell has no completed model qualification')
                    root = work/'regions'/area_id/str(year)
                    config_path = (root/f'sources-{year}.json').resolve()
                    require(config_path in configs, 'Qualified cell has no serving configuration')
                    used_configs.add(config_path)
                    config = configs[config_path]
                    require(config['version'] == 1 and config['embedding'] == EMBEDDING, 'Wrong embedding contract')
                    regions = config['qualified_regions']
                    require(len(regions) == 1 and regions[0]['years'] == [year]
                            and regions[0]['geometry'] == areas[area_id]['municipality']['geometry']
                            and regions[0]['bounds_3006'] == areas[area_id]['bounds_3006']
                            and regions[0]['country'] == 'SE'
                            and regions[0]['name'] == areas[area_id]['municipality']['properties']['name'],
                            'Serving region/year differs from the requested area')
                    entry_path = root/f'qualification-{year}/entry.json'
                    entry = read(pinned(entry_path, cell['qualification']))
                    require(regions[0]['verification'] == cell['qualification']['sha256'],
                            'Serving configuration names a different qualification')
                    centre = shape(areas[area_id]['municipality']['geometry']).representative_point()
                    grid = Grid.from_lonlat(centre.x, centre.y)
                    polygon = shape(regions[0]['geometry'])
                    require(polygon.geom_type in ('Polygon', 'MultiPolygon') and polygon.is_valid
                            and not polygon.is_empty, 'Invalid serving polygon')
                    projected = transform(Transformer.from_crs(4326, 3006, always_xy=True).transform, polygon)
                    bounds = grid.bounds()
                    require(projected.covers(Point(grid.east, grid.north)) and
                            box(*regions[0]['bounds_3006']).covers(box(
                                *(bounds[k] for k in ('west', 'south', 'east', 'north')))),
                            'Serving bounds do not contain the qualification window')
                    require(entry['year'] == year and entry['center_3006'] == [grid.east, grid.north]
                            and entry['public_area'] is True and entry['embedding'] == EMBEDDING,
                            'Qualification metadata differs from the requested window/year')
                    public_provenance(entry)
                    config_base = config_path.parent
                    verify_sources(config, config_base, entry, areas[area_id], plan, grid, year, sources)
                    static = {name: config[name] for name in ('skg_manifest', 'skg_vrts', 'dem_sources')}
                    require(static_by_area.setdefault(area_id, static) == static,
                            'Static auxiliary sources differ across years of the same area')
                    input_path = verify_file(entry['input'], entry_path.parent)
                    detail.update(validate_bundle(input_path, entry['input'], grid, year))
                    prediction = np.load(pinned(entry_path.parent/'prediction.npy', cell['prediction']),
                                         allow_pickle=False)
                    require(prediction.shape == (504, 504) and prediction.dtype == np.uint8
                            and np.all(prediction < 28), 'Prediction is not a native 504px R4 class array')
                    classes = [int(v) for v in np.unique(prediction)]
                    require(classes == cell['class_ids'], 'Prediction class inventory mismatch')
                    detail.update(class_ids=classes, center_class=int(prediction[252, 252]),
                                  input=public_identity(entry['input']), prediction=cell['prediction'])
            except (KeyError, ValueError, OSError, TypeError, IndexError) as error:
                detail['error'] = str(error)
                result['errors'].append(f'{area_id}/{year}: {error}')
            result['cells'].append(detail)
        require(used_configs == set(configs), 'Serving configuration advertises an unqualified area/year')
        result['counts'] = dict(Counter(c['status'] for c in result['cells']))
        result['source_verification'] = dict(sources.counts)
        result['campaign'] = public_identity(serving['campaign'])
        if api:
            def get(route):
                with urlopen(api.rstrip('/') + route, timeout=30) as response:
                    return json.load(response)
            info = get('/v1/lclu')
            require(info['ready'], 'Live API is not ready')
            live_areas = {a['properties']['name']: a for a in info['areas']}
            require(len(live_areas) == len(info['areas']) == len(areas), 'Live area count/names mismatch')
            require(sorted(info['candidate_years']) == sorted(plan['years']), 'Live year inventory mismatch')
            require(info['vpp_substitutions'] == vpp_substitutions(),
                    'Live HR-VPP year notice differs from the substitution table')
            for area_id, area in areas.items():
                name = area['municipality']['properties']['name']
                years = sorted(y for a, y in expected if a == area_id and coverage[(a, y)]['status'] == 'qualified')
                require(live_areas[name]['geometry'] == area['municipality']['geometry']
                        and sorted(live_areas[name]['properties']['years']) == years,
                        f'Live catalogue differs for {name}')
                centre = shape(area['municipality']['geometry']).representative_point()
                availability = get('/v1/lclu/availability?' + urlencode({'lon': centre.x, 'lat': centre.y}))
                require(sorted(availability['years']) == years
                        and sorted(availability['unavailable_years']) == sorted(set(plan['years']) - set(years)),
                        f'Live year availability differs for {name}')
                require(availability['geometry'] == Grid.from_lonlat(centre.x, centre.y).describe(),
                        f'Live native grid differs for {name}')
            result['live_catalogue'] = 'passed'
        result['passed'] = not result['errors']
    except (KeyError, ValueError, OSError, TypeError, IndexError) as error:
        result['errors'].append(str(error))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--api', help='Optional live API base URL; GET requests only')
    args = parser.parse_args()
    report = verify_campaign(args.campaign.resolve(), args.work.resolve(), args.api)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
