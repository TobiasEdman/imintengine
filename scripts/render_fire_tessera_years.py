#!/usr/bin/env python3
"""Per-year TESSERA R4 land-cover maps for the fire showcase AOI (task 0151).

Every class comes from the annual LCLU service's own path: ``stage`` (the
municipality campaign's staging) pins and qualifies one year's sources, and
``LcluService.run`` prepares, validates, classifies and exports each 504 px
window exactly as an API request does. This script adds only the AOI window
layout, the stitch of those exported GeoTIFFs onto the fire tab's 267x559
grid, and the MANIFEST.

TESSERA and HR-VPP come from the map year, except where
``lclu.VPP_YEAR_SUBSTITUTIONS`` names another HR-VPP year (2025 -> 2024); the
MANIFEST labels that year. Nothing else is substituted, filled or skipped.
The preflight stops the run before any download if a year lacks HR-VPP
coverage or TESSERA blocks; the checkpoint, forestry/markfukt mirror and DEM
are checked by ``stage`` and the service, so a problem there fails the run
during staging of the first year.

  scope   write config/lclu/fire_karbole.json from the pinned TESSERA registry
          (operator machine; needs pyarrow, like inventory_lclu_demo.py)
  render  stage, classify and stitch (k8s/fire-tessera-render-job.yaml)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from affine import Affine

from imint.inference import lclu
from imint.inference.lclu_legend import MAP_COLORS, legend
from scripts.cache_predict import contract_pin, file_identity

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'config/lclu/fire_karbole.json'
RES = 10
HALF_OUTPUT_M = lclu.OUTPUT_SIZE * RES // 2   # an exported window spans centre ± 2520 m

# The fire tab's overlay grid (docs/js/tab-data.js): 267x559 px at 10 m.
FIRE_AOI_3006 = (521870, 6865550, 524540, 6871140)
# Two windows stacked north-south on the AOI's x. Every AOI pixel keeps at
# least 1110 m of model context to its window's edge, and the split between
# the windows falls on a pixel edge (N 6868350), so no pixel is tied.
FIRE_WINDOWS_3006 = ((523200, 6869740), (523200, 6866960))
# Both 512 px inputs lie inside S2 tile 33VWJ only (corners checked with the
# mgrs library; /data/vpp_wekeo/index.json has no other overlapping tile).
FIRE_MGRS_TILE = '33VWJ'
# HR-VPP year -> index under the data root. /data/vpp_wekeo held 33VWJ for
# 2018, 2021, 2022 and 2024 on 2026-11-06. The other years are prefetched by
# k8s/fire-tessera-vpp-gapfill-job.yaml into their own cache, because the
# serving release pins the shared index.
FIRE_VPP_INDEX = {2017: 'vpp_wekeo_fire/index.json', 2018: 'vpp_wekeo/index.json',
                  2019: 'vpp_wekeo_fire/index.json', 2020: 'vpp_wekeo_fire/index.json',
                  2021: 'vpp_wekeo/index.json', 2022: 'vpp_wekeo/index.json',
                  2023: 'vpp_wekeo_fire/index.json', 2024: 'vpp_wekeo/index.json'}
STITCH_RULE = ('Each AOI pixel takes the class from the exported window whose centre is '
               'nearest (Chebyshev distance), i.e. the one giving it the most context; '
               'the layout has no ties.')


def aoi_shape(bounds) -> tuple[int, int]:
    west, south, east, north = bounds
    if any(v % RES for v in bounds) or east <= west or north <= south:
        raise ValueError('AOI must be a nonempty box on the 10 m lattice')
    return (north - south) // RES, (east - west) // RES


def aoi_transform(bounds) -> Affine:
    aoi_shape(bounds)
    return Affine(RES, 0, bounds[0], 0, -RES, bounds[3])


def window_grids(centres) -> list[lclu.Grid]:
    if not centres or any(e % RES or n % RES for e, n in centres):
        raise ValueError('Window centres must lie on the 10 m lattice')
    return [lclu.Grid(int(e), int(n)) for e, n in centres]


def input_union(grids) -> tuple[int, int, int, int]:
    """Bounds of all 512 px model inputs: the region the sources must cover."""
    boxes = [g.bounds() for g in grids]
    return (min(b['west'] for b in boxes), min(b['south'] for b in boxes),
            max(b['east'] for b in boxes), max(b['north'] for b in boxes))


def assign_windows(bounds, grids) -> np.ndarray:
    """Per AOI pixel, the index of the window whose centre is nearest (Chebyshev).

    Raises if a pixel centre lies outside every exported window or two
    windows are equally near.
    """
    height, width = aoi_shape(bounds)
    x = bounds[0] + RES * (np.arange(width) + 0.5)
    y = bounds[3] - RES * (np.arange(height) + 0.5)
    distance = np.stack([np.maximum(np.abs(x[None, :] - g.east), np.abs(y[:, None] - g.north))
                         for g in grids])
    nearest = np.sort(distance, axis=0)
    if np.any(nearest[0] >= HALF_OUTPUT_M):
        raise ValueError('AOI pixels outside every exported window')
    if len(grids) > 1 and np.any(nearest[0] == nearest[1]):
        raise ValueError('Ambiguous window assignment; move a window centre')
    return np.argmin(distance, axis=0)


def stitch(bounds, grids, maps, assignment) -> np.ndarray:
    """Crop the exported 504 px windows onto the AOI grid, window per ``assignment``."""
    height, width = aoi_shape(bounds)
    if len(maps) != len(grids) or assignment.shape != (height, width):
        raise ValueError('One exported map per window and an AOI-shaped assignment are required')
    rows, cols = np.mgrid[0:height, 0:width]
    out = np.empty((height, width), np.uint8)
    for i, (grid, array) in enumerate(zip(grids, maps)):
        if array.shape != (lclu.OUTPUT_SIZE, lclu.OUTPUT_SIZE) or array.dtype != np.uint8:
            raise ValueError('Expected a 504x504 uint8 class map')
        b = grid.bounds(lclu.OUTPUT_SIZE)
        row0, col0 = b['north'] - bounds[3], bounds[0] - b['west']
        if row0 % RES or col0 % RES:
            raise ValueError('Window and AOI are not on one 10 m lattice')
        take = assignment == i
        out[take] = array[rows[take] + row0 // RES, cols[take] + col0 // RES]
    return out


def read_window(path: Path, grid: lclu.Grid) -> np.ndarray:
    """One exported window, checked against the grid it was requested for."""
    import rasterio
    with rasterio.open(path) as src:
        if (src.crs.to_epsg() != 3006 or src.count != 1 or src.dtypes[0] != 'uint8'
                or src.transform != grid.transform(lclu.OUTPUT_SIZE)
                or src.shape != (lclu.OUTPUT_SIZE, lclu.OUTPUT_SIZE) or src.nodata != lclu.NODATA):
            raise ValueError('Exported window does not match its requested grid')
        return src.read(1)


def display_palette() -> np.ndarray:
    """The service overlay's colours (lclu.export_map): legend colour, nodata transparent."""
    palette = np.zeros((256, 4), np.uint8)
    for key, color in MAP_COLORS.items():
        palette[key] = (*color, 255)
    return palette


def colour(classes: np.ndarray) -> np.ndarray:
    if not set(np.unique(classes).tolist()) <= set(MAP_COLORS) | {lclu.NODATA}:
        raise ValueError('Class IDs outside the served legend')
    return display_palette()[classes]


def write_year(directory: Path, classes: np.ndarray, bounds) -> dict:
    """The AOI GeoTIFF of class IDs and its RGBA PNG, both reopened and checked."""
    import rasterio
    from PIL import Image

    transform = aoi_transform(bounds)
    height, width = aoi_shape(bounds)
    if classes.shape != (height, width) or classes.dtype != np.uint8:
        raise ValueError('Classes must be uint8 on the AOI grid')
    directory.mkdir(parents=True)
    tif, png = directory / 'classification.tif', directory / 'classification.png'
    with rasterio.open(tif, 'w', driver='GTiff', width=width, height=height, count=1,
                       dtype='uint8', crs='EPSG:3006', transform=transform,
                       nodata=lclu.NODATA, compress='deflate') as dst:
        dst.write(classes, 1)
        dst.write_colormap(1, {**{k: (*v, 255) for k, v in MAP_COLORS.items()},
                               lclu.NODATA: (0, 0, 0, 0)})
    with rasterio.open(tif) as src:
        if (src.crs.to_epsg() != 3006 or src.transform != transform or src.shape != (height, width)
                or src.nodata != lclu.NODATA or not np.array_equal(src.read(1), classes)):
            raise ValueError('AOI GeoTIFF round-trip verification failed')
    Image.fromarray(colour(classes)).save(png)
    with Image.open(png) as image:
        if image.size != (width, height) or image.mode != 'RGBA':
            raise ValueError('AOI PNG round-trip verification failed')
    return {path.name: lclu.public_identity(file_identity(path)) for path in (tif, png)}


def year_plan(scope: dict, years, data: Path) -> list[dict]:
    """Map year -> HR-VPP year (lclu.vpp_year_for) and the index that holds it."""
    if not years or len(set(years)) != len(years):
        raise ValueError('Years must be nonempty and unique')
    plan = []
    for year in sorted(years):
        if year not in lclu.YEARS or year not in scope['years']:
            raise lclu.InputUnavailable(f'{year} is not a scope year')
        vpp_year = lclu.vpp_year_for(year)
        index = scope['vpp_index_by_year'].get(str(vpp_year))
        if index is None:
            raise lclu.InputUnavailable(f'No HR-VPP {vpp_year} cache named for {year}')
        plan.append({'year': year, 'vpp_year': vpp_year, 'vpp_year_substituted': vpp_year != year,
                     'vpp_cache': index, 'vpp_index': data / index})
    return plan


def check_scope(scope: dict) -> tuple[tuple, list[lclu.Grid], np.ndarray]:
    """The committed scope's AOI, windows and pixel assignment, checked for consistency."""
    bounds = tuple(scope['aoi']['bounds_3006'])
    if (scope['version'] != 1 or scope['aoi']['crs'] != 'EPSG:3006'
            or scope['aoi']['shape'] != list(aoi_shape(bounds))
            or scope['aoi']['transform'] != list(aoi_transform(bounds))[:6]):
        raise ValueError('Scope AOI grid is inconsistent')
    grids = window_grids(scope['windows_center_3006'])
    if tuple(scope['bounds_3006']) != input_union(grids):
        raise ValueError('Scope region must equal the union of the model inputs')
    return bounds, grids, assign_windows(bounds, grids)


def preflight(scope: dict, plan: list[dict]) -> None:
    """Every year's HR-VPP coverage and TESSERA blocks, before any download."""
    from rasterio.warp import transform_bounds
    from scripts.stage_lclu_demo import select_vpp

    wgs = transform_bounds('EPSG:3006', 'EPSG:4326', *scope['bounds_3006'], densify_pts=41)
    for item in plan:
        selected = select_vpp(json.loads(item['vpp_index'].read_text()), item['vpp_year'], wgs)
        absent = sorted(n for n in selected if not (item['vpp_index'].parent / n).is_file())
        if absent:
            raise lclu.InputUnavailable(
                f"{item['year']}: HR-VPP {item['vpp_year']} files absent: {absent}")
        if not any(row['year'] == item['year'] for row in scope['blocks']):
            raise lclu.InputUnavailable(f"{item['year']}: no TESSERA blocks in the scope")


def build_manifest(*, image: str, git_sha: str, run_args: dict, scope: dict, checkpoint: dict,
                   inference: dict, runtime_lock: dict, legend_file: dict, bounds, grids,
                   years: dict, produced_at: str) -> dict:
    """The run's MANIFEST (CLAUDE.md item 7) with every input identity per year."""
    if not re.fullmatch(r'[a-z0-9./_-]+@sha256:[0-9a-f]{64}', image):
        raise ValueError('A digest-pinned image is required')
    if not re.fullmatch(r'[0-9a-f]{40}', git_sha):
        raise ValueError('A full git commit SHA is required')
    if not years:
        raise ValueError('No years rendered')
    for year, record in years.items():
        vpp_year = lclu.vpp_year_for(year)
        if record['vpp_year'] != vpp_year or record['vpp_year_substituted'] != (vpp_year != year):
            raise ValueError(f'{year}: HR-VPP year disagrees with lclu.VPP_YEAR_SUBSTITUTIONS')
        if not record['vpp_files'] or len(record['windows']) != len(grids):
            raise ValueError(f'{year}: incomplete record')
    substitutions = [{**entry, 'note_en': (
        f"TESSERA {entry['year']} with HR-VPP {entry['vpp_year']}: HR-VPP {entry['year']} "
        'is not published, so this map uses the previous year (lclu.VPP_YEAR_SUBSTITUTIONS).')}
        for entry in lclu.vpp_substitutions() if entry['year'] in years]
    notes = {entry['year']: entry['note_en'] for entry in substitutions}
    inputs = {'checkpoint': checkpoint, 'scope': scope['identity'], 'runtime_lock': runtime_lock,
              'years': {str(year): {'sources_config': r['sources_config'],
                                    'vpp_index': r['vpp_index'], 'vpp_files': r['vpp_files'],
                                    'inputs': [w['input'] for w in r['windows']]}
                        for year, r in sorted(years.items())}}
    return {
        'image': image, 'image_digest': image.split('@', 1)[1], 'git_sha': git_sha,
        'runtime_lock': runtime_lock, 'run_args': run_args, 'produced_at': produced_at,
        'input_data_hash': contract_pin(inputs),
        'model': {'family': 'tessera', 'name': 'TESSERA R4', 'checkpoint': checkpoint,
                  'inference_identity_sha256': contract_pin(inference),
                  'runtime': inference['runtime']},
        'scope': {'path': scope['path'], **scope['identity']},
        'aoi': {'crs': 'EPSG:3006', 'bounds_3006': list(bounds),
                'transform': list(aoi_transform(bounds))[:6], 'shape': list(aoi_shape(bounds))},
        'windows': [{'center_3006': [g.east, g.north], 'input_bounds_3006': g.bounds(),
                     'output_bounds_3006': g.bounds(lclu.OUTPUT_SIZE),
                     'output_transform': list(g.transform(lclu.OUTPUT_SIZE))[:6]}
                    for g in grids],
        'stitch': STITCH_RULE,
        'years': {str(year): {**record, **({'note': notes[year]} if year in notes else {})}
                  for year, record in sorted(years.items())},
        'vpp_year_substitutions': substitutions,
        'historical_note': lclu.HISTORICAL_NOTE,
        'legend': {'file': 'legend.json', **legend_file,
                   'png': 'service legend colours; nodata 255 transparent'},
    }


def render(scope_path: Path, years: list[int], data: Path, out: Path, run_id: str, *,
           image: str, git_sha: str, run_args: dict) -> Path:
    from scripts.stage_lclu_demo import stage

    if not re.fullmatch(r'[A-Za-z0-9._-]+', run_id):
        raise ValueError('Run ID must be a plain directory name')
    scope = json.loads(scope_path.read_text())
    bounds, grids, assignment = check_scope(scope)
    plan = year_plan(scope, years, data)
    preflight(scope, plan)
    run = out / run_id
    run.mkdir(parents=True)  # a run directory is never reused or overwritten
    staged = {}
    for item in plan:
        root = run / 'regions' / str(item['year'])
        # Verified downloads/COGs/receipts are shared across runs in out/shared.
        stage(scope_path, item['year'], data, root, shared_root=out / 'shared',
              vpp_index=item['vpp_index'])
        staged[item['year']] = json.loads((root / 'serving.json').read_text())
    first = staged[plan[0]['year']]
    if any(m['checkpoint'] != first['checkpoint'] for m in staged.values()):
        raise ValueError('All years must use the same pinned R4 checkpoint')
    serving = run / 'serving.json'
    lclu.write_json(serving, {'version': 1, 'device': 'cpu', 'checkpoint': first['checkpoint'],
                              'preparations': [staged[item['year']]['preparation'] for item in plan]})
    service = lclu.LcluService.from_manifest(serving, run / 'service')
    lclu.write_json(run / 'legend.json', legend())
    records = {}
    for item in plan:
        year = item['year']
        config_record = staged[year]['preparation']['config']
        sources = json.loads(lclu.verify_file(config_record, run).read_text())
        vpp_files = [{'file': name, **lclu.public_identity(record)}
                     for name, record in sorted(sources['vpp_files'].items())]
        maps, windows = [], []
        for grid in grids:
            result = service.run(grid, year, lambda phase, g=grid: print(
                f'{year} window {g.east},{g.north}: {phase}', flush=True))
            if (result['vpp_year'], result['vpp_year_substituted']) != (
                    item['vpp_year'], item['vpp_year_substituted']):
                raise ValueError(f'{year}: service HR-VPP year differs from the plan')
            directory = run / 'service/lclu' / result['cache_key']
            provenance = json.loads((directory / 'provenance.json').read_text())
            used = {r['sha256'] for r in provenance['auxiliary_provenance']['vpp']['files']}
            if not used or not used <= {r['sha256'] for r in vpp_files}:
                raise ValueError(f'{year}: window read HR-VPP outside the pinned files')
            maps.append(read_window(directory / 'classification.tif', grid))
            windows.append({'center_3006': [grid.east, grid.north], 'cache_key': result['cache_key'],
                            'input': provenance['input'],
                            'provenance': result['files']['provenance.json'],
                            'classification': result['files']['classification.tif']})
        classes = stitch(bounds, grids, maps, assignment)
        year_dir = run / str(year)
        outputs = write_year(year_dir, classes, bounds)
        for i, window in enumerate(windows):
            shutil.copyfile(run / 'service/lclu' / window['cache_key'] / 'provenance.json',
                            year_dir / f'window-{i}-provenance.json')
        records[year] = {
            'tessera_year': year, 'vpp_year': item['vpp_year'],
            'vpp_year_substituted': item['vpp_year_substituted'], 'vpp_cache': item['vpp_cache'],
            'sources_config': lclu.public_identity(config_record),
            'vpp_index': lclu.public_identity(sources['vpp_index']), 'vpp_files': vpp_files,
            'windows': windows, 'outputs': outputs,
            'valid_pixels': int((classes != lclu.NODATA).sum()),
            'nodata_pixels': int((classes == lclu.NODATA).sum()),
            'class_counts': {str(k): int(n) for k, n in zip(*np.unique(classes, return_counts=True))
                             if k != lclu.NODATA}}
        print(f'{year}: {year_dir}', flush=True)
    def identity(path: Path) -> dict:
        return lclu.public_identity(file_identity(path))

    lock = ROOT / 'docker/lclu/requirements.lock'
    manifest = build_manifest(
        image=image, git_sha=git_sha, run_args=run_args,
        scope={'path': str(scope_path.relative_to(ROOT)), 'identity': identity(scope_path)},
        checkpoint=lclu.public_identity(first['checkpoint']), inference=service.identity,
        runtime_lock={'path': str(lock.relative_to(ROOT)), **identity(lock)},
        legend_file=identity(run / 'legend.json'), bounds=bounds, grids=grids, years=records,
        produced_at=datetime.now(timezone.utc).isoformat())
    lclu.write_json(run / 'MANIFEST.json', manifest)
    return run


def build_scope(registry: Path) -> dict:
    """The fire scope from the campaign's pinned TESSERA v1 registry (no network)."""
    from pyproj import Transformer
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from scripts.inventory_lclu_demo import block_complete, registry_rows

    campaign = json.loads((ROOT / 'config/lclu/municipalities.json').read_text())
    if lclu.public_identity(file_identity(registry)) != campaign['registry']:
        raise ValueError('Registry differs from the campaign pin')
    grids = window_grids(FIRE_WINDOWS_3006)
    assign_windows(FIRE_AOI_3006, grids)
    region = input_union(grids)
    wgs = transform_bounds('EPSG:3006', 'EPSG:4326', *region, densify_pts=41)
    rows = sorted((r for r in registry_rows(registry, wgs, lclu.YEARS)
                   if box(r['lon'] - .05, r['lat'] - .05, r['lon'] + .05, r['lat'] + .05)
                   .intersects(box(*wgs))),
                  key=lambda r: (r['year'], r['lon'], r['lat']))
    if not all(map(block_complete, rows)) or {r['year'] for r in rows} != set(lclu.YEARS):
        raise ValueError('Registry lacks complete TESSERA blocks for some year')
    to_wgs = Transformer.from_crs(3006, 4326, always_xy=True).transform
    west, south, east, north = FIRE_AOI_3006
    ring = [list(to_wgs(x, y)) for x, y in
            ((west, south), (east, south), (east, north), (west, north), (west, south))]
    return {
        'version': 1, 'name': 'Kårbölebranden', 'task': '0151',
        'aoi': {'crs': 'EPSG:3006', 'bounds_3006': list(FIRE_AOI_3006),
                'shape': list(aoi_shape(FIRE_AOI_3006)),
                'transform': list(aoi_transform(FIRE_AOI_3006))[:6]},
        'windows_center_3006': [list(c) for c in FIRE_WINDOWS_3006],
        'bounds_3006': list(region), 'buffer_m': 0,
        # stage() reads the qualified area from 'municipality'; here it is the AOI.
        'municipality': {'type': 'Feature', 'properties': {'name': 'Kårbölebranden'},
                         'geometry': {'type': 'Polygon', 'coordinates': [ring]}},
        'mgrs_tile': FIRE_MGRS_TILE,
        'vpp_index_by_year': {str(y): index for y, index in FIRE_VPP_INDEX.items()},
        'years': list(lclu.YEARS),
        'registry': campaign['registry'], 'registry_url': campaign['registry_url'],
        'blocks': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    scope = commands.add_parser('scope', help='write the fire scope from the pinned registry')
    scope.add_argument('--registry', type=Path, required=True, help='TESSERA v1 manifest.parquet')
    scope.add_argument('--output', type=Path, default=SCOPE)
    maps = commands.add_parser('render', help='stage, classify and stitch the years')
    maps.add_argument('--scope', type=Path, default=SCOPE)
    maps.add_argument('--years', type=int, nargs='+', required=True)
    maps.add_argument('--data', type=Path, required=True, help='read-only reference data root')
    maps.add_argument('--out', type=Path, required=True, help='writable output root')
    maps.add_argument('--run-id', required=True)
    maps.add_argument('--image', required=True, help='repository@sha256:… of this pod')
    maps.add_argument('--git-sha', required=True, help='commit the source tree was fetched at')
    maps.add_argument('--output-location', required=True,
                      help='where --out lives, recorded in the MANIFEST (e.g. PVC:subPath)')
    args = parser.parse_args()
    if args.command == 'scope':
        lclu.write_json(args.output, build_scope(args.registry.resolve()))
        return
    run_args = {'scope': str(args.scope), 'years': args.years, 'data': str(args.data),
                'out': str(args.out), 'output_location': args.output_location,
                'run_id': args.run_id}
    print(render(args.scope.resolve(), args.years, args.data.resolve(), args.out.resolve(),
                 args.run_id, image=args.image, git_sha=args.git_sha, run_args=run_args))


if __name__ == '__main__':
    main()
