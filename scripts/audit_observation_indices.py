#!/usr/bin/env python3
"""Audit frozen observation identities and manifest geometry, never model scores.

Only the fixed identity/geometry columns below are decoded. Full file bytes are
hashed before parsing; this is not a claim that outcome-bearing bytes were never
read. No validator parquet, image, feature file or model is opened. This audit
does not change a split or grant permission to train/score on any population.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import platform
import resource
import runpy
import sys
import time

from scripts import crop_holdout_guard as guard

SCHEMA = 'imint-observation-index-audit-inputs-v1'
INPUTS = {
    'lucas': ('lucas/lucas_tile_index.parquet',
              ('tile_name', 'point_id', 'Easting', 'Northing', 'Year', 'row', 'col')),
    'nfi': ('nfi/nfi_index_unified_v2_512.parquet',
            ('tile_name', 'TractID', 'PlotID', 'Easting', 'Northing', 'Year', 'row', 'col')),
    'tiles': ('lucas/tile_bbox_manifest.parquet', ('tile_name', 'easting', 'northing', 'year')),
}
GEOMETRY = {'epsg': 3006, 'tile_size_px': 512, 'gsd_m': 10, 'near_distance_m': 10}
MAX_ROWS = 100_000
MAX_PAIRS = 2_000_000


def integer(value):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or int(value) != value or value < 0):
        raise guard.HoldoutError('invalid nonnegative integer metadata')
    return int(value)


def coordinate(value):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value)):
        raise guard.HoldoutError('invalid finite coordinate metadata')
    return float(value)


def year(value):
    value = integer(value)
    if not 1900 <= value <= 2100:
        raise guard.HoldoutError('invalid year metadata')
    return value


def validate_spec(spec):
    if (set(spec) != {'schema', 'split_manifest_sha256', 'split_source_git_sha', 'inputs', 'geometry'}
            or spec['schema'] != SCHEMA or spec['geometry'] != GEOMETRY
            or set(spec['inputs']) != set(INPUTS)):
        raise guard.HoldoutError('unexpected audit input contract')
    if (not isinstance(spec['split_manifest_sha256'], str)
            or not guard.SHA256.fullmatch(spec['split_manifest_sha256'])
            or not isinstance(spec['split_source_git_sha'], str)
            or not guard.GIT_SHA.fullmatch(spec['split_source_git_sha'])):
        raise guard.HoldoutError('invalid frozen identity pins')
    for entry in spec['inputs'].values():
        if (not isinstance(entry, dict) or set(entry) != {'sha256', 'size_bytes'}
                or not isinstance(entry['sha256'], str)
                or not guard.SHA256.fullmatch(entry['sha256'])
                or type(entry['size_bytes']) is not int
                or not 0 < entry['size_bytes'] <= 32 << 20):
            raise guard.HoldoutError('invalid index byte identity')


def read_table(path, pin, columns, forbidden_sha):
    # Both checks precede open, including a validator copied under another name.
    if path.name == guard.VALIDATOR or pin['sha256'] == forbidden_sha:
        raise guard.HoldoutError('validator parquet is outside this audit')
    raw = guard.read_pinned(path, pin['sha256'])
    if len(raw) != pin['size_bytes']:
        raise guard.HoldoutError('index size mismatch')
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(io.BytesIO(raw))
    if not 0 < parquet.metadata.num_rows <= MAX_ROWS:
        raise guard.HoldoutError('index row limit exceeded or empty index')
    if not set(columns) <= set(parquet.schema_arrow.names):
        raise guard.HoldoutError('required identity columns absent')
    return parquet.read(columns=list(columns), use_threads=False).to_pylist()


def observations(rows, kind):
    out = {}
    for r in rows:
        tile = guard._tile(r['tile_name'])
        identity = (guard.point_id(r['point_id']) if kind == 'lucas'
                    else (integer(r['TractID']), integer(r['PlotID'])))
        key = (tile, identity)
        if key in out:
            raise guard.HoldoutError('duplicate observation key')
        row, col = integer(r['row']), integer(r['col'])
        if row >= 512 or col >= 512:
            raise guard.HoldoutError('pixel index outside declared tile geometry')
        out[key] = {'tile': tile, 'id': identity, 'year': year(r['Year']),
                    'x': coordinate(r['Easting']), 'y': coordinate(r['Northing']),
                    'row': row, 'col': col}
    return out


def tile_geometries(rows):
    # Load the repo's pure geometry module without importing the model engine.
    path = Path(__file__).resolve().parents[1] / 'imint/training/tile_config.py'
    config = runpy.run_path(str(path))['TileConfig'](size_px=512, gsd_m=10)
    out, unresolved = {}, 0
    seen = set()
    for r in rows:
        name = guard._tile(r['tile_name'])
        if name in seen:
            raise guard.HoldoutError('duplicate tile metadata')
        seen.add(name)
        if any(r[k] is None for k in ('easting', 'northing', 'year')):
            unresolved += 1
            continue
        out[name] = {**config.bbox_from_center(coordinate(r['easting']), coordinate(r['northing'])),
                     'year': year(r['year'])}
    return out, unresolved


def population(rows):
    by_id = defaultdict(list)
    for r in rows:
        by_id[r['id']].append(r)
    return {
        'rows': len(rows), 'unique_ids': len(by_id),
        'unique_tiles': len({r['tile'] for r in rows}),
        'rows_by_year': dict(sorted(Counter(str(r['year']) for r in rows).items())),
        'ids_on_multiple_tiles': sum(len({r['tile'] for r in group}) > 1 for group in by_id.values()),
        'ids_in_multiple_years': sum(len({r['year'] for r in group}) > 1 for group in by_id.values()),
        'ids_with_multiple_coordinates': sum(len({(r['x'], r['y']) for r in group}) > 1 for group in by_id.values()),
    }


def identity_overlap(left, right):
    li, ri = {r['id'] for r in left}, {r['id'] for r in right}
    ly, ry = {(r['id'], r['year']) for r in left}, {(r['id'], r['year']) for r in right}
    shared = li & ri
    same_year = {p for p, y in ly & ry}
    return {'shared_ids': len(shared), 'left_rows_with_shared_id': sum(r['id'] in shared for r in left),
            'right_rows_with_shared_id': sum(r['id'] in shared for r in right),
            'shared_id_years': len(ly & ry), 'shared_ids_same_year': len(same_year),
            'shared_ids_only_different_years': len(shared - same_year)}


def near_points(left, right, *, same_namespace):
    """Distinct identity pairs within 10 m; a diagnostic radius, not a split rule."""
    cells = defaultdict(list)
    for r in right:
        cells[(math.floor(r['x'] / 10), math.floor(r['y'] / 10))].append(r)
    pairs, same_year, exact, left_rows = set(), set(), set(), 0
    comparisons = 0
    for a in left:
        cx, cy = math.floor(a['x'] / 10), math.floor(a['y'] / 10)
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for b in cells[(cx+dx, cy+dy)]:
                    comparisons += 1
                    if comparisons > MAX_PAIRS:
                        raise guard.HoldoutError('point proximity comparison budget exceeded')
                    if same_namespace and a['id'] == b['id']:
                        continue
                    distance2 = (a['x']-b['x'])**2 + (a['y']-b['y'])**2
                    if distance2 > 100:
                        continue
                    pair = (a['id'], b['id'])
                    pairs.add(pair)
                    found = True
                    if a['year'] == b['year']:
                        same_year.add(pair)
                    if distance2 == 0:
                        exact.add(pair)
        left_rows += found
    return {'radius_m': 10, 'pair_semantics': 'ordered(left_role_id,right_role_id)',
            'distinct_id_pairs': len(pairs),
            'pairs_with_same_year': len(same_year), 'pairs_at_identical_coordinates': len(exact),
            'left_rows_with_nearby_other_identity': left_rows}


def geometry_consistency(rows, tiles):
    missing = year_mismatch = pixel_mismatch = 0
    for r in rows:
        b = tiles.get(r['tile'])
        if b is None:
            missing += 1
            continue
        year_mismatch += r['year'] != b['year']
        col = math.floor((r['x'] - b['west']) / 10)
        row = math.floor((b['north'] - r['y']) / 10)
        pixel_mismatch += (row, col) != (r['row'], r['col'])
    return {'rows': len(rows), 'missing_manifest_geometry': missing,
            'year_mismatch': year_mismatch, 'pixel_mismatch': pixel_mismatch}


def tile_overlap(left_names, right_names, tiles):
    cells = defaultdict(list)
    for name in right_names:
        if name in tiles:
            b = tiles[name]
            cells[(b['west']//5120, b['south']//5120)].append(name)
    left_hits, right_hits = set(), set()
    pairs = same_year = comparisons = 0
    for name in left_names:
        if name not in tiles:
            continue
        a = tiles[name]
        cx, cy = a['west']//5120, a['south']//5120
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells[(cx+dx, cy+dy)]:
                    comparisons += 1
                    if comparisons > MAX_PAIRS:
                        raise guard.HoldoutError('tile overlap comparison budget exceeded')
                    b = tiles[other]
                    if (min(a['east'], b['east']) > max(a['west'], b['west'])
                            and min(a['north'], b['north']) > max(a['south'], b['south'])):
                        pairs += 1
                        same_year += a['year'] == b['year']
                        left_hits.add(name)
                        right_hits.add(other)
    return {'tile_pairs_positive_area': pairs, 'pairs_same_year': same_year,
            'pairs_different_years': pairs-same_year,
            'left_tiles_overlapping': len(left_hits), 'right_tiles_overlapping': len(right_hits),
            'left_tiles_missing_geometry': len(set(left_names) - tiles.keys()),
            'right_tiles_missing_geometry': len(set(right_names) - tiles.keys())}


def audit(root, spec):
    validate_spec(spec)
    split_dir = Path(root) / 'distill/crop_split'
    try:
        frozen = guard.load_crop_holdout(split_dir, manifest_sha256=spec['split_manifest_sha256'],
                                         source_git_sha=spec['split_source_git_sha'])['summary']
    except guard.HoldoutError as exc:
        if not exc.diagnostics or not exc.diagnostics.get('identity_metadata_verified'):
            raise
        frozen = exc.diagnostics
    split = guard._object(guard.read_pinned(split_dir / guard.SPLIT, frozen['split_sha256']))
    manifest = guard._object(guard.read_pinned(split_dir / guard.MANIFEST, spec['split_manifest_sha256']))
    if spec['inputs']['lucas']['sha256'] != split.get('source_index_sha256'):
        raise guard.HoldoutError('LUCAS source no longer matches frozen split provenance')
    tables = {name: read_table(Path(root)/relative, spec['inputs'][name], columns,
                               manifest['artifacts'][guard.VALIDATOR])
              for name, (relative, columns) in INPUTS.items()}
    lucas = observations(tables['lucas'], 'lucas')
    nfi = list(observations(tables['nfi'], 'nfi').values())
    roles = {}
    for name, field in [('distill', 'plots'), ('crop_holdout', 'holdout_plots'), ('prior_test', 'prior_test_plots')]:
        keys = guard._keys(split[field])
        if not set(keys) <= lucas.keys():
            raise guard.HoldoutError('frozen observations absent from pinned source index')
        roles[name] = [lucas[k] for k in keys]
    protected = { (r['tile'], r['id']): r for name in ('crop_holdout','prior_test') for r in roles[name] }
    protected_rows = list(protected.values())
    tiles, unresolved = tile_geometries(tables['tiles'])
    geometry_checks = {name: geometry_consistency(rows, tiles) for name, rows in {**roles, 'nfi_index':nfi}.items()}
    incomplete = any(any(v[k] for k in ('missing_manifest_geometry','year_mismatch','pixel_mismatch'))
                     for v in geometry_checks.values())
    distill_tiles = {r['tile'] for r in roles['distill']}
    protected_tiles = set(split['holdout_tiles']) | set(split['prior_test_tiles'])
    cross_tiles = tile_overlap(distill_tiles, protected_tiles, tiles)
    incomplete |= bool(cross_tiles['left_tiles_missing_geometry'] or cross_tiles['right_tiles_missing_geometry'])
    return {
        'schema': 'imint-observation-index-audit-v1',
        'status': 'incomplete_geometry' if incomplete else 'diagnostic_audit_complete',
        'cleared_for_scoring': False, 'frozen_identity': frozen,
        'geometry_assumptions': GEOMETRY, 'geometry_basis': 'pinned center/year manifest, not current NPZ verification',
        'unresolved_tile_manifest_rows': unresolved,
        'populations': {name: population(rows) for name, rows in {**roles, 'nfi_index':nfi}.items()},
        'identity_overlap': {name: identity_overlap(roles['distill'], roles[name]) for name in ('crop_holdout','prior_test')},
        'distill_protected_tile_overlap': cross_tiles,
        'different_lucas_ids_near_protected': near_points(roles['distill'], protected_rows, same_namespace=True),
        'nfi_near_protected_lucas': near_points(nfi, protected_rows, same_namespace=False),
        'nfi_protected_tile_overlap': tile_overlap({r['tile'] for r in nfi}, protected_tiles, tiles),
        'geometry_consistency': geometry_checks,
        'decoded_columns': {name:list(columns) for name, (_, columns) in INPUTS.items()},
        'input_pins': spec['inputs'], 'files_read': [str(split_dir/guard.MANIFEST), str(split_dir/guard.SPLIT)]
                      + [str(Path(root)/relative) for relative, _ in INPUTS.values()],
        'validator_parquet_read': False, 'outcome_columns_decoded': [], 'feature_files_read': False,
        'full_files_hashed': True, 'raw_rows_exported': False,
        'limits': ['historical fold assignments not audited', 'training population membership not established',
                   'NFI index is not a frozen evaluation/training role assignment',
                   'parcel identity and wider spatial dependence not assessed',
                   'no metric bias estimated', 'no resplit or eligibility decision'],
    }


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        spec = guard._object(guard.read_pinned(args.spec, args.spec_sha256))
        result = audit(args.input_root, spec)
        import pyarrow
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['execution'] = {'python':platform.python_version(), 'pyarrow':pyarrow.__version__,
                               'pod_uid':os.environ.get('POD_UID'), 'image':os.environ.get('AUDIT_IMAGE'),
                               'spec_sha256':args.spec_sha256,
                               'elapsed_seconds':round(time.monotonic()-started, 3),
                               'cpu_seconds':round(usage.ru_utime+usage.ru_stime, 3),
                               'max_rss_bytes':int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024))}
        result['source_sha256'] = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {
            'audit':Path(__file__), 'holdout_guard':Path(guard.__file__),
            'tile_config':Path(__file__).resolve().parents[1]/'imint/training/tile_config.py'}.items()}
        code = 2 if result['status'] == 'incomplete_geometry' else 0
    except (guard.HoldoutError, OSError, ValueError, KeyError, TypeError) as exc:
        # Do not expose input values, private row IDs or coordinates in errors.
        result = {'schema':'imint-observation-index-audit-v1', 'status':'failed',
                  'error_type':type(exc).__name__, 'cleared_for_scoring':False}
        code = 1
    raw = json.dumps(result, sort_keys=True, allow_nan=False)
    with args.output.open('x') as stream:
        stream.write(raw + '\n')
    print(raw, flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
