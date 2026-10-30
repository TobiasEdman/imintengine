#!/usr/bin/env python3
"""Preflight or stage exact municipality/year pairs with one shared COG store.

Preflight reads metadata and file presence only. Staging requires an explicit
original-byte budget; it never fills auxiliary gaps, substitutes years, deletes
originals, or changes cluster resources. Source-ready is not model-qualified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from imint.data.wekeo_vpp import _bounds_overlap, _VPP_PRODUCT_TYPES
from imint.inference.lclu import YEARS, public_identity, write_json
from scripts.cache_predict import file_identity
from scripts.stage_lclu_demo import cached_block, stage


def block_key(row: dict) -> str:
    return f"{row['year']}_{row['lon']:.2f}_{row['lat']:.2f}"


def load_campaign(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if (plan['version'] != 1 or not plan['areas'] or
            not plan['years'] or any(y not in YEARS for y in plan['years'])):
        raise ValueError('Unsupported campaign')
    ids = [a['id'] for a in plan['areas']]
    keys = [block_key(row) for row in plan['blocks']]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError('Duplicate areas or native blocks')
    if any(not re.fullmatch(r'[a-z0-9_]+', name) for name in ids):
        raise ValueError('Invalid area ID')
    for row in plan['blocks']:
        if (row['year'] not in plan['years'] or row['version'] != '1.0' or row['variant'] != 'vultr'
                or row['grid_size'] <= 128 or row['scales_size'] <= 128):
            raise ValueError('Unsupported native block')
    known = set(keys)
    for area in plan['areas']:
        if not area['block_keys'] or not set(area['block_keys']) <= known:
            raise ValueError('Unknown or empty area block inventory')
    return plan


def preflight(plan: dict, years: list[int], data: Path) -> dict:
    if not years or len(years) != len(set(years)) or not set(years) <= set(plan['years']):
        raise ValueError('Requested years must be unique and inventoried')
    index_path = data/'vpp_wekeo/index.json'
    index = json.loads(index_path.read_text())
    rows = {block_key(r): r for r in plan['blocks']}
    all_keys, eligible_keys, files = set(), set(), set()
    cells = []
    static = [data/'skg_open/MANIFEST.json', data/'skg_open/markfukt/markfukt.vrt',
              data/'skg_open/tradhojd/tradhojd_10m.vrt']
    for name in ('volym', 'grundyta', 'medeldiameter'):
        candidates = sorted((data/'skg_open/grunddata').glob(name+'_omdrev[123].vrt'))
        static.extend(candidates or [data/'skg_open/grunddata'/f'{name}_omdrev1.vrt'])
    missing_static = [str(p.relative_to(data)) for p in static if not p.is_file()]
    for area in plan['areas']:
        wgs = transform_bounds('EPSG:3006', 'EPSG:4326', *area['bounds_3006'], densify_pts=41)
        for year in sorted(years):
            keys = [k for k in area['block_keys'] if rows[k]['year'] == year]
            all_keys.update(keys)
            selected = {name: meta for name, meta in index.items()
                        if meta['year'] == year and meta['season'] == 1
                        and _bounds_overlap(meta['bounds_4326'], wgs)}
            missing_metrics = []
            for metric in _VPP_PRODUCT_TYPES:
                footprints = [box(*m['bounds_4326']) for name, m in selected.items()
                              if m['metric'] == metric and (index_path.parent/name).is_file()]
                if not footprints or not unary_union(footprints).covers(box(*wgs)):
                    missing_metrics.append(metric)
            missing_files = [name for name in selected if not (index_path.parent/name).is_file()]
            status = ('missing_tessera' if not keys else 'missing_static_aux' if missing_static else
                      'missing_vpp' if missing_metrics or missing_files else 'source_ready')
            cells.append({'area_id': area['id'], 'name': area['municipality']['properties']['name'],
                          'year': year, 'status': status, 'blocks': len(keys),
                          'original_bytes': sum(rows[k]['grid_size']+rows[k]['scales_size'] for k in keys),
                          'vpp_missing_metrics': missing_metrics, 'vpp_missing_files': missing_files})
            files.update(name for name in selected if (index_path.parent/name).is_file())
            if status == 'source_ready':
                eligible_keys.update(keys)
    sizes = lambda keys: sum(rows[k]['grid_size']+rows[k]['scales_size'] for k in keys)
    return {'version': 1, 'years': sorted(years), 'cells': cells,
            'vpp_index': public_identity(file_identity(index_path)),
            'shared_aux': {'temporal_policy': 'shared across all years; no annual duplication',
                           'missing_files': missing_static,
                           'dem': 'shared public COG windows, verified during staging'},
            'source_ready_unique_blocks': len(eligible_keys), 'source_ready_original_bytes': sizes(eligible_keys),
            'requested_unique_blocks': len(all_keys), 'requested_original_bytes': sizes(all_keys),
            'existing_vpp_files': len(files),
            'existing_vpp_bytes': sum((index_path.parent/name).stat().st_size for name in files),
            'note': 'Metadata and file presence only. Each pair still needs preparation and real model qualification.'}


def stage_campaign(path: Path, years: list[int], data: Path, work: Path,
                   max_original_bytes: int, *, reuse_roots: tuple[Path, ...] = ()) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    (work/'scopes').mkdir(exist_ok=True)
    plan = load_campaign(path)
    identity = file_identity(path)
    report = preflight(plan, years, data)
    write_json(work/'preflight.json', report)
    budget = report['source_ready_original_bytes']
    if max_original_bytes <= 0 or budget > max_original_bytes:
        raise ValueError(f'Unique source bytes {budget} exceed explicit budget {max_original_bytes}')
    rows = {block_key(r): r for r in plan['blocks']}
    eligible = {(c['area_id'], c['year']) for c in report['cells'] if c['status'] == 'source_ready'}
    keys = {k for area in plan['areas'] for k in area['block_keys']
            if (area['id'], rows[k]['year']) in eligible}
    remaining = sum(rows[k]['grid_size'] + rows[k]['scales_size'] for k in keys
                    if cached_block(rows[k], (work/'shared', *reuse_roots), work/'shared/receipts') is None)
    # Retain originals; allow conversion headroom only for missing blocks.
    # Qualifications/cache need room even when all COG pairs already exist.
    required = int(remaining * 2.3) + len(eligible) * 100 * 2**20 + 512 * 2**20
    report.update(remaining_original_bytes=remaining, required_free_bytes=required)
    write_json(work/'preflight.json', report)
    if shutil.disk_usage(work).free < required:
        raise ValueError('Insufficient free disk for remaining blocks and qualification workspace')
    manifests = []
    areas = {a['id']: a for a in plan['areas']}
    for cell in report['cells']:
        if cell['status'] != 'source_ready':
            continue
        area = areas[cell['area_id']]
        scope = {'version': 1, **area, 'registry': plan['registry'],
                 'blocks': [rows[k] for k in area['block_keys']]}
        scope_path = work/'scopes'/f"{area['id']}.json"
        write_json(scope_path, scope)
        root = work/'regions'/area['id']/str(cell['year'])
        stage(scope_path, cell['year'], data, root, shared_root=work/'shared',
              max_blocks=128, max_original_bytes=12_000_000_000, reuse_roots=reuse_roots)
        manifests.append(json.loads((root/'serving.json').read_text()))
        cell['status'] = 'prepared'
        write_json(work/'progress.json', report)
    if not manifests:
        raise ValueError('No exact-year region could be prepared')
    if any(m['checkpoint'] != manifests[0]['checkpoint'] for m in manifests):
        raise ValueError('All regions must use the same pinned checkpoint')
    # A source inventory alone cannot qualify a year: run the frozen R4 once
    # on every prepared representative window before publishing the manifest.
    from imint.inference.lclu import TesseraR4
    model = TesseraR4(manifests[0]['checkpoint'], work, 'cpu')
    for cell in report['cells']:
        if cell['status'] != 'prepared':
            continue
        root = work/'regions'/cell['area_id']/str(cell['year'])
        entry_path = root/f"qualification-{cell['year']}/entry.json"
        entry = json.loads(entry_path.read_text())
        from imint.inference.lclu import verify_file
        prediction = model.predict(verify_file(entry['input'], entry_path.parent), entry['input'])
        if prediction.shape != (504, 504):
            raise ValueError('Unexpected native R4 output shape')
        cell['status'] = 'qualified'
        cell['qualification'] = public_identity(file_identity(entry_path))
        prediction_path = entry_path.parent/'prediction.npy'
        np.save(prediction_path, prediction.astype(np.uint8), allow_pickle=False)
        cell['prediction'] = public_identity(file_identity(prediction_path))
        cell['class_ids'] = sorted(int(v) for v in np.unique(prediction))
        write_json(work/'progress.json', report)
    if public_identity(file_identity(path)) != public_identity(identity):
        raise ValueError('Campaign changed during staging')
    result = {'version': 1, 'device': 'cpu', 'checkpoint': manifests[0]['checkpoint'],
              'preparations': [m['preparation'] for m in manifests],
              'campaign': public_identity(identity), 'coverage': report['cells']}
    write_json(work/'serving.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--years', type=int, nargs='+', required=True)
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', action='store_true')
    parser.add_argument('--max-original-bytes', type=int, default=0)
    args = parser.parse_args()
    if args.stage:
        stage_campaign(args.campaign.resolve(), args.years, args.data.resolve(), args.output.resolve(),
                       args.max_original_bytes)
    else:
        report = preflight(load_campaign(args.campaign), args.years, args.data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != 'cells'}))


if __name__ == '__main__':
    main()
