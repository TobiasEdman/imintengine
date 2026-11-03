#!/usr/bin/env python3
"""Plan the WEkEO HR-VPP gap-fill for the IM-024 municipality × year campaign.

Read-only planner. It takes the campaign plan (``config/lclu/municipalities.json``)
and a campaign preflight receipt (``scripts/stage_lclu_campaign.py`` preflight
output) and, for every area-year whose status is ``missing_vpp``, derives the
Sentinel-2 MGRS tiles covering the area's buffered EPSG:3006 bounds. Output:

  * the exact (MGRS, year) pairs and ready-to-run ``scripts/prefetch_vpp_wekeo.py``
    invocations — WEkEO HDA only, never CDSE PU (CLAUDE.md VPP rule);
  * the area × year coverage matrix of the receipt as a Markdown table.

``prefetch_vpp_cogs`` skips COGs already present and maintains
``/data/vpp_wekeo/index.json``, which the campaign preflight reads, so pairs
that are partly cached are requested as a whole and nothing is subtracted
here. No network and no data root access; both inputs are hashed into the plan.

    python scripts/plan_lclu_vpp_gapfill.py \\
        --campaign config/lclu/municipalities.json \\
        --preflight docs/verification/lclu-service/campaign-preflight.json \\
        --output docs/verification/lclu-service/vpp-gapfill-plan.json \\
        --matrix docs/verification/lclu-service/coverage-matrix.md
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_missing_vpp_mgrs  # noqa: E402

# mgrs_tiles_for_bbox samples corners and centre, which is exact only for boxes
# far smaller than a 100 km MGRS cell. Municipalities span tens of kilometres
# and several straddle the 18°E UTM zone boundary, so each area is split.
SUB_BOX_M = 5000
STATUS_MARK = {'source_ready': '✓', 'missing_vpp': 'VPP', 'missing_tessera': 'TESSERA',
               'missing_static_aux': 'aux'}


def mgrs_tiles_for_area(bounds_3006: list[float], sub_box_m: float = SUB_BOX_M) -> set[str]:
    west, south, east, north = bounds_3006
    if not (west < east and south < north) or sub_box_m <= 0:
        raise ValueError('Area bounds must be a non-empty EPSG:3006 box')
    nx, ny = math.ceil((east - west) / sub_box_m), math.ceil((north - south) / sub_box_m)
    xs = [west + (east - west) * i / nx for i in range(nx + 1)]
    ys = [south + (north - south) * j / ny for j in range(ny + 1)]
    tiles: set[str] = set()
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            tiles |= derive_missing_vpp_mgrs.mgrs_tiles_for_bbox(
                {'west': x0, 'south': y0, 'east': x1, 'north': y1})
    return tiles


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(campaign: Path, preflight: Path) -> dict:
    areas = {area['id']: area for area in json.loads(campaign.read_text())['areas']}
    receipt = json.loads(preflight.read_text())
    unknown = {cell['area_id'] for cell in receipt['cells']} - set(areas)
    if unknown:
        raise ValueError(f'Preflight areas missing from the campaign plan: {sorted(unknown)}')
    tiles_by_area = {area_id: sorted(mgrs_tiles_for_area(area['bounds_3006']))
                     for area_id, area in areas.items()}
    cells, by_year = [], defaultdict(set)
    for cell in receipt['cells']:
        if cell['status'] != 'missing_vpp':
            continue
        tiles = tiles_by_area[cell['area_id']]
        cells.append({'area_id': cell['area_id'], 'name': cell['name'], 'year': cell['year'],
                      'vpp_missing_metrics': cell['vpp_missing_metrics'], 'mgrs_tiles': tiles})
        by_year[cell['year']].update(tiles)
    n_pairs = sum(len(tiles) for tiles in by_year.values())
    n_cogs = n_pairs * len(derive_missing_vpp_mgrs._REQUIRED_METRICS)
    mean_cog = receipt['existing_vpp_bytes'] / receipt['existing_vpp_files']
    return {
        'version': 1,
        'campaign_sha256': sha256(campaign), 'preflight_sha256': sha256(preflight),
        'preflight_vpp_index': receipt['vpp_index'],
        'status_counts': dict(sorted(Counter(cell['status'] for cell in receipt['cells']).items())),
        'missing_vpp_cells': cells,
        'pairs_by_year': {str(year): sorted(tiles) for year, tiles in sorted(by_year.items())},
        'n_pairs': n_pairs,
        # Upper bound: partly cached pairs are counted whole.
        'storage_estimate': {'cogs': n_cogs, 'bytes_at_cached_mean': round(n_cogs * mean_cog),
                             'basis': 'mean size of the COGs already cached for this footprint'},
        'prefetch_commands': [
            f"python scripts/prefetch_vpp_wekeo.py --mgrs-tiles {','.join(sorted(tiles))} "
            f'--years {year} --dest-dir /data/vpp_wekeo'
            for year, tiles in sorted(by_year.items())],
        'note': ('Rerun the campaign preflight after the prefetch: a year that WEkEO has '
                 'not published stays missing_vpp and is an explicit blocker, never '
                 'filled from another year.'),
    }


def coverage_matrix(preflight: Path) -> str:
    receipt = json.loads(preflight.read_text())
    years = receipt['years']
    status = {(cell['area_id'], cell['year']): cell['status'] for cell in receipt['cells']}
    names = {cell['area_id']: cell['name'] for cell in receipt['cells']}
    lines = ['| Område | ' + ' | '.join(map(str, years)) + ' |',
             '|---|' + '---|' * len(years)]
    for area_id, name in names.items():
        marks = [STATUS_MARK[status[(area_id, year)]] for year in years]
        lines.append(f'| {name} | ' + ' | '.join(marks) + ' |')
    counts = Counter(status.values())
    return '\n'.join([
        f'Källäge per område och år enligt preflight `{sha256(preflight)[:12]}` '
        f'(VPP-index `{receipt["vpp_index"]["sha256"][:12]}`). '
        '✓ = källklar, VPP = HR-VPP saknas i WEkEO-cachen. '
        'Källklar betyder filer på plats, inte en kvalificerad årskarta.', '',
        *lines, '',
        ', '.join(f'{STATUS_MARK[key]}: {value}' for key, value in sorted(counts.items())), ''])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.campaign, args.preflight)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    args.matrix.write_text(coverage_matrix(args.preflight))
    print(json.dumps({'n_pairs': result['n_pairs'], 'years': list(result['pairs_by_year']),
                      'missing_vpp_cells': len(result['missing_vpp_cells'])}))


if __name__ == '__main__':
    main()
