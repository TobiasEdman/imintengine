#!/usr/bin/env python3
"""Stage a bounded public municipality/year COG mirror on the demo volume.

Existing /data auxiliaries and the private checkpoint are read-only. Exact-year
VPP must already be present; no CDSE, DES, WEkEO login or year fallback occurs.
Interrupted downloads/conversions resume from verified receipts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape
from shapely.ops import unary_union

from imint.data.wekeo_vpp import _bounds_overlap, _VPP_PRODUCT_TYPES
from imint.inference.lclu import EMBEDDING, Grid, public_identity, verify_file, write_json
from imint.inference.lclu_cog import convert_block
from imint.training.copernicus_dem import _get_cog_urls
from scripts.cache_predict import file_identity
from scripts.prepare_lclu import prepare, verify_aux_metadata, verify_source_file


def fetch(url: str, path: Path, *, maximum: int, expected: int | None = None) -> Path:
    receipt = path.with_suffix(path.suffix + '.json')
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['url'] != url or saved['file']['size'] > maximum:
            raise ValueError('Download receipt disagrees with requested source')
        verify_file(saved['file'], path.parent)
        if expected is not None and saved['file']['size'] != expected:
            raise ValueError('Registry/download size mismatch')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    try:
        with requests.get(url, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError('Expected a complete source object')
            size = 0
            with partial.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > maximum:
                        raise ValueError('Public-source byte budget exceeded')
                    stream.write(chunk)
        if not size or (expected is not None and size != expected):
            raise ValueError('Incomplete public-source object')
        partial.replace(path)
        write_json(receipt, {'url': url, 'file': file_identity(path)})
    finally:
        partial.unlink(missing_ok=True)
    return path


def select_vpp(index: dict, year: int, bounds_4326: tuple) -> dict:
    selected = {name: meta for name, meta in index.items()
                if meta['year'] == year and meta['season'] == 1
                and _bounds_overlap(meta['bounds_4326'], bounds_4326)}
    for metric in _VPP_PRODUCT_TYPES:
        footprints = [box(*m['bounds_4326']) for m in selected.values() if m['metric'] == metric]
        if not footprints or not unary_union(footprints).covers(box(*bounds_4326)):
            raise ValueError(f'Exact-year {year} HR-VPP {metric} coverage missing; stage WEkEO data first')
    return selected


def cached_block(row: dict, roots: tuple[Path, ...], receipts: Path) -> dict | None:
    """Return a verified reusable COG pair, binding it to the native inventory."""
    name = f"{row['year']}_grid_{row['lon']:.2f}_{row['lat']:.2f}.json"
    for root in roots:
        path = root/'cogs'/name
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        if (record['year'], record['lon'], record['lat']) != (row['year'], row['lon'], row['lat']):
            raise ValueError('COG receipt block identity mismatch')
        if [v['size'] for v in record['original'][:2]] != [row['grid_size'], row['scales_size']]:
            raise ValueError('COG receipt disagrees with registry source sizes')
        for field in ('vectors', 'scales'):
            verified = verify_source_file(record[field], root, receipts)
            record[field] = {**record[field], 'path': str(verified.resolve())}
        return record
    return None


def stage(scope_path: Path, year: int, data: Path, root: Path, *,
          shared_root: Path | None = None, max_blocks: int = 32,
          max_original_bytes: int = 3_000_000_000,
          reuse_roots: tuple[Path, ...] = ()) -> None:
    start = time.perf_counter()
    scope_identity = file_identity(scope_path)
    scope = json.loads(scope_path.read_text())
    rows = [r for r in scope['blocks'] if r['year'] == year]
    budget = sum(r['grid_size'] + r['scales_size'] for r in rows)
    if (scope['version'] != 1 or not rows or len(rows) > max_blocks or budget > max_original_bytes
            or any(r['variant'] != 'vultr' or r['version'] != '1.0' for r in rows)):
        raise ValueError('Unsupported or oversized municipality inventory')
    root.mkdir(parents=True, exist_ok=True)
    shared = shared_root if shared_root is not None else root
    receipts = shared / 'receipts'
    (shared/'cogs').mkdir(parents=True, exist_ok=True)
    bounds = scope['bounds_3006']
    wgs = transform_bounds('EPSG:3006', 'EPSG:4326', *bounds, densify_pts=41)
    index_path = data / 'vpp_wekeo/index.json'
    selected = select_vpp(json.loads(index_path.read_text()), year, wgs)
    print(f'Exact {year} VPP: {len(selected)} files; TESSERA: {len(rows)} blocks / {budget} bytes', flush=True)
    skg = data / 'skg_open'
    config = {'version': 1, 'embedding': EMBEDDING, 'identity_cache': str(receipts),
              'vpp_index': file_identity(index_path), 'vpp_files': {},
              'skg_manifest': file_identity(skg/'MANIFEST.json'), 'qualified_regions': []}
    vrts = sorted((skg/'grunddata').glob('*_omdrev[123].vrt')) + [
        skg/'markfukt/markfukt.vrt', skg/'tradhojd/tradhojd_10m.vrt']
    config['skg_vrts'] = [{**file_identity(p), 'path': str(p.relative_to(skg))} for p in vrts]
    verify_aux_metadata(config, root)
    for name in selected:
        record = file_identity(index_path.parent/name)
        config['vpp_files'][name] = public_identity(record)
        verify_source_file(record, root, receipts)
    config['dem_sources'] = []
    for url in _get_cog_urls(wgs[1], wgs[3], wgs[0], wgs[2]):
        filename = url.rsplit('/', 1)[1]
        path = next((previous/'dem'/filename for previous in reuse_roots
                     if (previous/'dem'/(filename+'.json')).is_file()), shared/'dem'/filename)
        path = fetch(url, path, maximum=80_000_000)
        config['dem_sources'].append(file_identity(path))
    # All existing source rasters potentially read in the buffered region are
    # pinned and their identity receipts warmed before interactive requests.
    manifest = json.loads((skg/'MANIFEST.json').read_text())
    for relative, meta in manifest['derived'].items():
        if 'bounds' not in meta or not box(*meta['bounds']).intersects(box(*bounds)):
            continue
        path = skg/relative
        verify_source_file({'path': str(path), 'sha256': meta['sha256'],
                            'size': path.stat().st_size}, root, receipts)
    records = []
    for row in rows:
        name = f"grid_{row['lon']:.2f}_{row['lat']:.2f}"
        record_path = shared/'cogs'/f'{year}_{name}.json'
        record = cached_block(row, (shared, *reuse_roots), receipts)
        if record is not None:
            if not record_path.exists():
                write_json(record_path, record)
        else:
            base = f'https://data.source.coop/tessera/tessera/npy/v1/{year}/{name}'
            quantized = fetch(f'{base}/{name}.npy', shared/'original'/str(year)/f'{name}.npy',
                              maximum=row['grid_size'], expected=row['grid_size'])
            scales = fetch(f'{base}/{name}_scales.npy', shared/'original'/str(year)/f'{name}_scales.npy',
                           maximum=row['scales_size'], expected=row['scales_size'])
            reference = fetch(f'https://data.source.coop/tessera/tessera/landmasks/v1/{name}.tiff',
                              shared/'original'/f'{name}.tiff', maximum=1_000_000)
            record = convert_block(quantized, scales, reference, shared/'cogs', year, row['lon'], row['lat'])
            write_json(record_path, record)
        records.append(record)
        print(f'Verified compact COG {len(records)}/{len(rows)} {name}', flush=True)
    cog_index = root/f'cog-index-{year}.json'
    write_json(cog_index, {'version': 1, 'embedding': EMBEDDING, 'blocks': records,
                           'registry': scope['registry'], 'scope': public_identity(scope_identity)})
    config['cog_index'] = file_identity(cog_index)
    # Scope qualification is published only after a real input can be prepared.
    provisional = root/f'sources-{year}-checking.json'
    write_json(provisional, config)
    centre = shape(scope['municipality']['geometry']).representative_point()
    grid = Grid.from_lonlat(centre.x, centre.y)
    prepare(provisional, grid, year, root/f'qualification-{year}')
    qualification = file_identity(root/f'qualification-{year}/entry.json')
    config['qualified_regions'] = [{
        'country': 'SE', 'name': scope['municipality']['properties']['name'],
        'years': [year], 'bounds_3006': bounds,
        'geometry': scope['municipality']['geometry'],
        'verification': qualification['sha256']}]
    sources = root/f'sources-{year}.json'
    write_json(sources, config)
    checkpoint = file_identity(data/'checkpoints/ladder/tessera_r4/best_model.pt')
    if checkpoint['sha256'] != '1270bb2af88074840020f3b481ed5e98a72f0b7ceb51befac2b8fa4675385668':
        raise ValueError('Unexpected R4 checkpoint; explicit requalification required')
    write_json(root/'serving.json', {'version': 1, 'device': 'cpu', 'checkpoint': checkpoint,
        'preparation': {'config': file_identity(sources), 'python': sys.executable, 'timeout_seconds': 300}})
    if public_identity(file_identity(scope_path)) != public_identity(scope_identity):
        raise ValueError('Municipality inventory changed while staging')
    write_json(root/'staging-result.json', {'year': year, 'blocks': len(records),
        'original_bytes': budget, 'cog_bytes': sum(r[k]['size'] for r in records for k in ('vectors', 'scales')),
        'seconds': time.perf_counter()-start, 'scope': public_identity(scope_identity),
        'sources': public_identity(file_identity(sources)), 'qualification': public_identity(qualification)})
    print(f'Ready: {root / "serving.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--year', type=int, choices=range(2017, 2026), required=True)
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.scope.resolve(), args.year, args.data.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
