#!/usr/bin/env python3
"""Build a reproducible multi-area inventory from an already downloaded registry.

No network access or raster download. Native blocks are shared across areas;
static auxiliary layers are not multiplied by the number of TESSERA years.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

from imint.inference.lclu import YEARS, public_identity, write_json
from scripts.cache_predict import file_identity

BUFFER_M = 3700
REGISTRY_URL = 'https://data.source.coop/tessera/tessera/npy/v1/manifest.parquet'


def block_key(row: dict) -> str:
    return f"{row['year']}_{row['lon']:.2f}_{row['lat']:.2f}"


def build(registry: Path, registry_sha256: str, boundaries: Path,
          boundary_receipt: Path, kiruna: Path) -> dict:
    identity = file_identity(registry)
    if identity['sha256'] != registry_sha256:
        raise ValueError('Registry hash mismatch')
    features = json.loads(boundaries.read_text())['features']
    if len(features) != 10:
        raise ValueError('Expected the ten selected municipality boundaries')
    extra = json.loads(kiruna.read_text())
    features.append(extra['municipality'])
    to3006 = Transformer.from_crs(4326, 3006, always_xy=True).transform
    to4326 = Transformer.from_crs(3006, 4326, always_xy=True).transform
    areas, buffers = [], []
    for i, feature in enumerate(features):
        geometry = shape(feature['geometry'])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError('Invalid area geometry')
        buffered = transform(to3006, geometry).buffer(BUFFER_M)
        area_id = feature['properties']['code'] if i < 10 else 'kiruna_tatort'
        areas.append({'id': area_id, 'municipality': feature,
                      'bounds_3006': list(buffered.bounds), 'buffer_m': BUFFER_M,
                      'block_keys': []})
        buffers.append(transform(to4326, buffered))
    union = unary_union(buffers)
    west, south, east, north = union.bounds
    blocks = []
    columns = ['version', 'variant', 'year', 'lon', 'lat', 'grid_size', 'scales_size']
    for batch in pq.ParquetFile(registry).iter_batches(batch_size=65536, columns=columns):
        predicates = [pc.equal(batch['version'], '1.0'), pc.equal(batch['variant'], 'vultr'),
                      pc.is_in(batch['year'], value_set=pa.array(YEARS)),
                      pc.greater_equal(batch['lon'], west - .05), pc.less_equal(batch['lon'], east + .05),
                      pc.greater_equal(batch['lat'], south - .05), pc.less_equal(batch['lat'], north + .05)]
        mask = predicates[0]
        for predicate in predicates[1:]:
            mask = pc.and_(mask, predicate)
        for row in batch.filter(mask).to_pylist():
            footprint = box(row['lon']-.05, row['lat']-.05, row['lon']+.05, row['lat']+.05)
            matches = [i for i, buffered in enumerate(buffers) if buffered.intersects(footprint)]
            if not matches:
                continue
            q, s = row['grid_size'] - 128, row['scales_size'] - 128
            if q <= 0 or q % 128 or s % 4 or q // 128 != s // 4:
                raise ValueError(f'Incomplete block: {block_key(row)}')
            blocks.append(row)
            for i in matches:
                areas[i]['block_keys'].append(block_key(row))
    keys = [block_key(row) for row in blocks]
    if not blocks or len(keys) != len(set(keys)):
        raise ValueError('Empty or duplicated registry selection')
    for area in areas:
        if not area['block_keys']:
            raise ValueError(f"No blocks for {area['id']}")
        area['block_keys'].sort()
    return {'version': 1, 'years': list(YEARS), 'registry': public_identity(identity),
            'registry_url': REGISTRY_URL,
            'boundary_source': json.loads(boundary_receipt.read_text()),
            'boundary_files': {'municipalities': public_identity(file_identity(boundaries)),
                               'kiruna': public_identity(file_identity(kiruna))},
            'areas': areas, 'blocks': sorted(blocks, key=block_key),
            'temporal_policy': {'annual': ['TESSERA', 'HR-VPP'],
                                'shared': ['DEM', 'height', 'volume', 'basal_area', 'diameter', 'markfukt']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registry', 'boundaries', 'boundary-receipt', 'kiruna', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registry-sha256', required=True)
    args = parser.parse_args()
    result = build(args.registry, args.registry_sha256, args.boundaries, args.boundary_receipt, args.kiruna)
    # Compact JSON keeps the exact polygons within the immutable source bundle budget.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
    print(json.dumps({'areas': len(result['areas']), 'years': result['years'],
                      'unique_blocks': len(result['blocks']), 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
