"""Lossless compact TESSERA COGs and bounded native-window reads.

The pair retains the original int8 vectors and float32 scale factors. No
quantization, overview resampling or change of embedding dataset is applied.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from imint.inference.lclu import EMBEDDING, Grid, InputUnavailable, public_identity
from scripts.cache_predict import file_identity


def convert_block(quantized: Path, scales: Path, reference: Path,
                  output: Path, year: int, lon: float, lat: float) -> dict:
    """Write, reopen and compare every base-resolution pixel before publishing."""
    q = np.load(quantized, mmap_mode='r', allow_pickle=False)
    s = np.load(scales, mmap_mode='r', allow_pickle=False)
    if q.dtype != np.int8 or q.ndim != 3 or q.shape[2] != 128:
        raise ValueError('Expected original H×W×128 int8 vectors')
    if s.dtype != np.float32 or s.shape != q.shape[:2] or not np.isfinite(s).all():
        raise ValueError('Expected finite H×W float32 scale factors')
    with rasterio.open(reference) as ref:
        if ref.shape != s.shape or ref.crs is None:
            raise ValueError('Reference/embedding grid mismatch')
        crs, transform = ref.crs, ref.transform
    output.mkdir(parents=True, exist_ok=True)
    record = {'year': year, 'lon': lon, 'lat': lat, 'shape': list(s.shape),
              'crs': crs.to_string(), 'transform': list(transform)[:6],
              'original': [public_identity(file_identity(p))
                           for p in (quantized, scales, reference)]}
    for name, values in [('vectors', q.transpose(2, 0, 1)), ('scales', s[None])]:
        path = output / f'{year}_{lon:.2f}_{lat:.2f}_{name}.tif'
        temporary = path.with_suffix('.partial.tif')
        try:
            with rasterio.open(temporary, 'w', driver='COG', width=s.shape[1],
                               height=s.shape[0], count=values.shape[0],
                               dtype=values.dtype, crs=crs, transform=transform,
                               BLOCKSIZE=256, COMPRESS='ZSTD', LEVEL=9,
                               PREDICTOR=2 if name == 'vectors' else 3,
                               OVERVIEWS='NONE', NUM_THREADS=2) as dst:
                dst.write(values)
                dst.update_tags(dataset='v1', variant='vultr', year=str(year))
            with rasterio.open(temporary) as src:
                if (src.crs != crs or src.transform != transform
                        or src.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG'
                        or src.overviews(1)):
                    raise ValueError('COG geometry/layout verification failed')
                for _, window in src.block_windows(1):
                    rows, cols = window.toslices()
                    if not np.array_equal(src.read(window=window), values[:, rows, cols]):
                        raise ValueError('COG pixels differ from original source')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        record[name] = file_identity(path)
    return record


def _check_raster(src, block: dict, count: int, dtype: str) -> None:
    if (list(src.shape) != block['shape'] or src.count != count
            or any(d != dtype for d in src.dtypes) or src.nodata is not None
            or src.crs != rasterio.crs.CRS.from_string(block['crs'])
            or list(src.transform)[:6] != block['transform']
            or src.tags().get('dataset') != 'v1' or src.tags().get('variant') != 'vultr'
            or src.tags().get('year') != str(block['year'])
            or src.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG'):
        raise ValueError('Compact COG source contract mismatch')


def assemble_cog_embedding(index: dict, base: Path, grid: Grid, year: int,
                           receipts: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read all 128 bands only in the needed native windows, with a kernel halo."""
    from scripts.prepare_lclu import verify_source_file
    from scripts.enrich_tiles_tessera import _reproject_tessera_to_tile

    if index['version'] != 1 or index['embedding'] != EMBEDDING:
        raise ValueError('Unsupported COG dataset identity')
    b = grid.bounds()
    bounds = tuple(b[k] for k in ('west', 'south', 'east', 'north'))
    west, south, east, north = transform_bounds('EPSG:3006', 'EPSG:4326', *bounds)
    blocks = [r for r in index['blocks'] if r['year'] == year
              and r['lon'] + .05 >= west and r['lon'] - .05 <= east
              and r['lat'] + .05 >= south and r['lat'] - .05 <= north]
    if not blocks or len(blocks) > 4:
        raise InputUnavailable('Missing exact-year COG blocks or block budget exceeded')
    merged = np.zeros((128, 512, 512), np.float16)
    valid = np.zeros((512, 512), bool)
    observed = []
    for block in blocks:
        paths = [verify_source_file(block[name], base, receipts) for name in ('vectors', 'scales')]
        with rasterio.open(paths[0]) as vectors, rasterio.open(paths[1]) as scales:
            _check_raster(vectors, block, 128, 'int8')
            _check_raster(scales, block, 1, 'float32')
            native = transform_bounds('EPSG:3006', vectors.crs, *bounds, densify_pts=41)
            raw = from_bounds(*native, transform=vectors.transform)
            left, top = max(0, math.floor(raw.col_off) - 2), max(0, math.floor(raw.row_off) - 2)
            right = min(vectors.width, math.ceil(raw.col_off + raw.width) + 2)
            bottom = min(vectors.height, math.ceil(raw.row_off + raw.height) + 2)
            if right <= left or bottom <= top:
                continue
            window = Window(left, top, right - left, bottom - top)
            arr = vectors.read(window=window).astype(np.float32)
            arr *= scales.read(1, window=window)[None]
            arr = arr.transpose(1, 2, 0)
            transform = vectors.window_transform(window)
            if not np.isfinite(arr).all():
                raise ValueError('Nonfinite dequantized COG data')
            weights = np.zeros((512, 512), np.float32)
            reproject(np.any(arr != 0, axis=2).astype(np.float32), weights,
                      src_transform=transform, src_crs=vectors.crs,
                      dst_transform=grid.transform(), dst_crs='EPSG:3006',
                      resampling=Resampling.bilinear)
            chunk = _reproject_tessera_to_tile(arr, transform, vectors.crs, b, 512)
            take = (weights >= 1 - 1e-6) & np.any(chunk != 0, axis=0)
            merged[:, take] = chunk[:, take]
            valid |= take
        for name in ('vectors', 'scales'):
            verify_source_file(block[name], base, receipts)
            observed.append(public_identity(block[name]))
    return merged, valid, {'files': observed, 'blocks': len(blocks),
                            'uncached_source_bytes': 0, 'storage': 'compact-cog'}
