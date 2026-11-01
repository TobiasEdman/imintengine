#!/usr/bin/env python3
"""Prepare one public annual 512px input with pinned GeoTessera and repo readers.

Invoked by RepositorySources in a time-bounded child process. Source metadata
and local auxiliary inventories must already be staged; never bulk-prefetches
auxiliaries, calls DES/CDSE, or silently changes the year.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
import re
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.inference.lclu import (
    AUX_NAMES, EMBEDDING, YEARS, Grid, InputUnavailable, public_identity,
    validate_bundle, verify_file, write_json,
)
from scripts.cache_predict import atomic_write, file_identity, contract_pin


def verify_source_file(record: dict, base: Path, receipts: Path) -> Path:
    """Reuse an identity check only while local source inode/stats are unchanged."""
    from scripts.inference_comparison import _file_snapshot
    path = (base / record['path']).resolve()
    snapshot = list(_file_snapshot(path.stat()))
    key = contract_pin({'path': str(path), **public_identity(record)})
    receipt = receipts / f'{key}.json'
    try:
        saved = json.loads(receipt.read_text())
        if saved == {'identity': public_identity(record), 'stat': snapshot}:
            return path
    except (OSError, ValueError):
        pass
    verify_file(record, base)
    if list(_file_snapshot(path.stat())) != snapshot:
        raise ValueError('Source changed while verifying its identity')
    receipts.mkdir(parents=True, exist_ok=True)
    write_json(receipt, {'identity': public_identity(record), 'stat': snapshot})
    return path


def validate_embedding_cache(cache: Path) -> None:
    if not cache.exists() or not any(cache.iterdir()):
        return
    metadata = json.loads((cache / 'tessera_metadata.json').read_text())
    expected = {'dataset_version': '1.0', 'dataset_version_path': 'v1',
                'dataset_variant': 'vultr', 'dataset_path': 'v1',
                'embeddings_subdir': 'global_0.1_degree_representation'}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Embedding cache has missing or mismatched dataset identity')


def verify_aux_metadata(config: dict, base: Path):
    index = verify_file(config['vpp_index'], base)
    skg = verify_file(config['skg_manifest'], base)
    if index.name != 'index.json' or skg.name != 'MANIFEST.json':
        raise ValueError('Pin the exact index.json and MANIFEST.json opened by the canonical readers')
    required = {'markfukt/markfukt.vrt', 'tradhojd/tradhojd_10m.vrt'}
    required.update(str(p.relative_to(skg.parent)) for p in
                    (skg.parent / 'grunddata').glob('*_omdrev[123].vrt'))
    for variable in ('volym', 'grundyta', 'medeldiameter'):
        if not any(name.startswith(f'grunddata/{variable}_') for name in required):
            raise InputUnavailable(f'No staged forestry VRT for {variable}')
    records = {str((skg.parent / r['path']).resolve()): r for r in config['skg_vrts']}
    for relative in required:
        path = (skg.parent / relative).resolve()
        if str(path) not in records:
            raise ValueError(f'Unpinned canonical source mapping: {relative}')
        verify_file(records[str(path)], skg.parent)
    return index, skg


def embedding_files(gt, blocks):
    from geotessera.registry import (EMBEDDINGS_DIR_NAME, LANDMASKS_DIR_NAME,
                                    tile_to_embedding_paths, tile_to_landmask_filename)
    root = Path(gt.embeddings_dir)
    files = {}
    for year, lon, lat in blocks:
        grid, scales = tile_to_embedding_paths(lon, lat, year)
        files[root / EMBEDDINGS_DIR_NAME / grid] = gt.registry.get_tile_file_size(year, lon, lat)
        files[root / EMBEDDINGS_DIR_NAME / scales] = gt.registry.get_scales_file_size(year, lon, lat)
        files[root / LANDMASKS_DIR_NAME / tile_to_landmask_filename(lon, lat)] = gt.registry.get_landmask_file_size(lon, lat)
    if any(size <= 0 for size in files.values()):
        raise ValueError('Missing source-file sizes; download budget cannot be enforced')
    for path, size in files.items():
        if path.exists() and path.stat().st_size != size:
            raise ValueError('Cached source size disagrees with the pinned registry')
    return files


def verify_county_masks(manifest_path: Path, bounds: tuple, receipts: Path) -> dict:
    """Pin the validity-mask sources even when omdrev 3 supplies all auxiliaries."""
    import rasterio
    from imint.training.skg_open import COUNTY_MASKS
    manifest = json.loads(manifest_path.read_text())
    selected = {}
    west, south, east, north = bounds
    for relative, record in manifest['derived'].items():
        if not any(re.fullmatch(re.escape(prefix) + r'\d{2}\.tif', relative)
                   for prefix, _ in COUNTY_MASKS):
            continue
        left, bottom, right, top = record['bounds']
        overlaps = west < right and east > left and south < top and north > bottom
        path = manifest_path.parent / relative
        if not overlaps and not path.exists():
            continue
        # Check all available mask headers against the pinned footprint, so a
        # changed file cannot silently move from an excluded county into view.
        with rasterio.open(path) as src:
            if tuple(src.bounds) != tuple(record['bounds']) or src.crs.to_epsg() != 3006:
                raise ValueError('County-mask footprint differs from pinned metadata')
        if not overlaps:
            continue
        verify_source_file({'path': relative, 'sha256': record['sha256'],
                            'size': path.stat().st_size}, manifest_path.parent, receipts)
        source = manifest['files'][record['from']]
        selected[relative] = {'sha256': record['sha256'], 'source': record['from'],
                              'source_sha256': source['sha256']}
    return selected


def assemble_embedding(gt, grid: Grid, year: int, *, max_blocks: int, max_bytes: int):
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
    from scripts.enrich_tiles_tessera import _reproject_tessera_to_tile

    b = grid.bounds()
    bounds = transform_bounds('EPSG:3006', 'EPSG:4326', b['west'], b['south'], b['east'], b['north'])
    blocks = gt.registry.load_blocks_for_region(bounds=bounds, year=year)
    if not blocks or len(blocks) > max_blocks or any(y != year for y, _, _ in blocks):
        raise InputUnavailable('Missing exact-year blocks or block budget exceeded')
    files = embedding_files(gt, blocks)
    missing_bytes = sum(size for path, size in files.items() if not path.exists())
    if missing_bytes > max_bytes:
        raise InputUnavailable('Embedding download exceeds the configured byte budget')
    merged = np.zeros((128, 512, 512), np.float16)
    valid = np.zeros((512, 512), bool)
    for y, lon, lat in blocks:
        arr, crs, transform = gt.fetch_embedding(lon=lon, lat=lat, year=y, refresh=False)
        if arr.ndim != 3 or arr.shape[2] != 128 or not np.issubdtype(arr.dtype, np.floating):
            raise ValueError('Expected dequantized H×W×128 embedding')
        if not np.isfinite(arr).all():
            raise ValueError('Nonfinite source embeddings')
        mask_path = gt.registry.fetch_landmask(lon=lon, lat=lat)
        with rasterio.open(mask_path) as reference:
            if (reference.shape != arr.shape[:2] or reference.crs != crs
                    or reference.transform != transform):
                raise ValueError('Embedding/reference georeferencing mismatch')
        # A bilinear weight of one means every contributing source pixel was
        # available. This also prevents interpolated gap edges appearing valid.
        source_valid = np.any(arr != 0, axis=2).astype(np.float32)
        weights = np.zeros((512, 512), np.float32)
        reproject(source_valid, weights, src_transform=transform, src_crs=crs,
                  dst_transform=grid.transform(), dst_crs='EPSG:3006',
                  resampling=Resampling.bilinear)
        chunk = _reproject_tessera_to_tile(arr, transform, crs, b, 512)
        take = (weights >= 1 - 1e-6) & np.any(chunk != 0, axis=0)
        merged[:, take] = chunk[:, take]
        valid |= take
    observed = []
    for path, size in files.items():
        identity = file_identity(path)
        if identity['size'] != size:
            raise ValueError('Downloaded source size mismatch')
        observed.append({'file': path.name, **public_identity(identity)})
    return merged, valid, {'files': observed, 'download_budget_bytes': max_bytes,
                            'uncached_source_bytes': missing_bytes, 'blocks': len(blocks)}


BAND_ROWS = 256    # remote tiles are read and cached in fixed row bands
BAND_WORKERS = 8   # parallel range requests; one stream measured ~4.7 MB/s


def _band_path(cache: Path, url: str, k: int) -> Path:
    return cache / f'{Path(url).stem}_b{k}.npy'


def _remote_band(url: str, k: int, shape: tuple, dtype, cache: Path) -> np.ndarray:
    """Rows [k·BAND_ROWS, …) of a remote (H, W[, 128]) NPY tile, kept on disk."""
    from geotessera.remote import read_npy_window
    path = _band_path(cache, url, k)
    rows = min(BAND_ROWS, shape[0] - k * BAND_ROWS)
    if path.is_file():
        band = np.load(path, allow_pickle=False)
        os.utime(path)   # recency for the cache budget
    else:
        band = np.ascontiguousarray(read_npy_window(url, k * BAND_ROWS, k * BAND_ROWS + rows,
                                                    0, shape[1]))
        cache.mkdir(parents=True, exist_ok=True)
        atomic_write(path, lambda stream: np.save(stream, band))
    if band.shape != (rows, *shape[1:]) or band.dtype != dtype:
        raise ValueError('Remote tile band does not match the pinned tile shape')
    return band


def assemble_remote_embedding(gt, grid: Grid, year: int, *, root: str, cache: Path,
                              max_blocks: int):
    """The tile path's embedding, reading only the rows each remote tile contributes.

    Same blocks, landmask georeferencing, dequantization, resampling and merge
    as assemble_embedding and the compact COG path. Each tile is read in fixed
    BAND_ROWS bands by HTTP range (one request per band, BAND_WORKERS at a
    time) and the bands are kept on disk; a tile is only fetched after its
    remote size matches the pinned registry.
    """
    import math
    from concurrent.futures import ThreadPoolExecutor
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
    from rasterio.windows import Window, from_bounds, transform as window_transform
    from geotessera.registry import tile_to_embedding_paths
    from geotessera.remote import get_fs
    from scripts.enrich_tiles_tessera import _reproject_tessera_to_tile

    b = grid.bounds()
    bounds = tuple(b[k] for k in ('west', 'south', 'east', 'north'))
    blocks = gt.registry.load_blocks_for_region(
        bounds=transform_bounds('EPSG:3006', 'EPSG:4326', *bounds), year=year)
    if not blocks or len(blocks) > max_blocks or any(y != year for y, _, _ in blocks):
        raise InputUnavailable('Missing exact-year blocks or block budget exceeded')
    here = cache / str(year)
    # 1. Each block's native window and the bands it needs.
    windows, wanted, files = [], [], []
    for _, lon, lat in blocks:
        with rasterio.open(gt.registry.fetch_landmask(lon=lon, lat=lat)) as reference:
            crs, transform, (height, width) = reference.crs, reference.transform, reference.shape
        raw = from_bounds(*transform_bounds('EPSG:3006', crs, *bounds, densify_pts=41),
                          transform=transform)
        left, top = max(0, math.floor(raw.col_off) - 2), max(0, math.floor(raw.row_off) - 2)
        right = min(width, math.ceil(raw.col_off + raw.width) + 2)
        bottom = min(height, math.ceil(raw.row_off + raw.height) + 2)
        if right <= left or bottom <= top:
            continue
        ks = range(top // BAND_ROWS, (bottom - 1) // BAND_ROWS + 1)
        sizes = (gt.registry.get_tile_file_size(year, lon, lat),
                 gt.registry.get_scales_file_size(year, lon, lat))
        sources = []
        for relative, size, (dtype, shape) in zip(
                tile_to_embedding_paths(lon, lat, year), sizes,
                [(np.int8, (height, width, 128)), (np.float32, (height, width))]):
            url = f"{root.rstrip('/')}/{relative.as_posix()}"
            sources.append((url, dtype, shape))
            wanted += [(url, k, shape, dtype, size) for k in ks]
            files.append({'file': Path(url).name, 'size': size})
        windows.append((crs, transform, left, top, right, bottom, ks, sources))
    # 2. Fetch the missing bands in parallel: first each file's size check and
    #    header (memoised by geotessera, so read once), then the bands.
    from geotessera.remote import read_npy_window
    missing = [w for w in wanted if not _band_path(here, w[0], w[1]).is_file()]

    def check(url: str, size: int) -> None:
        if get_fs(url).size(url) != size:
            raise ValueError('Remote tile size disagrees with the pinned registry')
        read_npy_window(url, 0, 0, 0, 0)   # parses and memoises the header only

    with ThreadPoolExecutor(BAND_WORKERS) as pool:
        list(pool.map(lambda pair: check(*pair), {(w[0], w[4]) for w in missing}))
        list(pool.map(lambda w: _remote_band(w[0], w[1], w[2], w[3], here), missing))
    # 3. Merge exactly as the tile and COG paths do.
    merged = np.zeros((128, 512, 512), np.float16)
    valid = np.zeros((512, 512), bool)
    for crs, transform, left, top, right, bottom, ks, sources in windows:
        offset = ks[0] * BAND_ROWS
        q, scales = (np.concatenate([_remote_band(url, k, shape, dtype, here) for k in ks])
                     [top - offset:bottom - offset, left:right]
                     for url, dtype, shape in sources)
        arr = q.astype(np.float32) * scales[..., np.newaxis]
        if not np.isfinite(arr).all():
            raise ValueError('Nonfinite dequantized remote data')
        window = window_transform(Window(left, top, right - left, bottom - top), transform)
        weights = np.zeros((512, 512), np.float32)
        reproject(np.any(arr != 0, axis=2).astype(np.float32), weights,
                  src_transform=window, src_crs=crs,
                  dst_transform=grid.transform(), dst_crs='EPSG:3006',
                  resampling=Resampling.bilinear)
        chunk = _reproject_tessera_to_tile(arr, window, crs, b, 512)
        take = (weights >= 1 - 1e-6) & np.any(chunk != 0, axis=0)
        merged[:, take] = chunk[:, take]
        valid |= take
    return merged, valid, {'files': files, 'blocks': len(blocks),
                           'bands': {'fetched': len(missing), 'cached': len(wanted) - len(missing)},
                           'storage': 'remote-npy-range', 'root': root}


def prefetch_bands(gt, bounds_3006: tuple, year: int, *, root: str, cache: Path,
                   max_bytes: int) -> dict:
    """Fetch every band a box's windows could need, nearest the box centre first.

    For view and municipality prefetch: any later window inside the box reads
    only cached bands. Stops before exceeding ``max_bytes`` of new bands, so a
    large municipality is filled from the view outward within the budget.
    """
    import math
    from concurrent.futures import ThreadPoolExecutor
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from geotessera.registry import tile_to_embedding_paths
    from geotessera.remote import get_fs, read_npy_window

    west, south, east, north = bounds_3006
    cx, cy = (west + east) / 2, (south + north) / 2
    lonlat = transform_bounds('EPSG:3006', 'EPSG:4326', *bounds_3006, densify_pts=41)
    clon, clat = (lonlat[0] + lonlat[2]) / 2, (lonlat[1] + lonlat[3]) / 2
    blocks = sorted((b for b in gt.registry.load_blocks_for_region(bounds=lonlat, year=year)
                     if b[0] == year), key=lambda b: (b[1] - clon) ** 2 + (b[2] - clat) ** 2)
    here = cache / str(year)
    wanted, budget = [], {'bytes': 0, 'skipped_blocks': 0}
    for _, lon, lat in blocks:
        with rasterio.open(gt.registry.fetch_landmask(lon=lon, lat=lat)) as reference:
            crs, transform, (height, width) = reference.crs, reference.transform, reference.shape
        raw = from_bounds(*transform_bounds('EPSG:3006', crs, *bounds_3006, densify_pts=41),
                          transform=transform)
        top = max(0, math.floor(raw.row_off) - 2)
        bottom = min(height, math.ceil(raw.row_off + raw.height) + 2)
        if bottom <= top:
            continue
        ks = range(top // BAND_ROWS, (bottom - 1) // BAND_ROWS + 1)
        sizes = (gt.registry.get_tile_file_size(year, lon, lat),
                 gt.registry.get_scales_file_size(year, lon, lat))
        block = []
        for relative, size, (dtype, shape, item) in zip(
                tile_to_embedding_paths(lon, lat, year), sizes,
                [(np.int8, (height, width, 128), 128), (np.float32, (height, width), 4)]):
            url = f"{root.rstrip('/')}/{relative.as_posix()}"
            block += [(url, k, shape, dtype, size, BAND_ROWS * width * item)
                      for k in ks if not _band_path(here, url, k).is_file()]
        cost = sum(w[5] for w in block)
        if budget['bytes'] + cost > max_bytes:
            budget['skipped_blocks'] += 1
            continue
        budget['bytes'] += cost
        wanted += block

    def check(url: str, size: int) -> None:
        if get_fs(url).size(url) != size:
            raise ValueError('Remote tile size disagrees with the pinned registry')
        read_npy_window(url, 0, 0, 0, 0)

    with ThreadPoolExecutor(BAND_WORKERS) as pool:
        list(pool.map(lambda pair: check(*pair), {(w[0], w[4]) for w in wanted}))
        list(pool.map(lambda w: _remote_band(w[0], w[1], w[2], w[3], here), wanted))
    return {'blocks': len(blocks), 'fetched_bands': len(wanted),
            'fetched_bytes_estimate': budget['bytes'], 'skipped_blocks': budget['skipped_blocks']}


def enforce_band_budget(cache: Path, max_bytes: int, grace_s: float = 600) -> dict:
    """Delete least recently used bands until the cache fits ``max_bytes``.

    A band touched within ``grace_s`` seconds is never deleted, so a window
    being assembled keeps the bands it is reading.
    """
    import time
    bands = []
    for path in cache.rglob('*_b*.npy'):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        bands.append((stat.st_mtime, stat.st_size, path))
    total = sum(size for _, size, _ in bands)
    removed, freed, now = 0, 0, time.time()
    for mtime, size, path in sorted(bands):
        if total - freed <= max_bytes:
            break
        if now - mtime < grace_s:
            continue
        path.unlink(missing_ok=True)
        removed, freed = removed + 1, freed + size
    return {'bands': len(bands), 'bytes': total - freed, 'removed': removed, 'freed': freed}


def prepare_aux(config: dict, base: Path, grid: Grid, year: int):
    from imint.data.wekeo_vpp import (fetch_vpp_tiles_local, _bounds_overlap,
                                     _VPP_PRODUCT_TYPES)
    from rasterio.warp import transform_bounds
    from imint.training import skg_open
    from imint.training.skg_grunddata import (
        fetch_tree_height_tile, fetch_volume_tile, fetch_basal_area_tile, fetch_diameter_tile)
    from imint.training.slu_markfukt import fetch_markfukt_tile
    from imint.training.copernicus_dem import read_dem_window

    index_path, skg_path = verify_aux_metadata(config, base)
    receipts = (base / config['identity_cache']).resolve()
    # Environment scoped to this isolated child, never changed in the API.
    os.environ['SKG_OPEN_DIR'] = str(skg_path.parent)
    index = json.loads(index_path.read_text())
    b = grid.bounds()
    bounds = tuple(b[k] for k in ('west', 'south', 'east', 'north'))
    wgs = transform_bounds('EPSG:3006', 'EPSG:4326', *bounds)
    selected = {name: meta for name, meta in index.items()
                if meta['year'] == year and meta['season'] == 1
                and _bounds_overlap(meta['bounds_4326'], wgs)}
    if not set(_VPP_PRODUCT_TYPES).issubset({m['metric'] for m in selected.values()}):
        raise InputUnavailable('Exact-year HR-VPP source coverage is incomplete')
    # The serving inventory adds hashes to the canonical WEkEO index, which
    # alone contains only filenames and footprints. A mutable pathname is not
    # a source identity.
    vpp_inventory = config['vpp_files']
    vpp_sources = []
    for name in selected:
        record = vpp_inventory[name]
        verify_source_file({**record, 'path': name}, index_path.parent, receipts)
        vpp_sources.append({'file': name, **public_identity(record)})
    mask_sources = verify_county_masks(skg_path, bounds, receipts)
    skg_provenance = skg_open.window_provenance(*bounds, size_px=512)
    skg_provenance['rasters'].update(mask_sources)
    if not skg_provenance['counties']:
        raise InputUnavailable('No Swedish forestry coverage for this location')
    for relative, record in skg_provenance['rasters'].items():
        source = skg_path.parent / relative
        verify_source_file({'path': relative, 'sha256': record['sha256'],
                            'size': source.stat().st_size}, skg_path.parent, receipts)
    vpp = fetch_vpp_tiles_local(*bounds, size_px=512, vpp_cog_dir=index_path.parent,
                               year=year, season=1)
    aux = {'vpp_' + name: value.astype(np.float32) for name, value in vpp.items()}
    if not np.any(aux['vpp_sosd']):
        raise InputUnavailable('No usable exact-year phenology in this window')
    for name, fetch in [('height', fetch_tree_height_tile), ('volume', fetch_volume_tile),
                        ('basal_area', fetch_basal_area_tile), ('diameter', fetch_diameter_tile)]:
        aux[name] = fetch(*bounds, size_px=512).astype(np.float32)
    raw = fetch_markfukt_tile(*bounds, size_px=512)
    moisture = raw.astype(np.float32)
    moisture[(raw == 0) | (raw > 101)] = np.nan
    moisture[(raw >= 1) & (raw <= 100)] /= 100
    moisture[raw == 101] = 1.01
    aux['markfukt'] = moisture
    dem_sources = [verify_source_file(record, base, receipts) for record in config['dem_sources']]
    aux['dem'] = read_dem_window(dem_sources, *bounds, size_px=512)
    provenance = {
        'vpp': {'year': year, 'season': 1, 'index': public_identity(config['vpp_index']),
                'files': vpp_sources, 'source': 'Copernicus HR-VPP via WEkEO local COGs'},
        'forestry': {**skg_provenance, 'temporal_policy': 'latest scan per pixel in fixed release'},
        'markfukt': {'temporal_policy': 'fixed public mirror; differs from retired proxy version'},
        'dem': {'source': 'Copernicus GLO-30', 'temporal_policy': 'static DSM, primarily 2011–2015',
                'files': [{'file': path.name, **public_identity(record)}
                          for path, record in zip(dem_sources, config['dem_sources'])]},
    }
    # Detect metadata/source changes during the read before publishing arrays.
    verify_aux_metadata(config, base)
    for name in selected:
        verify_source_file({**vpp_inventory[name], 'path': name}, index_path.parent, receipts)
    for relative, record in skg_provenance['rasters'].items():
        source = skg_path.parent / relative
        verify_source_file({'path': relative, 'sha256': record['sha256'],
                            'size': source.stat().st_size}, skg_path.parent, receipts)
    for record in config['dem_sources']:
        verify_source_file(record, base, receipts)
    return aux, provenance


def pinned_geotessera(config: dict, base: Path):
    """GeoTessera 0.10.2 on the pinned v1/vultr registry, and its whole-tile cache."""
    if version('geotessera') != '0.10.2':
        raise ValueError('Preparation requires geotessera==0.10.2')
    registry = verify_file(config['registry'], base)
    landmasks = verify_file(config['landmasks'], base)
    if registry.name != 'manifest.parquet' or landmasks.name != 'landmasks.parquet' or registry.parent != landmasks.parent:
        raise ValueError('Pinned registry directory must contain both canonical parquet files')
    from geotessera import GeoTessera
    cache = (base / config['embedding_cache']).resolve() / 'geotessera-0.10.2-v1-vultr'
    validate_embedding_cache(cache)
    gt = GeoTessera(dataset_version='v1', dataset_variant='vultr', registry_dir=str(registry.parent),
                    cache_dir=str(cache / 'registry'), embeddings_dir=str(cache))
    return gt, cache


def remote_bands(config: dict, cache: Path) -> tuple[str, Path, int]:
    """The pinned remote tile root, the band cache beside the tile cache, and its budget.

    Range reads of the same v1/vultr tiles; bands never enter geotessera's own
    cache layout.
    """
    from geotessera.registry import TESSERA_NPY_MIRROR_URL
    if config['remote_tiles'] != f'{TESSERA_NPY_MIRROR_URL}/v1':
        raise ValueError('remote_tiles must be the published v1/vultr tile root')
    budget = config['band_cache_max_bytes']
    if type(budget) is not int or budget <= 0:
        raise ValueError('Explicit positive band cache budget is required')
    return config['remote_tiles'], cache.parent / 'geotessera-0.10.2-v1-vultr-bands', budget


def prepare_embedding(config: dict, base: Path, grid: Grid, year: int):
    """(embedding, validity, provenance) from the configured TESSERA source.

    ``cog_index``: the staged compact COGs. ``remote_tiles``: range reads of the
    published v1/vultr tiles. Otherwise whole-tile downloads into the pinned
    cache. All three read the same registry's tiles, so they share EMBEDDING.
    """
    if 'cog_index' in config:
        from imint.inference.lclu_cog import assemble_cog_embedding
        index_path = verify_file(config['cog_index'], base)
        embedding, valid, emb_provenance = assemble_cog_embedding(
            json.loads(index_path.read_text()), index_path.parent, grid, year,
            (base / config['identity_cache']).resolve())
        verify_file(config['cog_index'], base)
        emb_provenance['cog_index'] = public_identity(config['cog_index'])
    else:
        max_blocks = config['max_blocks']
        if type(max_blocks) is not int or not 1 <= max_blocks <= 4:
            raise ValueError('Explicit block budget of at most four blocks is required')
        gt, cache = pinned_geotessera(config, base)
        if 'remote_tiles' in config:
            root, bands, budget = remote_bands(config, cache)
            embedding, valid, emb_provenance = assemble_remote_embedding(
                gt, grid, year, root=root, max_blocks=max_blocks, cache=bands)
            emb_provenance['band_cache'] = enforce_band_budget(bands, budget)
        else:
            max_bytes = config['max_download_bytes']
            if type(max_bytes) is not int or max_bytes <= 0:
                raise ValueError('Explicit positive download budget is required')
            embedding, valid, emb_provenance = assemble_embedding(
                gt, grid, year, max_blocks=max_blocks, max_bytes=max_bytes)
        validate_embedding_cache(cache)
        verify_file(config['registry'], base)
        verify_file(config['landmasks'], base)
        emb_provenance.update(registry=public_identity(config['registry']),
                              landmasks=public_identity(config['landmasks']))
    return embedding, valid, emb_provenance


def prepare(config_path: Path, grid: Grid, year: int, output: Path):
    if year not in YEARS:
        raise InputUnavailable('Unsupported exact year')
    config_identity = file_identity(config_path)
    config = json.loads(config_path.read_text())
    if config['version'] != 1 or config['embedding'] != EMBEDDING:
        raise ValueError('Embedding source must be pinned to GeoTessera 0.10.2 v1/vultr')
    base = config_path.parent
    # Fail unavailable local auxiliaries before downloading any embedding data.
    aux, provenance = prepare_aux(config, base, grid, year)
    embedding, valid, emb_provenance = prepare_embedding(config, base, grid, year)
    # The county mask limits a cross-border window to Sweden; it does not infer
    # a class or turn missing data into model class zero.
    from imint.training import skg_open
    b = grid.bounds()
    bounds = tuple(b[k] for k in ('west', 'south', 'east', 'north'))
    counties = set(provenance['forestry']['counties'])
    skg_path = verify_file(config['skg_manifest'], base)
    receipts = (base / config['identity_cache']).resolve()
    verify_county_masks(skg_path, bounds, receipts)
    valid &= skg_open.county_cells(*bounds, (512, 512), counties)
    verify_county_masks(skg_path, bounds, receipts)
    verify_file(config['skg_manifest'], base)
    arrays = {**aux, 'tessera': embedding, 'valid_embedding': valid,
              'year': np.int32(year), 'tessera_year': np.int32(year), 'vpp_year': np.int32(year),
              'crs': np.array('EPSG:3006'), 'transform': np.array(list(grid.transform())[:6]),
              'easting': np.int32(grid.east), 'northing': np.int32(grid.north)}
    if set(AUX_NAMES) - arrays.keys():
        raise ValueError('Missing R4 auxiliaries')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'input.npz'
    atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))
    record = {**file_identity(path), 'path': 'input.npz'}
    validate_bundle(path, record, grid, year)
    if public_identity(file_identity(config_path)) != public_identity(config_identity):
        raise ValueError('Source configuration changed during preparation')
    write_json(output / 'entry.json', {
        'year': year, 'center_3006': [grid.east, grid.north], 'public_area': True,
        'input': record, 'embedding': EMBEDDING,
        'source_config': public_identity(config_identity),
        'auxiliary_provenance': provenance,
        'embedding_provenance': emb_provenance,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--east', required=True, type=int)
    parser.add_argument('--north', required=True, type=int)
    parser.add_argument('--year', required=True, type=int, choices=YEARS)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.east % 10 or args.north % 10:
        parser.error('Centre must be snapped to the EPSG:3006 10 m grid')
    prepare(args.config.resolve(), Grid(args.east, args.north), args.year, args.output.resolve())


if __name__ == '__main__':
    main()
