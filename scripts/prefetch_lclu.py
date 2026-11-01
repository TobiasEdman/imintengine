#!/usr/bin/env python3
"""Prefetch the TESSERA bands a box needs, within the band cache budget (IM-026).

Invoked by RepositorySources.prefetch in a time-bounded child process with the
preparation interpreter (geotessera 0.10.2), for the map viewer's current view
or chosen municipality. Only sources configured with ``remote_tiles`` have
anything to fetch; for staged sources this is a no-op.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.inference.lclu import EMBEDDING, YEARS, verify_file  # noqa: E402
from scripts import prepare_lclu as prep  # noqa: E402


def prefetch(config_path: Path, bounds_3006: tuple, year: int) -> dict:
    config = json.loads(config_path.read_text())
    if config['version'] != 1 or config['embedding'] != EMBEDDING:
        raise ValueError('Embedding source must be pinned to GeoTessera 0.10.2 v1/vultr')
    if 'remote_tiles' not in config:
        return {'state': 'not-needed'}
    limit = config['prefetch_max_bytes']
    if type(limit) is not int or limit <= 0:
        raise ValueError('Explicit positive prefetch budget is required')
    base = config_path.parent
    gt, cache = prep.pinned_geotessera(config, base)
    root, bands, budget = prep.remote_bands(config, cache)
    report = prep.prefetch_bands(gt, bounds_3006, year, root=root, cache=bands,
                                 max_bytes=min(limit, budget))
    report['band_cache'] = prep.enforce_band_budget(bands, budget)
    verify_file(config['registry'], base)
    verify_file(config['landmasks'], base)
    return {'state': 'prefetched', **report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--bounds', required=True, type=float, nargs=4,
                        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'), help='EPSG:3006')
    parser.add_argument('--year', required=True, type=int, choices=YEARS)
    args = parser.parse_args(argv)
    west, south, east, north = args.bounds
    if not (west < east and south < north):
        parser.error('Bounds must be a non-empty EPSG:3006 box')
    print(json.dumps(prefetch(args.config.resolve(), tuple(args.bounds), args.year)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
