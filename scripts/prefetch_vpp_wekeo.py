"""CLI: bulk-prefetch HR-VPP VPP COGs from WEkEO to a local cache.

Run once (e.g. on a k8s pod) to populate the WEkEO fallback cache that
imint.training.cdse_vpp falls back to when the CDSE Sentinel Hub quota
is exhausted.

    python scripts/prefetch_vpp_wekeo.py \\
        --mgrs-tiles 33VWJ,33VWH,33WXP \\
        --years 2021,2022 \\
        --dest-dir /data/vpp_wekeo

The S2 MGRS tile IDs must be given explicitly — the repo has no
EPSG:3006-bbox -> MGRS primitive, and the WEkEO HDA query keys on
``tileId``. WEkEO credentials: WEKEO_USERNAME / WEKEO_PASSWORD env
vars, or the hda client's ~/.hdarc.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imint.training.wekeo_vpp import prefetch_vpp_cogs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bulk-prefetch HR-VPP VPP COGs from WEkEO HDA",
    )
    p.add_argument(
        "--mgrs-tiles", required=True,
        help="Comma-separated Sentinel-2 MGRS tile IDs, e.g. 33VWJ,33VWH",
    )
    p.add_argument(
        "--years", required=True,
        help="Comma-separated product years, e.g. 2021,2022",
    )
    p.add_argument("--dest-dir", default="/data/vpp_wekeo")
    p.add_argument("--season", type=int, default=1, choices=[1, 2])
    args = p.parse_args()

    tiles = [t.strip() for t in args.mgrs_tiles.split(",") if t.strip()]
    years = [int(y) for y in args.years.split(",") if y.strip()]
    if not tiles or not years:
        raise SystemExit("--mgrs-tiles and --years must be non-empty")

    print("=== WEkEO VPP prefetch ===")
    print(f"  tiles:   {len(tiles)}  {tiles}")
    print(f"  years:   {years}")
    print(f"  season:  {args.season}")
    print(f"  dest:    {args.dest_dir}")

    index = prefetch_vpp_cogs(
        tiles, years, args.dest_dir, season=args.season,
    )
    print(f"=== done — {len(index)} COGs in cache ===")


if __name__ == "__main__":
    main()
