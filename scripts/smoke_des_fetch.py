"""DES openEO smoke test — exercises the production fetch code path.

Run:
    python scripts/smoke_des_fetch.py

Validates, in order:
    1. `_connect()` from imint.fetch — version-pinned discovery + auth.
    2. `list_collections()` includes the legacy `s2_msi_l2a`.
    3. A *minimal* `load_collection → filter_bands → reduce_dimension →
       rename_labels → download` graph — same shape as
       `imint/training/openeo_tile_graph.py::_build_slot_cube` — runs
       end-to-end. The graph deliberately includes `rename_labels`
       because that process is not advertised on `/processes`; this
       call answers whether the server still accepts it.
    4. Returned bytes parse via `_unpack_openeo_gtiff_bytes` into a
       (1, H, W) raster with non-trivial reflectance after the
       PB04.00 -1000 BOA dequant.

Tiny bbox (~50 m × 30 m) and a 1-day window keep the run sub-minute.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv():
    """Minimal .env loader (project-root only). The repo's secrets
    helper is heavier than we need for a smoke test."""
    import os
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    _load_dotenv()
    import numpy as np
    import rasterio

    from imint.fetch import (
        COLLECTION,
        OPENEO_URL,
        _connect,
        _unpack_openeo_gtiff_bytes,
    )
    from imint.utils import dn_to_reflectance

    print(f"[1/4] connecting to {OPENEO_URL} …", flush=True)
    t0 = time.time()
    conn = _connect()
    print(f"      connected in {time.time() - t0:.1f}s "
          f"(capabilities={conn.capabilities().api_version()})", flush=True)

    print(f"[2/4] listing collections …", flush=True)
    cols = conn.list_collections()
    cids = sorted(c["id"] for c in cols)
    print(f"      {len(cids)} collections: {', '.join(cids)}", flush=True)
    assert COLLECTION in cids, f"{COLLECTION!r} missing from collections"

    print(f"[3/4] running tile-graph shape (load→filter→reduce→"
          f"rename_labels→download) …", flush=True)
    # Tiny patch in Skåne — small enough that the openEO job stays well
    # under any timeout.
    bbox_3006 = {
        "west":  390000.0,
        "south": 6230000.0,
        "east":  390500.0,
        "north": 6230500.0,
        "crs":   3006,
    }
    date_start = "2024-06-15"
    date_end   = "2024-06-22"  # 7-day window — one Sentinel-2 pass for sure
    slot_idx   = 0
    bands      = ["b04"]  # one band keeps the download tiny

    cube = conn.load_collection(
        collection_id=COLLECTION,
        spatial_extent=bbox_3006,
        temporal_extent=[date_start, date_end],
        bands=bands,
    )
    cube = cube.reduce_dimension(dimension="t", reducer="first")
    cube = cube.filter_bands(bands=bands)
    # This is the unproven call — rename_labels isn't in /processes.
    cube = cube.rename_labels(
        dimension="bands",
        target=[f"s{slot_idx}_{b}" for b in bands],
    )

    t0 = time.time()
    try:
        raw_bytes = cube.download(format="gtiff")
    except Exception as exc:
        print(f"      ✗ download failed: {type(exc).__name__}: {exc}",
              flush=True)
        return 2
    dl_s = time.time() - t0
    print(f"      download OK in {dl_s:.1f}s ({len(raw_bytes)} bytes)",
          flush=True)

    raw_bytes = _unpack_openeo_gtiff_bytes(raw_bytes)
    print(f"[4/4] parsing raster + dequant check …", flush=True)
    with rasterio.open(io.BytesIO(raw_bytes)) as src:
        print(f"      shape={src.shape}, bands={src.count}, crs={src.crs}",
              flush=True)
        dn = src.read()
        # DES bakes the PB04.00 -1000 BOA offset into COGs.
        refl = dn_to_reflectance(dn, source="des")
        print(f"      DN  range: [{int(dn.min())}, {int(dn.max())}]",
              flush=True)
        print(f"      refl range: [{float(refl.min()):.4f}, "
              f"{float(refl.max()):.4f}]  mean={float(refl.mean()):.4f}",
              flush=True)
        assert src.count == len(bands), \
            f"expected {len(bands)} band(s), got {src.count}"
        assert refl.min() >= -0.05, \
            f"refl < -0.05 ({refl.min():.4f}) — dequant suspect"
        assert refl.max() <= 1.05, \
            f"refl > 1.05 ({refl.max():.4f}) — dequant suspect"
        assert np.isfinite(refl).all(), "non-finite values in refl"

    print("\n✅ DES openEO smoke test PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
