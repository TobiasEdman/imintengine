#!/usr/bin/env python3
"""Run AI2S1VesselAnalyzer on AI2's reference window and persist the full result.

``--mode ai2``  (run A): AI2's own materialised model input -> isolates the port.
``--mode port`` (run B): the port fetch's input over the same window, masked to
AI2's coverage (zero where AI2's raster is zero) -> isolates the input chain.
The port input must cover AI2's window (fetch the bbox 10.12 57.28 12.0 58.21).

    python scripts/marine/s1_port/run_analyzer_on.py --record <record> --mode ai2 --out run_a.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

import _record as rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--mode", choices=("ai2", "port"), required=True)
    ap.add_argument("--port-input", type=rec.Path,
                    help="cached port model input (.tif with .json sidecar), mode port")
    ap.add_argument("--infra-path", type=rec.Path,
                    help="Satlas GeoJSON (default: the reference run's, inputs/)")
    ap.add_argument("--out", type=rec.Path, required=True)
    a = ap.parse_args()
    if a.mode == "port" and a.port_input is None:
        ap.error("--mode port needs --port-input")

    from rasterio.transform import Affine

    from imint.analyzers.ai2_vessels_s1 import AI2S1VesselAnalyzer
    from imint.fetch import S1_HISTORICAL_BAND_KEYS, S1_TARGET_BAND_KEYS
    from imint.job import GeoContext

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ai2, (c0, r0, c1, r1) = rec.read_ai2_stack(a.record)
    if a.mode == "ai2":
        stack = ai2
    else:
        port, (pc0, pr0), _ = rec.read_port_input(a.port_input, chain=rec.AI2_PIXEL_CHAIN)
        sub = port[:, r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0]
        if sub.shape[1:] != ai2.shape[1:]:
            raise SystemExit(f"port input does not cover AI2's window: {sub.shape} vs {ai2.shape}")
        stack = np.where(ai2 > 0, sub, 0).astype(np.uint16)
    bands = dict(zip(S1_TARGET_BAND_KEYS + S1_HISTORICAL_BAND_KEYS, stack, strict=True))
    geo = GeoContext(crs="EPSG:32632", transform=Affine(10, 0, c0 * 10, 0, -10, r0 * -10),
                     bounds_projected=None, bounds_wgs84=None, shape=(r1 - r0, c1 - c0))
    infra = a.infra_path or a.record / "inputs" / "marine_infrastructure.geojson"
    t = time.time()
    res = AI2S1VesselAnalyzer({"infra_path": str(infra)}).run(None, bands=bands, geo=geo)
    if not res.success:
        raise SystemExit(f"analyzer failed: {res.error}")
    print("seconds", round(time.time() - t, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"outputs": res.outputs, "metadata": res.metadata,
                                 "origin_px": [c0, r0]}, indent=1, default=float))
    print(json.dumps({k: res.metadata[k] for k in ("patches", "raw_detections", "after_nms",
                                                   "final_detections")}))


if __name__ == "__main__":
    main()
