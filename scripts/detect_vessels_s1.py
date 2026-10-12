#!/usr/bin/env python3
"""Sentinel-1 vessel detection (AI2 detector + attributes) for one AOI and date.

Fetches the target and historical GRD scenes from Microsoft Planetary Computer
(``imint.fetch.fetch_s1_vessel_scenes``), runs ``AI2S1VesselAnalyzer`` and
writes the GeoJSON FeatureCollection marine-awareness reads
(``imint.exporters.export.s1_vessel_feature_collection``). A detection without
an AIS match is a candidate, not a confirmed dark vessel.

Example (Skagen test scene, ~10 min on an M1 CPU once the inputs are cached):

    python scripts/detect_vessels_s1.py --date 2026-06-28 \\
        --bbox 10.50 57.55 11.70 58.05 --cache-dir ~/imint_cache/s1 \\
        --out outputs/s1/skagen_2026-06-28_s1.geojson
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", required=True, help="target acquisition date (UTC), YYYY-MM-DD")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="WGS84 degrees")
    ap.add_argument("--cache-dir", required=True, type=Path,
                    help="cache for radar windows and model inputs")
    ap.add_argument("--out", required=True, type=Path, help="output GeoJSON")
    ap.add_argument("--run-json", type=Path,
                    help="optional: full-precision regions, metadata and scene provenance")
    ap.add_argument("--infra-path", type=Path,
                    help="local Satlas marine-infrastructure GeoJSON (default: cached download)")
    ap.add_argument("--score-threshold", type=float,
                    help="default: the analyzer's SCORE_THRESHOLD (0.7, AI2's API default)")
    ap.add_argument("--no-attributes", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from imint.analyzers.ai2_vessels_s1 import AI2S1VesselAnalyzer
    from imint.exporters.export import save_s1_vessel_geojson, s1_vessel_feature_collection
    from imint.fetch import S1_VESSEL_SOURCE, fetch_s1_vessel_scenes

    west, south, east, north = args.bbox
    coords = {"west": west, "south": south, "east": east, "north": north}
    t0 = time.time()
    scenes = fetch_s1_vessel_scenes(coords, args.date, cache_dir=args.cache_dir)
    print(f"fetch {time.time() - t0:.0f} s: target {scenes.target_scene.scene_id}, "
          f"historical {scenes.historical_scene.scene_id}, grid {scenes.geo.crs} "
          f"{scenes.geo.shape[1]}x{scenes.geo.shape[0]} px, AOI imaged at "
          f"{scenes.scene_time_utc}", flush=True)

    config = {"device": args.device, "predict_attributes": not args.no_attributes}
    if args.score_threshold is not None:
        config["score_threshold"] = args.score_threshold
    if args.infra_path:
        config["infra_path"] = str(args.infra_path)
    result = AI2S1VesselAnalyzer(config).run(
        None, bands=scenes.bands(), date=args.date, coords=coords, geo=scenes.geo)
    if not result.success:
        print(f"analyzer failed: {result.error}", file=sys.stderr)
        return 1
    md = result.metadata
    print(f"detector: {md['patches']} patches, {md['raw_detections']} raw, "
          f"{md['after_nms']} after NMS, {md['final_detections']} after infra filter "
          f"({md['infra_filter']}) in {md['seconds_total']} s", flush=True)

    fc = s1_vessel_feature_collection(
        result, scenes.geo, date=args.date, bbox=args.bbox,
        scene_id=scenes.target_scene.scene_id,
        historical_scene_id=scenes.historical_scene.scene_id,
        scene_time_utc=scenes.scene_time_utc, scene_start_utc=scenes.target_scene.start_utc,
        scene_end_utc=scenes.target_scene.end_utc, source=S1_VESSEL_SOURCE)
    print(f"GeoJSON: {len(fc['features'])} detections inside the bbox, "
          f"{fc['properties']['n_outside_bbox']} outside it dropped", flush=True)
    save_s1_vessel_geojson(fc, args.out)
    if args.run_json:
        import dataclasses

        args.run_json.parent.mkdir(parents=True, exist_ok=True)
        args.run_json.write_text(json.dumps({
            "outputs": result.outputs, "metadata": md,
            "target_scene": dataclasses.asdict(scenes.target_scene),
            "historical_scene": dataclasses.asdict(scenes.historical_scene),
            "scene_time_utc": scenes.scene_time_utc,
            "provenance": scenes.provenance,
        }, indent=1, default=float))
        print(f"    saved: {args.run_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
