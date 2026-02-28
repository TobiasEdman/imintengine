"""
run_vessel_heatmap.py — Multi-temporal vessel detection heatmap.

Downloads all cloud-free Sentinel-2 images in a date range, runs
vessel detection on each, and aggregates detections into a heatmap
showing frequently trafficked areas.

Usage:
    DES_USER=testuser DES_PASSWORD=secretpassword \
    .venv/bin/python run_vessel_heatmap.py \
        --west 11.25049 --south 58.42763 --east 11.30049 --north 58.47763 \
        --start 2025-07-01 --end 2025-07-31 \
        --output-dir outputs/full_bohuslan_2025-07-10

Optionally regenerate the IMINT showcase with the new heatmap panel:
    --showcase   Include heatmap in the showcase HTML report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-temporal vessel detection heatmap",
    )
    parser.add_argument("--west", type=float, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--east", type=float, required=True)
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--start", required=True, help="Start date ISO (e.g. 2025-07-01)")
    parser.add_argument("--end", required=True, help="End date ISO (e.g. 2025-07-31)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cloud-threshold", type=float, default=0.3,
                        help="Max AOI cloud fraction (default: 0.3)")
    parser.add_argument("--scene-cloud-max", type=float, default=50.0,
                        help="STAC scene cloud filter %% (default: 50)")
    parser.add_argument("--sigma", type=float, default=5.0,
                        help="Gaussian smoothing σ in pixels (default: 5.0)")
    parser.add_argument("--prefix", default="",
                        help="Date prefix for output files (e.g. '2025-07-10_')")
    parser.add_argument("--showcase", action="store_true",
                        help="Regenerate IMINT showcase HTML after heatmap")
    parser.add_argument("--analyzer", choices=["yolo", "ai2"], default="yolo",
                        help="Vessel detector: 'yolo' (YOLO11s) or 'ai2' (rslearn Swin V2 B)")
    parser.add_argument("--predict-attributes", action="store_true",
                        help="Run AI2 attribute model on each detection (speed/heading/type/length/width)")
    parser.add_argument("--fire-dir", default=None,
                        help="Fire analysis dir (for showcase, if --showcase)")
    parser.add_argument("--fire-date", default="", help="Fire date prefix")
    parser.add_argument("--marine-date", default="", help="Marine date prefix")
    args = parser.parse_args()

    coords = {
        "west": args.west, "south": args.south,
        "east": args.east, "north": args.north,
    }

    print("=" * 70)
    print("  IMINT Engine — Multi-temporal Vessel Heatmap")
    print(f"  Area:      {args.west:.5f}–{args.east:.5f}°E, "
          f"{args.south:.5f}–{args.north:.5f}°N")
    print(f"  Period:    {args.start} – {args.end}")
    print(f"  Cloud:     ≤{args.cloud_threshold:.0%} AOI, "
          f"≤{args.scene_cloud_max:.0f}% scene")
    print(f"  Sigma:     {args.sigma} px ({args.sigma * 10:.0f} m)")
    print(f"  Analyzer:  {args.analyzer}")
    if args.predict_attributes:
        print(f"  Attributes: AI2 (speed/heading/type/length/width)")
    print(f"  Output:    {args.output_dir}")
    print("=" * 70)

    from imint.fetch import fetch_vessel_heatmap

    result = fetch_vessel_heatmap(
        coords=coords,
        date_start=args.start,
        date_end=args.end,
        output_dir=args.output_dir,
        cloud_threshold=args.cloud_threshold,
        scene_cloud_max=args.scene_cloud_max,
        gaussian_sigma=args.sigma,
        prefix=args.prefix,
        analyzer_type=args.analyzer,
        predict_attributes=args.predict_attributes,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("  VESSEL HEATMAP RESULTS")
    print("=" * 70)
    print(f"  Dates used:    {len(result['dates_used'])}")
    print(f"  Dates skipped: {len(result['dates_skipped'])}")
    print(f"  Total vessels: {result['total_detections']}")
    if result["per_date"]:
        print("\n  Per-date breakdown:")
        for pd in result["per_date"]:
            cloud_str = f"{pd['cloud']:.1%}" if pd.get("cloud") is not None else "err"
            err = f"  ({pd['error']})" if pd.get("error") else ""
            print(f"    {pd['date']}  cloud={cloud_str}  vessels={pd['vessels']}{err}")
    if result["heatmap_path"]:
        print(f"\n  Heatmap PNG → {result['heatmap_path']}")
    print("=" * 70)

    # Optionally regenerate showcase
    if args.showcase:
        print("\n[showcase] Regenerating IMINT showcase...")
        from imint.exporters.html_report import save_tabbed_report
        fire_dir = args.fire_dir or ""
        save_tabbed_report(
            fire_dir=fire_dir,
            marine_dir=args.output_dir,
            output_path=str(Path(args.output_dir).parent / "imint_showcase.html"),
            fire_date=args.fire_date,
            marine_date=args.marine_date,
        )


if __name__ == "__main__":
    main()
