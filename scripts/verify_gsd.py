#!/usr/bin/env python3
"""Verify tile GSD by comparing on-disk tile against a freshly-fetched
scene at the same center with the correct 5120m bbox.

The hypothesis: fetch-tiles-512 jobs using --from-json with the stale
256-era manifest pass bbox=2560m + size_px=512 to Sentinel Hub, which
returns 5m-GSD upsampled rasters instead of native 10m. This script
proves or disproves the hypothesis with one extra API call.

Output: /tmp/gsd_compare.png with 4 panels:
  1. On-disk spectral (RGB composite, one frame)
  2. Fresh fetch with correct 5120m bbox (RGB composite, same date if possible)
  3. Absolute pixel-wise difference (amplified)
  4. High-frequency proxy (Laplacian) — upsampled data shows low HF content

Usage (in fetch pod):
    python3 scripts/verify_gsd.py \\
        --data-dir /data/unified_v2_512 \\
        --tile tile_885060_7420060 \\
        --out /tmp/gsd_compare.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Locate repo root
_candidates = [
    os.environ.get("IMINT_REPO"),
    str(Path(__file__).resolve().parents[1]),
    "/workspace/imintengine",
]
for c in _candidates:
    if c and (Path(c) / "imint").is_dir():
        sys.path.insert(0, c)
        break

import numpy as np


def fetch_correct_bbox(center_e: int, center_n: int, date: str, size_px: int = 512):
    """Fetch with the correct (size_px × 10 m) bbox at 10m GSD."""
    from imint.training.cdse_s2 import fetch_s2_scene
    half_m = size_px * 10 // 2  # 5120 // 2 = 2560m
    west = center_e - half_m
    south = center_n - half_m
    east = center_e + half_m
    north = center_n + half_m
    print(f"  Fetching correct-bbox scene: center=({center_e},{center_n}) "
          f"bbox=[{west},{south},{east},{north}] ({east-west}m × {north-south}m) "
          f"size_px={size_px} date={date}")
    result = fetch_s2_scene(
        west, south, east, north,
        date=date, size_px=size_px,
        cloud_threshold=0.20, haze_threshold=0.20,
        nodata_threshold=0.10,
    )
    if result is None:
        return None
    spectral, scl, cloud_frac = result
    return spectral, (west, south, east, north)


def rgb_composite(spectral_6band: np.ndarray) -> np.ndarray:
    """(6, H, W) Prithvi-order [B02, B03, B04, B8A, B11, B12] → (H, W, 3) uint8 RGB."""
    # Prithvi band order: B02 Blue, B03 Green, B04 Red → indices 0, 1, 2
    # Reflectance [0,1] scaled to [0,255], with percentile stretch for viewing.
    rgb = np.stack([spectral_6band[2], spectral_6band[1], spectral_6band[0]], axis=-1)  # R,G,B
    # Percentile stretch
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def laplacian_hf(gray: np.ndarray) -> np.ndarray:
    """Simple 3x3 Laplacian as a high-frequency content proxy.
    Native 10m data has more HF than bilinearly-upsampled 5m→10m data.
    """
    from scipy.ndimage import laplace
    return np.abs(laplace(gray.astype(np.float32)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/data/unified_v2_512")
    parser.add_argument("--tile", required=True, help="Tile name, e.g. tile_885060_7420060")
    parser.add_argument("--out", default="/tmp/gsd_compare.png")
    parser.add_argument("--frame", type=int, default=2,
                        help="Which temporal frame to compare (default 2 = peak summer)")
    args = parser.parse_args()

    tile_path = os.path.join(args.data_dir, f"{args.tile}.npz")
    print(f"Loading {tile_path}")
    d = np.load(tile_path, allow_pickle=True)
    spectral = d["spectral"]  # (T*6, H, W)
    dates = list(d["dates"])
    n_frames = int(d.get("num_frames", spectral.shape[0] // 6))
    print(f"  n_frames={n_frames}, spectral shape={spectral.shape}")

    fi = min(args.frame, n_frames - 1)
    frame = spectral[fi*6:(fi+1)*6]  # (6, H, W)
    date_str = str(dates[fi])[:10]
    print(f"  Using frame {fi}, date={date_str}")

    # Extract center from tile name
    parts = args.tile.split("_")
    if len(parts) != 3 or parts[0] != "tile":
        print(f"  Tile name doesn't match tile_{{E}}_{{N}} pattern: {args.tile}", file=sys.stderr)
        sys.exit(1)
    center_e = int(parts[1])
    center_n = int(parts[2])
    print(f"  Center: ({center_e}, {center_n})")

    print()
    print("=== Fetching reference scene at correct 5120m bbox ===")
    ref_result = fetch_correct_bbox(center_e, center_n, date_str, size_px=512)
    if ref_result is None:
        print(f"  Reference fetch failed for date {date_str} — trying another frame")
        # Try other frames
        for fi2 in range(n_frames):
            if fi2 == fi:
                continue
            d2 = str(dates[fi2])[:10]
            if not d2 or d2 == "":
                continue
            print(f"  Retry with frame {fi2}, date={d2}")
            ref_result = fetch_correct_bbox(center_e, center_n, d2, size_px=512)
            if ref_result is not None:
                fi = fi2
                frame = spectral[fi*6:(fi+1)*6]
                date_str = d2
                break
    if ref_result is None:
        print("All reference fetches failed — cannot verify")
        sys.exit(2)

    ref_spectral, ref_bbox = ref_result

    # Stats
    print()
    print("=== Comparison ===")
    print(f"On-disk spectral [frame {fi}]: shape={frame.shape}, "
          f"min={frame.min():.4f}, max={frame.max():.4f}, mean={frame.mean():.4f}")
    print(f"Reference spectral:            shape={ref_spectral.shape}, "
          f"min={ref_spectral.min():.4f}, max={ref_spectral.max():.4f}, mean={ref_spectral.mean():.4f}")

    # Per-band mean absolute difference (reflectance units)
    mad = np.mean(np.abs(frame - ref_spectral), axis=(1, 2))
    print(f"Per-band MAD: {mad.tolist()}")

    # High-frequency content comparison (Laplacian mean abs)
    hf_on = laplacian_hf(frame[0]).mean()  # use B02 band
    hf_ref = laplacian_hf(ref_spectral[0]).mean()
    print(f"High-freq content (B02 Laplacian mean abs):")
    print(f"  On-disk: {hf_on:.6f}")
    print(f"  Reference (true 10m): {hf_ref:.6f}")
    print(f"  Ratio on/ref: {hf_on/max(hf_ref,1e-9):.3f}  (<1.0 suggests on-disk is smoother = upsampled)")

    # Visualize
    print()
    print(f"=== Writing {args.out} ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(rgb_composite(frame))
    axes[0, 0].set_title(f"On-disk tile {args.tile}\nframe {fi}, date {date_str}\nstored in /data/unified_v2_512")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_composite(ref_spectral))
    axes[0, 1].set_title(f"Reference: fresh fetch, correct 5120m bbox\nsame center, same date, size_px=512 → 10m GSD")
    axes[0, 1].axis("off")

    diff_rgb = np.abs(frame.astype(np.float32) - ref_spectral.astype(np.float32)).mean(axis=0)
    im = axes[1, 0].imshow(diff_rgb, cmap="hot", vmin=0, vmax=max(diff_rgb.max(), 0.01))
    axes[1, 0].set_title(f"Absolute spectral difference\n(mean across 6 bands, reflectance units)\nMAD-mean={mad.mean():.4f}")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.04)

    # HF proxy comparison
    hf_ondisk = laplacian_hf(frame[0])
    hf_refer = laplacian_hf(ref_spectral[0])
    vmax = max(hf_ondisk.max(), hf_refer.max())
    axes[1, 1].imshow(np.hstack([hf_ondisk, hf_refer]), cmap="viridis", vmax=vmax)
    axes[1, 1].set_title(
        f"B02 Laplacian (HF content)\nLEFT: on-disk (mean={hf_on:.5f})  "
        f"RIGHT: reference (mean={hf_ref:.5f})\nRatio {hf_on/max(hf_ref,1e-9):.3f}"
    )
    axes[1, 1].axis("off")

    plt.suptitle(
        f"GSD verification: {args.tile} | date {date_str}\n"
        f"If on-disk = 5m upsampled, RIGHT (true 10m) should be sharper "
        f"and abs-diff map should show spatial misalignment patterns.",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"  Saved {args.out}  ({os.path.getsize(args.out)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
