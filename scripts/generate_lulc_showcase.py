#!/usr/bin/env python3
"""
scripts/generate_lulc_showcase.py — Generate LULC showcase images + chart data

Reads prediction .npz files (from predict_lulc.py) and generates:
  - Per-tile showcase images (NMD label, prediction, confidence, entropy, disagree)
  - Chart data JSON for the dashboard (per-class IoU and accuracy)

Picks a representative tile (highest disagreement + most diverse classes)
and renders it as showcase images for the dashboard Leaflet panels.

Usage:
    # Generate from val split predictions
    python scripts/generate_lulc_showcase.py \\
        --predictions-dir data/predictions/val \\
        --output-dir docs/showcase/lulc \\
        --chart-output docs/data/lulc-data.json

    # Use a specific tile
    python scripts/generate_lulc_showcase.py \\
        --predictions-dir data/predictions/val \\
        --tile tile_0042_pred.npz \\
        --output-dir docs/showcase/lulc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 10-class grouped palette (matches LULC_CLASS_NAMES_10) ────────────────

LULC_10_COLORS = {
    0: (51, 51, 51),        # background
    1: (0, 100, 0),         # forest_pine
    2: (34, 139, 34),       # forest_spruce
    3: (50, 205, 50),       # forest_deciduous
    4: (60, 179, 113),      # forest_mixed
    5: (46, 79, 46),        # forest_wetland
    6: (139, 90, 43),       # open_wetland
    7: (255, 215, 0),       # cropland
    8: (210, 180, 140),     # open_land
    9: (255, 0, 0),         # developed
    10: (0, 0, 255),        # water
}

LULC_10_NAMES_SV = {
    0: "Bakgrund",
    1: "Tallskog",
    2: "Granskog",
    3: "Lövskog",
    4: "Blandskog",
    5: "Sumpskog",
    6: "Öppen våtmark",
    7: "Åkermark",
    8: "Öppen mark",
    9: "Bebyggelse",
    10: "Vatten",
}

LULC_10_HEX = {
    1: "#006400", 2: "#228B22", 3: "#32CD32", 4: "#3CB371",
    5: "#2E4F2E", 6: "#8B5A2B", 7: "#FFD700", 8: "#D2B48C",
    9: "#FF0000", 10: "#0000FF",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.85) -> str:
    """Convert hex color to rgba string."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_class_map(class_map: np.ndarray, output_path: str) -> str:
    """Render a class index map as a color-coded PNG.

    Args:
        class_map: (H, W) uint8 with class indices 0-10.
        output_path: Output PNG path.

    Returns:
        Output file path.
    """
    from PIL import Image

    max_class = max(LULC_10_COLORS.keys())
    palette = np.zeros((max_class + 1, 3), dtype=np.uint8)
    for cls_id, color in LULC_10_COLORS.items():
        palette[cls_id] = color

    clamped = np.clip(class_map, 0, max_class)
    rgb = palette[clamped]
    Image.fromarray(rgb).save(output_path)
    print(f"    saved: {output_path}")
    return output_path


def render_heatmap(
    arr: np.ndarray,
    output_path: str,
    cmap_name: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> str:
    """Render a float array as a color-mapped heatmap PNG.

    Args:
        arr: (H, W) float array.
        output_path: Output PNG path.
        cmap_name: Matplotlib colormap name.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.

    Returns:
        Output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image

    norm = ((arr.astype(np.float32) - vmin) / (vmax - vmin + 1e-10)).clip(0, 1)
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgba).save(output_path)
    print(f"    saved: {output_path}")
    return output_path


def render_disagree_overlay(
    prediction: np.ndarray,
    label: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
    conf_threshold: float = 0.8,
) -> str:
    """Render disagreement overlay: green=correct, red=wrong, magenta=high-conf wrong.

    Args:
        prediction: (H, W) uint8 predicted class indices.
        label: (H, W) uint8 ground truth class indices.
        confidence: (H, W) float confidence scores.
        output_path: Output PNG path.
        conf_threshold: Threshold for high-confidence wrong (default 0.8).

    Returns:
        Output file path.
    """
    from PIL import Image

    h, w = prediction.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    valid = label > 0
    correct = valid & (prediction == label)
    wrong = valid & (prediction != label)
    high_conf_wrong = wrong & (confidence.astype(np.float32) > conf_threshold)
    regular_wrong = wrong & ~high_conf_wrong
    background = ~valid

    # Background: dark gray
    rgb[background] = [40, 40, 40]
    # Correct: green
    rgb[correct] = [46, 204, 64]
    # Wrong: red
    rgb[regular_wrong] = [255, 65, 54]
    # High-confidence wrong: bright magenta (label cleaning candidates)
    rgb[high_conf_wrong] = [255, 0, 255]

    Image.fromarray(rgb).save(output_path)
    print(f"    saved: {output_path}")
    return output_path


def pick_representative_tile(predictions_dir: Path) -> Path | None:
    """Pick the most interesting tile for showcase — highest disagree ratio
    with at least 4 distinct classes present.

    Args:
        predictions_dir: Directory containing *_pred.npz files.

    Returns:
        Path to the selected npz file, or None if no files found.
    """
    npz_files = sorted(predictions_dir.glob("*_pred.npz"))
    if not npz_files:
        return None

    best_file = None
    best_score = -1.0

    for f in npz_files:
        try:
            data = np.load(f)
            label = data["label"]
            disagree = data["disagree"]

            valid = label > 0
            n_valid = valid.sum()
            if n_valid < 100:
                continue

            # Disagreement ratio
            disagree_ratio = disagree.sum() / max(n_valid, 1)

            # Class diversity (number of unique classes present)
            unique_classes = len(np.unique(label[valid]))

            # Score: balance disagreement with diversity
            # We want tiles that show interesting disagreement across multiple classes
            score = disagree_ratio * 0.6 + (unique_classes / 10.0) * 0.4

            if score > best_score:
                best_score = score
                best_file = f
        except Exception:
            continue

    return best_file


def generate_showcase_images(tile_path: Path, output_dir: Path) -> dict:
    """Generate all showcase images from a prediction tile.

    Args:
        tile_path: Path to *_pred.npz file.
        output_dir: Output directory for images.

    Returns:
        Dict with image dimensions (imgH, imgW).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(tile_path)
    prediction = data["prediction"]
    confidence = data["confidence"].astype(np.float32)
    label = data["label"]
    disagree = data["disagree"]
    entropy = data["entropy"].astype(np.float32)

    h, w = prediction.shape
    print(f"\n  Generating showcase images from: {tile_path.name}")
    print(f"  Tile size: {w}x{h}")

    # 1. NMD ground truth (label)
    render_class_map(label, str(output_dir / "nmd_label.png"))

    # 2. Model prediction
    render_class_map(prediction, str(output_dir / "prediction.png"))

    # 3. Confidence heatmap (RdYlGn: red=low, green=high)
    render_heatmap(confidence, str(output_dir / "confidence.png"),
                   cmap_name="RdYlGn", vmin=0.0, vmax=1.0)

    # 4. Entropy heatmap (YlOrRd: yellow=certain, red=uncertain)
    max_entropy = float(entropy.max()) if entropy.max() > 0 else 1.0
    render_heatmap(entropy, str(output_dir / "entropy.png"),
                   cmap_name="YlOrRd", vmin=0.0, vmax=max_entropy)

    # 5. Disagreement overlay
    render_disagree_overlay(prediction, label, confidence,
                            str(output_dir / "disagree.png"))

    # Print tile stats
    valid = label > 0
    n_valid = valid.sum()
    n_disagree = disagree.sum()
    n_high_conf_wrong = ((confidence > 0.8) & disagree).sum()
    print(f"\n  Tile stats:")
    print(f"    Valid pixels:       {n_valid:,}")
    print(f"    Disagree:           {n_disagree:,} ({100*n_disagree/max(n_valid,1):.1f}%)")
    print(f"    High-conf wrong:    {n_high_conf_wrong:,}")
    print(f"    Mean confidence:    {confidence[valid].mean():.3f}")
    print(f"    Mean entropy:       {entropy[valid].mean():.3f}")

    return {"imgH": h, "imgW": w}


def generate_chart_data(predictions_dir: Path, chart_output: Path) -> dict:
    """Generate chart data JSON from prediction_summary.json.

    Args:
        predictions_dir: Directory containing prediction_summary.json.
        chart_output: Output path for lulc-data.json.

    Returns:
        The chart data dict.
    """
    summary_path = predictions_dir / "prediction_summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, generating placeholder data")
        return _placeholder_chart_data(chart_output)

    with open(summary_path) as f:
        summary = json.load(f)

    # Build per-class arrays
    labels = []
    iou_values = []
    acc_values = []
    colors = []

    per_class = summary.get("per_class", {})
    # Map English class names to Swedish + colors
    _EN_TO_SV = {
        "forest_pine": ("Tallskog", "#006400"),
        "forest_spruce": ("Granskog", "#228B22"),
        "forest_deciduous": ("Lövskog", "#32CD32"),
        "forest_mixed": ("Blandskog", "#3CB371"),
        "forest_wetland": ("Sumpskog", "#2E4F2E"),
        "open_wetland": ("Öppen våtmark", "#8B5A2B"),
        "cropland": ("Åkermark", "#FFD700"),
        "open_land": ("Öppen mark", "#D2B48C"),
        "developed": ("Bebyggelse", "#FF0000"),
        "water": ("Vatten", "#0000FF"),
    }

    for en_name, (sv_name, hex_color) in _EN_TO_SV.items():
        cls_data = per_class.get(en_name, {})
        labels.append(sv_name)
        acc_values.append(cls_data.get("accuracy_pct", 0))
        colors.append(_hex_to_rgba(hex_color))

        # IoU is not in prediction_summary.json, use accuracy as proxy
        # (real IoU comes from evaluate.py — we include it if available)
        iou_values.append(cls_data.get("iou_pct", cls_data.get("accuracy_pct", 0)))

    chart_data = {
        "summary": {
            "miou": summary.get("overall_agreement_pct", 0),
            "overall_accuracy": summary.get("overall_agreement_pct", 0),
            "tiles": summary.get("tiles", 0),
            "high_confidence_wrong": summary.get("high_confidence_wrong", 0),
            "low_confidence_correct": summary.get("low_confidence_correct", 0),
            "total_pixels": summary.get("total_pixels", 0),
            "disagree_pixels": summary.get("disagree_pixels", 0),
        },
        "per_class_iou": {
            "labels": labels,
            "values": iou_values,
            "colors": colors,
        },
        "per_class_accuracy": {
            "labels": labels,
            "values": acc_values,
            "colors": colors,
        },
    }

    chart_output.parent.mkdir(parents=True, exist_ok=True)
    with open(chart_output, "w") as f:
        json.dump(chart_data, f, indent=2)
    print(f"\n  Chart data saved: {chart_output}")

    return chart_data


def _placeholder_chart_data(chart_output: Path) -> dict:
    """Generate placeholder chart data when no predictions exist yet."""
    labels = ["Tallskog", "Granskog", "Lövskog", "Blandskog", "Sumpskog",
              "Öppen våtmark", "Åkermark", "Öppen mark", "Bebyggelse", "Vatten"]
    hex_colors = ["#006400", "#228B22", "#32CD32", "#3CB371", "#2E4F2E",
                  "#8B5A2B", "#FFD700", "#D2B48C", "#FF0000", "#0000FF"]
    colors = [_hex_to_rgba(h) for h in hex_colors]

    # Placeholder values from training val metrics (43.27% mIoU)
    iou_placeholder = [55.2, 48.1, 9.9, 24.3, 33.5, 41.2, 62.8, 35.7, 58.4, 63.6]
    acc_placeholder = [72.1, 65.3, 18.4, 38.9, 52.1, 58.7, 78.3, 51.2, 74.6, 82.1]

    chart_data = {
        "summary": {
            "miou": 43.27,
            "overall_accuracy": 0.0,
            "tiles": 0,
            "high_confidence_wrong": 0,
            "low_confidence_correct": 0,
            "total_pixels": 0,
            "disagree_pixels": 0,
        },
        "per_class_iou": {
            "labels": labels,
            "values": iou_placeholder,
            "colors": colors,
        },
        "per_class_accuracy": {
            "labels": labels,
            "values": acc_placeholder,
            "colors": colors,
        },
    }

    chart_output.parent.mkdir(parents=True, exist_ok=True)
    with open(chart_output, "w") as f:
        json.dump(chart_data, f, indent=2)
    print(f"\n  Placeholder chart data saved: {chart_output}")

    return chart_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate LULC showcase images + chart data",
    )
    parser.add_argument("--predictions-dir", type=str, required=True,
                        help="Directory with *_pred.npz files and prediction_summary.json")
    parser.add_argument("--output-dir", type=str, default="docs/showcase/lulc",
                        help="Output directory for showcase images")
    parser.add_argument("--chart-output", type=str, default="docs/data/lulc-data.json",
                        help="Output path for chart data JSON")
    parser.add_argument("--tile", type=str, default=None,
                        help="Specific tile .npz file to use (default: auto-pick)")
    parser.add_argument("--placeholder-only", action="store_true",
                        help="Only generate placeholder chart data, no images")

    args = parser.parse_args()
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    chart_output = Path(args.chart_output)

    print(f"\n{'='*60}")
    print(f"  LULC Showcase Generator")
    print(f"  Predictions: {predictions_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Chart data:  {chart_output}")
    print(f"{'='*60}")

    if args.placeholder_only:
        _placeholder_chart_data(chart_output)
        print("\n  Done (placeholder only).\n")
        return

    # Generate chart data from prediction summary
    generate_chart_data(predictions_dir, chart_output)

    # Pick or use specified tile
    if args.tile:
        tile_path = predictions_dir / args.tile
        if not tile_path.exists():
            print(f"  ERROR: Tile not found: {tile_path}")
            sys.exit(1)
    else:
        tile_path = pick_representative_tile(predictions_dir)
        if tile_path is None:
            print(f"  WARNING: No prediction tiles found in {predictions_dir}")
            print(f"           Run `make predict-aux` first to generate predictions.")
            print(f"           Generated placeholder chart data only.\n")
            return

    print(f"\n  Selected tile: {tile_path.name}")

    # Generate showcase images
    dims = generate_showcase_images(tile_path, output_dir)
    print(f"\n  Image dimensions: {dims['imgW']}x{dims['imgH']}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
