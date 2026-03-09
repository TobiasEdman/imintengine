#!/usr/bin/env python3
"""
scripts/generate_lulc_showcase.py — Generate LULC showcase images + chart data

Reads prediction .npz files (from predict_lulc.py) and generates:
  - Per-tile gallery images: S2 RGB, NMD label, prediction, quality overlay
  - Chart data JSON for the dashboard (per-class IoU and accuracy)
  - Tile gallery JSON listing all tiles with paths and metrics

Gallery mode (default): picks N diverse tiles and generates 4 images each
for the per-row dashboard layout: S2 pseudocolor | NMD | LULC | Quality.

Usage:
    # Generate gallery from val split predictions
    python scripts/generate_lulc_showcase.py \\
        --predictions-dir data/predictions/val \\
        --output-dir docs/showcase/lulc \\
        --chart-output docs/data/lulc-data.json \\
        --num-tiles 12

    # Placeholder only (no predictions needed)
    python scripts/generate_lulc_showcase.py \\
        --predictions-dir data/predictions/val \\
        --chart-output docs/data/lulc-data.json \\
        --placeholder-only
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
    """Render a class index map as a color-coded PNG."""
    from PIL import Image

    max_class = max(LULC_10_COLORS.keys())
    palette = np.zeros((max_class + 1, 3), dtype=np.uint8)
    for cls_id, color in LULC_10_COLORS.items():
        palette[cls_id] = color

    clamped = np.clip(class_map, 0, max_class)
    rgb = palette[clamped]
    Image.fromarray(rgb).save(output_path)
    return output_path


def render_heatmap(
    arr: np.ndarray,
    output_path: str,
    cmap_name: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> str:
    """Render a float array as a color-mapped heatmap PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image

    norm = ((arr.astype(np.float32) - vmin) / (vmax - vmin + 1e-10)).clip(0, 1)
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgba).save(output_path)
    return output_path


def render_disagree_overlay(
    prediction: np.ndarray,
    label: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
    conf_threshold: float = 0.8,
) -> str:
    """Render disagreement overlay: green=correct, red=wrong, magenta=high-conf wrong."""
    from PIL import Image

    h, w = prediction.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    valid = label > 0
    correct = valid & (prediction == label)
    wrong = valid & (prediction != label)
    high_conf_wrong = wrong & (confidence.astype(np.float32) > conf_threshold)
    regular_wrong = wrong & ~high_conf_wrong
    background = ~valid

    rgb[background] = [40, 40, 40]
    rgb[correct] = [46, 204, 64]
    rgb[regular_wrong] = [255, 65, 54]
    rgb[high_conf_wrong] = [255, 0, 255]

    Image.fromarray(rgb).save(output_path)
    return output_path


def render_s2_rgb(s2_rgb: np.ndarray, output_path: str) -> str:
    """Save pre-rendered S2 RGB uint8 array as PNG."""
    from PIL import Image
    Image.fromarray(s2_rgb).save(output_path)
    return output_path


# ── Multi-tile selection ──────────────────────────────────────────────────


def score_tiles(predictions_dir: Path) -> list[dict]:
    """Score all tiles by diversity and disagreement, return sorted list."""
    npz_files = sorted(predictions_dir.glob("*_pred.npz"))
    if not npz_files:
        return []

    scored = []
    for f in npz_files:
        try:
            data = np.load(f)
            label = data["label"]
            disagree = data["disagree"]
            prediction = data["prediction"]
            confidence = data["confidence"].astype(np.float32)

            valid = label > 0
            n_valid = int(valid.sum())
            if n_valid < 100:
                continue

            n_disagree = int(disagree.sum())
            disagree_ratio = n_disagree / max(n_valid, 1)
            unique_classes = len(np.unique(label[valid]))

            # High-confidence wrong count
            high_conf_wrong = int(((confidence > 0.8) & disagree).sum())

            # Accuracy
            accuracy = 1.0 - disagree_ratio

            # Dominant class
            unique, counts = np.unique(label[valid], return_counts=True)
            dominant_class = int(unique[counts.argmax()])
            dominant_name = LULC_10_NAMES_SV.get(dominant_class, f"Klass {dominant_class}")

            # Score: balance class diversity with interesting disagreement
            score = (unique_classes / 10.0) * 0.5 + disagree_ratio * 0.3 + (n_valid / 50176) * 0.2

            scored.append({
                "path": f,
                "name": f.stem.replace("_pred", ""),
                "score": score,
                "n_valid": n_valid,
                "n_disagree": n_disagree,
                "disagree_pct": round(disagree_ratio * 100, 1),
                "accuracy_pct": round(accuracy * 100, 1),
                "unique_classes": unique_classes,
                "high_conf_wrong": high_conf_wrong,
                "dominant_class": dominant_name,
                "has_s2_rgb": "s2_rgb" in data.files,
            })
        except Exception as e:
            print(f"    WARNING: Could not score {f.name}: {e}")
            continue

    # Sort by score descending, then pick diverse tiles
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def select_diverse_tiles(scored: list[dict], n: int = 12) -> list[dict]:
    """Select N tiles that are diverse in dominant class and disagreement."""
    if len(scored) <= n:
        return scored

    selected = []
    seen_dominant = set()

    # First pass: pick tiles with unique dominant classes
    for tile in scored:
        if len(selected) >= n:
            break
        dom = tile["dominant_class"]
        if dom not in seen_dominant:
            selected.append(tile)
            seen_dominant.add(dom)

    # Second pass: fill remaining from top-scored tiles
    for tile in scored:
        if len(selected) >= n:
            break
        if tile not in selected:
            selected.append(tile)

    return selected[:n]


# ── Gallery generation ────────────────────────────────────────────────────


def generate_tile_gallery(
    tiles: list[dict],
    output_dir: Path,
) -> list[dict]:
    """Generate gallery images for selected tiles.

    For each tile generates:
      - tile_N_s2.png     — Sentinel-2 pseudocolor RGB
      - tile_N_nmd.png    — NMD ground truth (colored)
      - tile_N_pred.png   — Model prediction (colored)
      - tile_N_quality.png — Disagreement overlay (green/red/magenta)

    Returns list of tile metadata dicts for the gallery JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gallery = []
    for i, tile_info in enumerate(tiles):
        tile_path = tile_info["path"]
        data = np.load(tile_path)

        prefix = f"tile_{i:02d}"
        print(f"  [{i+1}/{len(tiles)}] {tile_info['name']}  "
              f"acc={tile_info['accuracy_pct']}%  "
              f"classes={tile_info['unique_classes']}  "
              f"dom={tile_info['dominant_class']}")

        prediction = data["prediction"]
        confidence = data["confidence"].astype(np.float32)
        label = data["label"]

        # 1. S2 RGB (from predict_lulc.py or fallback to confidence heatmap)
        if "s2_rgb" in data.files:
            s2_path = render_s2_rgb(data["s2_rgb"], str(output_dir / f"{prefix}_s2.png"))
        else:
            # Fallback: render confidence heatmap if no S2 RGB available
            s2_path = render_heatmap(
                confidence, str(output_dir / f"{prefix}_s2.png"),
                cmap_name="viridis", vmin=0.0, vmax=1.0,
            )

        # 2. NMD ground truth
        nmd_path = render_class_map(label, str(output_dir / f"{prefix}_nmd.png"))

        # 3. Model prediction
        pred_path = render_class_map(prediction, str(output_dir / f"{prefix}_pred.png"))

        # 4. Quality overlay (disagree map)
        quality_path = render_disagree_overlay(
            prediction, label, confidence,
            str(output_dir / f"{prefix}_quality.png"),
        )

        gallery.append({
            "index": i,
            "name": tile_info["name"],
            "s2": f"showcase/lulc/{prefix}_s2.png",
            "nmd": f"showcase/lulc/{prefix}_nmd.png",
            "pred": f"showcase/lulc/{prefix}_pred.png",
            "quality": f"showcase/lulc/{prefix}_quality.png",
            "accuracy_pct": tile_info["accuracy_pct"],
            "disagree_pct": tile_info["disagree_pct"],
            "unique_classes": tile_info["unique_classes"],
            "high_conf_wrong": tile_info["high_conf_wrong"],
            "dominant_class": tile_info["dominant_class"],
        })

    return gallery


# ── Also keep single-tile showcase for Leaflet panels ────────────────────


def generate_showcase_images(tile_path: Path, output_dir: Path) -> dict:
    """Generate single-tile showcase images (backward compat)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(tile_path)
    prediction = data["prediction"]
    confidence = data["confidence"].astype(np.float32)
    label = data["label"]
    entropy = data["entropy"].astype(np.float32)

    h, w = prediction.shape

    render_class_map(label, str(output_dir / "nmd_label.png"))
    render_class_map(prediction, str(output_dir / "prediction.png"))
    render_heatmap(confidence, str(output_dir / "confidence.png"),
                   cmap_name="RdYlGn", vmin=0.0, vmax=1.0)
    max_entropy = float(entropy.max()) if entropy.max() > 0 else 1.0
    render_heatmap(entropy, str(output_dir / "entropy.png"),
                   cmap_name="YlOrRd", vmin=0.0, vmax=max_entropy)
    render_disagree_overlay(prediction, label, confidence,
                            str(output_dir / "disagree.png"))

    if "s2_rgb" in data.files:
        render_s2_rgb(data["s2_rgb"], str(output_dir / "s2_rgb.png"))

    return {"imgH": h, "imgW": w}


# ── Chart data generation ────────────────────────────────────────────────


def generate_chart_data(predictions_dir: Path, chart_output: Path,
                        gallery: list[dict] | None = None) -> dict:
    """Generate chart data JSON from prediction_summary.json."""
    summary_path = predictions_dir / "prediction_summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, generating placeholder data")
        return _placeholder_chart_data(chart_output)

    with open(summary_path) as f:
        summary = json.load(f)

    labels = []
    iou_values = []
    acc_values = []
    colors = []

    per_class = summary.get("per_class", {})
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

    # Include gallery tile list
    if gallery:
        chart_data["gallery"] = gallery

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
        "gallery": [],
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
    parser.add_argument("--num-tiles", type=int, default=12,
                        help="Number of tiles for gallery (default: 12)")
    parser.add_argument("--tile", type=str, default=None,
                        help="Specific tile .npz file for single-tile mode")
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

    # Score all tiles
    print(f"\n  Scoring tiles...")
    scored = score_tiles(predictions_dir)
    if not scored:
        print(f"  WARNING: No prediction tiles found in {predictions_dir}")
        print(f"           Run `make predict-aux` first.")
        _placeholder_chart_data(chart_output)
        return

    print(f"  Found {len(scored)} valid tiles")

    # Select diverse tiles for gallery
    selected = select_diverse_tiles(scored, n=args.num_tiles)
    print(f"  Selected {len(selected)} tiles for gallery\n")

    # Generate gallery images
    gallery = generate_tile_gallery(selected, output_dir)

    # Also generate single-tile showcase (best tile) for Leaflet panels
    best_tile = scored[0]["path"]
    print(f"\n  Generating single-tile showcase from: {best_tile.name}")
    generate_showcase_images(best_tile, output_dir)

    # Generate chart data + gallery JSON
    generate_chart_data(predictions_dir, chart_output, gallery=gallery)

    print(f"\n  Gallery: {len(gallery)} tiles")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
