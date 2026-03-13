#!/usr/bin/env python3
"""
scripts/predict_lulc.py — Run inference on all tiles and save predictions

Loads a trained checkpoint, runs inference on train/val/test tiles,
and saves per-tile prediction maps.  Useful for:
  - Visualising model predictions vs ground truth
  - Label cleaning (find high-disagreement pixels)
  - Confidence analysis (softmax entropy maps)
  - Generating full-coverage LULC maps

Output per tile (saved as .npz):
  - prediction:  (H, W) uint8  predicted class indices
  - confidence:  (H, W) float16  max softmax probability
  - label:       (H, W) uint8  ground truth (NMD)
  - disagree:    (H, W) bool   prediction != label (ignoring background)
  - entropy:     (H, W) float16  softmax entropy (high = uncertain)
  - easting:     int    tile center X (EPSG:3006)
  - northing:    int    tile center Y (EPSG:3006)

Usage:
    # Predict all splits using best AUX model
    python scripts/predict_lulc.py \\
        --checkpoint checkpoints/lulc_aux/best_model.pt \\
        --checkpoint-dir checkpoints/lulc_aux \\
        --enable-height --enable-volume --enable-basal-area \\
        --enable-diameter --enable-dem

    # Predict only val split
    python scripts/predict_lulc.py \\
        --checkpoint checkpoints/lulc_aux/best_model.pt \\
        --splits val \\
        --enable-height --enable-volume --enable-basal-area \\
        --enable-diameter --enable-dem

    # Quick summary (no save, just print metrics + disagreement stats)
    python scripts/predict_lulc.py \\
        --checkpoint checkpoints/lulc_aux/best_model.pt \\
        --splits val --summary-only \\
        --enable-height --enable-volume --enable-basal-area \\
        --enable-diameter --enable-dem
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.config.env import load_env
load_env()

from imint.training.config import TrainingConfig
from imint.training.dataset import LULCDataset
from imint.training.class_schema import get_class_names


def load_model(checkpoint_path: str, config: TrainingConfig, device):
    """Load trained model from checkpoint."""
    import torch
    from imint.fm.terratorch_loader import _load_prithvi_from_hf
    from imint.fm.upernet import PrithviSegmentationModel, get_default_pool_sizes

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get aux channel count from checkpoint config
    ckpt_config = ckpt.get("config", {})
    n_aux = ckpt_config.get("n_aux_channels", 0)

    print(f"  Loading Prithvi backbone...")
    backbone = _load_prithvi_from_hf(pretrained=True)

    model = PrithviSegmentationModel(
        encoder=backbone,
        feature_indices=config.feature_indices,
        decoder_channels=ckpt_config.get("decoder_channels", config.decoder_channels),
        num_classes=ckpt_config.get("num_classes", config.num_classes + 1),
        dropout=ckpt_config.get("dropout", config.dropout),
        n_aux_channels=n_aux,
        pool_sizes=get_default_pool_sizes(device),
    )

    # Load state dict (strip "model." prefix)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_sd = {}
    for k, v in state_dict.items():
        key = k[len("model."):] if k.startswith("model.") else k
        clean_sd[key] = v
    model.load_state_dict(clean_sd, strict=False)

    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    miou = metrics.get("miou", "?")
    print(f"  Loaded checkpoint: epoch {epoch}, mIoU {miou}")
    print(f"  Aux channels: {n_aux}")

    return model, n_aux


def predict_split(
    model,
    dataset: LULCDataset,
    config: TrainingConfig,
    device,
    output_dir: Path,
    n_aux: int,
    summary_only: bool = False,
):
    """Run inference on a dataset split, save per-tile predictions."""
    import torch

    _AUX_NAMES = (
        "height", "volume", "basal_area", "diameter", "dem",
        "vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = config.num_classes + 1
    class_names = get_class_names(config.num_classes)

    # Accumulators for summary stats
    total_pixels = 0
    disagree_pixels = 0
    per_class_disagree = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)       # pixels with label == c (for producer acc)
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_predicted = np.zeros(num_classes, dtype=np.int64)   # pixels with pred == c (for user acc)
    per_class_pred_correct = np.zeros(num_classes, dtype=np.int64)  # pred == c AND label == c
    high_conf_wrong = 0  # confident prediction != label (label cleaning candidates)
    low_conf_correct = 0  # uncertain prediction == label

    # AUC-ROC histograms: bin softmax probabilities for each class
    N_AUC_BINS = 1000
    auc_hist_pos = np.zeros((num_classes, N_AUC_BINS), dtype=np.int64)  # p(c) where label==c
    auc_hist_neg = np.zeros((num_classes, N_AUC_BINS), dtype=np.int64)  # p(c) where label!=c

    t0 = time.time()

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        label = sample["label"].numpy()
        metadata = sample.get("metadata", {})
        tile_name = metadata.get("tile", f"tile_{i:04d}.npz")
        easting = metadata.get("easting", 0)
        northing = metadata.get("northing", 0)

        # Add temporal dimension: (1, 6, H, W) → (1, 6, 1, H, W)
        image_5d = image.unsqueeze(2)

        # Collect auxiliary channels
        aux_parts = []
        for name in _AUX_NAMES:
            if name in sample:
                aux_parts.append(sample[name].unsqueeze(0).to(device))
        aux = torch.cat(aux_parts, dim=1) if aux_parts else None

        # Extract S2 NIR false-color for showcase: B8(NIR), B3(Green), B4(Red)
        # Band order in dataset: B02(0), B03(1), B04(2), B8A(3), B11(4), B12(5)
        # Visualization: R=B8A(idx3), G=B03(idx1), B=B04(idx2)
        # Stretch: min=400 DN, max=[4000, 1500, 1500] DN
        s2_raw = image.squeeze(0).cpu().numpy()  # (6, H, W)
        # Denormalize all 6 bands back to DN scale: dn = norm * std + mean
        _MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
        _STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])
        for b in range(s2_raw.shape[0]):
            s2_raw[b] = s2_raw[b] * _STD[b] + _MEAN[b]
        # NIR false-color: R=B8A, G=B03, B=B04
        _MIN = 400.0
        _MAX = np.array([4000.0, 1500.0, 1500.0])
        nir_fc = np.stack([s2_raw[3], s2_raw[1], s2_raw[2]], axis=-1)  # (H,W,3)
        s2_rgb = np.clip((nir_fc - _MIN) / (_MAX - _MIN) * 255, 0, 255).astype(np.uint8)

        with torch.no_grad():
            logits = model(image_5d, aux=aux).contiguous()

        # Softmax probabilities
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
        pred = probs.argmax(axis=0).astype(np.uint8)  # (H, W)
        confidence = probs.max(axis=0).astype(np.float16)  # (H, W)

        # Entropy: -sum(p * log(p))
        log_probs = np.log(np.clip(probs, 1e-8, 1.0))
        entropy = -(probs * log_probs).sum(axis=0).astype(np.float16)  # (H, W)

        # Disagreement mask (ignore background class 0)
        valid = label > 0
        disagree = np.zeros_like(pred, dtype=bool)
        disagree[valid] = pred[valid] != label[valid]

        # Accumulate stats
        n_valid = valid.sum()
        n_disagree = disagree.sum()
        total_pixels += n_valid
        disagree_pixels += n_disagree

        for c in range(num_classes):
            if c == 0:
                continue
            # Producer accuracy (recall): of pixels labeled c, how many predicted c?
            mask_label_c = label == c
            n_label = mask_label_c.sum()
            per_class_total[c] += n_label
            per_class_correct[c] += (pred[mask_label_c] == c).sum()
            per_class_disagree[c] += (pred[mask_label_c] != c).sum()
            # User accuracy (precision): of pixels predicted c, how many labeled c?
            mask_pred_c = (pred == c) & valid
            per_class_predicted[c] += mask_pred_c.sum()
            per_class_pred_correct[c] += (label[mask_pred_c] == c).sum()

        # AUC-ROC: bin per-class softmax probabilities
        for c in range(1, num_classes):
            p_c = probs[c].astype(np.float32)  # (H, W) probability of class c
            bins_c = np.clip((p_c * N_AUC_BINS).astype(np.int32), 0, N_AUC_BINS - 1)
            mask_is_c = (label == c) & valid
            mask_not_c = (label != c) & valid
            np.add.at(auc_hist_pos[c], bins_c[mask_is_c], 1)
            np.add.at(auc_hist_neg[c], bins_c[mask_not_c], 1)

        # High-confidence wrong: model confident (>0.8) but disagrees with label
        high_conf_mask = confidence.astype(np.float32) > 0.8
        high_conf_wrong += (high_conf_mask & disagree).sum()

        # Low-confidence correct: model uncertain (<0.5) but agrees
        low_conf_mask = confidence.astype(np.float32) < 0.5
        low_conf_correct += (low_conf_mask & ~disagree & valid).sum()

        # Save per-tile output
        if not summary_only:
            out_name = Path(tile_name).stem + "_pred.npz"
            np.savez_compressed(
                output_dir / out_name,
                prediction=pred,
                confidence=confidence,
                label=label.astype(np.uint8),
                disagree=disagree,
                entropy=entropy,
                s2_rgb=s2_rgb,
                easting=easting,
                northing=northing,
            )

        # Progress
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        if (i + 1) % 50 == 0 or i == len(dataset) - 1:
            pct = 100 * disagree_pixels / max(total_pixels, 1)
            print(f"  [{i+1:4d}/{len(dataset)}]  "
                  f"{rate:.1f} tiles/s  "
                  f"disagree: {pct:.1f}%  "
                  f"tile: {tile_name}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    agree_pct = 100 * (1 - disagree_pixels / max(total_pixels, 1))

    print(f"\n  {'='*60}")
    print(f"  Prediction Summary")
    print(f"  {'='*60}")
    print(f"  Tiles:           {len(dataset)}")
    print(f"  Time:            {elapsed:.0f}s ({len(dataset)/elapsed:.1f} tiles/s)")
    print(f"  Overall agree:   {agree_pct:.1f}%")
    print(f"  Total pixels:    {total_pixels:,} (excl. background)")
    print(f"  Disagree pixels: {disagree_pixels:,} ({100*disagree_pixels/max(total_pixels,1):.1f}%)")
    print(f"  High-conf wrong: {high_conf_wrong:,} (model >80% confident, wrong)")
    print(f"  Low-conf right:  {low_conf_correct:,} (model <50% confident, correct)")
    print()

    # Per-class breakdown
    print(f"  {'Class':<20s} {'Labeled':>9s} {'Pred':>9s} {'Prod.Acc':>9s} {'User.Acc':>9s} {'Label%':>7s}")
    print(f"  {'-'*70}")
    for c in range(1, num_classes):
        name = class_names.get(c, f"class_{c}")
        total_c = per_class_total[c]
        pred_c = per_class_predicted[c]
        producer_acc = 100 * per_class_correct[c] / max(total_c, 1)
        user_acc = 100 * per_class_pred_correct[c] / max(pred_c, 1)
        label_pct = 100 * total_c / max(total_pixels, 1)
        print(f"  {name:<20s} {total_c:>9,d} {pred_c:>9,d} {producer_acc:>8.1f}% {user_acc:>8.1f}% {label_pct:>6.1f}%")

    # AUC-ROC per class (from binned probability histograms)
    def _auc_from_hist(hist_pos, hist_neg, n_points=100):
        """Compute AUC-ROC and downsampled ROC curve from histograms."""
        total_pos = hist_pos.sum()
        total_neg = hist_neg.sum()
        if total_pos == 0 or total_neg == 0:
            return 0.0, [], []
        # Sweep from high to low threshold → accumulate TP, FP
        cum_tp, cum_fp = 0, 0
        fpr_list, tpr_list = [0.0], [0.0]
        for b in range(len(hist_pos) - 1, -1, -1):
            cum_tp += hist_pos[b]
            cum_fp += hist_neg[b]
            fpr_list.append(cum_fp / total_neg)
            tpr_list.append(cum_tp / total_pos)
        # Trapezoidal AUC
        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)
        auc = float(np.trapz(tpr_arr, fpr_arr))
        # Downsample to n_points for JSON/chart
        indices = np.linspace(0, len(fpr_arr) - 1, n_points, dtype=int)
        fpr_ds = [round(float(fpr_arr[i]), 4) for i in indices]
        tpr_ds = [round(float(tpr_arr[i]), 4) for i in indices]
        return round(auc, 4), fpr_ds, tpr_ds

    print()
    print(f"  {'Class':<20s} {'AUC-ROC':>9s}")
    print(f"  {'-'*30}")
    auc_results = {}
    for c in range(1, num_classes):
        name = class_names.get(c, f"class_{c}")
        auc_val, fpr_ds, tpr_ds = _auc_from_hist(auc_hist_pos[c], auc_hist_neg[c])
        auc_results[name] = {"auc": auc_val, "fpr": fpr_ds, "tpr": tpr_ds}
        print(f"  {name:<20s} {auc_val:>8.4f}")

    # Mean AUC (macro)
    mean_auc = np.mean([v["auc"] for v in auc_results.values()])
    print(f"  {'Mean AUC':<20s} {mean_auc:>8.4f}")

    print(f"  {'='*60}")

    # Save summary JSON
    summary = {
        "tiles": len(dataset),
        "total_pixels": int(total_pixels),
        "disagree_pixels": int(disagree_pixels),
        "overall_agreement_pct": round(agree_pct, 2),
        "high_confidence_wrong": int(high_conf_wrong),
        "low_confidence_correct": int(low_conf_correct),
        "mean_auc": round(float(mean_auc), 4),
        "per_class": {},
        "auc_roc": auc_results,
    }
    for c in range(1, num_classes):
        name = class_names.get(c, f"class_{c}")
        total_c = int(per_class_total[c])
        pred_c = int(per_class_predicted[c])
        correct_c = int(per_class_correct[c])
        pred_correct_c = int(per_class_pred_correct[c])
        summary["per_class"][name] = {
            "total_pixels": total_c,
            "predicted_pixels": pred_c,
            "correct": correct_c,
            "wrong": int(per_class_disagree[c]),
            "producer_accuracy_pct": round(100 * correct_c / max(total_c, 1), 2),
            "user_accuracy_pct": round(100 * pred_correct_c / max(pred_c, 1), 2),
            "auc": auc_results.get(name, {}).get("auc", 0),
            # Keep legacy field for backward compat
            "accuracy_pct": round(100 * correct_c / max(total_c, 1), 2),
        }

    summary_path = output_dir / "prediction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run inference and save per-tile predictions",
    )

    _defaults = TrainingConfig()

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, default=_defaults.data_dir)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data_dir/predictions/)")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"],
                        help="Which splits to predict (default: all)")
    parser.add_argument("--device", type=str, default=_defaults.device)
    parser.add_argument("--num-classes", type=int, default=_defaults.num_classes)
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print summary stats, don't save per-tile predictions")

    # Auxiliary channels (must match training config)
    parser.add_argument("--enable-height", action="store_true")
    parser.add_argument("--enable-volume", action="store_true")
    parser.add_argument("--enable-basal-area", action="store_true")
    parser.add_argument("--enable-diameter", action="store_true")
    parser.add_argument("--enable-dem", action="store_true")
    parser.add_argument("--enable-vpp", action="store_true")
    parser.add_argument("--enable-all-aux", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=_defaults.checkpoint_dir)

    args = parser.parse_args()

    import torch

    # Build config
    all_aux = args.enable_all_aux
    config = TrainingConfig(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        use_grouped_classes=(args.num_classes < 19),
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        enable_height_channel=all_aux or args.enable_height,
        enable_volume_channel=all_aux or args.enable_volume,
        enable_basal_area_channel=all_aux or args.enable_basal_area,
        enable_diameter_channel=all_aux or args.enable_diameter,
        enable_dem_channel=all_aux or args.enable_dem,
        enable_vpp_channels=all_aux or args.enable_vpp,
    )

    # Resolve device
    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # Load model
    model, n_aux = load_model(args.checkpoint, config, device)

    # Output directory
    output_base = Path(args.output_dir) if args.output_dir else Path(config.data_dir) / "predictions"

    print(f"\n{'='*60}")
    print(f"  LULC Prediction Pipeline")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data: {config.data_dir}")
    print(f"  Output: {output_base}")
    print(f"  Splits: {', '.join(args.splits)}")
    print(f"  Aux channels: {n_aux}")
    print(f"  Summary only: {args.summary_only}")
    print(f"{'='*60}\n")

    for split in args.splits:
        try:
            dataset = LULCDataset(config.data_dir, split=split, config=config)
        except FileNotFoundError:
            print(f"  Split '{split}' not found, skipping.\n")
            continue

        print(f"  === {split.upper()} ({len(dataset)} tiles) ===\n")

        split_dir = output_base / split
        predict_split(
            model=model,
            dataset=dataset,
            config=config,
            device=device,
            output_dir=split_dir,
            n_aux=n_aux,
            summary_only=args.summary_only,
        )
        print()

    print("  Done.\n")


if __name__ == "__main__":
    main()
