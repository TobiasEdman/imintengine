"""
imint/training/evaluate.py — Evaluation metrics for LULC segmentation

Computes per-class IoU, mean IoU, overall accuracy, and confusion matrix.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

from imint.schema.class_schema import get_class_names
from imint.schema.unified_schema import UNIFIED_CLASSES


from imint.metrics import compute_miou


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    config,
    device: torch.device,
    max_samples: int | None = None,
) -> dict:
    """Evaluate a segmentation model on a dataset.

    Args:
        model: Prithvi segmentation model in eval mode.
        dataset: LULCDataset instance.
        config: TrainingConfig.
        device: Torch device.
        max_samples: Limit number of samples (for quick checks).

    Returns:
        Dict with mIoU, per-class IoU, overall accuracy.
    """
    model.eval()
    all_preds = []
    all_targets = []

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    # Use canonical aux channel names from config
    aux_names = config.enabled_aux_names if hasattr(config, 'enabled_aux_names') else ()

    # Route the model input on backbone family (never on channel count):
    # tessera consumes the 4D (1, 128, H, W) embedding as-is; Prithvi needs
    # the 5D Conv3d layout. fm_spec is stashed on the model at build time.
    family = getattr(getattr(model, "fm_spec", None), "family", "prithvi")

    from imint.fm.forward_router import family_forward

    for i in range(n_samples):
        sample = dataset[i]
        label = sample["label"].numpy()                    # (H, W)

        # Add a batch dim of 1 to every tensor value so family_forward (which
        # expects batched tensors) can consume the sample directly. Non-tensor
        # values (e.g. metadata dict) are passed through untouched.
        batch = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in sample.items()
        }

        # Collect auxiliary channels if present
        aux_parts = []
        for name in aux_names:
            if name in sample:
                aux_parts.append(sample[name].unsqueeze(0).to(device))
        aux = torch.cat(aux_parts, dim=1) if aux_parts else None

        # Prithvi TL coordinate tensors
        temporal_coords = sample.get("temporal_coords")
        if temporal_coords is not None:
            temporal_coords = temporal_coords.unsqueeze(0).to(device)
        location_coords = sample.get("location_coords")
        if location_coords is not None:
            location_coords = location_coords.unsqueeze(0).to(device)

        logits = family_forward(
            model, family, batch, device,
            aux=aux,
            temporal_coords=temporal_coords,
            location_coords=location_coords,
        ).contiguous()  # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        all_preds.append(pred)
        all_targets.append(label)

    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])

    return compute_miou(
        all_preds, all_targets,
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
    )
