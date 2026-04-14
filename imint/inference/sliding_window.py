"""
imint/inference/sliding_window.py — Sliding window inference with overlap averaging

Runs overlapping patches across a full-resolution tile, averages softmax
predictions in overlap zones, and returns crisp per-pixel class probabilities.

Without overlap, each pixel is predicted by a single patch and boundary
artifacts appear at patch edges.  With 50% overlap each interior pixel
is covered by 4 patches, producing smoother and sharper boundaries.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    image_5d: torch.Tensor,
    aux: torch.Tensor | None = None,
    patch_size: int = 224,
    overlap: float = 0.5,
    temporal_coords: torch.Tensor | None = None,
    location_coords: torch.Tensor | None = None,
    num_classes: int = 23,
) -> torch.Tensor:
    """Averaged softmax predictions over overlapping windows.

    Args:
        model: Segmentation model returning (B, C, H, W) logits.
        image_5d: (1, C, T, H, W) spectral input.
        aux: Optional (1, N, H, W) auxiliary channels.
        patch_size: Window size in pixels.
        overlap: Overlap ratio in [0, 1). 0.5 = 50% overlap.
        temporal_coords: Optional (1, T, 2) [year, doy].
        location_coords: Optional (1, 2) [lat, lon].
        num_classes: Number of output classes.

    Returns:
        (1, num_classes, H, W) softmax probabilities at input resolution.
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    device = image_5d.device
    _, C, T, H, W = image_5d.shape
    stride = max(1, int(patch_size * (1 - overlap)))

    # Pad so that the grid of windows covers the entire image.
    # We need: H_pad >= patch_size and (H_pad - patch_size) % stride == 0
    if H <= patch_size:
        pad_h = patch_size - H
    else:
        remainder = (H - patch_size) % stride
        pad_h = (stride - remainder) % stride
    if W <= patch_size:
        pad_w = patch_size - W
    else:
        remainder = (W - patch_size) % stride
        pad_w = (stride - remainder) % stride

    if pad_h > 0 or pad_w > 0:
        # F.pad reflect doesn't support 5D, so reshape to 4D, pad, reshape back.
        # Use reflect when padding < spatial dim, zero-pad otherwise.
        B, Ci, T, _, _ = image_5d.shape
        pad_mode = "reflect" if pad_h < H and pad_w < W else "constant"
        img_4d = image_5d.reshape(B, Ci * T, H, W)
        img_4d = F.pad(img_4d, (0, pad_w, 0, pad_h), mode=pad_mode)
        image_5d = img_4d.reshape(B, Ci, T, H + pad_h, W + pad_w)
        if aux is not None:
            aux = F.pad(aux, (0, pad_w, 0, pad_h), mode=pad_mode)

    _, _, _, H_pad, W_pad = image_5d.shape

    # Accumulation buffers
    prob_sum = torch.zeros(1, num_classes, H_pad, W_pad, device=device)
    count = torch.zeros(1, 1, H_pad, W_pad, device=device)

    # Grid positions
    y_starts = list(range(0, H_pad - patch_size + 1, stride))
    x_starts = list(range(0, W_pad - patch_size + 1, stride))

    for y0 in y_starts:
        for x0 in x_starts:
            y1, x1 = y0 + patch_size, x0 + patch_size

            patch_img = image_5d[:, :, :, y0:y1, x0:x1]
            patch_aux = aux[:, :, y0:y1, x0:x1] if aux is not None else None

            logits = model(
                patch_img,
                aux=patch_aux,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
            )  # (1, C, patch_size, patch_size)

            probs = F.softmax(logits, dim=1)
            prob_sum[:, :, y0:y1, x0:x1] += probs
            count[:, :, y0:y1, x0:x1] += 1.0

    # Average and crop back to original size
    averaged = prob_sum / count.clamp(min=1.0)
    return averaged[:, :, :H, :W]


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image_5d: torch.Tensor,
    aux: torch.Tensor | None = None,
    patch_size: int = 224,
    overlap: float = 0.5,
    temporal_coords: torch.Tensor | None = None,
    location_coords: torch.Tensor | None = None,
    num_classes: int = 23,
) -> torch.Tensor:
    """Convenience: sliding window → argmax class indices.

    Returns:
        (1, H, W) int64 class predictions.
    """
    probs = sliding_window_inference(
        model, image_5d, aux,
        patch_size=patch_size, overlap=overlap,
        temporal_coords=temporal_coords,
        location_coords=location_coords,
        num_classes=num_classes,
    )
    return probs.argmax(dim=1)
