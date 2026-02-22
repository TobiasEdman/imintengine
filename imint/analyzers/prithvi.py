"""
imint/analyzers/prithvi.py — Prithvi-EO-2.0 Foundation Model analyzer

Runs IBM/NASA Prithvi-EO-2.0 (ViT with 3D MAE, trained on HLS Sentinel-2)
for feature extraction or segmentation.

Two modes:
  - embeddings (default): Extract backbone features. Returns embedding
    arrays and spatial statistics. Useful for downstream tasks, clustering,
    or feeding into other models.
  - segmentation: Run a fine-tuned segmentation head. Requires a saved
    model checkpoint (config: model_path). Returns per-pixel class map.

Loading priority:
  1. TerraTorch (pip install terratorch) — full framework, many models
  2. Direct HuggingFace loading — uses bundled prithvi_mae.py (torch only)

Band requirements: B02, B03, B04, B8A (not B08!), B11, B12
  These map to HLS bands B02–B07: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2.

The model expects DN-scale inputs (not [0,1] reflectance) and normalizes
internally using its trained mean/std statistics.
"""
from __future__ import annotations

import math
import numpy as np
from .base import BaseAnalyzer, AnalysisResult

# Prithvi-EO-2.0-300M-TL expects these 6 Sentinel-2 bands in this order.
# HLS naming: B02=Blue, B03=Green, B04=Red, B05=NarrowNIR(=S2 B8A),
#             B06=SWIR1(=S2 B11), B07=SWIR2(=S2 B12)
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
PRITHVI_BACKBONE = "prithvi_eo_v2_300m_tl"

# Model input patch size (ViT with 224x224 spatial resolution)
PATCH_SIZE = 224


class PrithviAnalyzer(BaseAnalyzer):
    """Prithvi-EO-2.0 foundation model analyzer.

    Config options:
        mode: "embeddings" (default) or "segmentation"
        backbone: Registry key (default: "prithvi_eo_v2_300m_tl")
        model_path: Path to fine-tuned checkpoint (required for segmentation)
        device: "cpu", "cuda", or "mps" (auto-detected if omitted)
    """

    name = "prithvi"

    def analyze(self, rgb, bands=None, date=None, coords=None, output_dir="outputs"):
        # Check availability: terratorch OR torch (direct HF loading)
        from ..fm.terratorch_loader import (
            check_terratorch_available, check_torch_available,
        )

        has_terratorch = check_terratorch_available()
        has_torch = check_torch_available()

        if not has_terratorch and not has_torch:
            return AnalysisResult(
                analyzer=self.name,
                success=False,
                error=(
                    "Neither terratorch nor torch is installed. "
                    "Install with: pip install terratorch  (recommended)\n"
                    "  or: pip install torch  (minimal, Prithvi-only)"
                ),
            )

        # Validate bands
        missing = self._check_bands(bands)
        if missing:
            has_b08 = bands is not None and "B08" in bands and "B8A" in missing
            b08_note = (
                " Note: B8A (865nm, 20m narrow NIR) != B08 (842nm, 10m wide NIR)."
                if has_b08 else ""
            )
            return AnalysisResult(
                analyzer=self.name,
                success=False,
                error=(
                    f"Missing required bands: {missing}. "
                    f"Prithvi-EO-2.0 requires: {PRITHVI_BANDS}.{b08_note}"
                ),
            )

        mode = self.config.get("mode", "embeddings")
        if mode == "segmentation":
            return self._run_segmentation(bands)
        return self._run_embeddings(bands)

    def _check_bands(self, bands: dict | None) -> list[str]:
        """Return list of missing required bands, or empty list if all present."""
        if bands is None:
            return PRITHVI_BANDS.copy()
        return [b for b in PRITHVI_BANDS if b not in bands]

    def _run_embeddings(self, bands: dict) -> AnalysisResult:
        """Extract backbone feature embeddings.

        Uses a sliding-window approach over 224x224 patches to handle
        images larger than the model's native input size. Each patch
        is normalized with the model's trained mean/std, reshaped to
        5D (B, C, T, H, W), and passed through the encoder.
        """
        from ..fm.terratorch_loader import (
            load_backbone, get_device, get_prithvi_config,
            extract_encoder_features,
        )
        import torch

        device = get_device(self.config.get("device"))
        backbone_name = self.config.get("backbone", PRITHVI_BACKBONE)

        # Load backbone
        model = load_backbone(backbone_name, pretrained=True)
        model = model.to(device)
        model.eval()

        # Get model config for normalization
        config = get_prithvi_config()
        mean = np.array(config["mean"], dtype=np.float32)
        std = np.array(config["std"], dtype=np.float32)

        # Stack bands in correct order: (C, H, W) in reflectance [0, 1]
        band_arrays = [bands[b] for b in PRITHVI_BANDS]
        stacked = np.stack(band_arrays, axis=0)  # (6, H, W)

        # Convert reflectance [0,1] -> DN scale matching model training data
        # Model was trained on HLS reflectance x 10000 (surface reflectance DN)
        stacked_dn = stacked * 10000.0

        # Normalize with model's mean/std
        # mean/std shape: (6,) -> (6, 1, 1) for broadcasting
        mean_3d = mean.reshape(-1, 1, 1)
        std_3d = std.reshape(-1, 1, 1)
        stacked_norm = (stacked_dn - mean_3d) / std_3d

        _, h, w = stacked_norm.shape

        # Process in 224x224 patches with sliding window
        patch_embeddings = []
        patch_positions = []

        # Calculate patch positions (no overlap for simplicity)
        y_steps = list(range(0, h, PATCH_SIZE))
        x_steps = list(range(0, w, PATCH_SIZE))

        # If the image is smaller than 224, pad it
        if h < PATCH_SIZE or w < PATCH_SIZE:
            pad_h = max(PATCH_SIZE - h, 0)
            pad_w = max(PATCH_SIZE - w, 0)
            stacked_norm = np.pad(
                stacked_norm,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )
            y_steps = [0]
            x_steps = [0]

        for y in y_steps:
            for x in x_steps:
                # Extract patch, handling edge cases
                y_end = min(y + PATCH_SIZE, stacked_norm.shape[1])
                x_end = min(x + PATCH_SIZE, stacked_norm.shape[2])
                patch = stacked_norm[:, y:y_end, x:x_end]

                # Pad if patch is smaller than 224x224
                ph, pw = patch.shape[1], patch.shape[2]
                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    pad_h = PATCH_SIZE - ph
                    pad_w = PATCH_SIZE - pw
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="reflect",
                    )

                # Convert to 5D tensor: (1, C, T=1, H, W) for PrithviMAE
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(2)
                tensor = tensor.to(device)  # (1, 6, 1, 224, 224)

                # Extract encoder features
                features = extract_encoder_features(model, tensor)
                patch_embeddings.append(features)
                patch_positions.append((y, x))

        # If single patch, use it directly
        if len(patch_embeddings) == 1:
            embedding_np = patch_embeddings[0]
        else:
            # Combine patch embeddings: average them for a global representation
            # Each is (1, embed_dim, grid_h, grid_w)
            embedding_np = np.mean(patch_embeddings, axis=0)

        # Compute spatial statistics over the embedding
        stats = {
            "embedding_shape": list(embedding_np.shape),
            "mean": float(np.mean(embedding_np)),
            "std": float(np.std(embedding_np)),
            "min": float(np.min(embedding_np)),
            "max": float(np.max(embedding_np)),
            "device": device,
            "backbone": backbone_name,
            "n_patches": len(patch_embeddings),
            "image_size": [h, w],
            "patch_size": PATCH_SIZE,
        }

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "embedding": embedding_np,
                "stats": stats,
            },
            metadata={
                "mode": "embeddings",
                "backbone": backbone_name,
                "device": device,
                "bands_used": PRITHVI_BANDS,
                "n_patches": len(patch_embeddings),
            },
        )

    def _run_segmentation(self, bands: dict) -> AnalysisResult:
        """Run fine-tuned segmentation model.

        Supports two loading modes:
          1. task_head (recommended): Name from TASK_HEAD_REGISTRY
             (e.g. "sen1floods11", "burn_scars"). Downloads from HuggingFace.
          2. model_path (legacy): Direct path to a checkpoint file.

        The task_head approach uses our built-in UperNet decoder and handles
        weight mapping automatically.
        """
        from ..fm.terratorch_loader import (
            get_device, get_prithvi_config, load_segmentation_model,
            TASK_HEAD_REGISTRY,
        )
        import torch

        task_head = self.config.get("task_head")
        model_path = self.config.get("model_path")

        if not task_head and not model_path:
            available = ", ".join(sorted(TASK_HEAD_REGISTRY.keys()))
            return AnalysisResult(
                analyzer=self.name,
                success=False,
                error=(
                    "Segmentation mode requires either:\n"
                    f"  - task_head: one of [{available}]\n"
                    "  - model_path: path to a fine-tuned checkpoint"
                ),
            )

        device = get_device(self.config.get("device"))

        # Get model config for normalization
        prithvi_config = get_prithvi_config()
        mean = np.array(prithvi_config["mean"], dtype=np.float32)
        std = np.array(prithvi_config["std"], dtype=np.float32)

        # Load model
        task_config = None
        try:
            if task_head:
                model, task_config = load_segmentation_model(task_head, device=device)
            else:
                model = torch.load(model_path, map_location=device, weights_only=False)
                if hasattr(model, "eval"):
                    model.eval()
        except Exception as e:
            source = task_head or model_path
            return AnalysisResult(
                analyzer=self.name,
                success=False,
                error=f"Failed to load segmentation model '{source}': {e}",
            )

        # Stack and normalize bands
        band_arrays = [bands[b] for b in PRITHVI_BANDS]
        stacked = np.stack(band_arrays, axis=0)  # (6, H, W)

        # Reflectance [0,1] -> DN scale -> normalize
        stacked_dn = stacked * 10000.0
        mean_3d = mean.reshape(-1, 1, 1)
        std_3d = std.reshape(-1, 1, 1)
        stacked_norm = (stacked_dn - mean_3d) / std_3d

        _, h, w = stacked_norm.shape

        # Run sliding-window inference for large images
        seg_mask = self._sliding_window_segmentation(
            model, stacked_norm, device, patch_size=PATCH_SIZE,
        )

        # Compute class statistics
        unique, counts = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        class_names = task_config.get("class_names", {}) if task_config else {}
        class_stats = {}
        for cls, cnt in zip(unique, counts):
            cls_int = int(cls)
            entry = {
                "pixel_count": int(cnt),
                "fraction": round(int(cnt) / total, 4),
            }
            if cls_int in class_names:
                entry["name"] = class_names[cls_int]
            class_stats[cls_int] = entry

        metadata = {
            "mode": "segmentation",
            "device": device,
            "bands_used": PRITHVI_BANDS,
            "image_size": [h, w],
        }
        if task_head:
            metadata["task_head"] = task_head
            if task_config:
                metadata["description"] = task_config.get("description", "")
        if model_path:
            metadata["model_path"] = model_path

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "seg_mask": seg_mask,
                "class_stats": class_stats,
                "n_classes": len(unique),
            },
            metadata=metadata,
        )

    def _sliding_window_segmentation(
        self, model, stacked_norm: np.ndarray, device: str, patch_size: int = 224,
    ) -> np.ndarray:
        """Run segmentation with sliding window, handling images of any size.

        Args:
            model: Segmentation model with forward(tensor) -> logits.
            stacked_norm: (C, H, W) normalized input.
            device: PyTorch device string.
            patch_size: Window size (default 224).

        Returns:
            (H, W) uint8 array of class indices.
        """
        import torch

        _, h, w = stacked_norm.shape

        # Pad to be divisible by patch_size
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            stacked_padded = np.pad(
                stacked_norm,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )
        else:
            stacked_padded = stacked_norm

        padded_h, padded_w = stacked_padded.shape[1:]
        n_rows = padded_h // patch_size
        n_cols = padded_w // patch_size

        # Process each patch
        seg_patches = np.zeros((padded_h, padded_w), dtype=np.uint8)

        with torch.no_grad():
            for row in range(n_rows):
                for col in range(n_cols):
                    y0 = row * patch_size
                    x0 = col * patch_size
                    patch = stacked_padded[:, y0:y0 + patch_size, x0:x0 + patch_size]

                    # (C, H, W) -> (1, C, H, W)
                    tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

                    # Forward pass — fall back to CPU if MPS has issues
                    # (MPS adaptive pool doesn't support non-divisible sizes)
                    try:
                        output = model(tensor)
                    except RuntimeError as e:
                        if "mps" in device.lower() or "Adaptive pool MPS" in str(e):
                            print(f"    MPS error, falling back to CPU: {e}")
                            model = model.to("cpu")
                            tensor = tensor.to("cpu")
                            device = "cpu"
                            output = model(tensor)
                        else:
                            raise

                    # Handle different output formats
                    if isinstance(output, dict):
                        logits = output.get("logits", output.get("output"))
                        if logits is None:
                            logits = next(iter(output.values()))
                    elif isinstance(output, (list, tuple)):
                        logits = output[0]
                    else:
                        logits = output

                    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    seg_patches[y0:y0 + patch_size, x0:x0 + patch_size] = pred

        # Crop back to original size
        return seg_patches[:h, :w]
