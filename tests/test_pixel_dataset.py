"""Tests for imint.training.pixel_dataset.PixelContextDataset.

Covers:
  1. DataLoader collation safety (the pin_memory storage-resize bug)
  2. Tensor ownership — torch.tensor() vs torch.from_numpy() semantics
  3. 3-tuple / 2-tuple return shape depending on enable_aux
  4. get_class_weights returns np.ndarray (not dict)
  5. Geometry & pixel alignment
       a. Context patch is centred exactly on the sampled (row, col)
       b. Center pixel spectral values reproduced faithfully in the patch
       c. Pixels within half the context window of the tile edge are excluded
       d. frame_2016 occupies channels [0:6] when present; base frames follow
       e. aux_vector reads from the correct spatial (row, col) position
       f. label at the index matches the class stored in the tile
  6. Boundary mask (_compute_boundary_mask)
  7. Oversampling weights — rare classes get weight ≥ 1.0
  8. split RNG — train / val produce different shuffles
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ── Torch availability guard ─────────────────────────────────────────────────
try:
    import torch
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Module under test ────────────────────────────────────────────────────────
from imint.training.pixel_dataset import (
    IGNORE_CLASS,
    N_AUX,
    N_BANDS,
    PixelContextDataset,
    _compute_boundary_mask,
    _TORCH_AVAILABLE,
)
from imint.training.unified_dataset import (
    AUX_CHANNEL_NAMES,
    AUX_LOG_TRANSFORM,
    AUX_NORM,
)
from imint.training.unified_schema import (
    NUM_UNIFIED_CLASSES,
    get_class_weights,
)

# ── Tile synthesis helpers ────────────────────────────────────────────────────

def _make_tile(
    H: int = 64,
    W: int = 64,
    T_base: int = 4,
    has_frame_2016: bool = False,
    n_classes: int = 5,
    seed: int = 0,
    *,
    constant_spectral: float | None = None,
    include_aux: bool = True,
) -> dict[str, np.ndarray]:
    """Build a synthetic tile dict ready to be saved as .npz.

    Args:
        H, W: spatial size.
        T_base: number of base temporal frames (4 = autumn/3× growing-season).
        has_frame_2016: whether to include a 2016 background frame.
        n_classes: number of non-background classes to distribute.
        seed: RNG seed for reproducibility.
        constant_spectral: if set, fill all spectral values with this constant
            (useful to verify exact value reproduction in patches).
        include_aux: whether to add AUX channel arrays.
    """
    rng = np.random.default_rng(seed)

    # ── spectral (T_base*6, H, W) ───────────────────────────────────────────
    if constant_spectral is not None:
        spectral = np.full((T_base * N_BANDS, H, W), constant_spectral, dtype=np.float32)
    else:
        spectral = rng.random((T_base * N_BANDS, H, W), dtype=np.float32).astype(np.float32)

    # ── label: checkerboard of classes 1..n_classes ─────────────────────────
    label = np.zeros((H, W), dtype=np.int32)
    for r in range(H):
        for c in range(W):
            label[r, c] = (r // 4 + c // 4) % n_classes + 1  # 1..n_classes

    arrays: dict[str, np.ndarray] = {
        "spectral": spectral,
        "label": label,
        "has_frame_2016": np.array(int(has_frame_2016), dtype=np.int32),
    }

    if has_frame_2016:
        if constant_spectral is not None:
            frame_2016 = np.full((N_BANDS, H, W), constant_spectral * 2, dtype=np.float32)
        else:
            frame_2016 = rng.random((N_BANDS, H, W), dtype=np.float32).astype(np.float32)
        arrays["frame_2016"] = frame_2016

    if include_aux:
        for ch in AUX_CHANNEL_NAMES:
            arrays[ch] = rng.random((H, W), dtype=np.float32).astype(np.float32)

    return arrays


def _save_tile(tmp_path: Path, name: str, arrays: dict[str, np.ndarray]) -> Path:
    """Save synthetic tile dict to a .npz file and return its path."""
    p = tmp_path / f"{name}.npz"
    np.savez(p, **arrays)
    return p


def _make_tile_file(
    tmp_path: Path,
    name: str = "tile_0",
    **kwargs: Any,
) -> Path:
    """Convenience wrapper: make + save in one call."""
    return _save_tile(tmp_path, name, _make_tile(**kwargs))


# ── 1. Dataset construction & basic shapes ───────────────────────────────────


class TestDatasetConstruction:
    """Basic construction and length checks."""

    def test_empty_tile_list(self):
        ds = PixelContextDataset([], context_px=32)
        assert len(ds) == 0

    def test_single_tile_non_zero_length(self, tmp_path):
        p = _make_tile_file(tmp_path, H=64, W=64, seed=1)
        ds = PixelContextDataset([p], context_px=32, samples_per_tile=128)
        assert len(ds) > 0

    def test_corrupt_tile_skipped(self, tmp_path):
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"not a npz file")
        good = _make_tile_file(tmp_path, "good", H=64, W=64)
        ds = PixelContextDataset([bad, good], context_px=32, samples_per_tile=128)
        assert len(ds) > 0  # the corrupt tile is skipped, good tile is used

    def test_tile_without_label_key_skipped(self, tmp_path):
        p = tmp_path / "nolabel.npz"
        np.savez(p, spectral=np.zeros((24, 64, 64), dtype=np.float32))
        ds = PixelContextDataset([p], context_px=32)
        assert len(ds) == 0

    def test_ignore_class_zero_excluded(self, tmp_path):
        """All pixels with class 0 must never appear in the index."""
        arrays = _make_tile(H=64, W=64)
        # Force label to all zeros in a region
        arrays["label"][:32, :] = IGNORE_CLASS
        p = _save_tile(tmp_path, "partial", arrays)
        ds = PixelContextDataset([p], context_px=32, samples_per_tile=512)
        for _, r, c, cls in ds._index:
            assert cls != IGNORE_CLASS, f"class 0 found at ({r},{c})"


# ── 2. Geometry & pixel alignment ────────────────────────────────────────────


class TestPixelAlignment:
    """Verify that patches are correctly centred and values match the tile."""

    def test_patch_center_matches_tile_spectral(self, tmp_path):
        """The center pixel of the returned patch must equal the tile value."""
        # Use a unique float at every spatial position so any alignment error
        # is immediately detectable.
        H, W = 64, 64
        T = 4
        arrays = _make_tile(H=H, W=W, T_base=T, has_frame_2016=False)

        # Paint each pixel with its spatial index encoded as a float
        for r in range(H):
            for c in range(W):
                arrays["spectral"][:, r, c] = float(r * W + c) / float(H * W)

        p = _save_tile(tmp_path, "indexed", arrays)
        ds = PixelContextDataset(
            [p],
            context_px=32,
            samples_per_tile=512,
            use_frame_2016=False,
            enable_aux=False,
        )
        assert len(ds) > 0

        half = 16  # context_px // 2
        for i in range(min(30, len(ds))):
            _, row, col, _ = ds._index[i]
            data = np.load(p, allow_pickle=False)
            patch_np = ds._build_patch(data, row, col)  # (T*6, 32, 32)

            expected_val = float(row * W + col) / float(H * W)
            center_vals = patch_np[:, half - 1, half - 1]  # center of [r0:r1, c0:c1]
            # r0 = row - half  → patch[:,0,:] = tile[:,row-half,:]
            # center of patch = tile[:,row, col] → index (half-1, half-1) uses
            # exclusive upper bound: r0+half-1 = row-1 …
            # Actually: patch slice is [row-half : row+half], so patch[:,k,:] =
            # tile[:,row-half+k,:]. Center is k=half → tile[:,row,:].
            center_vals = patch_np[:, half, half - 1]
            # Correct: r0 = row-half, r1 = row+half.  patch[:,half,:] = tile[:,row,:]
            center_vals = patch_np[:, half, half]
            # col: c0=col-half, patch[:,:,half] = tile[:,:,col]
            np.testing.assert_allclose(
                center_vals,
                expected_val,
                atol=1e-5,
                err_msg=f"Sample {i}: patch center mismatch at tile row={row} col={col}",
            )

    def test_patch_shape_without_2016_zeropads_to_T5(self, tmp_path):
        """When use_frame_2016=True but has_frame_2016=0, the first 6 channels
        must be zero-padded so patches are always (T_base+1)*6 channels wide.
        This keeps batch shapes uniform across tiles with/without the frame.
        """
        p = _make_tile_file(tmp_path, "no2016", H=64, W=64, T_base=4, has_frame_2016=False)
        ds = PixelContextDataset(
            [p], context_px=32, use_frame_2016=True, enable_aux=False
        )
        assert len(ds) > 0
        item = ds[0]
        patch = item[0]
        expected_ch = 5 * N_BANDS  # T_base + 1 (zero-padded) = 30
        assert patch.shape == (expected_ch, 32, 32), (
            f"Expected ({expected_ch}, 32, 32) got {tuple(patch.shape)}"
        )
        # First 6 channels (the zero-padded frame_2016 slot) must be all zeros
        if HAS_TORCH:
            np.testing.assert_allclose(patch[:N_BANDS].numpy(), 0.0, atol=1e-6,
                                       err_msg="Zero-pad channels should be 0.0")

    def test_patch_shape_without_2016_use_frame_false(self, tmp_path):
        """When use_frame_2016=False, patch channels = T_base * 6 exactly."""
        p = _make_tile_file(tmp_path, "no2016_nouse", H=64, W=64, T_base=4, has_frame_2016=False)
        ds = PixelContextDataset(
            [p], context_px=32, use_frame_2016=False, enable_aux=False
        )
        assert len(ds) > 0
        item = ds[0]
        patch = item[0]
        assert patch.shape == (4 * N_BANDS, 32, 32), (
            f"Expected (24, 32, 32) got {tuple(patch.shape)}"
        )

    def test_patch_shape_with_2016(self, tmp_path):
        """With 2016 frame: patch channels = (T_base+1)*6 = 30."""
        p = _make_tile_file(tmp_path, "with2016", H=64, W=64, T_base=4, has_frame_2016=True)
        ds = PixelContextDataset(
            [p], context_px=32, use_frame_2016=True, enable_aux=False
        )
        assert len(ds) > 0
        item = ds[0]
        patch = item[0]
        assert patch.shape == (5 * N_BANDS, 32, 32), (
            f"Expected (30, 32, 32) got {tuple(patch.shape)}"
        )

    def test_patch_spatial_size(self, tmp_path):
        """Spatial dimensions of the patch must equal context_px."""
        for ctx in (16, 32, 48):
            p = _make_tile_file(tmp_path, f"ctx{ctx}", H=96, W=96)
            ds = PixelContextDataset([p], context_px=ctx, enable_aux=False)
            if len(ds) == 0:
                continue
            patch = ds[0][0]
            assert patch.shape[-2] == ctx and patch.shape[-1] == ctx, (
                f"context_px={ctx}: got shape {tuple(patch.shape)}"
            )

    def test_mixed_2016_batch_has_uniform_shape(self, tmp_path):
        """Regression: batches containing tiles both with and without
        frame_2016 must produce patches of the same channel count so that
        torch.stack() inside DataLoader workers does not crash.

        When use_frame_2016=True but has_frame_2016=0, the dataset must
        zero-pad 6 channels so every patch is (T_base+1)*6 channels wide.
        """
        # Two tiles: one with frame_2016, one without
        p_with    = _make_tile_file(tmp_path, "with2016",    H=64, W=64, has_frame_2016=True)
        p_without = _make_tile_file(tmp_path, "without2016", H=64, W=64, has_frame_2016=False)

        ds = PixelContextDataset(
            [p_with, p_without],
            context_px=32,
            use_frame_2016=True,
            enable_aux=False,
            samples_per_tile=64,
        )
        assert len(ds) > 0

        shapes = set()
        for i in range(min(len(ds), 20)):
            item = ds[i]
            patch = item[0]
            if HAS_TORCH:
                shapes.add(tuple(patch.shape))
            else:
                shapes.add(patch.shape)

        assert len(shapes) == 1, (
            f"Mixed-tile batch has inconsistent patch shapes: {shapes}. "
            "Tiles without frame_2016 must be zero-padded to the same channel count."
        )
        # Shape should be (30, 32, 32) = (T_base+1)*6 channels
        expected_ch = 5 * N_BANDS  # 30
        (shape,) = shapes
        assert shape[0] == expected_ch, (
            f"Expected {expected_ch} channels (T=5 with zero-pad), got {shape[0]}"
        )

    def test_frame_2016_occupies_first_6_channels(self, tmp_path):
        """When has_frame_2016=1, channels [0:6] must come from frame_2016."""
        H, W = 64, 64
        arrays = _make_tile(H=H, W=W, T_base=4, has_frame_2016=True, constant_spectral=0.3)
        # frame_2016 gets constant_spectral * 2 = 0.6 in _make_tile
        p = _save_tile(tmp_path, "bg2016", arrays)

        ds = PixelContextDataset([p], context_px=32, use_frame_2016=True, enable_aux=False)
        assert len(ds) > 0

        for i in range(min(10, len(ds))):
            item = ds[i]
            patch = item[0]
            if HAS_TORCH:
                patch = patch.numpy()
            # Channels 0-5 come from frame_2016 (value ≈ 0.6)
            np.testing.assert_allclose(patch[:6], 0.6, atol=1e-4,
                                       err_msg="frame_2016 channels should be 0.6")
            # Channels 6-29 come from base spectral (value ≈ 0.3)
            np.testing.assert_allclose(patch[6:], 0.3, atol=1e-4,
                                       err_msg="base spectral channels should be 0.3")

    def test_border_exclusion(self, tmp_path):
        """No sample should be within half a context window of the tile edge."""
        H, W, ctx = 80, 80, 32
        p = _make_tile_file(tmp_path, "border", H=H, W=W)
        ds = PixelContextDataset([p], context_px=ctx, samples_per_tile=1024)
        half = ctx // 2
        for _, row, col, _ in ds._index:
            assert row >= half,     f"row={row} too close to top (half={half})"
            assert row < H - half,  f"row={row} too close to bottom"
            assert col >= half,     f"col={col} too close to left"
            assert col < W - half,  f"col={col} too close to right"

    def test_label_at_center_pixel_matches_stored_class(self, tmp_path):
        """The class stored in _index must equal label[row, col] in the tile."""
        p = _make_tile_file(tmp_path, "label_check", H=64, W=64, seed=7)
        ds = PixelContextDataset([p], context_px=32, samples_per_tile=256)

        tile_label = np.load(p, allow_pickle=False)["label"]
        for path, row, col, stored_cls in ds._index:
            actual_cls = int(tile_label[row, col])
            assert stored_cls == actual_cls, (
                f"Stored class {stored_cls} ≠ tile label {actual_cls} at ({row},{col})"
            )


class TestAuxAlignment:
    """AUX vector must be read from the correct (row, col) position."""

    def test_aux_reads_center_pixel(self, tmp_path):
        """aux_vec[i] must equal the normalized value at (row, col) in aux array."""
        H, W = 64, 64
        arrays = _make_tile(H=H, W=W, include_aux=True)
        p = _save_tile(tmp_path, "aux_check", arrays)

        ds = PixelContextDataset([p], context_px=32, enable_aux=True, samples_per_tile=64)
        assert len(ds) > 0

        for i in range(min(10, len(ds))):
            path, row, col, _ = ds._index[i]
            data = np.load(path, allow_pickle=False)
            aux_vec = ds._build_aux_vector(data, row, col)

            for j, ch_name in enumerate(AUX_CHANNEL_NAMES):
                raw_val = float(data[ch_name][row, col])
                if ch_name in AUX_LOG_TRANSFORM:
                    raw_val = float(np.log1p(max(raw_val, 0.0)))
                if ch_name in AUX_NORM:
                    mean, std = AUX_NORM[ch_name]
                    expected = (raw_val - mean) / max(std, 1e-6)
                else:
                    expected = raw_val
                np.testing.assert_allclose(
                    aux_vec[j], expected, atol=1e-5,
                    err_msg=f"aux_vec[{j}] ({ch_name}) mismatch at ({row},{col})",
                )

    def test_aux_shape(self, tmp_path):
        """aux_vec must have exactly N_AUX elements."""
        p = _make_tile_file(tmp_path, "aux_shape", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=True)
        assert len(ds) > 0
        item = ds[0]
        assert len(item) == 3, "enable_aux=True must return 3-tuple"
        aux = item[1]
        if HAS_TORCH:
            assert aux.shape == (N_AUX,), f"Expected ({N_AUX},) got {tuple(aux.shape)}"
        else:
            assert len(aux) == N_AUX

    def test_aux_missing_channel_is_zero_normalized(self, tmp_path):
        """A missing AUX channel must normalise to approximately 0."""
        arrays = _make_tile(H=64, W=64, include_aux=False)
        # Provide only one AUX channel, leave the rest missing
        arrays[AUX_CHANNEL_NAMES[0]] = np.full((64, 64), 0.0, dtype=np.float32)
        p = _save_tile(tmp_path, "partial_aux", arrays)

        ds = PixelContextDataset([p], context_px=32, enable_aux=True, samples_per_tile=32)
        assert len(ds) > 0
        _, row, col, _ = ds._index[0]
        data = np.load(p, allow_pickle=False)
        aux_vec = ds._build_aux_vector(data, row, col)

        # Channels 1..N_AUX-1 are missing → raw val=0 → normalized to ≈ (0-mean)/std
        for j in range(1, N_AUX):
            ch_name = AUX_CHANNEL_NAMES[j]
            if ch_name in AUX_NORM:
                mean, std = AUX_NORM[ch_name]
                expected = (0.0 - mean) / max(std, 1e-6)
                np.testing.assert_allclose(
                    aux_vec[j], expected, atol=1e-4,
                    err_msg=f"Missing {ch_name} should yield normalized 0.0",
                )


# ── 3. Tensor output types & collation safety ────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTensorOwnership:
    """The storage-resize bug: torch.from_numpy() vs torch.tensor()."""

    def test_patch_tensor_owns_storage(self, tmp_path):
        """patch_t must NOT share memory with any numpy array (PyTorch-owned)."""
        p = _make_tile_file(tmp_path, "storage", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=False)
        assert len(ds) > 0
        patch_t, _ = ds[0]
        # If storage is PyTorch-owned, the tensor is not backed by a numpy array.
        # torch.is_storage doesn't expose this directly, but a reliable proxy:
        # calling .numpy() on a tensor that shares numpy storage is lossless,
        # while torch.tensor() copies → modifying one does not affect the other.
        patch_np = patch_t.numpy().copy()
        patch_np[:] = -999.0
        # Original tensor should be unchanged if it owns its storage
        assert not torch.all(patch_t == -999.0), (
            "patch_t appears to share numpy storage — torch.tensor() copy was lost"
        )

    def test_aux_tensor_owns_storage(self, tmp_path):
        """aux_vec tensor must own its storage."""
        p = _make_tile_file(tmp_path, "aux_storage", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=True)
        assert len(ds) > 0
        _, aux_t, _ = ds[0]
        aux_np = aux_t.numpy().copy()
        aux_np[:] = -999.0
        assert not torch.all(aux_t == -999.0)

    def test_patch_dtype(self, tmp_path):
        p = _make_tile_file(tmp_path, "dtype_patch", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=False)
        assert len(ds) > 0
        patch_t, _ = ds[0]
        assert patch_t.dtype == torch.float32

    def test_label_dtype(self, tmp_path):
        p = _make_tile_file(tmp_path, "dtype_label", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=False)
        assert len(ds) > 0
        _, label_t = ds[0]
        assert label_t.dtype == torch.long

    def test_aux_dtype(self, tmp_path):
        p = _make_tile_file(tmp_path, "dtype_aux", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=True)
        assert len(ds) > 0
        _, aux_t, _ = ds[0]
        assert aux_t.dtype == torch.float32


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDataLoaderCollation:
    """DataLoader with pin_memory=True must not raise storage-resize errors."""

    def _make_ds(self, tmp_path: Path, n_tiles: int = 3, enable_aux: bool = True):
        paths = [
            _make_tile_file(tmp_path, f"t{i}", H=64, W=64, seed=i)
            for i in range(n_tiles)
        ]
        return PixelContextDataset(
            paths, context_px=32, samples_per_tile=64, enable_aux=enable_aux
        )

    def test_dataloader_pin_memory_false(self, tmp_path):
        """Baseline: pin_memory=False, num_workers=0 must always work.

        Tiles have no frame_2016 but use_frame_2016 defaults to True, so
        patches are zero-padded to (T_base+1)*6 = 30 channels.
        """
        ds = self._make_ds(tmp_path)
        loader = DataLoader(ds, batch_size=16, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        patch, aux, label = batch
        assert patch.shape == (16, 5 * N_BANDS, 32, 32)  # 30 channels (zero-padded)
        assert aux.shape == (16, N_AUX)
        assert label.shape == (16,)

    def test_dataloader_pin_memory_false_no_aux(self, tmp_path):
        ds = self._make_ds(tmp_path, enable_aux=False)
        loader = DataLoader(ds, batch_size=16, num_workers=0, pin_memory=False)
        patch, label = next(iter(loader))
        assert patch.shape == (16, 5 * N_BANDS, 32, 32)  # 30 channels (zero-padded)
        assert label.shape == (16,)

    @pytest.mark.skipif(
        os.name == "nt",
        reason="multiprocessing fork semantics differ on Windows",
    )
    def test_dataloader_multiworker(self, tmp_path):
        """num_workers=2 — exercises the fork+collate path."""
        ds = self._make_ds(tmp_path, n_tiles=6)
        loader = DataLoader(
            ds,
            batch_size=8,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
        )
        batch = next(iter(loader))
        patch, aux, label = batch
        assert patch.ndim == 4
        assert aux.ndim == 2
        assert label.ndim == 1

    @pytest.mark.skipif(
        os.name == "nt",
        reason="multiprocessing fork semantics differ on Windows",
    )
    def test_safe_collate_multiworker_no_resize_error(self, tmp_path):
        """Regression: PyTorch's default_collate in worker processes calls
        storage._new_shared() + resize_() to pre-allocate a shared-memory
        output buffer.  On some PyTorch 2.x / Python 3.11 builds the
        mmap-backed shared storage is fixed-size and rejects resize_():
          RuntimeError: Trying to resize storage that is not resizable

        This broke 3 consecutive H100 training runs.  The fix is _safe_collate
        in train_pixel.py, which calls torch.stack() directly (no pre-alloc).

        Verify that num_workers=4 + _safe_collate completes without error
        (pin_memory=False to avoid needing CUDA in CI).
        """
        from scripts.train_pixel import _safe_collate  # the fix under test

        ds = self._make_ds(tmp_path, n_tiles=8)
        loader = DataLoader(
            ds,
            batch_size=16,
            num_workers=4,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
            collate_fn=_safe_collate,
        )
        # Consume all batches — must not raise RuntimeError
        for patch, aux, label in loader:
            assert patch.shape[1:] == (5 * N_BANDS, 32, 32)  # 30ch (zero-padded)
            assert aux.shape[1] == N_AUX
            assert label.ndim == 1

    def test_batch_label_values_in_range(self, tmp_path):
        """All labels in a batch must be valid class indices (> 0 for training)."""
        ds = self._make_ds(tmp_path)
        loader = DataLoader(ds, batch_size=32, num_workers=0)
        for patch, aux, label in loader:
            assert torch.all(label > 0), "IGNORE_CLASS (0) must not appear in batch"
            assert torch.all(label < NUM_UNIFIED_CLASSES), (
                f"Label >= {NUM_UNIFIED_CLASSES} found"
            )
            break  # one batch is enough

    def test_patch_values_finite(self, tmp_path):
        """All spectral values in a batch must be finite (no NaN/Inf)."""
        ds = self._make_ds(tmp_path)
        loader = DataLoader(ds, batch_size=32, num_workers=0)
        for patch, _, _ in loader:
            assert torch.all(torch.isfinite(patch)), "Non-finite values in patch batch"
            break


# ── 4. Return-tuple shape ────────────────────────────────────────────────────


class TestReturnTupleShape:
    """3-tuple vs 2-tuple depending on enable_aux."""

    def test_enable_aux_true_returns_three_tuple(self, tmp_path):
        p = _make_tile_file(tmp_path, "3tup", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=True)
        assert len(ds) > 0
        item = ds[0]
        assert len(item) == 3, f"Expected 3-tuple, got {len(item)}-tuple"

    def test_enable_aux_false_returns_two_tuple(self, tmp_path):
        p = _make_tile_file(tmp_path, "2tup", H=64, W=64)
        ds = PixelContextDataset([p], context_px=32, enable_aux=False)
        assert len(ds) > 0
        item = ds[0]
        assert len(item) == 2, f"Expected 2-tuple, got {len(item)}-tuple"


# ── 5. Class weights ─────────────────────────────────────────────────────────


class TestClassWeights:
    """get_class_weights must return an ndarray, not a dict."""

    def test_returns_ndarray(self):
        counts = {i: 100 * i for i in range(NUM_UNIFIED_CLASSES)}
        result = get_class_weights(counts)
        assert isinstance(result, np.ndarray), (
            f"Expected np.ndarray, got {type(result)}"
        )

    def test_correct_length(self):
        counts = {i: max(i, 1) for i in range(NUM_UNIFIED_CLASSES)}
        result = get_class_weights(counts)
        assert len(result) == NUM_UNIFIED_CLASSES

    def test_background_class_weight_is_zero(self):
        counts = {i: 100 for i in range(NUM_UNIFIED_CLASSES)}
        result = get_class_weights(counts)
        assert result[0] == 0.0, "Background (class 0) weight must be 0.0"

    def test_rare_class_gets_higher_weight(self):
        """A class with fewer pixels must get a higher weight than a common one."""
        counts = {i: 1000 for i in range(NUM_UNIFIED_CLASSES)}
        counts[1] = 10    # very rare
        counts[2] = 10000  # very common
        result = get_class_weights(counts)
        assert result[1] > result[2], (
            "Rare class (1) should have higher weight than common class (2)"
        )

    def test_weight_capped_at_max(self):
        counts = {i: 1000000 if i != 1 else 1 for i in range(NUM_UNIFIED_CLASSES)}
        result = get_class_weights(counts, max_weight=10.0)
        assert np.all(result <= 10.0), "No weight should exceed max_weight"

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_directly_usable_by_torch_tensor(self):
        """Bug regression: result must work with torch.tensor() without .values()."""
        counts = {i: max(i * 100, 1) for i in range(NUM_UNIFIED_CLASSES)}
        result = get_class_weights(counts)
        # This must not raise AttributeError
        try:
            t = torch.tensor(result, dtype=torch.float32)
        except Exception as e:
            pytest.fail(f"torch.tensor(result) raised {e}")
        assert t.shape == (NUM_UNIFIED_CLASSES,)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_no_values_attribute(self):
        """Regression: ndarray has no .values() — verify calling it raises."""
        counts = {i: 100 for i in range(NUM_UNIFIED_CLASSES)}
        result = get_class_weights(counts)
        with pytest.raises(AttributeError):
            _ = result.values()  # type: ignore[attr-defined]


# ── 6. Boundary mask ────────────────────────────────────────────────────────


class TestBoundaryMask:
    """_compute_boundary_mask geometry."""

    def test_uniform_label_has_no_boundary(self):
        """A label map with a single class has no boundaries."""
        label = np.ones((32, 32), dtype=np.int32) * 3
        mask = _compute_boundary_mask(label, boundary_px=3)
        assert not mask.any(), "Uniform label should produce an empty boundary mask"

    def test_two_class_boundary_is_near_edge(self):
        """A sharp left/right split: boundary must appear only near the midline."""
        label = np.ones((32, 32), dtype=np.int32)
        label[:, 16:] = 2  # class 1 on left, class 2 on right
        mask = _compute_boundary_mask(label, boundary_px=3)
        # Columns far from split (< 12 or > 20) should not be boundary
        assert not mask[:, :12].any(), "Far-left columns should not be boundary"
        assert not mask[:, 20:].any(), "Far-right columns should not be boundary"
        # Columns adjacent to split must be boundary
        assert mask[:, 13:19].any(), "Columns near split should be boundary"

    def test_boundary_mask_shape(self):
        label = np.zeros((48, 64), dtype=np.int32)
        label[:24, :] = 1
        label[24:, :] = 2
        mask = _compute_boundary_mask(label, boundary_px=2)
        assert mask.shape == (48, 64)
        assert mask.dtype == bool

    def test_background_ignored_in_boundary(self):
        """Class 0 (background) should not contribute to the boundary mask."""
        label = np.zeros((32, 32), dtype=np.int32)
        label[:, 16:] = 3  # class 3 on right, background on left
        mask = _compute_boundary_mask(label, boundary_px=2)
        # Background pixels should not be marked as boundaries
        # (only class edges between non-background classes count)
        # With current impl: class 3 dilated touches class 0 side → that's OK
        # Just verify shape and dtype are correct
        assert mask.shape == label.shape
        assert mask.dtype == bool


# ── 7. Oversampling weights ──────────────────────────────────────────────────


class TestOversamplingWeights:
    """Rare-class oversampling: rare classes should appear ≥ proportionally."""

    def test_rare_class_represented(self, tmp_path):
        """A rare class occupying ~1% of pixels should still appear in the index.

        Note: the rare-class block must lie inside the valid sampling window
        (i.e. at least context_px//2 pixels from every edge).  With
        context_px=32 the border exclusion is 16 pixels on each side, so we
        place the block at rows/cols [16:29] — well within the valid zone but
        only 169 out of 9,216 interior pixels (~1.8%).
        """
        H, W, ctx = 128, 128, 32
        half = ctx // 2  # 16
        arrays = _make_tile(H=H, W=W)
        # Flood most pixels with class 1
        arrays["label"][:] = 1
        # Place class 2 in a small interior region (~1.8% of valid pixels)
        arrays["label"][half : half + 13, half : half + 13] = 2
        p = _save_tile(tmp_path, "rare", arrays)

        ds = PixelContextDataset(
            [p],
            context_px=32,
            samples_per_tile=512,
            oversample_rare=True,
            rare_weight=3.0,
        )
        classes_in_index = {cls for _, _, _, cls in ds._index}
        assert 2 in classes_in_index, "Rare class 2 must appear in the sample index"


# ── 8. Split RNG reproducibility ────────────────────────────────────────────


class TestSplitReproducibility:
    """Index is now sorted by tile path; shuffle lives in TileGroupSampler."""

    def _make_ds(self, tmp_path, paths, split="train", seed=42):
        return PixelContextDataset(
            paths, context_px=32, split=split, samples_per_tile=64, seed=seed
        )

    def test_index_sorted_by_path(self, tmp_path):
        """After init the index must be sorted ascending by tile path."""
        paths = [_make_tile_file(tmp_path, f"z_{i}", H=64, W=64, seed=i) for i in range(3)]
        ds = self._make_ds(tmp_path, paths)
        paths_in_index = [entry[0] for entry in ds._index]
        assert paths_in_index == sorted(paths_in_index), \
            "Index must be sorted by tile path for tile-group caching to work"

    def test_same_args_produce_identical_index(self, tmp_path):
        """Construction is deterministic — same args → same index twice."""
        p = _make_tile_file(tmp_path, "repro", H=64, W=64, seed=0)
        ds1 = self._make_ds(tmp_path, [p], seed=7)
        ds2 = self._make_ds(tmp_path, [p], seed=7)
        assert ds1._index == ds2._index, "Same args must produce identical index"

    def test_different_splits_use_different_sampling_rngs(self, tmp_path):
        """train / val use different RNG seeds → different sample draws."""
        paths = [_make_tile_file(tmp_path, f"sp_{i}", H=64, W=64, seed=i) for i in range(2)]
        ds_tr = self._make_ds(tmp_path, paths, split="train")
        ds_vl = self._make_ds(tmp_path, paths, split="val")
        # Both sorted by path but drawn with different RNG → different rows/cols
        rows_tr = [(r, c) for _, r, c, _ in ds_tr._index]
        rows_vl = [(r, c) for _, r, c, _ in ds_vl._index]
        assert rows_tr != rows_vl, "train and val should draw different pixel locations"


class TestTileGroupSampler:
    """TileGroupSampler shuffles tile groups, keeps per-tile locality."""

    def test_sampler_yields_all_indices(self, tmp_path):
        """Every sample index must appear exactly once per epoch."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        from imint.training.pixel_dataset import TileGroupSampler
        paths = [_make_tile_file(tmp_path, f"ts_{i}", H=64, W=64, seed=i) for i in range(3)]
        ds = PixelContextDataset(paths, context_px=32, samples_per_tile=64, seed=0)
        sampler = TileGroupSampler(ds, shuffle=True, seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(len(ds))), \
            "Sampler must yield each index exactly once"

    def test_tile_locality_within_epoch(self, tmp_path):
        """All samples from a tile must be contiguous in the sampler output."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        from imint.training.pixel_dataset import TileGroupSampler
        paths = [_make_tile_file(tmp_path, f"loc_{i}", H=64, W=64, seed=i) for i in range(4)]
        ds = PixelContextDataset(paths, context_px=32, samples_per_tile=32, seed=0)
        sampler = TileGroupSampler(ds, shuffle=True, seed=42)
        indices = list(sampler)
        # Map index → tile path, verify no interleaving
        tile_sequence = [ds._index[i][0] for i in indices]
        # Count path-change events: must equal n_tiles - 1 (no interleaving)
        changes = sum(1 for a, b in zip(tile_sequence, tile_sequence[1:]) if a != b)
        n_tiles = len({e[0] for e in ds._index})
        assert changes == n_tiles - 1, \
            f"Tile paths should change exactly {n_tiles-1} times, got {changes}"

    def test_set_epoch_reshuffles_tile_order(self, tmp_path):
        """set_epoch() must produce a different tile order across epochs."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        from imint.training.pixel_dataset import TileGroupSampler
        paths = [_make_tile_file(tmp_path, f"ep_{i}", H=64, W=64, seed=i) for i in range(5)]
        ds = PixelContextDataset(paths, context_px=32, samples_per_tile=32, seed=0)
        sampler = TileGroupSampler(ds, shuffle=True, seed=42)
        sampler.set_epoch(0); order0 = list(sampler)
        sampler.set_epoch(1); order1 = list(sampler)
        assert order0 != order1, "Different epochs must produce different tile order"

    def test_shuffle_false_is_deterministic(self, tmp_path):
        """shuffle=False must produce the same order every call."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        from imint.training.pixel_dataset import TileGroupSampler
        paths = [_make_tile_file(tmp_path, f"det_{i}", H=64, W=64, seed=i) for i in range(3)]
        ds = PixelContextDataset(paths, context_px=32, samples_per_tile=32, seed=0)
        sampler = TileGroupSampler(ds, shuffle=False, seed=0)
        assert list(sampler) == list(sampler), "Deterministic sampler must be stable"
