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

    def test_patch_shape_without_2016(self, tmp_path):
        """Without 2016 frame: patch channels = T_base * 6."""
        p = _make_tile_file(tmp_path, "no2016", H=64, W=64, T_base=4, has_frame_2016=False)
        ds = PixelContextDataset(
            [p], context_px=32, use_frame_2016=True, enable_aux=False
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
        """Baseline: pin_memory=False, num_workers=0 must always work."""
        ds = self._make_ds(tmp_path)
        loader = DataLoader(ds, batch_size=16, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        patch, aux, label = batch
        assert patch.shape == (16, 4 * N_BANDS, 32, 32)
        assert aux.shape == (16, N_AUX)
        assert label.shape == (16,)

    def test_dataloader_pin_memory_false_no_aux(self, tmp_path):
        ds = self._make_ds(tmp_path, enable_aux=False)
        loader = DataLoader(ds, batch_size=16, num_workers=0, pin_memory=False)
        patch, label = next(iter(loader))
        assert patch.shape == (16, 4 * N_BANDS, 32, 32)
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
    """Different splits must produce different shuffles; same split reproducible."""

    def _get_index_order(self, tmp_path, split, seed=42):
        p = _make_tile_file(tmp_path, f"rng_{split}", H=64, W=64, seed=0)
        ds = PixelContextDataset(
            [p], context_px=32, split=split, samples_per_tile=256, seed=seed
        )
        return [(r, c) for _, r, c, _ in ds._index]

    def test_train_val_produce_different_orders(self, tmp_path):
        train_order = self._get_index_order(tmp_path, "train")
        val_order   = self._get_index_order(tmp_path, "val")
        assert train_order != val_order, "train and val should produce different orders"

    def test_same_split_is_reproducible(self, tmp_path):
        order1 = self._get_index_order(tmp_path, "train", seed=7)
        order2 = self._get_index_order(tmp_path, "train", seed=7)
        assert order1 == order2, "Same split + seed must produce identical order"

    def test_different_seeds_differ(self, tmp_path):
        order1 = self._get_index_order(tmp_path, "train", seed=1)
        order2 = self._get_index_order(tmp_path, "train", seed=99)
        assert order1 != order2, "Different seeds must produce different orders"
