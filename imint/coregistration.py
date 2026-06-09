"""
imint/coregistration.py — Multi-temporal image co-registration

Reusable alignment utilities for any multi-temporal analysis pipeline:
change detection, LSTM networks, time-series compositing, etc.

Two levels of alignment:

1. **Integer-pixel alignment** — deterministic, grid-based. Computes
   the pixel offset between two raster transforms and crops/places
   arrays to their overlapping region. No reference image needed.

2. **Sub-pixel alignment** — image-based. Uses phase correlation on a
   reference band to detect fractional-pixel shifts caused by
   Sentinel-2 orbit geometry and reprojection, then corrects via
   Fourier phase shift (sinc interpolation).

Typical usage in an analyzer::

    from imint.coregistration import coregister_to_reference

    # Align a multi-band target stack to a reference stack
    aligned_target, aligned_ref, meta = coregister_to_reference(
        target=current_stack,          # (H, W, C) float
        reference=baseline_stack,      # (H, W, C) float
        target_transform=cur_geo,      # Affine transform list [a,b,c,d,e,f]
        reference_transform=bl_geo,    # Affine transform list [a,b,c,d,e,f]
        subpixel=True,                 # enable sub-pixel refinement
        reference_band=2,              # band index for phase correlation
    )

For fetch-level grid snapping (no reference image, just transforms)::

    from imint.coregistration import (
        compute_grid_offset, subpixel_shift, align_arrays,
    )
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
#  Integer-pixel alignment (transform-based, deterministic)
# ---------------------------------------------------------------------------

def compute_grid_offset(
    current_transform: list | tuple,
    baseline_transform: list | tuple,
    pixel_size: float = 10.0,
) -> tuple[int, int]:
    """Compute integer pixel offset between two raster grids.

    Args:
        current_transform:  Affine transform of the current image
                            [a, b, c, d, e, f] where c=x-origin, f=y-origin.
        baseline_transform: Affine transform of the baseline/reference image.
        pixel_size:         Pixel size in metres (default 10 m for Sentinel-2).

    Returns:
        (drow, dcol) — integer pixel offset of the baseline origin relative
        to the current image origin.
        Positive drow → baseline origin is *south* of current origin.
        Positive dcol → baseline origin is *east* of current origin.
    """
    cur_x0, cur_y0 = current_transform[2], current_transform[5]
    bl_x0, bl_y0 = baseline_transform[2], baseline_transform[5]

    dcol = round((bl_x0 - cur_x0) / pixel_size)
    drow = round((cur_y0 - bl_y0) / pixel_size)
    return drow, dcol


def align_arrays(
    current: np.ndarray,
    baseline: np.ndarray,
    drow: int,
    dcol: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Crop *current* and *baseline* to their overlapping region.

    Works for 2-D (H, W) and 3-D (H, W, C) arrays.

    Args:
        current:  Current image array.
        baseline: Baseline/reference image array.
        drow:     Row offset from :func:`compute_grid_offset`.
        dcol:     Column offset from :func:`compute_grid_offset`.

    Returns:
        (current_crop, baseline_crop, row_start, col_start)
        where row_start/col_start indicate where the crop begins in
        the *current* image's pixel coordinates.
    """
    ch, cw = current.shape[:2]
    bh, bw = baseline.shape[:2]

    # Overlap in current-image pixel coordinates
    cur_r0 = max(0, drow)
    cur_c0 = max(0, dcol)
    cur_r1 = min(ch, bh + drow)
    cur_c1 = min(cw, bw + dcol)

    # Corresponding region in baseline pixel coordinates
    bl_r0 = max(0, -drow)
    bl_c0 = max(0, -dcol)
    bl_r1 = bl_r0 + (cur_r1 - cur_r0)
    bl_c1 = bl_c0 + (cur_c1 - cur_c0)

    if current.ndim == 3:
        return (
            current[cur_r0:cur_r1, cur_c0:cur_c1, :],
            baseline[bl_r0:bl_r1, bl_c0:bl_c1, :],
            cur_r0, cur_c0,
        )
    return (
        current[cur_r0:cur_r1, cur_c0:cur_c1],
        baseline[bl_r0:bl_r1, bl_c0:bl_c1],
        cur_r0, cur_c0,
    )


# ---------------------------------------------------------------------------
#  Sub-pixel alignment (image-based, needs reference)
# ---------------------------------------------------------------------------

def subpixel_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift a 2-D image by sub-pixel amounts using Fourier phase shift.

    Equivalent to sinc interpolation — the most accurate resampling for
    band-limited signals like satellite imagery.

    Args:
        image: 2-D (H, W) float array.
        dy:    Sub-pixel shift in row direction (fractional pixels).
        dx:    Sub-pixel shift in column direction (fractional pixels).

    Returns:
        Shifted 2-D array of the same shape and dtype.
    """
    if abs(dy) < 1e-6 and abs(dx) < 1e-6:
        return image

    h, w = image.shape
    freq_y = np.fft.fftfreq(h)
    freq_x = np.fft.fftfreq(w)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing="ij")

    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    shifted = np.real(np.fft.ifft2(np.fft.fft2(image) * phase))
    return shifted.astype(image.dtype)


def estimate_subpixel_offset(
    current_band: np.ndarray,
    reference_band: np.ndarray,
    window_frac: float = 0.3,
    max_peak_px: float | None = 1.0,
) -> tuple[float, float]:
    """Estimate sub-pixel offset between two co-registered images.

    Uses phase correlation (cross-power spectrum) with parabolic peak
    fitting to detect fractional-pixel shifts between Sentinel-2
    acquisitions from different orbits or dates.

    The result is the *residual* shift AFTER integer-pixel alignment.

    Args:
        current_band:   2-D (H, W) float array — single band from the
                        current (target) image.
        reference_band: 2-D (H, W) float array — matching band from the
                        reference image.
        window_frac:    Fraction of image centre to use for correlation
                        (avoids edge artefacts).
        max_peak_px:    Reject the correlation as spurious (return ``0,0``)
                        when the integer peak exceeds this many pixels on
                        either axis. ``1.0`` (default) suits callers that
                        already did integer alignment from transforms, where
                        the residual must be sub-pixel. Pass a larger budget
                        (e.g. the halo width) — or ``None`` to disable — when
                        no integer alignment preceded the call, so genuine
                        multi-pixel drift is returned instead of dropped.

    Returns:
        (dy, dx) — fractional-pixel offset of the reference relative to
        the current image.  Apply ``subpixel_shift(reference, -dy, -dx)``
        to align the reference to the current image.
    """
    h, w = current_band.shape

    # Central window to avoid edge effects and focus on stable features
    cy, cx = h // 2, w // 2
    wh = max(32, int(h * window_frac))
    ww = max(32, int(w * window_frac))
    crop_c = current_band[cy - wh:cy + wh, cx - ww:cx + ww].astype(np.float64)
    crop_r = reference_band[cy - wh:cy + wh, cx - ww:cx + ww].astype(np.float64)

    # Hann window to reduce spectral leakage
    win_y = np.hanning(crop_c.shape[0])
    win_x = np.hanning(crop_c.shape[1])
    window = np.outer(win_y, win_x)
    crop_c = (crop_c - crop_c.mean()) * window
    crop_r = (crop_r - crop_r.mean()) * window

    # Phase correlation
    fc = np.fft.fft2(crop_c)
    fr = np.fft.fft2(crop_r)
    cross_power = (fc * np.conj(fr)) / (np.abs(fc * np.conj(fr)) + 1e-10)
    corr = np.real(np.fft.ifft2(cross_power))

    # Integer peak
    peak = np.unravel_index(corr.argmax(), corr.shape)
    py, px = peak

    # Wrap negative offsets (FFT convention)
    if py > corr.shape[0] // 2:
        py -= corr.shape[0]
    if px > corr.shape[1] // 2:
        px -= corr.shape[1]

    # Parabolic fitting for sub-pixel precision
    def _parabolic_peak(arr, idx, size):
        i0 = idx % size
        im1 = (idx - 1) % size
        ip1 = (idx + 1) % size
        v0, vm1, vp1 = arr[i0], arr[im1], arr[ip1]
        denom = 2.0 * (2.0 * v0 - vm1 - vp1)
        if abs(denom) < 1e-10:
            return 0.0
        return (vm1 - vp1) / denom

    sub_dy = _parabolic_peak(
        corr[:, peak[1] % corr.shape[1]], peak[0], corr.shape[0]
    )
    sub_dx = _parabolic_peak(
        corr[peak[0] % corr.shape[0], :], peak[1], corr.shape[1]
    )

    dy = float(py) + sub_dy
    dx = float(px) + sub_dx

    # Reject a spurious peak as a no-op. Transform-aligned callers (the
    # default) require a sub-pixel residual, so a >1 px peak means the
    # correlation mis-latched. Callers that skip integer alignment raise
    # ``max_peak_px`` so real multi-pixel drift survives (``None`` disables it).
    if max_peak_px is not None and (abs(py) > max_peak_px or abs(px) > max_peak_px):
        print(
            f"    [coreg] WARNING: phase-correlation peak ({py},{px}) exceeds "
            f"max_peak_px={max_peak_px} — ignoring (likely spurious)"
        )
        return 0.0, 0.0

    return dy, dx


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 48) -> float:
    """Mutual information between two equal-shape arrays via a joint histogram.

    ``MI = Σ p(a,b) · log[ p(a,b) / (p(a)·p(b)) ]`` — high when the two images'
    intensities are statistically dependent, regardless of the actual intensity
    *mapping*. That invariance is why it survives appearance change (different
    season, sensor, or illumination) where intensity-difference metrics fail.
    """
    hist, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    p = hist / hist.sum()
    pa = p.sum(axis=1, keepdims=True)
    pb = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (pa @ pb)[nz])))


def estimate_mi_offset(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    search_px: float = 4.0,
    bins: int = 48,
    window_frac: float = 0.55,
) -> tuple[float, float]:
    """Sub-pixel shift that registers ``moving`` onto ``reference`` by maximising
    mutual information — robust to content/appearance change.

    For multi-temporal Sentinel-2 the frames span seasons (autumn stubble vs
    summer canopy), so the same ground point looks radiometrically different and
    phase correlation latches onto phenology, not geometry. MI scores the
    *statistical dependence* of the joint histogram, so a sub-pixel shift that
    lines the two frames up geometrically maximises it regardless of how crop
    brightness changed. A continuous optimiser (Powell) over ``(dy, dx)``, each
    trial applied with the same Fourier ``subpixel_shift`` used elsewhere, gives
    a precise optimum (≈0.05 px on a synthetic season-change fixture, vs ≈0.75 px
    for phase correlation on the same structured imagery).

    Args:
        moving:      2-D frame to register.
        reference:   2-D frame defining the target position.
        search_px:   reject (return ``0,0``) if the optimum exceeds this on
                     either axis — a wrap-/halo-safety bound.
        bins:        joint-histogram bins.
        window_frac: central window fraction scored (avoids edge wrap).

    Returns:
        ``(dy, dx)`` — the shift to **apply** to ``moving`` (via
        :func:`subpixel_shift`) to register it onto ``reference``; ``(0,0)`` if
        the optimiser does not improve on no shift or the optimum is out of
        range.
    """
    from scipy.optimize import minimize

    h, w = moving.shape
    cy, cx = h // 2, w // 2
    wh = max(48, int(h * window_frac / 2))
    ww = max(48, int(w * window_frac / 2))
    sl = (slice(cy - wh, cy + wh), slice(cx - ww, cx + ww))
    ref_win = np.asarray(reference, np.float64)[sl]
    mov = np.asarray(moving, np.float64)

    base_mi = _mutual_information(ref_win, mov[sl], bins)

    def neg_mi(s: np.ndarray) -> float:
        if abs(s[0]) > search_px or abs(s[1]) > search_px:
            return 0.0  # 0 > any −MI, so the optimiser is repelled from out-of-range
        shifted = subpixel_shift(mov, float(s[0]), float(s[1]))
        return -_mutual_information(ref_win, shifted[sl], bins)

    res = minimize(neg_mi, np.zeros(2), method="Powell",
                   options={"xtol": 0.02, "ftol": 1e-4})
    dy, dx = float(res.x[0]), float(res.x[1])
    # Reject: no MI gain over no-shift, or the optimum sits at the search wall
    # (the true drift wants to go further than ``search_px`` — the halo can't
    # absorb it and a clamped partial shift would mis-register), → keep M1.
    if -res.fun <= base_mi or max(abs(dy), abs(dx)) >= 0.95 * search_px:
        return 0.0, 0.0
    return dy, dx


# ---------------------------------------------------------------------------
#  High-level co-registration pipeline
# ---------------------------------------------------------------------------

def coregister_to_reference(
    target: np.ndarray,
    reference: np.ndarray,
    target_transform: list | tuple | None = None,
    reference_transform: list | tuple | None = None,
    pixel_size: float = 10.0,
    subpixel: bool = True,
    reference_band: int = 2,
    subpixel_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Co-register a target image to a reference image.

    Combines integer-pixel alignment (from transforms) and optional
    sub-pixel refinement (from phase correlation) in a single call.

    This is the main entry point for any multi-temporal analysis that
    needs aligned image pairs: change detection, LSTM time-series,
    multi-date compositing, etc.

    Args:
        target:              (H, W) or (H, W, C) target image array.
        reference:           (H, W) or (H, W, C) reference image array.
        target_transform:    Affine transform [a,b,c,d,e,f] of target.
                             If None, integer alignment is skipped.
        reference_transform: Affine transform [a,b,c,d,e,f] of reference.
                             If None, integer alignment is skipped.
        pixel_size:          Grid pixel size in metres (default 10 m).
        subpixel:            Whether to apply sub-pixel refinement.
        reference_band:      Band index to use for phase correlation
                             (default 2 = B04/Red in CHANGE_BANDS order).
        subpixel_threshold:  Minimum sub-pixel offset to correct (pixels).

    Returns:
        (aligned_target, aligned_reference, metadata) where metadata is::

            {
                "integer_offset": (drow, dcol),
                "subpixel_offset": (dy, dx),
                "aligned_shape": (H, W),
                "crop_origin": (row0, col0),  # in target coords
                "original_shape": (H, W),     # target's original shape
            }
    """
    full_shape = target.shape[:2]
    drow, dcol = 0, 0
    row0, col0 = 0, 0
    sub_dy, sub_dx = 0.0, 0.0

    # ── Integer-pixel alignment ──────────────────────────────────────
    if target_transform is not None and reference_transform is not None:
        drow, dcol = compute_grid_offset(
            target_transform, reference_transform, pixel_size
        )
        if drow != 0 or dcol != 0:
            target, reference, row0, col0 = align_arrays(
                target, reference, drow, dcol
            )
            print(
                f"    [coreg] Integer offset: drow={drow} dcol={dcol} "
                f"({drow * pixel_size:.0f}m / {dcol * pixel_size:.0f}m) → "
                f"overlap {target.shape[0]}×{target.shape[1]}"
            )

    # ── Sub-pixel alignment ──────────────────────────────────────────
    if subpixel and target.shape[:2] == reference.shape[:2]:
        try:
            # Pick band for correlation
            if target.ndim == 3 and target.shape[2] > reference_band:
                cur_band = target[..., reference_band]
                ref_band = reference[..., reference_band]
            elif target.ndim == 3:
                cur_band = target[..., 0]
                ref_band = reference[..., 0]
            else:
                cur_band = target
                ref_band = reference

            sub_dy, sub_dx = estimate_subpixel_offset(cur_band, ref_band)

            if abs(sub_dy) > subpixel_threshold or abs(sub_dx) > subpixel_threshold:
                print(
                    f"    [coreg] Sub-pixel shift: "
                    f"dy={sub_dy:+.3f} dx={sub_dx:+.3f} px "
                    f"({sub_dy * pixel_size:+.1f}m / {sub_dx * pixel_size:+.1f}m)"
                )
                # Apply correction to each band of the reference
                if reference.ndim == 3:
                    for b in range(reference.shape[2]):
                        reference[..., b] = subpixel_shift(
                            reference[..., b], -sub_dy, -sub_dx
                        )
                else:
                    reference = subpixel_shift(reference, -sub_dy, -sub_dx)
                print(f"    [coreg] Sub-pixel correction applied")
            else:
                sub_dy, sub_dx = 0.0, 0.0
        except Exception as e:
            print(f"    [coreg] Sub-pixel alignment skipped: {e}")
            sub_dy, sub_dx = 0.0, 0.0

    meta = {
        "integer_offset": (drow, dcol),
        "subpixel_offset": (sub_dy, sub_dx),
        "aligned_shape": target.shape[:2],
        "crop_origin": (row0, col0),
        "original_shape": full_shape,
    }

    return target, reference, meta


def coregister_timeseries(
    images: list[np.ndarray],
    transforms: list[list | tuple] | None = None,
    reference_idx: int = 0,
    pixel_size: float = 10.0,
    subpixel: bool = True,
    reference_band: int = 2,
) -> tuple[list[np.ndarray], dict]:
    """Co-register a time-series of images to a common reference.

    Useful for LSTM networks, multi-date compositing, and any analysis
    that processes a temporal stack of images.

    Args:
        images:        List of (H, W, C) or (H, W) arrays.
        transforms:    List of Affine transforms matching ``images``.
                       If None, only sub-pixel alignment is performed.
        reference_idx: Index of the reference image in ``images``.
        pixel_size:    Grid pixel size in metres.
        subpixel:      Whether to apply sub-pixel refinement.
        reference_band: Band index for phase correlation.

    Returns:
        (aligned_images, metadata) where aligned_images are all cropped
        to the common overlap region.
    """
    n = len(images)
    if n == 0:
        return [], {}
    if n == 1:
        return images, {"n_images": 1, "offsets": [(0, 0)]}

    reference = images[reference_idx].copy()
    ref_transform = transforms[reference_idx] if transforms else None

    aligned = [None] * n
    aligned[reference_idx] = reference
    offsets = [(0, 0)] * n
    subpixel_offsets = [(0.0, 0.0)] * n

    for i in range(n):
        if i == reference_idx:
            continue
        tgt_transform = transforms[i] if transforms else None
        aligned_tgt, aligned_ref, meta = coregister_to_reference(
            target=images[i].copy(),
            reference=reference.copy(),
            target_transform=tgt_transform,
            reference_transform=ref_transform,
            pixel_size=pixel_size,
            subpixel=subpixel,
            reference_band=reference_band,
        )
        aligned[i] = aligned_tgt
        offsets[i] = meta["integer_offset"]
        subpixel_offsets[i] = meta["subpixel_offset"]

        # Update reference to the aligned version (common region)
        if meta["integer_offset"] != (0, 0):
            reference = aligned_ref

    meta = {
        "n_images": n,
        "reference_idx": reference_idx,
        "offsets": offsets,
        "subpixel_offsets": subpixel_offsets,
    }

    return aligned, meta
