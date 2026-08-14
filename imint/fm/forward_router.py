"""imint/fm/forward_router.py — per-family model forward routing.

The training loop (`imint.training.trainer`) and the evaluation loop
(`imint.training.evaluate`) both need to turn a dataset batch dict into a
model forward call. Each foundation-model family has a different forward
signature:

    prithvi / tessera : model(x, aux=, temporal_coords=, location_coords=,
                              return_fractions=)
                        — x is the (B, 6, T, H, W) Conv3d stack (prithvi)
                          or the (B, 128, H, W) embedding (tessera), both
                          carried in batch["spectral"].
    croma             : model(sar=, optical=, aux=, output_size=,
                              return_fractions=)
                        — sar from batch["s1_vv_vh"], optical from
                          batch["s2_croma"] (raw reflectance stacks).
    clay              : model(chips=, timestamps=, wavelengths=, aux=,
                              return_fractions=)
                        — chips from batch["s2_clay"].
    terramind         : model({"S2L2A":.., "S1GRD":..}, aux=,
                              return_fractions=)
                        — S2L2A is the raw 6-band frame; S1GRD from
                          batch["s1_vv_vh"].

Rather than duplicate this routing in two loops (drift risk), both import
``family_forward`` from here. Routing is on the model family string — never
on tensor shape.

Normalization: CROMA/Clay/TerraMind consume RAW reflectance / linear-σ⁰
tensors from the dataset (the dataset emits the per-model stacks unscaled).
Their encoders were pretrained on normalized inputs, so this router applies
the matching normalizer from ``imint.fm.normalize`` before the encoder call.
Prithvi/Tessera keep their existing behaviour byte-for-byte (the dataset
already z-scores Prithvi reflectance; Tessera embeddings ship normalized) —
this router does NOT touch their inputs.
"""
from __future__ import annotations

import torch


# Clay's full 10-band S2 order (matches build_s2_clay_tensor default).
_CLAY_BANDS = (
    "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "swir16", "swir22",
)


def _prithvi_input(spectral: torch.Tensor) -> torch.Tensor:
    """(B, T*6, H, W) or (B, 6, H, W) → (B, 6, T, H, W) Conv3d layout."""
    B, CT, H, W = spectral.shape
    if CT > 6:
        T = CT // 6
        return spectral.view(B, T, 6, H, W).permute(0, 2, 1, 3, 4)
    return spectral.unsqueeze(2)


def family_forward(
    model: torch.nn.Module,
    family: str,
    batch: dict,
    device: torch.device,
    *,
    aux: torch.Tensor | None,
    temporal_coords: torch.Tensor | None = None,
    location_coords: torch.Tensor | None = None,
    return_fractions: bool = False,
):
    """Run ``model`` on a dataset batch, routing on ``family``.

    Args:
        model: The segmentation model (its ``fm_spec.family`` == ``family``).
        family: "prithvi" | "tessera" | "croma" | "clay" | "terramind".
        batch: Dataset batch dict. Must carry the keys the family needs
            (``spectral`` for prithvi/tessera; ``s2_croma``/``s1_vv_vh`` for
            croma; ``s2_clay`` for clay; ``spectral``/``s1_vv_vh`` for
            terramind).
        device: Target device.
        aux: Pre-collected (B, n_aux, H, W) aux stack or None.
        temporal_coords, location_coords: Prithvi TL coords (ignored by the
            single-date families).
        return_fractions: Forwarded to the model so the Trädslag frac head
            (when enabled) also returns fraction logits.

    Returns:
        ``logits`` (B, C, H, W), or ``(logits, frac_logits)`` when
        ``return_fractions`` and the frac head is enabled.
    """
    if family in ("prithvi", "tessera"):
        spectral = batch["spectral"].to(device)
        model_input = spectral if family == "tessera" else _prithvi_input(spectral)
        return model(
            model_input, aux=aux,
            temporal_coords=temporal_coords,
            location_coords=location_coords,
            return_fractions=return_fractions,
        )

    if family == "croma":
        from imint.fm.normalize import CromaNormalizer
        optical = batch["s2_croma"].to(device).float()   # (B, 12, H, W) raw
        sar = batch["s1_vv_vh"].to(device).float()       # (B, 2, H, W) linear
        norm = _get_normalizer(model, "croma", CromaNormalizer, device)
        n = norm({"s2_full": optical, "s1": sar})
        return model(
            sar=n["s1"], optical=n["s2_full"], aux=aux,
            return_fractions=return_fractions,
        )

    if family == "clay":
        from imint.fm.normalize import ClayNormalizer  # noqa: F401 (parity)
        from imint.fm.loaders.clay import get_clay_wavelengths, get_clay_norm
        chips = batch["s2_clay"].to(device).float()      # (B, 10, H, W) raw
        B = chips.shape[0]
        # Clay's dynamic embedding takes explicit per-band wavelengths and
        # DN-scale normalization (x*10000 - mean)/std over the 10-band stack.
        mean, std = get_clay_norm(_CLAY_BANDS)            # (10,), (10,)
        mean = mean.view(1, -1, 1, 1).to(device)
        std = std.view(1, -1, 1, 1).to(device)
        chips = (chips * 10000.0 - mean) / std
        wls = get_clay_wavelengths(_CLAY_BANDS).to(device).unsqueeze(0).expand(B, -1)
        # timestamps: (B, 4) [week, hour, lat, lon]. Use location coords when
        # available; zeros otherwise (Clay tolerates unknown time/place).
        ts = torch.zeros(B, 4, device=device)
        if location_coords is not None:
            lc = location_coords.to(device)
            ts[:, 2] = lc[:, 0]
            ts[:, 3] = lc[:, 1]
        return model(
            chips=chips, timestamps=ts, wavelengths=wls, aux=aux,
            return_fractions=return_fractions,
        )

    if family == "terramind":
        from imint.fm.normalize import TerraMindNormalizer
        # S2L2A: the dataset emits the RAW 6-band frame under `s2_terramind`
        # (un-z-scored, Prithvi-order == TerraMind's band order); S1GRD is the
        # raw linear-σ⁰ 2-band frame. TerraMind's own normalizer scales both.
        s2_raw = batch["s2_terramind"].to(device).float()  # (B, 6, H, W)
        sar = batch["s1_vv_vh"].to(device).float()          # (B, 2, H, W)
        norm = _get_normalizer(model, "terramind", TerraMindNormalizer, device)
        n = norm({"s2": s2_raw, "s1": sar})
        return model(
            {"S2L2A": n["s2"], "S1GRD": n["s1"]}, aux=aux,
            return_fractions=return_fractions,
        )

    raise ValueError(f"Unknown backbone family {family!r} in forward router.")


def _get_normalizer(model, family, cls, device):
    """Cache one normalizer instance on the model so buffers move with .to()."""
    attr = f"_norm_{family}"
    norm = getattr(model, attr, None)
    if norm is None:
        norm = cls().to(device)
        setattr(model, attr, norm)
    return norm
