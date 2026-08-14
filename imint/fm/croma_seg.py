"""
imint/fm/croma_seg.py — CROMA segmentation wrapper.

CROMA forward returns a dict:
    SAR_encodings:      (B, N, D) — S1 patches
    optical_encodings:  (B, N, D) — S2 patches
    joint_encodings:    (B, N, D) — fused (modality='both' only)
    plus *_GAP globally-pooled vectors we ignore.

For dense segmentation we use ``joint_encodings`` when both modalities
are available, falling back to ``optical_encodings`` otherwise. N =
(img_size/8)² tokens. Linear-probe head follows the same pattern as
TerraMindSegmentationModel / ClaySegmentationModel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CromaSegmentationModel(nn.Module):
    """Linear-probe segmentation head on top of CROMA encodings.

    Args:
        encoder: PretrainedCROMA instance from ``load_croma()``.
        num_classes: Output classes.
        img_size: Input resolution. Must be multiple of patch_size=8.
        patch_size: 8 (CROMA-base and CROMA-large both).
        embed_dim: 768 (base) or 1024 (large).
        modality: Which encoding to use: 'joint' (S1+S2 fused),
            'optical' (S2 only), or 'sar' (S1 only).
        n_aux_channels: Optional aux raster channels fused at output res.
        dropout: Dropout before classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 23,
        img_size: int = 120,
        patch_size: int = 8,
        embed_dim: int = 768,
        modality: str = "joint",
        n_aux_channels: int = 0,
        dropout: float = 0.1,
        enable_tradslag_head: bool = False,
        num_tradslag: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.modality = modality
        self.n_aux_channels = n_aux_channels
        self.enable_tradslag_head = enable_tradslag_head
        self.num_tradslag = num_tradslag

        grid = img_size // patch_size
        if grid * patch_size != img_size:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={patch_size}"
            )
        self.grid_size = grid
        self.expected_n_patches = grid * grid

        if modality not in ("joint", "optical", "sar"):
            raise ValueError(
                f"modality={modality!r}; must be 'joint', 'optical', or 'sar'."
            )

        # Multi-level UPerNet head (fairness parity with Prithvi). CROMA's
        # cross_encoder (joint SAR-optical) exposes a 6-block ModuleList
        # (``cross_encoder.layers``), so we hook 4 evenly-spaced blocks and
        # feed the SAME UPerNet decoder Prithvi uses — not a linear probe.
        from imint.fm.upernet import ViTUPerNetHead, get_default_pool_sizes
        pool_sizes = get_default_pool_sizes(
            device=None, img_size=img_size, patch_size=patch_size,
        )
        self.decoder_head = ViTUPerNetHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            decoder_channels=256,
            dropout=dropout,
            n_aux_channels=n_aux_channels,
            pool_sizes=pool_sizes,
            enable_tradslag_head=enable_tradslag_head,
            num_tradslag=num_tradslag,
        )

    def _pick_encoding(self, enc_dict: dict) -> torch.Tensor:
        """Select the token sequence per ``self.modality``, with a
        graceful fallback when joint encodings are unavailable."""
        key_map = {
            "joint": "joint_encodings",
            "optical": "optical_encodings",
            "sar": "SAR_encodings",
        }
        primary = key_map[self.modality]
        if primary in enc_dict:
            return enc_dict[primary]
        # Fallbacks
        if self.modality == "joint":
            for fb in ("optical_encodings", "SAR_encodings"):
                if fb in enc_dict:
                    return enc_dict[fb]
        raise KeyError(
            f"CROMA encoder did not return any usable encoding. "
            f"Wanted {primary!r}; got keys {list(enc_dict)}."
        )

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        if N != self.expected_n_patches:
            raise ValueError(
                f"CROMA returned {N} tokens; expected "
                f"{self.expected_n_patches} (grid {self.grid_size}²). "
                f"Check img_size/patch_size."
            )
        if D != self.embed_dim:
            raise ValueError(
                f"CROMA returned embed_dim={D}; expected {self.embed_dim}."
            )
        return tokens.transpose(1, 2).reshape(
            B, D, self.grid_size, self.grid_size,
        )

    def forward(
        self,
        sar: torch.Tensor | None = None,
        optical: torch.Tensor | None = None,
        aux: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        return_fractions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Per-pixel class logits from CROMA encodings.

        Args:
            sar: (B, 2, H, W) S1 VV/VH. Required for modality in
                {'sar', 'joint'}.
            optical: (B, 12, H, W) S2 12-band stack from
                build_s2_croma_tensor. Required for modality in
                {'optical', 'joint'}.
            aux: Optional (B, n_aux, Ho, Wo) at the desired output res.
            output_size: Optional (H, W) for the output logits. Defaults
                to the input sar/optical tensor's (H, W).
            return_fractions: When True AND the fraction head is enabled,
                also return the (B, num_tradslag, Ho, Wo) fraction logits
                computed from the SAME feature that feeds the classifier.
                Head disabled → second element is ``None``. Default False →
                logits only (byte-identical to the pre-frac signature).

        Returns:
            (B, num_classes, Ho, Wo) logits, or ``(logits, frac_logits)``
            when ``return_fractions`` is True.
        """
        if self.modality in ("joint", "sar") and sar is None:
            raise ValueError(f"CROMA modality={self.modality!r} requires 'sar'.")
        if self.modality in ("joint", "optical") and optical is None:
            raise ValueError(f"CROMA modality={self.modality!r} requires 'optical'.")

        ref = optical if optical is not None else sar
        if output_size is None:
            output_size = tuple(ref.shape[-2:])

        # Collect 4 evenly-spaced multi-level features from the joint stream.
        # CROMA's cross_encoder (SAR↔optical fusion) is a 6-block ModuleList
        # at ``cross_encoder.layers``; we hook it, run the encoder once, and
        # feed the 4 captured block token-sequences into the UPerNet head.
        # Fake encoders (tests) with no cross_encoder fall back to replicating
        # the single joint encoding to 4 levels.
        block_tokens = self._extract_multi_level(sar, optical)
        feats = [self._tokens_to_spatial(t) for t in block_tokens]  # 4× (B,D,g,g)
        return self.decoder_head(
            feats, output_size=output_size, aux=aux,
            return_fractions=return_fractions,
        )

    def _extract_multi_level(
        self, sar: torch.Tensor | None, optical: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Return 4 evenly-spaced (B, N, D) block token-sequences.

        Hooks the joint cross-attention stream (``cross_encoder.layers``) so
        the multi-level features carry fused SAR+optical at increasing depth —
        the same information the linear probe's ``joint_encodings`` used, but
        at 4 depths for the UPerNet pyramid. Falls back to the final joint
        encoding replicated ×4 when no hookable block list is present.
        """
        layers = None
        cross = getattr(self.encoder, "cross_encoder", None)
        if cross is not None and hasattr(cross, "layers"):
            layers = cross.layers

        if layers is not None and len(layers) >= 4:
            L = len(layers)
            idxs = [L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1, L - 1]
            idxs = [max(0, i) for i in idxs]
            captured: dict[int, torch.Tensor] = {}
            handles = []

            def mk(slot):
                def hook(mod, inp, out):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    captured[slot] = t
                return hook

            for slot, i in enumerate(idxs):
                handles.append(layers[i].register_forward_hook(mk(slot)))
            try:
                _ = self.encoder(SAR_images=sar, optical_images=optical)
            finally:
                for h in handles:
                    h.remove()
            if len(captured) == 4:
                return [captured[s] for s in range(4)]

        # Fallback: single final joint encoding → 4 levels.
        enc = self.encoder(SAR_images=sar, optical_images=optical)
        tokens = self._pick_encoding(enc)
        return [tokens, tokens, tokens, tokens]
