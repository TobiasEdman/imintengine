"""
imint/fm/clay_seg.py — Clay v1.5 segmentation wrapper.

Clay's documented ``model.encoder(chips, timestamps, wavelengths)`` returns
a pooled ``(B, 1024)`` embedding per image — useful for classification
and retrieval, but not for per-pixel segmentation.

For dense prediction we hook into the encoder's final transformer block
BEFORE the pooling to grab the (B, N+1, D) token sequence. At the
native Clay config (256 px input, patch_size=8) this gives N = 32*32 =
1024 patch tokens plus one CLS token. We drop the CLS, reshape the
rest to a (B, D=1024, 32, 32) spatial feature map, then run a linear-
probe head to predict per-pixel class logits.

Architecture (same philosophy as TerraMindSegmentationModel):
    1. encoder hook → pre-pool tokens (B, N+1, 1024)
    2. drop CLS, reshape → (B, 1024, grid, grid)
    3. 2x ConvTranspose up (1024→512, 512→256)
    4. 3x3 smooth → 128
    5. optional aux fusion
    6. 1x1 classifier → num_classes
    7. bilinear resize to input resolution

If mIoU disappoints, swap in a multi-level hook (hook blocks 5/11/17/23)
to feed a real UPerNet.

Usage:
    from imint.fm.clay_seg import ClaySegmentationModel
    model = ClaySegmentationModel(
        encoder=clay_encoder,        # from loader.load_clay
        num_classes=23,
        img_size=256,
        patch_size=8,
        embed_dim=1024,
    )
    logits = model(
        chips=s2_clay_tensor,        # (B, 10, H, W)
        wavelengths=wls,             # (B, 10) in nanometers
        timestamps=timestamps,       # (B, 4): [week, hour, lat, lon]
        aux=None,
    )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClaySegmentationModel(nn.Module):
    """Hook-based linear-probe segmentation head on top of a Clay encoder.

    Args:
        encoder: Clay encoder from ``load_clay()``. Expected to be a
            ViT with a ``.blocks`` (or equivalent) module list.
        num_classes: Output classes.
        img_size: Input image size. Must be divisible by patch_size.
        patch_size: ViT patch size (Clay v1.5 uses 8).
        embed_dim: Encoder hidden size (Clay v1.5: 1024).
        n_aux_channels: Auxiliary raster channels, concatenated at
            output resolution before the classifier. 0 = no aux.
        dropout: Dropout before classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 23,
        img_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 1024,
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
        self.n_aux_channels = n_aux_channels
        self.enable_tradslag_head = enable_tradslag_head
        self.num_tradslag = num_tradslag

        grid = img_size // patch_size
        if grid * patch_size != img_size:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={patch_size}"
            )
        self.grid_size = grid
        self.expected_n_patches = grid * grid  # 32×32 = 1024 at native

        # Multi-level UPerNet head (fairness parity with Prithvi). Clay's
        # encoder.transformer.layers is a 24-block ModuleList, so we hook 4
        # evenly-spaced blocks and feed the SAME UPerNet decoder Prithvi uses
        # — not a linear probe.
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

    def _extract_tokens(
        self,
        chips: torch.Tensor,
        timestamps: torch.Tensor,
        wavelengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return 4 evenly-spaced (B, 1+L, D) block token-sequences.

        Clay v1.5's real ``Encoder.forward(datacube)`` takes a dict and
        returns a tuple whose first element is the encoded patch sequence
        ``(B, 1+L, D)`` (CLS + all patch tokens). It must be loaded with
        ``mask_ratio=0`` (see load_clay) so ALL patches are returned; the
        default MAE mask drops ~75%, which cannot be reshaped to a grid.
        We hook 4 evenly-spaced blocks of ``encoder.transformer.layers``
        (24-block ModuleList) for the multi-level UPerNet head.

        A fake encoder in tests may instead expose a ``.blocks`` ModuleList
        and a positional ``forward(chips, timestamps, wavelengths)``. The
        real-vs-fake path is chosen by whether the encoder accepts a
        datacube dict (has a DynamicEmbedding-style ``patch_embedding``).
        """
        # load_clay returns the ClayMAE wrapper; the datacube-forward
        # Encoder is at ``.encoder``. Unwrap it (a bare Encoder is used
        # directly). Fake test encoders expose neither and fall through to
        # the .blocks hook path below.
        real_enc = self.encoder
        if not (hasattr(real_enc, "patch_embedding")
                or hasattr(real_enc, "transformer")):
            inner = getattr(real_enc, "encoder", None)
            if inner is not None and (
                hasattr(inner, "patch_embedding")
                or hasattr(inner, "transformer")
            ):
                real_enc = inner

        # Real Clay encoder path: build the datacube dict.
        if hasattr(real_enc, "patch_embedding") or hasattr(
            real_enc, "transformer"
        ):
            # Clay's Encoder.add_encodings builds an 8-dim metadata vector
            # from ``hstack((time, latlon))`` and concatenates it onto a
            # positional encoding of dim ``self.dim - 8``. That arithmetic
            # requires time and latlon to be 4-wide EACH (total 8) — a 2+2
            # split yields dim-4 metadata and a 1020≠1024 mismatch (observed).
            # Clay's convention: time=[week_sin,week_cos,hour_sin,hour_cos],
            # latlon=[lat_sin,lat_cos,lon_sin,lon_cos]. We derive sin/cos from
            # the (B,4)=[week,hour,lat,lon] timestamps; unknown → zeros, which
            # Clay tolerates (metadata is additive, not gating).
            B = chips.shape[0]
            week = timestamps[:, 0]
            hour = timestamps[:, 1]
            lat = timestamps[:, 2]
            lon = timestamps[:, 3]
            time = torch.stack([
                torch.sin(week), torch.cos(week),
                torch.sin(hour), torch.cos(hour),
            ], dim=1)                                       # (B, 4)
            latlon = torch.stack([
                torch.sin(lat), torch.cos(lat),
                torch.sin(lon), torch.cos(lon),
            ], dim=1)                                       # (B, 4)
            gsd = torch.tensor(10.0, device=chips.device)  # Sentinel-2 10 m
            datacube = {
                "pixels": chips,
                "time": time,
                "latlon": latlon,
                "waves": wavelengths[0] if wavelengths.dim() == 2 else wavelengths,
                "gsd": gsd,
            }
            # Hook 4 evenly-spaced transformer blocks for the multi-level
            # UPerNet head (transformer.layers is a 24-block ModuleList).
            tr = getattr(real_enc, "transformer", None)
            layers = getattr(tr, "layers", None) if tr is not None else None
            if layers is not None and len(layers) >= 4:
                levels = self._hook_and_run(
                    layers, lambda: real_enc(datacube),
                )
                if levels is not None:
                    return levels
            # Fallback: single final encoding → 4 levels.
            out = real_enc(datacube)
            tokens = out[0] if isinstance(out, (tuple, list)) else out
            return [tokens, tokens, tokens, tokens]

        # Fake-encoder fallback (tests): hook .blocks/.layers if present.
        blocks = None
        for attr in ("blocks", "layers"):
            if hasattr(self.encoder, attr):
                blocks = getattr(self.encoder, attr)
                break
        if blocks is None:
            raise AttributeError(
                "Clay encoder exposes neither a datacube forward "
                "(patch_embedding/transformer) nor a hookable .blocks/"
                ".layers ModuleList. Inspect the encoder structure."
            )
        if len(blocks) >= 4:
            levels = self._hook_and_run(
                blocks, lambda: self.encoder(chips, timestamps, wavelengths),
                no_grad=True,
            )
            if levels is not None:
                return levels
        # Single-block fake → replicate its output ×4.
        captured: dict[str, torch.Tensor] = {}

        def hook(module, inputs, output):
            captured["t"] = output[0] if isinstance(output, tuple) else output

        h = blocks[-1].register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = self.encoder(chips, timestamps, wavelengths)
        finally:
            h.remove()
        if "t" not in captured:
            raise RuntimeError("Clay hook captured no tokens.")
        t = captured["t"]
        return [t, t, t, t]

    def _hook_and_run(self, layers, run_fn, no_grad: bool = False):
        """Hook 4 evenly-spaced blocks in ``layers``, run ``run_fn`` once,
        return the 4 captured (B, N, D) token-sequences (or None on miss)."""
        L = len(layers)
        idxs = [L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1, L - 1]
        idxs = [max(0, i) for i in idxs]
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def mk(slot):
            def hook(mod, inp, out):
                captured[slot] = out[0] if isinstance(out, (tuple, list)) else out
            return hook

        for slot, i in enumerate(idxs):
            handles.append(layers[i].register_forward_hook(mk(slot)))
        try:
            if no_grad:
                with torch.no_grad():
                    _ = run_fn()
            else:
                _ = run_fn()
        finally:
            for h in handles:
                h.remove()
        if len(captured) == 4:
            return [captured[s] for s in range(4)]
        return None

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N+1, D) or (B, N, D) → (B, D, grid, grid)."""
        B, N, D = tokens.shape
        if N == self.expected_n_patches + 1:
            tokens = tokens[:, 1:, :]  # drop CLS
        elif N != self.expected_n_patches:
            raise ValueError(
                f"Clay encoder returned {N} tokens; expected "
                f"{self.expected_n_patches} (grid {self.grid_size}²) or "
                f"{self.expected_n_patches + 1} (with CLS) for "
                f"img_size={self.img_size} patch_size={self.patch_size}. "
                f"Verify chips were passed at native resolution."
            )
        if D != self.embed_dim:
            raise ValueError(
                f"Clay encoder returned embed_dim={D}; expected {self.embed_dim}."
            )
        return tokens.transpose(1, 2).reshape(
            B, D, self.grid_size, self.grid_size,
        )

    def forward(
        self,
        chips: torch.Tensor,
        timestamps: torch.Tensor,
        wavelengths: torch.Tensor,
        aux: torch.Tensor | None = None,
        return_fractions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Per-pixel class logits from Clay encoder features.

        Args:
            chips: (B, n_bands, H, W) stacked Sentinel-2 tensor from
                ``build_s2_clay_tensor``. n_bands=10 for the default
                Clay spec; Clay's dynamic embedding block accepts
                variable band counts as long as ``wavelengths`` matches.
            timestamps: (B, 4) tensor [week, hour, lat, lon]. Pass
                zeros if time/location unknown.
            wavelengths: (B, n_bands) tensor of per-band central
                wavelength in nanometers.
            aux: Optional (B, n_aux, H, W) auxiliary raster channels.
            return_fractions: When True AND the fraction head is enabled,
                also return the (B, num_tradslag, H, W) fraction logits from
                the SAME feature that feeds the classifier. Head disabled →
                second element is ``None``. Default False → logits only.

        Returns:
            (B, num_classes, H, W) logits, or ``(logits, frac_logits)`` when
            ``return_fractions`` is True.
        """
        input_h, input_w = chips.shape[-2:]

        block_tokens = self._extract_tokens(chips, timestamps, wavelengths)
        feats = [self._tokens_to_spatial(t) for t in block_tokens]  # 4× (B,D,g,g)
        return self.decoder_head(
            feats, output_size=(input_h, input_w), aux=aux,
            return_fractions=return_fractions,
        )
