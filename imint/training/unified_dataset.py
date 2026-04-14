"""
imint/training/unified_dataset.py -- Unified PyTorch Dataset for LULC + Crop segmentation

Loads tiles from unified_v2/ producing 23-class per-pixel segmentation labels.
Labels are pre-built offline by build_labels.py (NMD 19-class + gated LPIS + gated SKS)
and stored directly in data["label"] — no runtime merging.

Tile sources (unified_v2/*.npz):
    spectral  (24, 256, 256) = 4 frames × 6 bands, reflectance [0, 1]
    label     (256, 256)     unified 23-class label (pre-built by build_labels.py)
    label_mask (256, 256)    raw SJV grödkoder (uint16, for reference)
    harvest_mask (256, 256)  binary SKS harvest indicator
    bbox_3006 (4,)           [west, south, east, north] in SWEREF 99 TM
    height/volume/basal_area/diameter/dem (256, 256) forest/terrain aux
    vpp_sosd/vpp_eosd (256, 256) phenology aux
    dates (4,), temporal_mask (4,), doy (4,)

Output dict (matches LULCDataset format for trainer.py compatibility):
    image:               (6, 224, 224) float32, single-date Prithvi-normalized
    label:               (224, 224) int64, unified 19-class per-pixel
    height:              (1, 224, 224) float32, z-score normalized
    volume:              (1, 224, 224) float32
    basal_area:          (1, 224, 224) float32
    diameter:            (1, 224, 224) float32
    dem:                 (1, 224, 224) float32
    vpp_sosd:            (1, 224, 224) float32
    vpp_eosd:            (1, 224, 224) float32
    harvest_probability: (1, 224, 224) float32
    metadata:            {"tile": str, "source": "lulc" | "crop"}

Unified 19-class schema (from unified_schema.py + harvest):
    0  bakgrund (ignore_index)
    1  tallskog            (NMD pine)
    2  granskog            (NMD spruce)
    3  lovskog             (NMD deciduous)
    4  blandskog           (NMD mixed)
    5  sumpskog            (NMD wetland forest)
    6  vatmark             (NMD open wetland)
    7  oppen mark          (NMD open land)
    8  bebyggelse          (NMD developed)
    9  vatten              (NMD water)
    10 vete                (LPIS wheat)
    11 korn                (LPIS barley)
    12 havre               (LPIS oats)
    13 oljevaxter          (LPIS rapeseed)
    14 vall                (LPIS ley/grass)
    15 potatis             (LPIS potato)
    16 trindsad            (LPIS pulses)
    17 ovrig aker          (LPIS other / unmapped cropland)
    18 hygge               (harvested forest)
"""
from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
except ImportError:
    raise ImportError(
        "PyTorch is required for training. Install with: pip install torch"
    )

from .unified_schema import NUM_UNIFIED_CLASSES, HARVEST_CLASS
from .losses import parcel_area_to_pixel_weights
from .sampler import _sweref99_to_wgs84

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Total classes: 0-22 (23 classes including hygge/harvest at index 22)
NUM_CLASSES = NUM_UNIFIED_CLASSES  # 23

# Prithvi-EO-2.0 normalization constants (DN scale, per band)
# Bands: B02, B03, B04, B8A, B11, B12
PRITHVI_MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32,
)
PRITHVI_STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32,
)

# DOY target for peak-summer frame selection (mid-July)
PEAK_SUMMER_DOY = 195

# Number of spectral bands per frame (Prithvi 6-band)
N_BANDS = 6

# Auxiliary channel z-score normalization: (mean, std)
# volume and basal_area use log(1+x) pre-transform (lognormal distributions)
AUX_NORM = {
    "height": (7.36, 6.55),
    "volume": (3.56, 1.14),          # log1p-transformed
    "basal_area": (2.42, 0.71),      # log1p-transformed
    "diameter": (16.33, 7.84),
    "dem": (264.03, 215.37),
    "vpp_sosd": (21130.90, 49.13),
    "vpp_eosd": (21280.29, 78.28),
    "vpp_length": (141.61, 41.39),
    "vpp_maxv": (0.88, 0.57),
    "vpp_minv": (0.04, 0.05),
    "harvest_probability": (0.1, 0.2),
}

# Channels that need log(1+x) pre-transform before z-score
AUX_LOG_TRANSFORM = {"volume", "basal_area"}

# Canonical ordered list of auxiliary channels
# Must match order from config.enabled_aux_names
AUX_CHANNEL_NAMES = [
    "height", "volume", "basal_area", "diameter", "dem",
    "vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv",
    "harvest_probability",
]

# Forest classes in the unified schema (eligible for harvest override)
_FOREST_CLASSES = frozenset({1, 2, 3, 4, 5})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnifiedDataset(Dataset):
    """PyTorch Dataset that loads both LULC and crop tiles for
    unified 19-class segmentation training.

    Merges NMD land-cover classes with LPIS crop detail and a
    harvest-detection class into a single per-pixel label space.
    Outputs are single-date, Prithvi-normalized, and compatible
    with the LULCDataset dict format expected by trainer.py.

    Args:
        lulc_dir: Directory containing LULC seasonal tiles
            (expected layout: ``<lulc_dir>/tiles/*.npz``).
        crop_dir: Directory containing crop tiles
            (expected layout: ``<crop_dir>/*.npz``).
        split: Dataset split -- ``"train"`` or ``"val"``.
            If ``split_<split>.txt`` files exist in the respective
            directories, they are used. Otherwise all tiles are loaded.
        patch_size: Output spatial dimension (default 224 for Prithvi).
        enable_aux: Whether to include auxiliary channels in output.
        augment_override: Explicit augmentation flag. If None (default),
            augmentation is enabled when ``split == "train"``.

    Raises:
        FileNotFoundError: If both tile directories are empty or missing.
    """

    def __init__(
        self,
        lulc_dir: str | Path | None = None,
        crop_dir: str | Path | None = None,
        split: str = "train",
        patch_size: int = 224,
        enable_aux: bool = True,
        augment_override: bool | None = None,
        multitemporal: bool = False,
        num_temporal_frames: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.enable_aux = enable_aux
        self.augment = (split == "train") if augment_override is None else augment_override
        self.multitemporal = multitemporal
        self.num_temporal_frames = num_temporal_frames

        # Prithvi normalization reshaped for broadcasting over (6, H, W)
        self._mean = PRITHVI_MEAN.reshape(N_BANDS, 1, 1)
        self._std = PRITHVI_STD.reshape(N_BANDS, 1, 1)

        # Discover tiles from both sources
        self._entries: list[dict] = []
        self._load_tile_list(lulc_dir, "lulc", split)
        self._load_tile_list(crop_dir, "crop", split)

        if not self._entries:
            raise FileNotFoundError(
                f"No tiles found for split='{split}'. "
                f"Searched lulc_dir={lulc_dir}, crop_dir={crop_dir}"
            )

        logger.info(
            "UnifiedDataset[%s]: %d tiles (%d LULC, %d crop)",
            split,
            len(self._entries),
            sum(1 for e in self._entries if e["source"] == "lulc"),
            sum(1 for e in self._entries if e["source"] == "crop"),
        )

    # ------------------------------------------------------------------
    # Tile discovery
    # ------------------------------------------------------------------

    def _load_tile_list(
        self,
        data_dir: str | Path | None,
        source: str,
        split: str,
    ) -> None:
        """Discover tiles from a directory and add them to ``_entries``.

        If a ``split_<split>.txt`` file is present, only listed tiles
        are included. Otherwise, all ``.npz`` files are loaded.

        Args:
            data_dir: Root directory for this tile source.
            source: ``"lulc"`` or ``"crop"`` tag.
            split: ``"train"`` or ``"val"``.
        """
        if data_dir is None:
            return

        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.warning("Tile directory not found: %s", data_dir)
            return

        # LULC tiles are under a 'tiles' subdirectory; crop tiles are
        # directly in the directory
        if source == "lulc":
            tiles_dir = data_dir / "tiles"
            if not tiles_dir.exists():
                tiles_dir = data_dir
        else:
            tiles_dir = data_dir

        # Check for a split file
        split_file = data_dir / f"split_{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                allowed = {line.strip() for line in f if line.strip()}
            tile_paths = sorted(
                p for p in tiles_dir.glob("*.npz")
                if p.name in allowed
            )
        else:
            # Deterministic 90/10 train/val split by MD5 hash of tile name.
            # This ensures consistent splits without needing split files.
            all_tiles = sorted(tiles_dir.glob("*.npz"))
            VAL_FRACTION = 10  # 10% val (every tile where hash % 100 < 10)
            if split == "val":
                tile_paths = [
                    p for p in all_tiles
                    if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 100 < VAL_FRACTION
                ]
            else:  # train
                tile_paths = [
                    p for p in all_tiles
                    if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 100 >= VAL_FRACTION
                ]

        for p in tile_paths:
            self._entries.append({
                "path": p,
                "source": source,
                "name": p.name,
            })

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tile_names(self) -> list[str]:
        """Tile filenames (for sampler compatibility with LULCDataset)."""
        return [e["name"] for e in self._entries]

    # ------------------------------------------------------------------
    # Core __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int, _retries: int = 0) -> dict:
        """Load a single tile and return the trainer-compatible dict.

        Returns:
            Dict with keys ``image``, ``label``, auxiliary channels,
            and ``metadata``.
        """
        entry = self._entries[idx]
        try:
            data = np.load(entry["path"], allow_pickle=True)
            if "spectral" not in data and "image" not in data:
                raise KeyError("missing spectral")
        except Exception:
            if _retries >= 50:
                raise RuntimeError(f"No valid tiles found after {_retries} retries from idx {idx}")
            alt = (idx + 1) % len(self._entries)
            return self.__getitem__(alt, _retries=_retries + 1)

        source = entry["source"]

        # --- Spectral image ---------------------------------------------
        if self.multitemporal:
            image, temporal_mask, doy = self._extract_all_frames(
                data, source, self.num_temporal_frames
            )
        else:
            if source == "lulc":
                image = self._extract_lulc_frame(data)
            else:
                image = self._extract_crop_frame(data)
            temporal_mask = None
            doy = None

        # --- Label construction ----------------------------------------
        label = self._build_label(data, source)

        # --- Auxiliary channels ----------------------------------------
        h, w = label.shape

        # Unified area map for inverse-area pixel weighting.
        # LPIS parcel area takes precedence for crop pixels; NMD connected-
        # component area fills in for all other labeled pixels (forest, water,
        # wetland, urban etc.).  Both are float32 ha-per-pixel rasters built
        # by build_labels.py.  Tiles that predate the NMD area stamp fall back
        # to zeros, which parcel_area_to_pixel_weights treats as weight=1.0.
        raw_lpis = data.get("parcel_area_ha", None)
        raw_nmd  = data.get("nmd_area_ha",    None)

        lpis_arr = (np.asarray(raw_lpis, dtype=np.float32)
                    if raw_lpis is not None else np.zeros((h, w), dtype=np.float32))
        nmd_arr  = (np.asarray(raw_nmd,  dtype=np.float32)
                    if raw_nmd  is not None else np.zeros((h, w), dtype=np.float32))

        if lpis_arr.shape != (h, w): lpis_arr = lpis_arr[:h, :w]
        if nmd_arr.shape  != (h, w): nmd_arr  = nmd_arr[:h, :w]

        # LPIS area where crop parcel exists, NMD component area everywhere else
        area_map = np.where(lpis_arr > 0, lpis_arr, nmd_arr)

        aux_stack = self._load_aux_channels(data, h, w) if self.enable_aux else None

        # Fold area_map into aux_stack as channel 0 for spatial consistency
        area_as_channel = area_map[np.newaxis]  # (1, H, W)
        if aux_stack is not None:
            aug_stack = np.concatenate([area_as_channel, aux_stack], axis=0)
        else:
            aug_stack = area_as_channel

        # --- Prithvi normalization: reflectance [0,1] -> DN -> z-score -
        # Normalize all T frames identically (mean/std tile across frames)
        n_frames = image.shape[0] // N_BANDS
        mean_t = np.tile(self._mean, (n_frames, 1, 1))  # (T*6, 1, 1)
        std_t = np.tile(self._std, (n_frames, 1, 1))
        image = (image * 10000.0 - mean_t) / std_t

        # --- Spatial augmentation / crop --------------------------------
        if self.augment:
            image, label, aug_stack = self._augment(image, label, aug_stack)
        else:
            image, label, aug_stack = self._center_crop(image, label, aug_stack)

        # Extract area_map back out (was channel 0); restore aux_stack
        area_map_cropped = aug_stack[0]                          # (H', W')
        aux_stack = aug_stack[1:] if self.enable_aux else None   # (N, H', W')

        # Compute per-pixel loss weights from cropped area map
        area_t = torch.from_numpy(np.ascontiguousarray(area_map_cropped))
        pixel_weight = parcel_area_to_pixel_weights(
            area_t,
            mmu_ha=getattr(self, "_mmu_ha", 0.25),
            max_weight=getattr(self, "_area_weight_max", 4.0),
        )

        # --- Prithvi TL coordinate tensors ------------------------------
        n_frames = image.shape[0] // N_BANDS
        temporal_coords, location_coords = self._build_coords(
            data, doy, n_frames,
        )

        # --- Build output dict ------------------------------------------
        result: dict = {
            "spectral":     torch.from_numpy(np.ascontiguousarray(image)),
            "label":        torch.from_numpy(np.ascontiguousarray(label)),
            "pixel_weight": pixel_weight,
            "temporal_coords": temporal_coords,
            "location_coords": location_coords,
            "metadata": {
                "tile":   entry["name"],
                "source": source,
            },
        }

        # Multitemporal metadata
        if temporal_mask is not None:
            result["temporal_mask"] = torch.from_numpy(
                np.ascontiguousarray(temporal_mask)
            )
        if doy is not None:
            result["doy"] = torch.from_numpy(np.ascontiguousarray(doy))

        # Attach each auxiliary channel as (1, H, W) tensor
        if aux_stack is not None:
            for i, ch_name in enumerate(AUX_CHANNEL_NAMES):
                result[ch_name] = torch.from_numpy(
                    np.ascontiguousarray(aux_stack[i:i + 1])
                )  # (1, H', W')

        return result

    # ------------------------------------------------------------------
    # Prithvi TL coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def _build_coords(
        data: dict,
        doy: np.ndarray | None,
        n_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Prithvi TL coordinate tensors from tile metadata.

        Returns:
            temporal_coords: (T, 2) float32 [year, doy] per frame.
            location_coords: (2,) float32 [lat, lon] in WGS84 degrees.
        """
        year = int(data.get("year", data.get("lpis_year", 0)))
        if year == 0:
            dates = data.get("dates", [])
            if len(dates) > 0:
                try:
                    year = int(str(dates[0])[:4])
                except (ValueError, IndexError):
                    year = 2022
            else:
                year = 2022

        temporal_coords = np.zeros((n_frames, 2), dtype=np.float32)
        temporal_coords[:, 0] = float(year)
        if doy is not None:
            temporal_coords[:len(doy), 1] = doy[:n_frames].astype(np.float32)

        easting = float(data.get("easting", 500_000))
        northing = float(data.get("northing", 6_500_000))
        lat, lon = _sweref99_to_wgs84(easting, northing)
        location_coords = np.array([lat, lon], dtype=np.float32)

        return (
            torch.from_numpy(temporal_coords),
            torch.from_numpy(location_coords),
        )

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_lulc_frame(data: np.lib.npyio.NpzFile) -> np.ndarray:
        """Select peak-summer frame from a 4-frame LULC tile.

        The ``image`` array has shape (24, H, W) = 4 frames x 6 bands.
        We pick the frame whose ``doy`` value is closest to
        ``PEAK_SUMMER_DOY`` (195, mid-July). Falls back to frame index 1
        if ``doy`` is missing or all zeros.

        Args:
            data: Loaded .npz file.

        Returns:
            (6, H, W) float32 single-date reflectance.
        """
        raw = data.get("spectral", data.get("image"))
        image = raw.astype(np.float32)  # (24, H, W)
        n_frames = image.shape[0] // N_BANDS

        # Determine best frame via day-of-year
        doy = data.get("doy", None)
        if doy is not None:
            doy = np.asarray(doy).ravel()
            if doy.shape[0] >= n_frames and np.any(doy > 0):
                # Pick frame closest to peak summer DOY
                valid_mask = doy[:n_frames] > 0
                if np.any(valid_mask):
                    distances = np.abs(doy[:n_frames].astype(np.float64) - PEAK_SUMMER_DOY)
                    distances[~valid_mask] = 9999
                    frame_idx = int(np.argmin(distances))
                else:
                    frame_idx = min(1, n_frames - 1)
            else:
                frame_idx = min(1, n_frames - 1)
        else:
            frame_idx = min(1, n_frames - 1)

        start = frame_idx * N_BANDS
        return image[start:start + N_BANDS]  # (6, H, W)

    @staticmethod
    def _extract_crop_frame(data: np.lib.npyio.NpzFile) -> np.ndarray:
        """Select peak-summer frame from a 3-frame crop tile.

        The ``spectral`` array has shape (18, H, W) = 3 frames x 6 bands.
        Frame index 1 corresponds to Jun-Jul (peak summer).

        Args:
            data: Loaded .npz file.

        Returns:
            (6, H, W) float32 single-date reflectance.
        """
        spectral = data["spectral"].astype(np.float32)  # (18, H, W)
        n_frames = spectral.shape[0] // N_BANDS

        # Prefer frame 1 (Jun-Jul); verify via seasons_valid if available
        frame_idx = min(1, n_frames - 1)
        seasons_valid = data.get("seasons_valid", None)
        if seasons_valid is not None:
            sv = np.asarray(seasons_valid).ravel()
            if sv.shape[0] >= n_frames and not sv[frame_idx]:
                # Frame 1 invalid -- pick first valid frame
                valid = np.where(sv[:n_frames])[0]
                if len(valid) > 0:
                    frame_idx = int(valid[0])

        start = frame_idx * N_BANDS
        return spectral[start:start + N_BANDS]  # (6, H, W)

    # ------------------------------------------------------------------
    # Label construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_label(data: np.lib.npyio.NpzFile, source: str) -> np.ndarray:
        """Return the pre-built unified 23-class label from the tile.

        Tiles built by build_labels.py store the fully-merged label in
        data["label"] (NMD + gated LPIS + gated SKS). Return it directly —
        no runtime merging needed.

        Args:
            data: Loaded .npz file.
            source: ``"lulc"`` or ``"crop"``.

        Returns:
            (H, W) int64 unified label array.
        """
        # Always use "label" — it contains the unified 23-class output from
        # build_labels.py.  "nmd_label" holds raw NMD indices (0-255) and
        # must NOT be used as training labels (causes CUDA assert on
        # values >= num_classes).
        if "label" not in data:
            img = data.get("spectral", data.get("image"))
            h, w = img.shape[1], img.shape[2]
            return np.zeros((h, w), dtype=np.int64)
        label = np.asarray(data["label"]).astype(np.int64)
        # Safety clamp: any out-of-range values → background (0)
        label[label >= NUM_CLASSES] = 0
        label[label < 0] = 0
        return label

    @staticmethod
    def _extract_all_frames(
        data: np.lib.npyio.NpzFile,
        source: str,
        num_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract all temporal frames for multi-temporal training.

        Returns:
            image: (T*6, H, W) float32 stacked frames.
            temporal_mask: (T,) uint8, 1 = valid frame, 0 = padded.
            doy: (T,) int32 day-of-year per frame.
        """
        raw = data.get("spectral", data.get("image")).astype(np.float32)
        c, h, w = raw.shape
        tile_frames = c // N_BANDS

        # Temporal metadata from tile
        tile_mask = data.get("temporal_mask", None)
        tile_doy = data.get("doy", None)
        if tile_mask is not None:
            tile_mask = np.asarray(tile_mask).ravel()[:tile_frames]
        else:
            tile_mask = np.ones(tile_frames, dtype=np.uint8)
        if tile_doy is not None:
            tile_doy = np.asarray(tile_doy).ravel()[:tile_frames].astype(np.int32)
        else:
            tile_doy = np.zeros(tile_frames, dtype=np.int32)

        if tile_frames >= num_frames:
            # Tile has enough frames — take first num_frames
            image = raw[:num_frames * N_BANDS]
            temporal_mask = tile_mask[:num_frames]
            doy = tile_doy[:num_frames]
        else:
            # Pad with zeros (and mask)
            image = np.zeros((num_frames * N_BANDS, h, w), dtype=np.float32)
            image[:tile_frames * N_BANDS] = raw
            temporal_mask = np.zeros(num_frames, dtype=np.uint8)
            temporal_mask[:tile_frames] = tile_mask
            doy = np.zeros(num_frames, dtype=np.int32)
            doy[:tile_frames] = tile_doy

        # Replace zero-padded frames with nearest valid frame
        for t in range(num_frames):
            if temporal_mask[t] == 0:
                # Find nearest valid frame
                valid_indices = np.where(temporal_mask > 0)[0]
                if len(valid_indices) > 0:
                    nearest = valid_indices[
                        np.argmin(np.abs(valid_indices - t))
                    ]
                    start_dst = t * N_BANDS
                    start_src = nearest * N_BANDS
                    image[start_dst:start_dst + N_BANDS] = (
                        image[start_src:start_src + N_BANDS]
                    )

        return image, temporal_mask, doy

    # ------------------------------------------------------------------
    # Auxiliary channels
    # ------------------------------------------------------------------

    @staticmethod
    def _load_aux_channels(
        data: np.lib.npyio.NpzFile,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Load, zero-fill, and z-score-normalize auxiliary channels.

        Args:
            data: Loaded .npz file.
            h: Spatial height of label array.
            w: Spatial width of label array.

        Returns:
            (N, H, W) float32 stacked aux channels, normalized.
        """
        aux_arrays: list[np.ndarray] = []
        for ch_name in AUX_CHANNEL_NAMES:
            if ch_name in data:
                arr = data[ch_name].astype(np.float32)
                # Ensure spatial dimensions match
                if arr.shape != (h, w):
                    arr = arr[:h, :w]
                    # Pad if smaller (edge case)
                    if arr.shape[0] < h or arr.shape[1] < w:
                        padded = np.zeros((h, w), dtype=np.float32)
                        padded[:arr.shape[0], :arr.shape[1]] = arr
                        arr = padded
            else:
                arr = np.zeros((h, w), dtype=np.float32)
            aux_arrays.append(arr)

        aux_stack = np.stack(aux_arrays, axis=0)  # (N, H, W)

        # Log-transform skewed channels, then z-score normalize
        for i, ch_name in enumerate(AUX_CHANNEL_NAMES):
            if ch_name in AUX_LOG_TRANSFORM:
                aux_stack[i] = np.log1p(np.maximum(aux_stack[i], 0.0))
            if ch_name in AUX_NORM:
                mean, std = AUX_NORM[ch_name]
                aux_stack[i] = (aux_stack[i] - mean) / max(std, 1e-6)

        return aux_stack

    # ------------------------------------------------------------------
    # Augmentation and cropping
    # ------------------------------------------------------------------

    def _augment(
        self,
        image: np.ndarray,
        label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Apply data augmentation (training only).

        Augmentations (all applied consistently to image, label, aux):
            - Random crop from fetch size (256) to patch size (224)
            - Random horizontal flip (p=0.5)
            - Random vertical flip (p=0.5)
            - Random 90-degree rotation (k in {0, 1, 2, 3})

        Args:
            image: (C, H, W) float32.
            label: (H, W) int64.
            aux: Optional (N, H, W) float32 stacked aux channels.

        Returns:
            Tuple of (image, label, aux) after augmentation.
        """
        # Random crop
        image, label, aux = self._random_crop(image, label, aux)

        # Random horizontal flip
        if random.random() > 0.5:
            image = image[:, :, ::-1]
            label = label[:, ::-1]
            if aux is not None:
                aux = aux[:, :, ::-1]

        # Random vertical flip
        if random.random() > 0.5:
            image = image[:, ::-1, :]
            label = label[::-1, :]
            if aux is not None:
                aux = aux[:, ::-1, :]

        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))
            if aux is not None:
                aux = np.rot90(aux, k, axes=(1, 2))

        return image, label, aux

    def _random_crop(
        self,
        image: np.ndarray,
        label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Random spatial crop to ``patch_size``.

        Args:
            image: (C, H, W).
            label: (H, W).
            aux: Optional (N, H, W).

        Returns:
            Cropped (image, label, aux).
        """
        p = self.patch_size
        _, h, w = image.shape
        if h <= p and w <= p:
            return image, label, aux

        y = random.randint(0, max(h - p, 0))
        x = random.randint(0, max(w - p, 0))

        a_crop = aux[:, y:y + p, x:x + p] if aux is not None else None
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p], a_crop

    def _center_crop(
        self,
        image: np.ndarray,
        label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Center spatial crop to ``patch_size``.

        Args:
            image: (C, H, W).
            label: (H, W).
            aux: Optional (N, H, W).

        Returns:
            Cropped (image, label, aux).
        """
        p = self.patch_size
        _, h, w = image.shape
        if h <= p and w <= p:
            return image, label, aux

        y = max((h - p) // 2, 0)
        x = max((w - p) // 2, 0)

        a_crop = aux[:, y:y + p, x:x + p] if aux is not None else None
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p], a_crop


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_unified_class_weights(
    dataset: UnifiedDataset,
    max_weight: float = 10.0,
    max_tiles: int | None = None,
) -> np.ndarray:
    """Compute inverse-frequency class weights by scanning all tiles.

    Loads each tile's label, counts per-class pixels, and returns
    weights inversely proportional to class frequency.  Weights are
    capped at ``max_weight`` to prevent training instability on very
    rare classes.  Background (class 0) always receives weight 0.

    This can be slow for large datasets.  Use ``max_tiles`` to limit
    the scan to a random subset for faster approximation.

    Args:
        dataset: A UnifiedDataset instance.
        max_weight: Maximum per-class weight (default 10.0).
        max_tiles: If set, scan at most this many tiles (randomly
            selected) instead of the full dataset.

    Returns:
        (NUM_CLASSES,) float32 array of class weights.  Index 0 is
        background (weight = 0).
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    # Determine which tiles to scan
    indices = list(range(len(dataset)))
    if max_tiles is not None and max_tiles < len(indices):
        rng = random.Random(42)
        indices = rng.sample(indices, max_tiles)

    for idx in indices:
        entry = dataset._entries[idx]
        try:
            data = np.load(entry["path"], allow_pickle=True)
        except Exception:
            continue

        label = UnifiedDataset._build_label(data, entry["source"])
        classes, pixel_counts = np.unique(label, return_counts=True)
        for cls, cnt in zip(classes, pixel_counts):
            if 0 <= cls < NUM_CLASSES:
                counts[cls] += cnt

    # Inverse-frequency weighting
    total = counts.sum()
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for i in range(1, NUM_CLASSES):
        if counts[i] > 0:
            w = total / (NUM_CLASSES * counts[i])
            weights[i] = min(w, max_weight)
        else:
            weights[i] = max_weight  # Never-seen class gets max weight

    weights[0] = 0.0  # Ignore background
    return weights


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

def build_unified_sampler(
    dataset: UnifiedDataset,
    max_tile_weight: float = 5.0,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples tiles containing
    rare classes.

    For each tile, we check which classes are present and assign the
    tile a weight equal to the maximum inverse-frequency weight of its
    constituent classes.  This ensures tiles with rare classes (e.g.
    harvest, potato, pulses) are drawn more often.

    Args:
        dataset: A UnifiedDataset instance.
        max_tile_weight: Cap on per-tile sampling weight to prevent
            extreme oversampling of any single tile.

    Returns:
        A ``WeightedRandomSampler`` suitable for ``DataLoader(sampler=...)``.
    """
    n_tiles = len(dataset)
    class_tile_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    tile_classes: list[set[int]] = []

    # First pass: discover which classes each tile contains
    for entry in dataset._entries:
        try:
            data = np.load(entry["path"], allow_pickle=True)
            label = UnifiedDataset._build_label(data, entry["source"])
            present = set(np.unique(label).tolist())
        except Exception:
            present = set()

        tile_classes.append(present)
        for cls in present:
            if 0 < cls < NUM_CLASSES:
                class_tile_counts[cls] += 1

    # Compute inverse tile-frequency per class
    class_tile_counts = np.maximum(class_tile_counts, 1.0)
    class_rarity = n_tiles / (NUM_CLASSES * class_tile_counts)

    # Second pass: assign per-tile weight = max rarity of present classes
    sample_weights: list[float] = []
    for present in tile_classes:
        if not present:
            sample_weights.append(1.0)
            continue
        w = max(
            class_rarity[cls] for cls in present
            if 0 < cls < NUM_CLASSES
        ) if any(0 < cls < NUM_CLASSES for cls in present) else 1.0
        sample_weights.append(min(float(w), max_tile_weight))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_tiles,
        replacement=True,
    )
