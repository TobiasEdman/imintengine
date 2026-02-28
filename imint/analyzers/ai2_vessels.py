"""Allen AI (rslearn) Sentinel-2 vessel detection analyzer.

Implements the detection model from the rslearn_projects repository
(https://github.com/allenai/rslearn_projects) using only standard
PyTorch / torchvision components so that it works on Python 3.9+.

The model architecture:
  Swin V2 B (9-ch input) → FPN → Faster R-CNN

Input bands (in order):  B04, B03, B02, B05, B06, B07, B08, B11, B12
Normalization:
  - Bands 0–2 (B04/B03/B02): divide by 3000, clip to [0, 1]
  - Bands 3–8 (B05–B12):     divide by 8160, clip to [0, 1]

Inference uses 512×512 sliding windows with 10 % overlap.
"""

from __future__ import annotations

import collections
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
from torchvision.models.detection import rpn as _rpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    TwoMLPHead,
)
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

from imint.analyzers.base import AnalysisResult, BaseAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_FM_DIR = Path(__file__).resolve().parent.parent / "fm" / "ai2_vessels"
_DEFAULT_CKPT = _FM_DIR / "detect_best.ckpt"
_DEFAULT_ATTR_CKPT = _FM_DIR / "attribute_best.ckpt"

# ---------------------------------------------------------------------------
# Model hyper-parameters (from rslearn config.yaml)
# ---------------------------------------------------------------------------
_INPUT_CHANNELS = 9
_SWIN_OUTPUT_LAYERS = [1, 3, 5, 7]
_FPN_IN_CHANNELS = [128, 256, 512, 1024]
_FPN_OUT_CHANNELS = 128
_ANCHOR_SIZES = [[32], [64], [128], [256]]
_NUM_CLASSES = 2  # unknown + vessel
_PATCH_SIZE = 512
_OVERLAP_RATIO = 0.1
_SCORE_THRESHOLD = 0.5  # lower than training (0.8) to capture more candidates

# Band order expected by the model:
#   B04, B03, B02, B05, B06, B07, B08, B11, B12
BAND_ORDER = ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]

# Normalization constants per band group
_NORM_RGB_STD = 3000.0    # for B04, B03, B02 (indices 0,1,2)
_NORM_SWIR_STD = 8160.0   # for B05-B12 (indices 3-8)

# Attribute model constants
_ATTR_NORM_STD = 10000.0  # attribute model normalizes ALL bands by /10000
_CROP_SIZE = 128  # 128×128 crop around each detection
_ATTR_SCALE_FACTOR = 0.01  # model outputs raw → multiply by 1/scale to get real values
SHIP_TYPES = [
    "cargo", "tanker", "passenger", "service", "tug",
    "pleasure", "fishing", "enforcement", "sar",
]


# ===================================================================
# Standalone model builder (no rslearn dependency)
# ===================================================================

class _SwinFeatureExtractor(torch.nn.Module):
    """Swin V2 B backbone that returns multi-scale feature maps.

    Matches rslearn.models.swin.Swin with output_layers=[1,3,5,7].
    """

    def __init__(self, input_channels: int = 9) -> None:
        super().__init__()
        self.model = torchvision.models.swin_v2_b()
        # Replace first conv to accept *input_channels* instead of 3
        if input_channels != 3:
            self.model.features[0][0] = torch.nn.Conv2d(
                input_channels,
                self.model.features[0][0].out_channels,
                kernel_size=(4, 4),
                stride=(4, 4),
            )
        self.output_layers = _SWIN_OUTPUT_LAYERS

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return feature maps at the requested layers.

        Args:
            x: (B, C, H, W) tensor

        Returns:
            list of feature maps, each (B, C_i, H_i, W_i)
        """
        layer_features: list[torch.Tensor] = []
        for layer in self.model.features:
            x = layer(x)
            # Swin outputs (B, H, W, C) — permute to (B, C, H, W)
            layer_features.append(x.permute(0, 3, 1, 2))
        return [layer_features[i] for i in self.output_layers]


class _FPN(torch.nn.Module):
    """Thin wrapper around torchvision FPN."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=out_channels,
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        inp = collections.OrderedDict(
            [(f"feat{i}", f) for i, f in enumerate(features)]
        )
        out = self.fpn(inp)
        return list(out.values())


class _NoopTransform(torch.nn.Module):
    """Matches rslearn.models.faster_rcnn.NoopTransform."""

    def __init__(self) -> None:
        super().__init__()
        self.transform = GeneralizedRCNNTransform(
            min_size=800, max_size=800, image_mean=[], image_std=[]
        )

    def forward(self, images, targets):
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(img.shape[1], img.shape[2]) for img in images]
        image_list = ImageList(images, image_sizes)
        return image_list, targets


class _FasterRCNNHead(torch.nn.Module):
    """Faster R-CNN detection head matching rslearn config."""

    def __init__(
        self,
        num_channels: int = 128,
        num_classes: int = 2,
        anchor_sizes: list[list[int]] | None = None,
        box_score_thresh: float = 0.05,
    ) -> None:
        super().__init__()
        if anchor_sizes is None:
            anchor_sizes = _ANCHOR_SIZES

        n_scales = len(anchor_sizes)
        featmap_names = [f"feat{i}" for i in range(n_scales)]

        self.noop_transform = _NoopTransform()

        aspect_ratios = ((0.5, 1.0, 2.0),) * n_scales
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = _rpn.RPNHead(
            num_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )
        self.rpn = _rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=2000),
            post_nms_top_n=dict(training=2000, testing=2000),
            nms_thresh=0.7,
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names, output_size=7, sampling_ratio=2
        )
        box_head = TwoMLPHead(
            num_channels * box_roi_pool.output_size[0] ** 2, 1024
        )
        box_predictor = FastRCNNPredictor(1024, num_classes)
        self.roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=box_score_thresh,
            nms_thresh=0.5,
            detections_per_img=100,
        )

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        raw_images: list[torch.Tensor],
    ) -> list[dict]:
        """Run detection on feature maps.

        Args:
            feature_maps: list of FPN outputs [(B,C,H_i,W_i), ...]
            raw_images:   list of per-image tensors for computing image sizes

        Returns:
            list of dicts with 'boxes', 'labels', 'scores'
        """
        images, _ = self.noop_transform(raw_images, None)

        feature_dict = collections.OrderedDict()
        for i, fm in enumerate(feature_maps):
            feature_dict[f"feat{i}"] = fm

        proposals, _ = self.rpn(images, feature_dict, None)
        detections, _ = self.roi_heads(
            feature_dict, proposals, images.image_sizes, None
        )
        return detections


class AI2VesselDetector(torch.nn.Module):
    """Full detection model: Swin V2 B → FPN → Faster R-CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.swin = _SwinFeatureExtractor(input_channels=_INPUT_CHANNELS)
        self.fpn = _FPN(in_channels=_FPN_IN_CHANNELS, out_channels=_FPN_OUT_CHANNELS)
        self.head = _FasterRCNNHead(
            num_channels=_FPN_OUT_CHANNELS,
            num_classes=_NUM_CLASSES,
            box_score_thresh=_SCORE_THRESHOLD,
        )

    def forward(self, x: torch.Tensor) -> list[dict]:
        """Detect vessels in a batch of normalised 9-band images.

        Args:
            x: (B, 9, H, W) float tensor (already normalised to [0,1])

        Returns:
            list of dicts per image with 'boxes' (N,4), 'labels' (N,), 'scores' (N,)
        """
        features = self.swin(x)
        fpn_out = self.fpn(features)
        raw_images = [x[i] for i in range(x.shape[0])]
        return self.head(fpn_out, raw_images)


# ===================================================================
# Checkpoint loading
# ===================================================================

def _load_detection_model(ckpt_path: str | Path) -> AI2VesselDetector:
    """Load detection model from a PyTorch Lightning checkpoint.

    The rslearn checkpoint stores keys like:
      model.encoder.0.model.features.0.0.weight   →  swin.model.features.0.0.weight
      model.encoder.1.fpn.inner_blocks.0.weight    →  fpn.fpn.inner_blocks.0.weight
      model.decoders.detect.0.rpn.*                →  head.rpn.*
      model.decoders.detect.0.roi_heads.*          →  head.roi_heads.*
      model.decoders.detect.0.noop_transform.*     →  head.noop_transform.*
    """
    ckpt_path = Path(ckpt_path)
    logger.info("Loading AI2 detection checkpoint from %s", ckpt_path)

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    # Remap keys from rslearn → our module
    remap = {
        "model.encoder.0.model.": "swin.model.",
        "model.encoder.1.fpn.": "fpn.fpn.",
        "model.decoders.detect.0.": "head.",
    }

    new_state = {}
    for key, val in state.items():
        new_key = key
        for src_prefix, dst_prefix in remap.items():
            if new_key.startswith(src_prefix):
                new_key = dst_prefix + new_key[len(src_prefix):]
                break
        new_state[new_key] = val

    model = AI2VesselDetector()
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing[:10])
    if unexpected:
        logger.debug("Unexpected keys in checkpoint: %s", unexpected[:10])

    model.eval()
    return model


# ===================================================================
# Attribute predictor model (SatlasPretrain + PoolingDecoder heads)
# ===================================================================

class _PoolingDecoder(torch.nn.Module):
    """Decoder that pools multi-scale features into a flat vector.

    Matches rslearn.models.pooling_decoder.PoolingDecoder exactly:
      Conv(in→conv_ch, 3×3) → ReLU  (× num_conv_layers)
      → GlobalMaxPool
      → Linear(prev→fc_ch) → ReLU   (× num_fc_layers)
      → Linear(fc_ch→out_ch)

    Only uses the **last** feature map from the encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int = 0,
        num_fc_layers: int = 0,
        conv_channels: int = 128,
        fc_channels: int = 512,
    ) -> None:
        super().__init__()
        conv_layers = []
        prev_ch = in_channels
        for _ in range(num_conv_layers):
            conv_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(prev_ch, conv_channels, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
            )
            prev_ch = conv_channels
        self.conv_layers = torch.nn.Sequential(*conv_layers)

        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(prev_ch, fc_channels),
                    torch.nn.ReLU(inplace=True),
                )
            )
            prev_ch = fc_channels
        self.fc_layers = torch.nn.Sequential(*fc_layers)

        self.output_layer = torch.nn.Linear(prev_ch, out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (B, C, H, W) — the last FPN feature map.

        Returns:
            (B, out_channels) tensor.
        """
        x = self.conv_layers(features)
        x = torch.amax(x, dim=(2, 3))       # global max pool → (B, C)
        x = self.fc_layers(x)
        return self.output_layer(x)


class AI2AttributePredictor(torch.nn.Module):
    """Vessel attribute prediction using SatlasPretrain + 6 PoolingDecoder heads.

    Predicts: length (m), width (m), speed (knots), heading (deg), ship type.

    Architecture (from rslearn_projects config):
      Encoder:  SatlasPretrain("Sentinel2_SwinB_SI_MS", fpn=True)
      Decoders: 6 × PoolingDecoder(128→{1|9}, conv=1, fc=2)
        - length, width, speed, heading_x, heading_y: out=1 (regression)
        - ship_type: out=9 (classification)
    """

    HEAD_NAMES = ["length", "width", "speed", "heading_x", "heading_y", "ship_type"]

    def __init__(self) -> None:
        super().__init__()
        try:
            import satlaspretrain_models
        except ImportError:
            raise ImportError(
                "satlaspretrain_models is required for attribute prediction. "
                "Install with: pip install satlaspretrain_models"
            )

        weights_manager = satlaspretrain_models.Weights()
        self.encoder = weights_manager.get_pretrained_model(
            model_identifier="Sentinel2_SwinB_SI_MS",
            fpn=True,
            device="cpu",
        )

        # 6 decoder heads (matching rslearn config exactly)
        self.decoders = torch.nn.ModuleDict()
        for name in self.HEAD_NAMES:
            out_ch = 9 if name == "ship_type" else 1
            self.decoders[name] = _PoolingDecoder(
                in_channels=128,
                out_channels=out_ch,
                num_conv_layers=1,
                num_fc_layers=2,
                conv_channels=128,
                fc_channels=512,
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict vessel attributes from a batch of 9-band crops.

        Args:
            x: (B, 9, 128, 128) float tensor normalised by /10000, clipped [0,1].

        Returns:
            dict mapping head name → (B, out_ch) tensor.
        """
        feature_maps = self.encoder(x)         # list of FPN feature maps
        last_fm = feature_maps[-1]             # smallest scale, (B, 128, H', W')
        return {name: head(last_fm) for name, head in self.decoders.items()}


def _load_attribute_model(ckpt_path: str | Path) -> AI2AttributePredictor:
    """Load attribute model from a PyTorch Lightning checkpoint.

    The rslearn checkpoint stores keys like:
      model.encoder.0.model.backbone.backbone.*  → encoder.backbone.backbone.*
      model.encoder.0.model.fpn.*                → encoder.fpn.*
      model.decoders.{head}.0.conv_layers.*      → decoders.{head}.conv_layers.*
      model.decoders.{head}.0.fc_layers.*        → decoders.{head}.fc_layers.*
      model.decoders.{head}.0.output_layer.*     → decoders.{head}.output_layer.*
    """
    ckpt_path = Path(ckpt_path)
    logger.info("Loading AI2 attribute checkpoint from %s", ckpt_path)

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)

    # Build remap prefixes
    remap = {"model.encoder.0.model.": "encoder."}
    for head in AI2AttributePredictor.HEAD_NAMES:
        remap[f"model.decoders.{head}.0."] = f"decoders.{head}."

    new_state = {}
    skipped = 0
    for key, val in state.items():
        new_key = key
        matched = False
        for src, dst in remap.items():
            if new_key.startswith(src):
                new_key = dst + new_key[len(src):]
                matched = True
                break
        if not matched:
            skipped += 1
            continue
        new_state[new_key] = val

    if skipped:
        logger.debug("Skipped %d non-model keys from attribute checkpoint", skipped)

    model = AI2AttributePredictor()
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logger.warning("Missing keys in attribute model: %s", missing[:10])
    if unexpected:
        logger.debug("Unexpected keys in attribute model: %s", unexpected[:10])

    model.eval()
    return model


# ===================================================================
# Attribute inference helpers
# ===================================================================

def _normalize_bands_attr(arr: np.ndarray) -> np.ndarray:
    """Normalize a (9, H, W) array for the attribute model.

    All bands divided by 10000, clipped to [0, 1].
    (Different from the detection model which uses split normalisation.)
    """
    return np.clip(arr.astype(np.float32) / _ATTR_NORM_STD, 0.0, 1.0)


def _crop_centered(image: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """Extract a (C, size, size) crop centred at (cy, cx), zero-padding edges.

    Args:
        image: (C, H, W) array.
        cy, cx: centre pixel in image coords.
        size: crop size.

    Returns:
        (C, size, size) array.
    """
    c, h, w = image.shape
    half = size // 2
    y0 = cy - half
    x0 = cx - half

    crop = np.zeros((c, size, size), dtype=image.dtype)

    # Source region (clipped to image bounds)
    sy0 = max(0, y0)
    sx0 = max(0, x0)
    sy1 = min(h, y0 + size)
    sx1 = min(w, x0 + size)

    # Destination region in the crop
    dy0 = sy0 - y0
    dx0 = sx0 - x0
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)

    crop[:, dy0:dy1, dx0:dx1] = image[:, sy0:sy1, sx0:sx1]
    return crop


def _predict_attributes(
    model: AI2AttributePredictor,
    image_9ch_dn: np.ndarray,
    detections: list[dict],
    device: str = "cpu",
) -> list[dict]:
    """Run attribute prediction on detected vessels.

    Args:
        model: loaded attribute model (eval mode).
        image_9ch_dn: (9, H, W) raw DN image (0–10000 range).
        detections: list of detection dicts with x_min/y_min/x_max/y_max.
        device: torch device.

    Returns:
        list of attribute dicts (same order as detections), each containing:
            length_m, width_m, speed_knots, heading_deg, vessel_type, type_confidence
    """
    if not detections:
        return []

    # Normalise for attribute model (all bands /10000)
    image_norm = _normalize_bands_attr(image_9ch_dn)

    # Prepare batch of crops
    crops = []
    for det in detections:
        cy = (det["y_min"] + det["y_max"]) // 2
        cx = (det["x_min"] + det["x_max"]) // 2
        crop = _crop_centered(image_norm, cy, cx, _CROP_SIZE)
        crops.append(crop)

    batch = torch.from_numpy(np.stack(crops, axis=0)).to(device)  # (N, 9, 128, 128)

    # Run inference (in batches of 16 to avoid OOM)
    all_attrs = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            sub = batch[i:i + batch_size]
            preds = model(sub)

            for j in range(sub.shape[0]):
                # Regression heads: model output / scale_factor → real value
                length_m = float(preds["length"][j, 0]) / _ATTR_SCALE_FACTOR
                width_m = float(preds["width"][j, 0]) / _ATTR_SCALE_FACTOR
                speed_kn = float(preds["speed"][j, 0]) / _ATTR_SCALE_FACTOR

                # Heading: model outputs cos(θ) and sin(θ)
                hx = float(preds["heading_x"][j, 0])
                hy = float(preds["heading_y"][j, 0])
                heading_deg = math.degrees(math.atan2(hy, hx)) % 360

                # Ship type: argmax of 9-class logits
                type_logits = preds["ship_type"][j]  # (9,)
                type_probs = torch.softmax(type_logits, dim=0)
                type_idx = int(torch.argmax(type_probs))
                type_conf = float(type_probs[type_idx])

                all_attrs.append({
                    "length_m": max(0.0, round(length_m, 1)),
                    "width_m": max(0.0, round(width_m, 1)),
                    "speed_knots": max(0.0, round(speed_kn, 1)),
                    "heading_deg": round(heading_deg, 1),
                    "vessel_type": SHIP_TYPES[type_idx],
                    "type_confidence": round(type_conf, 3),
                })

    return all_attrs


# ===================================================================
# Preprocessing helpers
# ===================================================================

def _normalize_bands(arr: np.ndarray) -> np.ndarray:
    """Normalize a (9, H, W) float array to [0, 1].

    Bands 0-2 (B04/B03/B02):  /3000
    Bands 3-8 (B05-B12):       /8160

    Input should be raw L2A DN values (float32, typically 0-10000).
    """
    out = arr.copy().astype(np.float32)
    out[:3] = np.clip(out[:3] / _NORM_RGB_STD, 0.0, 1.0)
    out[3:] = np.clip(out[3:] / _NORM_SWIR_STD, 0.0, 1.0)
    return out


def _bands_dict_to_9ch(bands: dict) -> np.ndarray:
    """Convert a bands dict to a (9, H, W) array in model band order.

    The bands dict uses IMINT naming: B02, B03, B04, B05, B06, B07, B08, B11, B12
    and values are in DN (digital numbers, 0-10000 range for L2A).

    If bands are already reflectance (0-1 range), they are scaled up to DN
    by multiplying by 10000.
    """
    stack = []
    for bname in BAND_ORDER:
        if bname not in bands:
            raise KeyError(f"Missing band {bname} — AI2 model needs all 9 bands: {BAND_ORDER}")
        b = bands[bname].astype(np.float32)
        # If values look like reflectance (max < 2), scale to DN
        if b.max() < 2.0:
            b = b * 10000.0
        stack.append(b)
    return np.stack(stack, axis=0)


# ===================================================================
# Sliding-window inference
# ===================================================================

def _sliding_window_inference(
    model: AI2VesselDetector,
    image: np.ndarray,
    patch_size: int = _PATCH_SIZE,
    overlap: float = _OVERLAP_RATIO,
    score_threshold: float = _SCORE_THRESHOLD,
    device: str = "cpu",
) -> list[dict]:
    """Run sliding-window inference over a large image.

    Args:
        model: the detection model (eval mode)
        image: (9, H, W) normalised float array
        patch_size: size of each patch
        overlap: overlap ratio between patches
        score_threshold: minimum confidence
        device: torch device

    Returns:
        list of detection dicts with 'box' (y1,x1,y2,x2 in image coords),
        'score', 'label'
    """
    _, h, w = image.shape
    stride = int(patch_size * (1 - overlap))
    all_dets = []

    # Compute patch origins
    y_starts = list(range(0, max(h - patch_size, 0) + 1, stride))
    if not y_starts or y_starts[-1] + patch_size < h:
        y_starts.append(max(h - patch_size, 0))
    x_starts = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not x_starts or x_starts[-1] + patch_size < w:
        x_starts.append(max(w - patch_size, 0))

    # Handle images smaller than patch_size
    if h < patch_size or w < patch_size:
        padded = np.zeros((image.shape[0], max(h, patch_size), max(w, patch_size)), dtype=image.dtype)
        padded[:, :h, :w] = image
        image = padded
        y_starts = [0]
        x_starts = [0]

    n_patches = len(y_starts) * len(x_starts)
    logger.info("Running AI2 detector on %d patches (%d×%d, stride=%d)",
                n_patches, patch_size, patch_size, stride)

    with torch.no_grad():
        for yi, y0 in enumerate(y_starts):
            for xi, x0 in enumerate(x_starts):
                patch = image[:, y0:y0 + patch_size, x0:x0 + patch_size]
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                results = model(tensor)

                for det in results:
                    boxes = det["boxes"].cpu().numpy()
                    scores = det["scores"].cpu().numpy()
                    labels = det["labels"].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        if score < score_threshold:
                            continue
                        if label == 0:
                            continue  # class 0 = "unknown"
                        # box is (x1, y1, x2, y2) in patch coords
                        # Convert to image coords
                        x1, y1_b, x2, y2_b = box
                        all_dets.append({
                            "x_min": int(x1 + x0),
                            "y_min": int(y1_b + y0),
                            "x_max": int(x2 + x0),
                            "y_max": int(y2_b + y0),
                            "score": float(score),
                            "label": "vessel",
                        })

    # Simple distance-based NMS to merge overlapping detections from overlap zones
    all_dets = _distance_nms(all_dets, dist_threshold=15)
    return all_dets


def _distance_nms(dets: list[dict], dist_threshold: float = 15) -> list[dict]:
    """Distance-based NMS: merge detections whose centroids are within threshold pixels.

    Keeps the detection with highest score in each cluster.
    """
    if not dets:
        return []

    # Sort by score descending
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    used = [False] * len(dets)

    for i, d in enumerate(dets):
        if used[i]:
            continue
        keep.append(d)
        ci_x = (d["x_min"] + d["x_max"]) / 2
        ci_y = (d["y_min"] + d["y_max"]) / 2

        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            cj_x = (dets[j]["x_min"] + dets[j]["x_max"]) / 2
            cj_y = (dets[j]["y_min"] + dets[j]["y_max"]) / 2
            dist = math.sqrt((ci_x - cj_x) ** 2 + (ci_y - cj_y) ** 2)
            if dist < dist_threshold:
                used[j] = True

    return keep


# ===================================================================
# BaseAnalyzer implementation
# ===================================================================

class AI2VesselAnalyzer(BaseAnalyzer):
    """Vessel detection + attribute prediction using Allen AI rslearn models.

    Detection model: Swin V2 B + FPN + Faster R-CNN (9-band Sentinel-2).
    Attribute model: SatlasPretrain + 6 PoolingDecoder heads predicting
    vessel length, width, speed, heading, and type.

    Required bands: B04, B03, B02, B05, B06, B07, B08, B11, B12
    (passed via the ``bands`` dict to ``analyze()``).

    Config options::

        checkpoint: str           — path to detection checkpoint (.ckpt)
        attr_checkpoint: str      — path to attribute checkpoint (.ckpt)
        predict_attributes: bool  — run attribute prediction (default True)
        score_threshold: float    — minimum detection confidence (default 0.5)
        patch_size: int           — sliding window size (default 512)
        overlap: float            — overlap ratio (default 0.1)
        water_filter: bool        — filter detections to SCL water pixels (default True)
        max_bbox_m: float         — max bounding-box size in metres (default 750)
        device: str               — torch device (default "cpu")
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._model: AI2VesselDetector | None = None
        self._attr_model: AI2AttributePredictor | None = None
        self._attr_failed: bool = False  # set True if attribute model fails to load

    def _ensure_model(self) -> AI2VesselDetector:
        """Lazy-load the detection model on first use."""
        if self._model is None:
            ckpt = self.config.get("checkpoint", str(_DEFAULT_CKPT))
            self._model = _load_detection_model(ckpt)
            device = self.config.get("device", "cpu")
            self._model = self._model.to(device)
        return self._model

    def _ensure_attr_model(self) -> Optional[AI2AttributePredictor]:
        """Lazy-load the attribute model. Returns None if unavailable."""
        if self._attr_failed:
            return None
        if self._attr_model is None:
            attr_ckpt = self.config.get("attr_checkpoint", str(_DEFAULT_ATTR_CKPT))
            if not Path(attr_ckpt).exists():
                logger.warning("Attribute checkpoint not found: %s", attr_ckpt)
                self._attr_failed = True
                return None
            try:
                self._attr_model = _load_attribute_model(attr_ckpt)
                device = self.config.get("device", "cpu")
                self._attr_model = self._attr_model.to(device)
            except Exception as exc:
                logger.warning("Failed to load attribute model: %s", exc)
                self._attr_failed = True
                return None
        return self._attr_model

    # noinspection PyMethodOverriding
    def analyze(
        self,
        rgb: np.ndarray,
        bands: dict | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
        previous_results: dict | None = None,
        scl: np.ndarray | None = None,
    ) -> AnalysisResult:
        """Detect vessels in a Sentinel-2 image.

        Args:
            rgb: (H, W, 3) uint8 RGB — used as fallback, but AI2 model
                 prefers the raw bands dict.
            bands: dict mapping band names to (H, W) float32 arrays.
                   Must contain B02, B03, B04, B05, B06, B07, B08, B11, B12.
            date: acquisition date (metadata only)
            coords: bounding box (metadata only)
            output_dir: output directory (not used)
            previous_results: ignored
            scl: (H, W) SCL array for water filtering

        Returns:
            AnalysisResult with regions list.
        """
        if bands is None:
            return AnalysisResult(
                analyzer="ai2_vessels",
                success=False,
                error="AI2 model requires bands dict with B02-B12 — got None",
            )

        # Check all 9 bands present
        missing = [b for b in BAND_ORDER if b not in bands]
        if missing:
            return AnalysisResult(
                analyzer="ai2_vessels",
                success=False,
                error=f"Missing bands for AI2 model: {missing}",
            )

        # Build 9-channel input
        image_9ch = _bands_dict_to_9ch(bands)  # (9, H, W) DN
        image_norm = _normalize_bands(image_9ch)  # (9, H, W) [0,1]

        # Run detection
        model = self._ensure_model()
        device = self.config.get("device", "cpu")
        score_thresh = self.config.get("score_threshold", _SCORE_THRESHOLD)
        patch_sz = self.config.get("patch_size", _PATCH_SIZE)
        overlap = self.config.get("overlap", _OVERLAP_RATIO)

        raw_dets = _sliding_window_inference(
            model, image_norm,
            patch_size=patch_sz,
            overlap=overlap,
            score_threshold=score_thresh,
            device=device,
        )

        # Convert to standard region format
        regions = []
        n_raw = len(raw_dets)
        n_water = 0
        n_size = 0
        water_filter = self.config.get("water_filter", True)
        max_bbox_m = self.config.get("max_bbox_m", 750)
        max_bbox_px = max_bbox_m / 10.0  # 10 m/pixel

        h, w = rgb.shape[:2] if rgb is not None else (image_9ch.shape[1], image_9ch.shape[2])

        for det in raw_dets:
            y_min = max(0, det["y_min"])
            y_max = min(h, det["y_max"])
            x_min = max(0, det["x_min"])
            x_max = min(w, det["x_max"])

            bw = x_max - x_min
            bh = y_max - y_min

            # Size filter
            if bw > max_bbox_px or bh > max_bbox_px:
                n_size += 1
                continue

            # Water filter via SCL
            if water_filter and scl is not None:
                cy = (y_min + y_max) // 2
                cx = (x_min + x_max) // 2
                if 0 <= cy < scl.shape[0] and 0 <= cx < scl.shape[1]:
                    if scl[cy, cx] != 6:  # SCL class 6 = water
                        n_water += 1
                        continue

            regions.append({
                "bbox": {
                    "y_min": int(y_min),
                    "y_max": int(y_max),
                    "x_min": int(x_min),
                    "x_max": int(x_max),
                },
                "pixel_count": int(bw * bh),
                "score": det["score"],
                "label": "vessel",
            })

        # --- Attribute prediction ---
        has_attributes = False
        do_attrs = self.config.get("predict_attributes", True)
        if do_attrs and regions:
            attr_model = self._ensure_attr_model()
            if attr_model is not None:
                try:
                    # Build detection list in the format _predict_attributes expects
                    det_for_attr = [
                        {
                            "x_min": r["bbox"]["x_min"],
                            "y_min": r["bbox"]["y_min"],
                            "x_max": r["bbox"]["x_max"],
                            "y_max": r["bbox"]["y_max"],
                        }
                        for r in regions
                    ]
                    attrs = _predict_attributes(
                        attr_model, image_9ch, det_for_attr, device=device,
                    )
                    # Merge attributes into regions
                    for region, attr in zip(regions, attrs):
                        region["attributes"] = attr
                    has_attributes = True
                    logger.info(
                        "Attribute prediction complete for %d vessels", len(regions),
                    )
                except Exception as exc:
                    logger.warning("Attribute prediction failed: %s", exc)

        area_km2 = 0.0
        if coords:
            import math as _m
            lat_mid = (coords.get("south", 0) + coords.get("north", 0)) / 2
            dx = (coords.get("east", 0) - coords.get("west", 0)) * 111.32 * _m.cos(_m.radians(lat_mid))
            dy = (coords.get("north", 0) - coords.get("south", 0)) * 111.32
            area_km2 = dx * dy

        return AnalysisResult(
            analyzer="ai2_vessels",
            success=True,
            outputs={
                "regions": regions,
                "vessel_count": len(regions),
                "vessel_density_per_km2": (
                    len(regions) / area_km2 if area_km2 > 0 else 0.0
                ),
                "has_attributes": has_attributes,
            },
            metadata={
                "model": "rslearn_sentinel2_vessels_swinv2b",
                "attr_model": "rslearn_sentinel2_vessel_attribute_swinv2b" if has_attributes else None,
                "checkpoint": self.config.get("checkpoint", str(_DEFAULT_CKPT)),
                "score_threshold": score_thresh,
                "patch_size": patch_sz,
                "overlap": overlap,
                "raw_detections": n_raw,
                "water_filtered": n_water,
                "size_filtered": n_size,
                "final_detections": len(regions),
                "area_km2": area_km2,
            },
        )
