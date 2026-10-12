"""Allen AI (rslearn) Sentinel-1 vessel detection analyzer.

Port of AI2's Sentinel-1 vessel pipeline (``allenai/rslearn_projects``,
``rslp/sentinel1_vessels``; the detector behind Skylight) to plain
torch/torchvision, built from the rslearn blocks shared with the Sentinel-2
port (``imint/analyzers/ai2_rslearn_blocks.py``). The reference run, input
contract and checkpoint layout are in ``docs/marine/s1_vessels_reference.md``.

Detector (``data/sentinel1_vessels/config.yaml``):
  one Swin V2 B with a 2-channel stem, applied to the target and to one
  historical scene with shared weights -> features of both concatenated per
  scale (256/512/1024/2048 ch) -> FPN (256) -> Faster R-CNN. 512 px patches,
  51 px overlap, no image transform, score > 0.7, then distance NMS (10 px
  between box centres) across patches.

Attribute model (``data/sentinel1_vessel_attribute/config.yaml``):
  Swin V2 B (2-channel stem) on a 128 px target-only window centred on the
  detection -> six pooling heads: length, width, speed, heading_x, heading_y,
  ship_type.

Inputs are uncalibrated GRD DN (detected amplitude, NOT sigma0 and NOT dB) on
AI2's 10 m UTM window grid; ``imint.fetch.fetch_s1_vessel_scenes`` produces
them (uint16). Normalisation: detector ``clip(DN / 250, 0, 4)``, attribute
model ``DN / 250`` without clip, applied per 512 px patch and per 128 px crop
so the full-size bands stay in their input dtype.

There is no water mask: Sentinel-1 has no SCL and AI2 applies none. Instead
AI2 drops detections near known marine infrastructure (Satlas, ODC-BY); this
analyzer does the same and records in its metadata whether the filter ran.

A detection without an AIS match is a candidate, not a confirmed dark vessel.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from pyproj import Transformer

from imint.analyzers.base import AnalysisResult, BaseAnalyzer
from imint.fetch import S1_HISTORICAL_BAND_KEYS, S1_TARGET_BAND_KEYS, S1_VESSEL_GSD_M
from imint.utils import haversine_m

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and band contract
# ---------------------------------------------------------------------------
_FM_DIR = Path(__file__).resolve().parent.parent / "fm" / "ai2_vessels_s1"
DEFAULT_DETECT_CKPT = _FM_DIR / "detect_best.ckpt"
DEFAULT_ATTR_CKPT = _FM_DIR / "attribute_best.ckpt"

# ---------------------------------------------------------------------------
# Model constants (rslearn_projects configs; see the reference dossier)
# ---------------------------------------------------------------------------
PATCH_SIZE = 512
OVERLAP_PX = 51                  # round(512 * overlap_ratio 0.1)
SCORE_THRESHOLD = 0.7            # SENTINEL1_SCORE_THRESHOLD default; kept if score > it
NMS_DISTANCE_PX = 10             # NMSDistanceMerger distance_threshold
DETECT_BATCH_SIZE = 8            # AI2 predict batch size (patches per forward)
_DN_SCALE = 250.0                # normalisation std (mean 0)
_DETECT_CLIP = 4.0
_FPN_IN_CHANNELS = [256, 512, 1024, 2048]   # target + historical per scale
_FPN_OUT_CHANNELS = 256
_BOX_SCORE_THRESH = 0.05         # RoIHeads score_thresh inside the detector
ATTR_CROP_PX = 128
ATTR_BATCH_SIZE = 16             # crops per attribute-model forward
_ATTR_SCALE = 100.0              # regression heads were trained on value * 0.01
ATTR_HEADS = {"length": 1, "width": 1, "speed": 1, "heading_x": 1, "heading_y": 1,
              "ship_type": 9}
SHIP_TYPES = ["cargo", "tanker", "passenger", "service", "tug",
              "pleasure", "fishing", "enforcement", "sar"]

# Satlas marine infrastructure (Allen AI, ODC-BY), as AI2's NearInfraFilter.
INFRA_URL = ("https://storage.googleapis.com/satlas-explorer-public/outputs/marine/"
             "latest.geojson")
INFRA_DISTANCE_M = 200.0         # SENTINEL1_INFRA_DISTANCE_KM default 0.2 (docs say 50 m)
_INFRA_CACHE_DIR = Path.home() / ".cache" / "imint" / "satlas_marine_infrastructure"


# ===================================================================
# Pure logic (no torch)
# ===================================================================

def normalize_detector(dn: np.ndarray) -> np.ndarray:
    """Detector input: ``clip(DN / 250, 0, 4)`` as float32."""
    return np.clip(np.asarray(dn, dtype=np.float32) / _DN_SCALE, 0.0, _DETECT_CLIP)


def normalize_attribute(dn: np.ndarray) -> np.ndarray:
    """Attribute-model input: ``DN / 250`` as float32, no clip."""
    return np.asarray(dn, dtype=np.float32) / _DN_SCALE


def crop_starts(start: int, stop: int, size: int = PATCH_SIZE,
                overlap: int = OVERLAP_PX) -> list[int]:
    """1-D crop origins over ``[start, stop)``, as rslearn's ``get_window_crop_options``.

    The first crop starts at ``start``, interior crops follow at stride
    ``size - overlap`` and the last one is snapped to ``stop - size`` so that
    no crop leaves the bounds. A range shorter than ``size`` gets one crop
    (zero-padded by the caller).
    """
    stride = size - overlap
    starts = [start, *range(start + stride, stop - size, stride)]
    if stop - size > start:
        starts.append(stop - size)
    return starts


def patch_origins(height: int, width: int) -> list[tuple[int, int]]:
    """(col, row) origins of all patches, in rslearn's order (columns outer)."""
    return [(c, r) for c in crop_starts(0, width) for r in crop_starts(0, height)]


def distance_nms(centers: np.ndarray, scores: np.ndarray,
                 threshold: float = NMS_DISTANCE_PX) -> list[int]:
    """AI2's distance NMS (``rslp.utils.nms.distance_nms``), exactly.

    Detections are visited from the lowest score up; one is eliminated when
    another detection not yet eliminated lies within ``threshold`` (inclusive)
    of its centre and has a higher score, or the same score and a lower index.

    Returns:
        Kept indices, in ascending score order.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    eliminated = np.zeros(len(scores), dtype=bool)
    keep: list[int] = []
    for idx in order:
        if eliminated[idx]:
            continue
        dist = np.hypot(*(centers - centers[idx]).T)
        stronger = (scores > scores[idx]) | ((scores == scores[idx])
                                             & (np.arange(len(scores)) < idx))
        if np.any((dist <= threshold) & stronger & ~eliminated):
            eliminated[idx] = True
        else:
            keep.append(int(idx))
    return keep


def ai2_point(cx_abs: float, cy_abs: float) -> tuple[int, int]:
    """AI2's detection point: ``int()`` of the box centre in absolute pixel units.

    ``int`` truncates towards zero, so with rslearn's negative row coordinates
    (``y / -10``) the row rounds up (south) and the column rounds down (west).
    The reported position is that pixel's corner, ``(col * 10, row * -10)``.
    """
    return int(cx_abs), int(cy_abs)


def wrap_heading(deg: float | np.ndarray) -> float | np.ndarray:
    """Wrap a heading to [0, 360). AI2 reports ``90 - atan2`` in (-90, 270].

    ``np.mod`` alone can return exactly 360.0 for a tiny negative input
    (``np.mod(-1e-14, 360) == 360.0``); that case maps to 0.
    """
    wrapped = np.mod(deg, 360.0)
    return wrapped - 360.0 * (wrapped >= 360.0)


def decode_attributes(raw: dict[str, np.ndarray]) -> list[dict]:
    """Decode the six attribute heads for a batch (arrays of shape (N, out)).

    length/width/speed are the raw outputs x 100 (trained on value * 0.01);
    heading = 90 - degrees(atan2(heading_y, heading_x)), wrapped to [0, 360);
    type = argmax of the softmax, with its probability as confidence.
    """
    logits = np.asarray(raw["ship_type"], dtype=np.float64)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    heading = wrap_heading(90.0 - np.degrees(np.arctan2(
        np.asarray(raw["heading_y"], dtype=np.float64)[:, 0],
        np.asarray(raw["heading_x"], dtype=np.float64)[:, 0])))
    out = []
    for i in range(len(logits)):
        k = int(np.argmax(probs[i]))
        out.append({
            "length_m": float(raw["length"][i, 0]) * _ATTR_SCALE,
            "width_m": float(raw["width"][i, 0]) * _ATTR_SCALE,
            "speed_knots": float(raw["speed"][i, 0]) * _ATTR_SCALE,
            "heading_deg": float(heading[i]),
            "vessel_type": SHIP_TYPES[k],
            "type_confidence": float(probs[i, k]),
        })
    return out


def grid_origin_px(transform) -> tuple[int, int]:
    """Absolute pixel coordinates (col, row) of the image origin on AI2's grid.

    AI2 works in pixel units ``(x / 10, y / -10)`` of a 10 m north-up UTM grid
    whose origin sits on integer multiples of 10 m; the detection points depend
    on that (``ai2_point``), so any other grid is rejected.
    """
    a, b, c, d, e, f = (float(v) for v in list(transform)[:6])
    if (a, b, d, e) != (S1_VESSEL_GSD_M, 0.0, 0.0, -S1_VESSEL_GSD_M):
        raise ValueError(f"expected a north-up {S1_VESSEL_GSD_M} m grid, "
                         f"got transform {transform}")
    col, row = c / S1_VESSEL_GSD_M, f / -S1_VESSEL_GSD_M
    if col != int(col) or row != int(row):
        raise ValueError(f"grid origin ({c}, {f}) is not on multiples of {S1_VESSEL_GSD_M} m")
    return int(col), int(row)


def load_infrastructure(path: str | Path | None = None,
                        cache_dir: str | Path = _INFRA_CACHE_DIR) -> tuple[np.ndarray, dict]:
    """Satlas marine-infrastructure points as an (N, 2) lon/lat array + provenance.

    ``path`` is a local GeoJSON; without it the file is downloaded once from
    ``INFRA_URL`` into ``cache_dir`` and reused from there.
    """
    import hashlib

    if path is None:
        cache_dir = Path(cache_dir)
        path = cache_dir / "latest.geojson"
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(INFRA_URL, timeout=120) as resp:
                last_modified = resp.headers.get("Last-Modified")
                tmp.write_bytes(resp.read())
            tmp.rename(path)
            (cache_dir / "latest.provenance.json").write_text(json.dumps(
                {"url": INFRA_URL, "last_modified": last_modified,
                 "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    path = Path(path)
    blob = path.read_bytes()
    lonlat = np.array([f["geometry"]["coordinates"][:2]
                       for f in json.loads(blob)["features"]], dtype=np.float64)
    return lonlat.reshape(-1, 2), {"path": str(path), "points": int(len(lonlat)),
                                   "sha256": hashlib.sha256(blob).hexdigest()}


def nearest_infra_m(lon: float, lat: float, infra_lonlat: np.ndarray) -> float:
    """Distance (m) from a point to the nearest infrastructure point."""
    if len(infra_lonlat) == 0:
        return math.inf
    return float(haversine_m(lon, lat, infra_lonlat[:, 0], infra_lonlat[:, 1]).min())


def _region(box: np.ndarray, score: float, col0: int, row0: int, to_wgs84) -> dict:
    """One kept detection: local box, AI2 point (local px) and its lon/lat."""
    x1, y1, x2, y2 = (float(v) for v in box)
    col, row = ai2_point(col0 + (x1 + x2) / 2, row0 + (y1 + y2) / 2)
    lon, lat = to_wgs84(col * S1_VESSEL_GSD_M, row * -S1_VESSEL_GSD_M)
    return {
        "bbox": {"x_min": math.floor(x1), "y_min": math.floor(y1),
                 "x_max": math.ceil(x2), "y_max": math.ceil(y2)},
        "box_px": [x1, y1, x2, y2],
        "point_px": [col - col0, row - row0],
        "lon": float(lon),
        "lat": float(lat),
        "pixel_count": int(round((x2 - x1) * (y2 - y1))),
        "score": score,
        "label": "vessel",
    }


# ===================================================================
# Models (torch is imported lazily so the engine import stays light)
# ===================================================================

def _load_state(path: str | Path, remap) -> dict:
    """Lightning checkpoint -> state dict renamed by ``remap(key) -> key``."""
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    return {remap(k): v for k, v in ckpt["state_dict"].items()}


def build_detector():
    """Detector from the shared rslearn blocks (random weights)."""
    import torch

    from imint.analyzers.ai2_rslearn_blocks import FPN, FasterRCNNHead, SwinFeatureExtractor

    return torch.nn.ModuleDict({
        "swin": SwinFeatureExtractor(input_channels=2),
        "fpn": FPN(in_channels=_FPN_IN_CHANNELS, out_channels=_FPN_OUT_CHANNELS),
        "head": FasterRCNNHead(num_channels=_FPN_OUT_CHANNELS, num_classes=2,
                               box_score_thresh=_BOX_SCORE_THRESH),
    })


def load_detector(path: str | Path):
    """Load the detector checkpoint strictly (every key must match)."""
    prefixes = {"model.encoder.0.encoder.model.": "swin.model.",
                "model.encoder.1.fpn.": "fpn.fpn.",
                "model.decoders.detect.0.": "head."}

    def remap(key: str) -> str:
        for src, dst in prefixes.items():
            if key.startswith(src):
                return dst + key[len(src):]
        raise KeyError(f"unexpected detector checkpoint key {key}")

    model = build_detector()
    model.load_state_dict(_load_state(path, remap), strict=True)
    return model.eval()


def detector_forward(model, target, historical) -> list[dict]:
    """(B, 2, 512, 512) normalised target + historical -> torchvision detections."""
    import torch

    feats = [torch.cat(pair, dim=1)
             for pair in zip(model["swin"](target), model["swin"](historical), strict=True)]
    return model["head"](model["fpn"](feats), list(target))


def build_attribute_model():
    """Attribute model: Swin V2 B (2-ch stem, last stage only) + six pooling heads."""
    import torch

    from imint.analyzers.ai2_rslearn_blocks import PoolingDecoder, SwinFeatureExtractor

    # rslearn Swin output_layers=[7]: stride 32, 1024 channels.
    backbone = SwinFeatureExtractor(input_channels=2, output_layers=(7,))
    heads = torch.nn.ModuleDict({
        name: PoolingDecoder(in_channels=1024, out_channels=n, num_conv_layers=1,
                             num_fc_layers=2)
        for name, n in ATTR_HEADS.items()})
    return torch.nn.ModuleDict({"backbone": backbone, "heads": heads})


def load_attribute_model(path: str | Path):
    """Load the attribute checkpoint strictly."""
    def remap(key: str) -> str:
        if key.startswith("model.encoder.0.model."):
            return "backbone.model." + key[len("model.encoder.0.model."):]
        if key.startswith("model.decoders."):
            head, rest = key[len("model.decoders."):].split(".0.", 1)
            return f"heads.{head}.{rest}"
        raise KeyError(f"unexpected attribute checkpoint key {key}")

    model = build_attribute_model()
    model.load_state_dict(_load_state(path, remap), strict=True)
    return model.eval()


def attribute_forward(model, x) -> dict:
    """(B, 2, 128, 128) normalised target crops -> raw head outputs."""
    (feat,) = model["backbone"](x)
    return {name: head(feat) for name, head in model["heads"].items()}


# ===================================================================
# Analyzer
# ===================================================================

class AI2S1VesselAnalyzer(BaseAnalyzer):
    """Sentinel-1 vessel detection + attributes with AI2's rslearn models.

    Inputs (``bands``): ``VV_DN``, ``VH_DN`` of the target scene and
    ``VV_DN_HIST``, ``VH_DN_HIST`` of the historical scene
    (``imint.fetch.S1_TARGET_BAND_KEYS`` / ``S1_HISTORICAL_BAND_KEYS``), GRD
    DN on one grid; ``geo`` must be AI2's window grid (10 m UTM, origin on
    multiples of 10 m). The bands are read in place (no full-size copies) and
    normalised per patch / per crop. Without these bands the analyzer reports
    itself skipped.

    Config options::

        checkpoint: str        detector checkpoint (default imint/fm/ai2_vessels_s1/)
        attr_checkpoint: str   attribute checkpoint
        predict_attributes: bool   default True
        score_threshold: float     default 0.7 (AI2 API default)
        infra_filter: bool         default True
        infra_path: str            local Satlas GeoJSON (default: cached download)
        infra_distance_m: float    default 200 (AI2 code default)
        batch_size: int            detector batch, default 8 (AI2)
        device: str                default "cpu"

    Edges: a detection within 64 px of the image edge gets a zero-padded
    attribute window where AI2 would read real pixels; fetch the AOI with a
    margin when attributes at the edge matter. ``metadata["attr_crops_padded"]``
    counts them.

    Coverage: ``metadata["valid_frac_target"]`` / ``["valid_frac_historical"]``
    are the fractions of non-zero VV pixels. Where only the target has data the
    detector runs without its historical reference; a WARNING is logged when
    the historical coverage is below the target's.
    """

    name = "ai2_vessels_s1"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._detector = None
        self._attr_model = None

    # noinspection PyMethodOverriding
    def analyze(
        self,
        rgb: np.ndarray | None,
        bands: dict | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
        geo=None,
    ) -> AnalysisResult:
        keys = S1_TARGET_BAND_KEYS + S1_HISTORICAL_BAND_KEYS
        missing = [k for k in keys if not bands or k not in bands]
        if missing:
            return AnalysisResult(
                analyzer=self.name, success=False, skipped=True,
                error=(f"no Sentinel-1 input (missing {missing}); fetch it with "
                       "imint.fetch.fetch_s1_vessel_scenes"))
        if geo is None:
            return AnalysisResult(analyzer=self.name, success=False,
                                  error="geo is required: detections are placed on AI2's grid")
        col0, row0 = grid_origin_px(geo.transform)
        target = tuple(np.asarray(bands[k]) for k in S1_TARGET_BAND_KEYS)
        historical = tuple(np.asarray(bands[k]) for k in S1_HISTORICAL_BAND_KEYS)
        shapes = {b.shape for b in target + historical}
        if len(shapes) != 1 or len(target[0].shape) != 2:
            return AnalysisResult(analyzer=self.name, success=False,
                                  error=f"bands are not 2-D arrays on one grid: {shapes}")
        h, w = target[0].shape
        valid_t = np.count_nonzero(target[0]) / target[0].size
        valid_h = np.count_nonzero(historical[0]) / historical[0].size
        if valid_h < valid_t:
            logger.warning("S1 vessels: historical scene covers %.1f %% of the AOI, the target "
                           "%.1f %%; detections where only the target has data are made "
                           "without the historical reference", 100 * valid_h, 100 * valid_t)

        cfg = self.config
        device = cfg.get("device", "cpu")
        score_thr = float(cfg.get("score_threshold", SCORE_THRESHOLD))
        t_start = time.time()

        # ── Detector over AI2's patch grid ─────────────────────────────
        raw_boxes, raw_scores, n_patches = self._detect(target, historical, score_thr, device)
        centers = np.column_stack([(raw_boxes[:, 0] + raw_boxes[:, 2]) / 2,
                                   (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2]) \
            if len(raw_boxes) else np.zeros((0, 2))
        keep = distance_nms(centers, raw_scores)
        t_detect = time.time() - t_start

        to_wgs84 = Transformer.from_crs(geo.crs, 4326, always_xy=True).transform
        regions = [_region(raw_boxes[i], float(raw_scores[i]), col0, row0, to_wgs84)
                   for i in keep]

        # ── Marine-infrastructure filter (AI2 NearInfraFilter) ────────
        infra_meta, regions = self._infra_filter(regions)

        # ── Attributes ────────────────────────────────────────────────
        n_padded = 0
        if cfg.get("predict_attributes", True) and regions:
            n_padded = self._attributes(target, regions, device)
        regions.sort(key=lambda r: -r["score"])

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "regions": regions,
                "vessel_count": len(regions),
                "has_attributes": bool(regions) and "attributes" in regions[0],
            },
            metadata={
                "model": "rslearn sentinel1_vessels (Swin V2 B x2 + FPN + Faster R-CNN)",
                "attr_model": "rslearn sentinel1_vessel_attribute (Swin V2 B + pooling heads)",
                "checkpoint": str(cfg.get("checkpoint", DEFAULT_DETECT_CKPT)),
                "attr_checkpoint": str(cfg.get("attr_checkpoint", DEFAULT_ATTR_CKPT)),
                "score_threshold": score_thr,
                "patch_size": PATCH_SIZE,
                "overlap_px": OVERLAP_PX,
                "patches": n_patches,
                "raw_detections": int(len(raw_scores)),
                "after_nms": len(keep),
                "infra_filter": infra_meta,
                "final_detections": len(regions),
                "attr_crops_padded": n_padded,
                "water_mask": "none (Sentinel-1 has no SCL; AI2 applies none)",
                "valid_frac_target": round(valid_t, 6),
                "valid_frac_historical": round(valid_h, 6),
                "grid_px": [w, h],
                "device": device,
                "seconds_detector": round(t_detect, 1),
                "seconds_total": round(time.time() - t_start, 1),
            },
        )

    # ── internals ──────────────────────────────────────────────────────

    def _ensure_detector(self, device: str):
        if self._detector is None:
            path = self.config.get("checkpoint", DEFAULT_DETECT_CKPT)
            self._detector = load_detector(path).to(device)
        return self._detector

    def _ensure_attr_model(self, device: str):
        if self._attr_model is None:
            path = self.config.get("attr_checkpoint", DEFAULT_ATTR_CKPT)
            self._attr_model = load_attribute_model(path).to(device)
        return self._attr_model

    def _detect(self, target: tuple[np.ndarray, ...], historical: tuple[np.ndarray, ...],
                score_thr: float, device: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Boxes (image px, float64 x1, y1, x2, y2) and scores above the threshold.

        ``target`` / ``historical`` are the (H, W) [vv, vh] DN bands; each patch
        is cut, normalised and zero-padded on its own.
        """
        import torch

        model = self._ensure_detector(device)
        batch_size = int(self.config.get("batch_size", DETECT_BATCH_SIZE))
        h, w = target[0].shape
        origins = patch_origins(h, w)
        boxes: list[np.ndarray] = []
        scores: list[np.ndarray] = []

        def patch(scene: tuple[np.ndarray, ...], c: int, r: int) -> np.ndarray:
            out = np.zeros((len(scene), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            for i, band in enumerate(scene):
                sub = band[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
                out[i, :sub.shape[0], :sub.shape[1]] = normalize_detector(sub)
            return out

        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(origins), batch_size):
                chunk = origins[i:i + batch_size]
                bt = torch.from_numpy(np.stack([patch(target, c, r) for c, r in chunk]))
                bh = torch.from_numpy(np.stack([patch(historical, c, r) for c, r in chunk]))
                dets = detector_forward(model, bt.to(device), bh.to(device))
                for (c, r), det in zip(chunk, dets, strict=True):
                    s = det["scores"].cpu().numpy()
                    wanted = s > score_thr
                    b = det["boxes"].cpu().numpy()[wanted].astype(np.float64)
                    b[:, [0, 2]] += c
                    b[:, [1, 3]] += r
                    boxes.append(b)
                    scores.append(s[wanted].astype(np.float64))
                done = min(i + batch_size, len(origins))
                logger.info("S1 detector %d/%d patches, %.1f s", done, len(origins),
                            time.time() - t0)
        if not boxes:
            return np.zeros((0, 4)), np.zeros(0), len(origins)
        return np.concatenate(boxes), np.concatenate(scores), len(origins)

    def _infra_filter(self, regions: list[dict]) -> tuple[dict, list[dict]]:
        """Drop detections closer than ``infra_distance_m`` to Satlas infrastructure.

        The Satlas file is only loaded (and on first use downloaded) when there
        is at least one detection to test.
        """
        dist_m = float(self.config.get("infra_distance_m", INFRA_DISTANCE_M))
        if not self.config.get("infra_filter", True):
            logger.warning("S1 vessels: marine-infrastructure filter DISABLED by config")
            return {"applied": False, "reason": "disabled by config"}, regions
        if not regions:
            return {"applied": False, "reason": "no detections to filter"}, regions
        try:
            infra, prov = load_infrastructure(self.config.get("infra_path"))
        except Exception as exc:  # noqa: BLE001 — surfaced loudly, never silent
            logger.warning("S1 vessels: marine-infrastructure filter NOT applied: %s", exc)
            return {"applied": False, "reason": f"{type(exc).__name__}: {exc}",
                    "source": INFRA_URL}, regions
        kept = []
        for region in regions:
            region["infra_distance_m"] = nearest_infra_m(region["lon"], region["lat"], infra)
            if region["infra_distance_m"] >= dist_m:
                kept.append(region)
        return {"applied": True, "distance_m": dist_m, "removed": len(regions) - len(kept),
                "dataset": "Satlas marine infrastructure (Allen AI, ODC-BY)", **prov}, kept

    def _attributes(self, target: tuple[np.ndarray, ...], regions: list[dict],
                    device: str) -> int:
        """Attach decoded attributes to ``regions``; return how many crops were padded."""
        import torch

        from imint.analyzers.ai2_rslearn_blocks import crop_centered

        model = self._ensure_attr_model(device)
        h, w = target[0].shape
        half = ATTR_CROP_PX // 2
        padded = 0
        crops = []
        for region in regions:
            col, row = region["point_px"]
            # AI2's window is [col - 64, col + 64) x [row - 64, row + 64).
            crops.append(normalize_attribute(np.concatenate(
                [crop_centered(band[None], row, col, ATTR_CROP_PX) for band in target])))
            padded += not (half <= col <= w - half and half <= row <= h - half)
        with torch.inference_mode():
            for i in range(0, len(crops), ATTR_BATCH_SIZE):
                batch = regions[i:i + ATTR_BATCH_SIZE]
                x = torch.from_numpy(np.stack(crops[i:i + ATTR_BATCH_SIZE])).to(device)
                raw = {k: v.cpu().numpy() for k, v in attribute_forward(model, x).items()}
                for region, attrs in zip(batch, decode_attributes(raw), strict=True):
                    region["attributes"] = attrs
        return padded
