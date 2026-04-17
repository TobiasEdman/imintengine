#!/usr/bin/env python3
"""On-demand tile image server for ImintEngine tile viewer.

Endpoints:
  GET /index.json           → {ready, tiles:{id:{lat,lon,bbox3006}}}
  GET /tile/{id}/rgb.png    → True-colour composite (B04/B03/B02)
  GET /tile/{id}/nmd.png    → NMD unified label colourmap
  GET /tile/{id}/thumb.png  → 48px label thumbnail
  GET /tile/{id}/meta.json  → per-tile class distribution
"""
from __future__ import annotations
import io, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace")
from imint.training.unified_schema import UNIFIED_COLORS, UNIFIED_CLASS_NAMES

TILES_DIR = "/data/unified_v2"
PORT      = 8000
CACHE_MAX = 300

# ── image helpers ──────────────────────────────────────────────────────────────

def make_rgb(spectral: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """True-colour B04/B03/B02.

    Frame selection: pick the temporal frame with the highest mean brightness
    (proxy for least cloud cover). Stretch: adaptive p2→0 / p98→1 per tile so
    both dark-water and bright-farmland tiles look natural. Then gamma 1.4.
    """
    bpf = 6
    nf  = spectral.shape[0] // bpf

    # --- pick least-cloudy frame (max mean across R+G+B) ---
    best_si, best_mean = 0, -1.0
    for fi in range(nf):
        b = fi * bpf
        m = float(spectral[b+2].mean() + spectral[b+1].mean() + spectral[b+0].mean())
        if m > best_mean:
            best_mean = m
            best_si   = fi

    b     = best_si * bpf
    stack = np.stack([spectral[b+2], spectral[b+1], spectral[b+0]], -1).astype(np.float32)

    # --- adaptive stretch: p2 → 0, p98 → 1 ---
    p2, p98 = np.percentile(stack, [2, 98])
    stack   = np.clip((stack - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

    # --- gamma ---
    stack = np.power(stack, 1.0 / gamma)
    return (stack * 255).astype(np.uint8)

def label_to_rgb(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, c in UNIFIED_COLORS.items():
        m = label == cls
        if m.any():
            rgb[m] = c
    return rgb

# ── index builder ──────────────────────────────────────────────────────────────

INDEX : dict = {}
INDEX_READY  = False
INDEX_LOCK   = threading.Lock()

def _load_meta(p: Path):
    tid = p.stem
    try:
        d = np.load(p, allow_pickle=True)
        if "bbox_3006" in d:
            b = d["bbox_3006"].flatten().astype(float)
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        elif "easting" in d and "northing" in d:
            cx, cy = float(d["easting"]), float(d["northing"])
            b = np.array([cx-1280, cy-1280, cx+1280, cy+1280])
        else:
            return tid, None
        return tid, (cx, cy, b)
    except Exception:
        return tid, None

def build_index():
    global INDEX, INDEX_READY
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
    except ImportError:
        tr = None
        print("[index] pyproj not available — coords unavailable", flush=True)

    tiles = sorted(Path(TILES_DIR).glob("*.npz"))
    print(f"[index] scanning {len(tiles)} tiles...", flush=True)
    idx = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(_load_meta, p): p for p in tiles}
        for i, f in enumerate(as_completed(futs)):
            tid, val = f.result()
            if val is None:
                idx[tid] = {}
                continue
            cx, cy, b = val
            entry = {"bbox3006": [int(b[0]),int(b[1]),int(b[2]),int(b[3])]}
            if tr:
                lon, lat = tr.transform(cx, cy)
                entry["lat"] = round(lat, 6)
                entry["lon"] = round(lon, 6)
            idx[tid] = entry
            if (i+1) % 1000 == 0:
                print(f"[index] {i+1}/{len(tiles)}", flush=True)

    with INDEX_LOCK:
        INDEX       = idx
        INDEX_READY = True
    print(f"[index] complete — {len(idx)} tiles", flush=True)

threading.Thread(target=build_index, daemon=True).start()

# ── PNG cache ──────────────────────────────────────────────────────────────────

_cache: dict = {}
_keys : list = []
_clock       = threading.Lock()

def cache_get(k):
    with _clock: return _cache.get(k)

def cache_set(k, v):
    with _clock:
        if k not in _cache:
            if len(_keys) >= CACHE_MAX:
                _cache.pop(_keys.pop(0), None)
            _keys.append(k)
        _cache[k] = v

# ── HTTP handler ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path   = urlparse(self.path).path
        parts  = path.strip("/").split("/")

        if path == "/index.json":
            with INDEX_LOCK:
                snap  = dict(INDEX)
                ready = INDEX_READY
            body = json.dumps({"ready": ready, "tiles": snap}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif len(parts) == 3 and parts[0] == "tile":
            self._serve_tile(parts[1], parts[2])

        else:
            self.send_error(404)

    def _serve_tile(self, tid, kind):
        ck = f"{tid}/{kind}"
        hit = cache_get(ck)
        if hit and kind.endswith(".png"):
            self._send_png(hit); return

        p = Path(TILES_DIR) / f"{tid}.npz"
        if not p.exists():
            self.send_error(404); return

        try:
            d = np.load(p, allow_pickle=True)

            if kind == "rgb.png":
                sp  = np.array(d.get("spectral", d.get("image")), dtype=np.float32)
                img = Image.fromarray(make_rgb(sp))
                buf = io.BytesIO(); img.save(buf, "PNG"); data = buf.getvalue()
                cache_set(ck, data); self._send_png(data)

            elif kind == "nmd.png":
                lbl = np.asarray(d.get("label", np.zeros((256,256), np.uint8)), np.uint8)
                img = Image.fromarray(label_to_rgb(lbl))
                buf = io.BytesIO(); img.save(buf, "PNG"); data = buf.getvalue()
                cache_set(ck, data); self._send_png(data)

            elif kind == "thumb.png":
                lbl = np.asarray(d.get("label", np.zeros((256,256), np.uint8)), np.uint8)
                img = Image.fromarray(label_to_rgb(lbl)).resize((48,48), Image.NEAREST)
                buf = io.BytesIO(); img.save(buf, "PNG"); data = buf.getvalue()
                cache_set(ck, data); self._send_png(data)

            elif kind == "meta.json":
                lbl = np.asarray(d.get("label", np.zeros((256,256), np.uint8)), np.uint8)
                cls, cnts = np.unique(lbl, return_counts=True)
                total = lbl.size
                meta  = {"id": tid, "classes": {
                    int(c): {"name": UNIFIED_CLASS_NAMES[c] if c < len(UNIFIED_CLASS_NAMES) else f"cls_{c}",
                             "pct": round(100*n/total,1)}
                    for c, n in zip(cls, cnts)
                }}
                body = json.dumps(meta).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(400)

        except Exception as e:
            self.send_error(500, str(e)[:200])

    def _send_png(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "public, max-age=3600")
        self._cors()
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

print(f"[server] listening on :{PORT}", flush=True)
ThreadedHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
