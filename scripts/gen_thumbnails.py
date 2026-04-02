#!/usr/bin/env python3
"""Generate base64 thumbnails of the 5 latest tiles — summer RGB at 256x256."""
import numpy as np, glob, os, json, base64, io
from PIL import Image

tiles = sorted(glob.glob("/data/unified_v2/*.npz"), key=os.path.getmtime, reverse=True)
tiles = tiles[2:7] if len(tiles) > 6 else tiles[:5]  # skip 2 newest (might be partial)
results = []

for p in tiles:
    try:
        d = np.load(p, allow_pickle=True)
        img = d.get("spectral", d.get("image"))
        name = os.path.basename(p).replace(".npz", "")
        n_frames = img.shape[0] // 6

        # Pick best frame: summer (frame 2) preferred, fallback to highest-signal
        si = min(2, n_frames - 1) * 6
        frame = img[si:si + 6]
        if frame.max() < 0.01 and n_frames > 1:
            # Summer frame empty — pick frame with highest signal
            best = max(range(n_frames), key=lambda fi: float(img[fi * 6:(fi + 1) * 6].max()))
            si = best * 6
            frame = img[si:si + 6]

        # B4/B3/B2 natural color with percentile stretch (reflectance 0-1)
        r, g, b = frame[2], frame[1], frame[0]
        rgb = np.stack([r, g, b], axis=-1)
        valid = rgb[rgb > 0.001]
        if len(valid) > 100:
            p2, p98 = np.percentile(valid, [2, 98])
            rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
        else:
            rgb = np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        dates = [str(dd) for dd in d.get("dates", [])]
        frame_idx = si // 6
        date_str = dates[frame_idx] if frame_idx < len(dates) else ""

        results.append({
            "name": name,
            "img": b64,
            "date": date_str,
            "frame": frame_idx,
        })
    except:
        continue

print(json.dumps(results))
