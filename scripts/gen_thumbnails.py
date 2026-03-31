#!/usr/bin/env python3
"""Generate base64 thumbnails from the 3 latest tiles on PVC.

Shows all 4 temporal frames as RGB (B4/B3/B2) at full 256x256 resolution.
Uses percentile stretch for reflectance data (0-1 range from CDSE).
"""
import numpy as np, glob, os, json, base64, io
from PIL import Image

FRAME_LABELS = ["Autumn", "Spring", "Summer", "Late"]

# Colors from unified_schema.py UNIFIED_COLORS
COLORS = [
    (0, 0, 0),         # 0  background
    (0, 100, 0),       # 1  tallskog
    (34, 139, 34),     # 2  granskog
    (50, 205, 50),     # 3  lövskog
    (60, 179, 113),    # 4  blandskog
    (46, 79, 46),      # 5  sumpskog
    (160, 200, 120),   # 6  tillfälligt ej skog
    (139, 90, 43),     # 7  våtmark
    (210, 180, 140),   # 8  öppen mark
    (255, 0, 0),       # 9  bebyggelse
    (0, 0, 255),       # 10 vatten
    (230, 180, 34),    # 11 vete
    (212, 130, 23),    # 12 korn
    (255, 255, 100),   # 13 havre
    (45, 180, 90),     # 14 oljeväxter
    (100, 200, 100),   # 15 vall
    (180, 80, 40),     # 16 potatis
    (140, 180, 50),    # 17 trindsäd
    (170, 170, 170),   # 18 övrig åker
    (180, 120, 60),    # 19 hygge
]

tiles = sorted(glob.glob("/data/unified_v2/*.npz"), key=os.path.getmtime, reverse=True)
# Skip the very latest (might be partially written)
tiles = tiles[2:5] if len(tiles) > 4 else tiles[:3]
results = []

for p in tiles:
    try:
        d = np.load(p, allow_pickle=True)
        img = d["image"]  # (T*6, 256, 256) reflectance 0-1
        name = os.path.basename(p).replace(".npz", "")
        n_frames = img.shape[0] // 6

        frames = []
        for fi in range(min(n_frames, 4)):
            offset = fi * 6
            # RGB from B4(idx2), B3(idx1), B2(idx0) — natural color
            r, g, b = img[offset + 2], img[offset + 1], img[offset]
            rgb = np.stack([r, g, b], axis=-1)

            # Percentile stretch for reflectance (0-1 range)
            valid = rgb[rgb > 0.001]
            if len(valid) > 100:
                p2, p98 = np.percentile(valid, [2, 98])
                rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
            else:
                rgb = np.zeros_like(rgb, dtype=np.uint8)

            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "JPEG", quality=75)
            frames.append(base64.b64encode(buf.getvalue()).decode())

        # Pad missing frames
        while len(frames) < 4:
            frames.append(frames[-1] if frames else "")

        dates = d.get("dates", [])
        dates_list = [str(dd) for dd in dates]
        tmask = d.get("temporal_mask", None)
        frames_ok = int(tmask.sum()) if tmask is not None else n_frames

        results.append({
            "name": name,
            "frames": frames,
            "frame_labels": FRAME_LABELS[:4],
            "dates": dates_list[:4],
            "frames_ok": frames_ok,
        })
    except:
        continue

print(json.dumps(results))
