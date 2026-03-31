#!/usr/bin/env python3
"""Generate base64 PNG thumbnails from the 3 latest tiles on PVC.

Shows all 4 temporal frames as RGB (B4/B3/B2) at full 256x256 resolution.
Frame labels: Autumn (Sep-Oct), Spring (Apr-May), Summer (Jun-Jul), Late (Aug)
"""
import numpy as np, glob, os, json, base64, io
from PIL import Image

FRAME_LABELS = ["Autumn", "Spring", "Summer", "Late"]

tiles = sorted(glob.glob("/data/unified_v2/*.npz"), key=os.path.getmtime, reverse=True)[:3]
results = []

for p in tiles:
    try:
        d = np.load(p, allow_pickle=True)
        img = d["image"]  # (T*6, 256, 256)
        name = os.path.basename(p).replace(".npz", "")
        n_frames = img.shape[0] // 6

        frames = []
        for fi in range(min(n_frames, 4)):
            offset = fi * 6
            # RGB from B4(idx2), B3(idx1), B2(idx0) — natural color
            r, g, b = img[offset + 2], img[offset + 1], img[offset]
            rgb = np.stack([r, g, b], axis=-1)
            rgb = np.clip(rgb / 3000 * 255, 0, 255).astype(np.uint8)

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
            "frames": frames,  # 4 base64 JPEG strings
            "frame_labels": FRAME_LABELS[:4],
            "dates": dates_list[:4],
            "frames_ok": frames_ok,
        })
    except:
        continue

print(json.dumps(results))
