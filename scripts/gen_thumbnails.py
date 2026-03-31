#!/usr/bin/env python3
"""Generate base64 thumbnails from the 3 latest tiles on PVC."""
import numpy as np, glob, os, json, base64, io
from PIL import Image

tiles = sorted(glob.glob("/data/unified_v2/*.npz"), key=os.path.getmtime, reverse=True)[:3]
results = []
COLORS = [(55,65,81),(22,101,52),(21,128,61),(74,222,128),(134,239,172),(6,95,70),
          (163,230,53),(6,182,212),(251,191,36),(239,68,68),(59,130,246),
          (245,158,11),(217,119,6),(180,83,9),(234,179,8),(132,204,22),
          (161,98,7),(202,138,4),(146,64,14),(147,51,234)]

for p in tiles:
    try:
        d = np.load(p, allow_pickle=True)
        img = d["image"]
        name = os.path.basename(p).replace(".npz", "")
        n_frames = img.shape[0] // 6
        summer_idx = min(2, n_frames - 1) * 6
        r, g, b = img[summer_idx + 2], img[summer_idx + 1], img[summer_idx]
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb / 3000 * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).resize((64, 64)).save(buf, "PNG")
        rgb_b64 = base64.b64encode(buf.getvalue()).decode()

        if n_frames >= 4:
            r, g, b = img[2], img[1], img[0]
            aut = np.stack([r, g, b], axis=-1)
            aut = np.clip(aut / 3000 * 255, 0, 255).astype(np.uint8)
        else:
            aut = rgb
        buf2 = io.BytesIO()
        Image.fromarray(aut).resize((64, 64)).save(buf2, "PNG")
        aut_b64 = base64.b64encode(buf2.getvalue()).decode()

        label = d.get("label", None)
        if label is not None:
            h, w = label.shape[-2:]
            lbl_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for ci, c in enumerate(COLORS):
                if ci < 20:
                    lbl_rgb[label.squeeze() == ci] = c
        else:
            lbl_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        buf3 = io.BytesIO()
        Image.fromarray(lbl_rgb).resize((64, 64), Image.NEAREST).save(buf3, "PNG")
        nmd_b64 = base64.b64encode(buf3.getvalue()).decode()

        dates = d.get("dates", [])
        dates_str = ", ".join(str(dd) for dd in dates if str(dd))
        tmask = d.get("temporal_mask", None)
        frames_ok = int(tmask.sum()) if tmask is not None else n_frames

        results.append({"name": name, "rgb": rgb_b64, "nmd": nmd_b64,
                        "autumn": aut_b64, "dates": dates_str[:60], "frames_ok": frames_ok})
    except:
        continue
print(json.dumps(results))
