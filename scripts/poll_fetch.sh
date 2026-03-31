#!/bin/bash
# Poll fetch job logs and update fetch_status.json for dashboard
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
OUT="/Users/tobiasedman/Developer/ImintEngine/fetch_status.json"
LOGFILE="/tmp/fetch_raw.log"
JOB_LABEL="job-name=fetch-lulc"

CLASS_STATS="/Users/tobiasedman/Developer/ImintEngine/class_stats.json"
echo "Polling $JOB_LABEL logs every 10s → $OUT"

# One-time: scan source tiles on PVC for class distribution
if [ ! -f "$CLASS_STATS" ]; then
  echo "Scanning source tiles for class distribution..."
  POD_INIT=$(kubectl get pods -n prithvi-training-default -l "$JOB_LABEL" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -n "$POD_INIT" ]; then
    kubectl exec -n prithvi-training-default "$POD_INIT" -- python3 -c "
import numpy as np, glob, json, os
from collections import Counter

CLASS_NAMES = ['background','tallskog','granskog','lövskog','blandskog','sumpskog',
    'tillfälligt ej skog','våtmark','öppen mark','bebyggelse','vatten',
    'vete','korn','havre','oljeväxter','vall','potatis','trindsäd','övrig åker','hygge']

pixel_counts = Counter()
tile_dominant = Counter()
tiles_with_crop = 0
tiles_scanned = 0

for d in ['/data/lulc_seasonal/tiles', '/data/crop_tiles']:
    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    for f in files[:3000]:  # Sample up to 3000
        try:
            data = np.load(f, allow_pickle=True)
            label = data.get('label', None)
            if label is None:
                continue
            vals, counts = np.unique(label.flatten(), return_counts=True)
            for v, c in zip(vals, counts):
                if 0 <= v < len(CLASS_NAMES):
                    pixel_counts[int(v)] += int(c)
            dominant = int(vals[np.argmax(counts)])
            if dominant < len(CLASS_NAMES):
                tile_dominant[dominant] += 1
            if any(11 <= v <= 18 for v in vals):
                tiles_with_crop += 1
            tiles_scanned += 1
        except:
            continue

result = {
    'pixel_counts': {CLASS_NAMES[k]: v for k, v in pixel_counts.items()},
    'tile_dominant_class': {CLASS_NAMES[k]: v for k, v in tile_dominant.items()},
    'tiles_with_crop': tiles_with_crop,
    'tiles_scanned': tiles_scanned,
    'class_names': CLASS_NAMES,
}
print(json.dumps(result))
" 2>/dev/null > "$CLASS_STATS" && echo "Class stats saved" || echo "Class stats scan failed"
  fi
fi

POLL_COUNT=0
export POLL_COUNT
while true; do
  POLL_COUNT=$((POLL_COUNT + 1))
  export POLL_COUNT
  POD=$(kubectl get pods -n prithvi-training-default -l "$JOB_LABEL" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  STATUS=$(kubectl get pods -n prithvi-training-default -l "$JOB_LABEL" --field-selector=status.phase=Running -o jsonpath='{.items[0].status.phase}' 2>/dev/null)

  # Fallback: any pod if no running one
  if [ -z "$POD" ]; then
    POD=$(kubectl get pods -n prithvi-training-default -l "$JOB_LABEL" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)
    STATUS=$(kubectl get pods -n prithvi-training-default -l "$JOB_LABEL" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].status.phase}' 2>/dev/null)
  fi

  if [ -z "$POD" ]; then
    echo "$(date): No pod found..."
    sleep 10
    continue
  fi

  kubectl logs "$POD" -n prithvi-training-default --tail=500 > "$LOGFILE" 2>&1

  # Ground truth: count actual tiles on PVC
  PVC_COUNT=$(kubectl exec -n prithvi-training-default "$POD" -- python3 -c "import glob; print(len(glob.glob('/data/unified_v2/*.npz')))" 2>/dev/null || echo "0")
  export PVC_COUNT

  python3 -c "
import re, json, sys
from datetime import datetime

with open('$LOGFILE') as f:
    log = f.read()

lines = log.strip().split('\n')

tiles_ok = 0
tiles_failed = 0
tiles_skipped = 0
tiles_total = 0
rate = 0
processed = 0
last_progress = []

# Parse all progress lines: [N/TOTAL] name: status | rate/h
for line in lines:
    m = re.match(r'\s*\[(\d+)/(\d+)\]\s+(\S+):\s+(\w+)\s+\|\s+(\d+)/h', line)
    if m:
        processed = int(m.group(1))
        tiles_total = int(m.group(2))
        name = m.group(3)
        status_word = m.group(4).lower()
        rate = int(m.group(5))
        if status_word == 'ok':
            tiles_ok += 1
        elif status_word == 'failed':
            tiles_failed += 1
        elif status_word == 'skipped':
            tiles_skipped += 1
        last_progress.append({'n': processed, 'total': tiles_total, 'name': name, 'status': status_word, 'rate': rate})
        continue

    # Simpler [N/TOTAL] without rate
    m2 = re.match(r'\s*\[(\d+)/(\d+)\]', line)
    if m2:
        p = int(m2.group(1))
        t = int(m2.group(2))
        if t > tiles_total:
            tiles_total = t
        if p > processed:
            processed = p

# Use processed count as best estimate
if processed > tiles_ok + tiles_failed + tiles_skipped:
    tiles_ok = processed - tiles_failed - tiles_skipped

# Parse 'Total: N tiles'
m = re.search(r'Total:\s+(\d+)\s+tiles', log)
if m:
    tiles_total = max(tiles_total, int(m.group(1)))

# Final stats line
m = re.search(r'OK=(\d+)\s+Skipped=(\d+)\s+Failed=(\d+)', log)
if m:
    tiles_ok = int(m.group(1))
    tiles_skipped = int(m.group(2))
    tiles_failed = int(m.group(3))

# Phase detection
phase = 'fetching'
if 'Done in' in log:
    phase = 'completed'
elif 'pip install' in log.lower() or 'Installing' in log:
    if processed == 0:
        phase = 'installing'
elif 'Found' in log and processed == 0:
    phase = 'scanning'

# Build progress history from the log (sample every 50 tiles)
history = []
for p in last_progress:
    if p['n'] % 50 == 0 or p['n'] == tiles_total:
        history.append({'n': p['n'], 'rate': p['rate']})

# Get last 15 meaningful log lines (skip 'Fetching spectral bands...')
meaningful = [l for l in lines if l.strip() and 'Fetching spectral bands' not in l and 'pip' not in l.lower()]
last_lines = meaningful[-15:]

# Parse adaptive worker count
workers = 6  # default
for line in reversed(lines):
    m_w = re.search(r'workers=(\d+)', line)
    if m_w:
        workers = int(m_w.group(1))
        break
    m_a = re.search(r'to (\d+) workers', line)
    if m_a:
        workers = int(m_a.group(1))
        break

# Load class stats if available
class_stats = None
try:
    with open('$CLASS_STATS') as cf:
        class_stats = json.load(cf)
except:
    pass

# Use PVC tile count as ground truth for progress
pvc_count = int(os.environ.get('PVC_COUNT', '0'))
if pvc_count > processed:
    processed = pvc_count
    tiles_ok = max(tiles_ok, pvc_count - tiles_failed - tiles_skipped)
    if phase == 'installing' and pvc_count > 0:
        phase = 'fetching'

result = {
    'phase': phase,
    'pod': '$POD',
    'pod_status': '$STATUS',
    'tiles_ok': tiles_ok,
    'tiles_failed': tiles_failed,
    'tiles_skipped': tiles_skipped,
    'tiles_total': tiles_total,
    'processed': processed,
    'pvc_count': pvc_count,
    'rate_per_hour': rate,
    'workers': workers,
    'history': history,
    'class_stats': class_stats,
    'last_update': datetime.now().isoformat(),
    'last_log_lines': last_lines,
}

# Fetch latest tile thumbnails every 6th poll (~60s)
latest_tiles = None
poll_count = int(os.environ.get('POLL_COUNT', '0'))
if poll_count % 6 == 0 and processed > 50:
    try:
        import subprocess, base64
        thumb_script = '''
import numpy as np, glob, os, json, base64, io
from PIL import Image

tiles = sorted(glob.glob("/data/unified_v2/*.npz"), key=os.path.getmtime, reverse=True)[:3]
results = []
for p in tiles:
    try:
        d = np.load(p, allow_pickle=True)
        img = d["image"]  # (24, 256, 256) or (6, 256, 256)
        name = os.path.basename(p).replace(".npz", "")

        # Summer RGB (frame 2 if multitemporal, else frame 0)
        n_frames = img.shape[0] // 6
        summer_idx = min(2, n_frames - 1) * 6
        r, g, b = img[summer_idx + 2], img[summer_idx + 1], img[summer_idx]
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb / 3000 * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).resize((64, 64)).save(buf, "PNG")
        rgb_b64 = base64.b64encode(buf.getvalue()).decode()

        # Autumn RGB (frame 0 if multitemporal)
        if n_frames >= 4:
            r, g, b = img[2], img[1], img[0]
            aut = np.stack([r, g, b], axis=-1)
            aut = np.clip(aut / 3000 * 255, 0, 255).astype(np.uint8)
        else:
            aut = rgb  # fallback
        buf2 = io.BytesIO()
        Image.fromarray(aut).resize((64, 64)).save(buf2, "PNG")
        aut_b64 = base64.b64encode(buf2.getvalue()).decode()

        # NMD label
        label = d.get("label", None)
        COLORS = [(55,65,81),(22,101,52),(21,128,61),(74,222,128),(134,239,172),(6,95,70),
                  (163,230,53),(6,182,212),(251,191,36),(239,68,68),(59,130,246),
                  (245,158,11),(217,119,6),(180,83,9),(234,179,8),(132,204,22),
                  (161,98,7),(202,138,4),(146,64,14),(147,51,234)]
        if label is not None:
            h, w = label.shape[-2:]
            lbl_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for ci, c in enumerate(COLORS):
                lbl_rgb[label.squeeze() == ci] = c
        else:
            lbl_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        buf3 = io.BytesIO()
        Image.fromarray(lbl_rgb).resize((64, 64), Image.NEAREST).save(buf3, "PNG")
        nmd_b64 = base64.b64encode(buf3.getvalue()).decode()

        # Dates + frame count
        dates = d.get("dates", [])
        dates_str = ", ".join(str(dd) for dd in dates if str(dd))
        tmask = d.get("temporal_mask", None)
        frames_ok = int(tmask.sum()) if tmask is not None else n_frames

        results.append({"name": name, "rgb": rgb_b64, "nmd": nmd_b64,
                        "autumn": aut_b64, "dates": dates_str[:60], "frames_ok": frames_ok})
    except:
        continue
print(json.dumps(results))
'''
        r = subprocess.run(
            ['kubectl', 'exec', '-n', 'prithvi-training-default', '$POD', '--', 'python3', '-c', thumb_script],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0 and r.stdout.strip():
            latest_tiles = json.loads(r.stdout.strip())
    except:
        pass

if latest_tiles:
    result['latest_tiles'] = latest_tiles

json.dump(result, open('$OUT', 'w'), indent=2)
pct = (processed / tiles_total * 100) if tiles_total > 0 else 0
print(f'{phase}: {processed}/{tiles_total} ({pct:.1f}%) | ok={tiles_ok} fail={tiles_failed} skip={tiles_skipped} | {rate}/h')
" 2>&1

  if [ "$STATUS" = "Succeeded" ] || [ "$STATUS" = "Failed" ]; then
    echo "Job finished: $STATUS"
    break
  fi

  sleep 10
done
