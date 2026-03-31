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

for d in ['/data/unified_v2']:
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
import re, json, sys, os
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
if poll_count % 6 == 0 and pvc_count > 50:
    try:
        import subprocess
        r = subprocess.run(
            ['kubectl', 'exec', '-n', 'prithvi-training-default', '$POD', '--',
             'python3', '/workspace/imintengine/scripts/gen_thumbnails.py'],
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
