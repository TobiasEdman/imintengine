#!/bin/bash
# Poll training logs and save to JSON for dashboard
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
OUT="/Users/tobiasedman/Developer/ImintEngine/training_log.json"
LOGFILE="/Users/tobiasedman/Developer/ImintEngine/training_raw.log"

echo "Polling unified-train-v1 logs every 30s → $OUT"

while true; do
  # Get full logs from specific pod
  POD=$(kubectl get pods -n prithvi-training-default -l app=unified-training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  LOGS=$(kubectl logs "$POD" -n prithvi-training-default 2>/dev/null)

  if [ -z "$LOGS" ]; then
    echo "$(date): No logs yet..."
    sleep 30
    continue
  fi

  # Save raw log (use proper method to avoid pipe truncation)
  POD_FULL=$(kubectl get pods -n prithvi-training-default -l app=unified-training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  kubectl logs "$POD_FULL" -n prithvi-training-default > "$LOGFILE" 2>&1

  # Parse with python
  python3 -c "
import re, json, sys

log = open('$LOGFILE').read()

epochs = []
for m in re.finditer(r'Epoch\s+(\d+)/(\d+)\s+\|\s+loss=([\d.]+)\s+\|\s+val_mIoU=([\d.]+)\s+\|\s+worst=([\d.]+)\s+\((\w+)\)\s+\|\s+lr=([\d.e+-]+)\s+\|\s+(\d+)s', log):
    ep = {
        'epoch': int(m.group(1)),
        'total': int(m.group(2)),
        'loss': float(m.group(3)),
        'val_miou': float(m.group(4)),
        'worst_iou': float(m.group(5)),
        'worst_class': m.group(6),
        'lr': float(m.group(7)),
        'time_s': int(m.group(8)),
    }
    epochs.append(ep)

# Parse last per-class IoU block
iou_block = {}
for m in re.finditer(r'^\s+([\w_]+)\s+([\d.]+)\s+', log, re.MULTILINE):
    iou_block[m.group(1)] = float(m.group(2))

# Parse model info
train_tiles = 0
val_tiles = 0
m = re.search(r'Train:\s+(\d+)\s+tiles.*Val:\s+(\d+)\s+tiles', log)
if m:
    train_tiles = int(m.group(1))
    val_tiles = int(m.group(2))

num_classes = 19
m = re.search(r'Num classes:\s+(\d+)', log)
if m:
    num_classes = int(m.group(1))

best_miou = max((e['val_miou'] for e in epochs), default=0)

result = {
    'epochs': epochs,
    'per_class_iou': iou_block,
    'train_tiles': train_tiles,
    'val_tiles': val_tiles,
    'num_classes': num_classes,
    'best_miou': best_miou,
    'last_update': __import__('datetime').datetime.now().isoformat(),
}

json.dump(result, open('$OUT', 'w'), indent=2)
print(f'Epoch {len(epochs)}/{epochs[-1][\"total\"] if epochs else \"?\"} | mIoU={best_miou:.4f} | {len(iou_block)} classes')
" 2>&1

  # Check if job still running
  STATUS=$(kubectl get pods -n prithvi-training-default -l app=unified-training -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
  if [ "$STATUS" = "Succeeded" ] || [ "$STATUS" = "Failed" ]; then
    echo "Job finished with status: $STATUS"
    break
  fi

  sleep 30
done
