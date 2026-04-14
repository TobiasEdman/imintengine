#!/bin/bash
# watch_and_infer_local.sh — Synkar checkpoint från k8s PVC och kör M1-inference
#
# Kör detta lokalt när träningen har startat:
#   bash scripts/watch_and_infer_local.sh
#
# Vad det gör:
#   1. Synkar best_model.pt från k8s PVC via kubectl cp (kollar var 60s)
#   2. När ett nytt checkpoint finns (nyare mtime) kör inference lokalt på MPS/CPU
#   3. Uppdaterar dashboards/pixel_live/tile_viz/comparison.html med ny kolumn 6

set -euo pipefail

NS=prithvi-training-default
CKPT_LOCAL="/Users/tobiasedman/Developer/ImintEngine/checkpoints/pixel_v1/best_model.pt"
TILES_DIR="/Users/tobiasedman/Developer/ImintEngine/data/viz_tiles"
VIZ_OUT="/Users/tobiasedman/Developer/ImintEngine/data/viz_tiles/col6_inference.json"
REPO="/Users/tobiasedman/Developer/ImintEngine"
PYTHON=python3

mkdir -p "$(dirname $CKPT_LOCAL)"

last_epoch=-1

echo "[$(date '+%H:%M:%S')] Watching for pixel_v1 checkpoint on k8s PVC..."
echo "  Checkpoint dest: $CKPT_LOCAL"
echo "  Tiles: $TILES_DIR"

while true; do
  # Hämta aktiv träningspod
  TRAIN_POD=$(kubectl get pod -n $NS -l job-name=train-pixel-v1 \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

  if [ -z "$TRAIN_POD" ]; then
    echo "[$(date '+%H:%M:%S')] Ingen träningspod hittad — väntar..."
    sleep 30
    continue
  fi

  # Kolla om checkpoint finns och vilken epoch
  REMOTE_EPOCH=$(kubectl exec -n $NS $TRAIN_POD -- \
    python3 -c "
import json,sys
try:
  import torch
  ckpt=torch.load('/checkpoints/pixel_v1/best_model.pt',map_location='cpu',weights_only=True)
  print(ckpt.get('epoch',-1))
except:
  print(-1)
" 2>/dev/null || echo "-1")

  if [ "$REMOTE_EPOCH" != "-1" ] && [ "$REMOTE_EPOCH" != "$last_epoch" ]; then
    echo "[$(date '+%H:%M:%S')] Nytt checkpoint: epoch $REMOTE_EPOCH (var $last_epoch) — synkar..."

    kubectl cp "$NS/$TRAIN_POD:/checkpoints/pixel_v1/best_model.pt" \
      "$CKPT_LOCAL" 2>/dev/null && \
      echo "[$(date '+%H:%M:%S')] Checkpoint synkad: $(ls -lh $CKPT_LOCAL | awk '{print $5}')"

    # Kör M1-inference
    echo "[$(date '+%H:%M:%S')] Kör inference på $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'lokal CPU')..."

    cd "$REPO"
    $PYTHON scripts/infer_local_mps.py \
      --checkpoint "$CKPT_LOCAL" \
      --tiles-dir "$TILES_DIR" \
      --out "$VIZ_OUT" \
      --context-px 32 \
      --use-frame-2016 \
      --stride 1 2>&1 | tail -5

    if [ -f "$VIZ_OUT" ]; then
      echo "[$(date '+%H:%M:%S')] Inferens klar — regenererar HTML..."
      $PYTHON scripts/build_comparison_html.py 2>&1 | tail -3
      echo "[$(date '+%H:%M:%S')] HTML uppdaterad ✓"
    fi

    last_epoch=$REMOTE_EPOCH
  fi

  sleep 60
done
