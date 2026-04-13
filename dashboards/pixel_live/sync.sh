#!/bin/bash
# Syncs train_log.json + tile_preds/ from k8s checkpoint PVC to local dashboard dir.
NS=prithvi-training-default
OUTDIR="$(dirname "$0")"

while true; do
    POD=$(kubectl get pod -n $NS -l job-name=train-pixel-v1 \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$POD" ]; then
        # Training log (always)
        kubectl cp "$NS/$POD:/checkpoints/pixel_v1/train_log.json" \
            "$OUTDIR/training_log.json" 2>/dev/null && \
            echo "[$(date '+%H:%M:%S')] synced train_log.json"

        # Tile prediction manifest + images (when available)
        MANIFEST_REMOTE="/checkpoints/pixel_v1/tile_preds/manifest.json"
        if kubectl exec -n "$NS" "$POD" -- test -f "$MANIFEST_REMOTE" 2>/dev/null; then
            mkdir -p "$OUTDIR/tile_preds"
            kubectl cp "$NS/$POD:/checkpoints/pixel_v1/tile_preds" \
                "$OUTDIR/tile_preds_tmp" 2>/dev/null
            if [ -d "$OUTDIR/tile_preds_tmp" ]; then
                # Atomic swap: only copy new/changed files
                rsync -a --delete "$OUTDIR/tile_preds_tmp/" "$OUTDIR/tile_preds/" 2>/dev/null || \
                    cp -r "$OUTDIR/tile_preds_tmp/." "$OUTDIR/tile_preds/"
                rm -rf "$OUTDIR/tile_preds_tmp"
                echo "[$(date '+%H:%M:%S')] synced tile_preds/"
            fi
        fi
    fi
    sleep 30
done
