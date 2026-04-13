#!/usr/bin/env bash
# run-full-train-pipeline.sh — chains all steps after refetch-missing completes
#
# Sequence:
#   1. Wait for refetch-missing      (writes new tiles to training-data RWO)
#   2. Migrate RWO → CephFS          (rsync; deletes stale train.txt)
#   3. Eval smoke-test checkpoint    (mIoU + per-class table; fast ~10 min)
#   4. build-labels-v2               (rebuilds labels for all tiles on CephFS)
#   5. add-background-frame          (idempotent; only patches new tiles)
#   6. train-pixel-v1                (35-epoch full run; auto-resumes)
#
# Usage (already running in background via nohup):
#   nohup ./k8s/run-full-train-pipeline.sh > /tmp/pipeline.log 2>&1 &
#   tail -f /tmp/pipeline.log
#
set -euo pipefail

NS=prithvi-training-default
export PATH="/usr/local/bin:$PATH"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_job() {
    local job=$1 timeout=${2:-7200}
    log "Waiting for job/$job to complete (timeout ${timeout}s) …"
    kubectl wait --for=condition=complete "job/$job" \
        -n "$NS" --timeout="${timeout}s"
    log "job/$job completed ✓"
}

log "=== ImintEngine Full Training Pipeline ==="
log "Namespace: $NS"
log ""

# ── Step 1/6: Wait for refetch-missing ────────────────────────────────────────
log "Step 1/6 — Waiting for refetch-missing …"
kubectl wait --for=condition=complete job/refetch-missing \
    -n "$NS" --timeout=86400s     # 24h cap (matches job deadline)
log "refetch-missing done ✓"
log ""

# Quick tile count on old RWO PVC before migration
TILE_COUNT=$(kubectl run --rm -i --restart=Never \
    --image=alpine:3.19 count-tiles \
    --overrides='{
      "spec":{
        "volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"training-data"}}],
        "containers":[{"name":"c","image":"alpine:3.19",
          "command":["sh","-c","ls /data/unified_v2/*.npz 2>/dev/null | wc -l"],
          "volumeMounts":[{"name":"d","mountPath":"/data"}]}]
      }
    }' \
    -n "$NS" 2>/dev/null | tail -1 || echo "?")
log "Tiles on training-data (RWO): $TILE_COUNT"
log ""

# ── Step 2/6: Migrate RWO → CephFS ────────────────────────────────────────────
log "Step 2/6 — Migrating training-data (RWO) → training-data-cephfs (RWX) …"
kubectl delete job migrate-to-cephfs -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/migrate-to-cephfs-job.yaml"
wait_for_job migrate-to-cephfs 7200
log ""

# ── Step 3/6: Eval smoke-test checkpoint on CephFS tiles ──────────────────────
# Runs on GPU; uses training-data-cephfs (RWX) + training-checkpoints (RWO).
# Must complete before training starts to avoid training-checkpoints RWO conflict.
log "Step 3/6 — Evaluating smoke-test checkpoint (mIoU) …"
kubectl delete job eval-pixel-v1 -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/eval-pixel-v1-job.yaml"
wait_for_job eval-pixel-v1 1800   # 30 min cap; eval takes ~10 min
log ""
log "=== Smoke-test mIoU results ==="
kubectl logs job/eval-pixel-v1 -n "$NS" 2>/dev/null | \
    python3 -c "
import sys
lines = sys.stdin.readlines()
in_results = False
for l in lines:
    if 'mIoU' in l or 'mAcc' in l or 'Overall' in l or 'Per-class' in l or in_results:
        print(l.rstrip())
        in_results = True
" || true
log ""

# ── Step 4/6: Rebuild labels on CephFS ────────────────────────────────────────
log "Step 4/6 — build-labels-v2 (all tiles including new ones) …"
kubectl delete job build-labels-v2 -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/build-labels-job.yaml"
wait_for_job build-labels-v2 7200
log ""

# ── Step 5/6: Add 2016 background frames ──────────────────────────────────────
log "Step 5/6 — add-background-frame (idempotent, only patches new tiles) …"
kubectl delete job add-background-frame -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/add-background-frame-job.yaml"
wait_for_job add-background-frame 5400
log ""

# ── Step 6/6: Full 35-epoch training run ──────────────────────────────────────
log "Step 6/6 — Starting full 35-epoch training on CephFS PVC …"
kubectl delete job train-pixel-v1 -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/train-pixel-job.yaml"

sleep 15
POD=$(kubectl get pods -n "$NS" \
    --selector=job-name=train-pixel-v1 \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "starting")
log "Training pod: $POD"
log ""
log "=== Pipeline complete — training is running ==="
log "Monitor: kubectl logs job/train-pixel-v1 -n $NS -f"
log ""
log "When the 10h deadline hits, restart with:"
log "  kubectl delete job train-pixel-v1 -n $NS"
log "  kubectl apply -f k8s/train-pixel-job.yaml"
