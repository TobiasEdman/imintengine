#!/usr/bin/env bash
# run-full-train-pipeline.sh — chains all steps after refetch-missing completes
#
# Run this from your local machine. It polls refetch-missing until done,
# then: migrate → build-labels → add-background-frame → train-pixel
#
# Usage:
#   chmod +x k8s/run-full-train-pipeline.sh
#   ./k8s/run-full-train-pipeline.sh 2>&1 | tee k8s/pipeline-$(date +%Y%m%d-%H%M).log
#
set -euo pipefail

NS=prithvi-training-default
K8S=/usr/local/bin
export PATH="$K8S:$PATH"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

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

# ── Step 1: Wait for refetch-missing ──────────────────────────────────────────
log "Step 1/5 — Waiting for refetch-missing (may take several hours) …"
kubectl wait --for=condition=complete job/refetch-missing \
    -n "$NS" --timeout=86400s     # 24h cap (matches job deadline)
log "refetch-missing done ✓"
log ""

# Quick tile count
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
log "Tiles on training-data PVC: $TILE_COUNT"
log ""

# ── Step 2: Migrate RWO → CephFS ──────────────────────────────────────────────
log "Step 2/5 — Migrating training-data (RWO) → training-data-cephfs (RWX) …"
kubectl delete job migrate-to-cephfs -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/migrate-to-cephfs-job.yaml"
wait_for_job migrate-to-cephfs 7200
log ""

# ── Step 3: Rebuild labels on CephFS PVC ──────────────────────────────────────
log "Step 3/5 — build-labels-v2 (all tiles including new ones) …"
kubectl delete job build-labels-v2 -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/build-labels-job.yaml"
wait_for_job build-labels-v2 3600
log ""

# ── Step 4: Add 2016 background frames ────────────────────────────────────────
log "Step 4/5 — add-background-frame (idempotent, only patches new tiles) …"
kubectl delete job add-background-frame -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/add-background-frame-job.yaml"
wait_for_job add-background-frame 5400
log ""

# ── Step 5: Full 35-epoch training run ────────────────────────────────────────
log "Step 5/5 — Starting full 35-epoch training on CephFS PVC …"
kubectl delete job train-pixel-v1 -n "$NS" --ignore-not-found
kubectl apply -f "$(dirname "$0")/train-pixel-job.yaml"

# Brief pause then confirm it's running
sleep 15
POD=$(kubectl get pods -n "$NS" \
    --selector=job-name=train-pixel-v1 \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "starting")
log "Training pod: $POD"
log ""
log "=== Pipeline complete — training is running ==="
log "Monitor with:"
log "  kubectl logs job/train-pixel-v1 -n $NS -f"
log ""
log "When the 10h deadline hits, the job auto-restarts:"
log "  kubectl delete job train-pixel-v1 -n $NS"
log "  kubectl apply -f k8s/train-pixel-job.yaml"
