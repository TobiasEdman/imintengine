#!/bin/bash
# ── k8s/cleanup.sh — Download results and clean up k8s resources ──────────
#
# Run this after training completes to:
#   1. Download model checkpoints to local machine
#   2. Download training logs
#   3. Delete all k8s resources (pods, PVCs, namespace)
#
# Usage:
#   ./k8s/cleanup.sh                    # Download + cleanup
#   ./k8s/cleanup.sh --download-only    # Just download, keep resources
#   ./k8s/cleanup.sh --status           # Just check job status
#
set -euo pipefail
K="kubectl --insecure-skip-tls-verify"
NS="prithvi-training-default"

if [[ "${1:-}" == "--status" ]]; then
    echo "  Job status:"
    $K get job prithvi-seasonal-train -n $NS 2>/dev/null || echo "  No job found"
    echo ""
    echo "  Pods:"
    $K get pods -n $NS 2>/dev/null
    echo ""
    POD=$($K get pods -n $NS -l app=prithvi-training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "$POD" ]; then
        echo "  Last 20 log lines:"
        $K logs "$POD" -n $NS --tail=20 2>/dev/null || true
    fi
    exit 0
fi

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  Downloading results from ICE k8s                       ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# Create download pod to access PVCs
echo "  [1/4] Starting download pod..."
cat <<YAML | $K apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: data-download
  namespace: $NS
spec:
  containers:
    - name: dl
      image: busybox:1.36
      command: ["sleep", "3600"]
      resources:
        requests:
          cpu: "2"
          memory: 4Gi
        limits:
          cpu: "2"
          memory: 4Gi
      volumeMounts:
        - name: checkpoints
          mountPath: /checkpoints
        - name: data
          mountPath: /data
  volumes:
    - name: checkpoints
      persistentVolumeClaim:
        claimName: training-checkpoints
    - name: data
      persistentVolumeClaim:
        claimName: training-data
  restartPolicy: Never
YAML

$K wait --for=condition=Ready pod/data-download -n $NS --timeout=120s

# Download checkpoints
echo "  [2/4] Downloading checkpoints..."
mkdir -p checkpoints/lulc_seasonal
$K cp $NS/data-download:/checkpoints/ checkpoints/lulc_seasonal/ 2>/dev/null || \
    echo "  WARNING: No checkpoints found (training may not have completed)"

# Download training log
echo "  [3/4] Downloading training log..."
$K cp $NS/data-download:/data/lulc_seasonal/training_log.json data/lulc_seasonal/training_log.json 2>/dev/null || true
$K cp $NS/data-download:/data/lulc_seasonal/train.log data/lulc_seasonal/train.log 2>/dev/null || true

echo "  Downloaded to:"
ls -lh checkpoints/lulc_seasonal/ 2>/dev/null || echo "    (no checkpoints)"
ls -lh data/lulc_seasonal/training_log.json 2>/dev/null || true

if [[ "${1:-}" == "--download-only" ]]; then
    $K delete pod data-download -n $NS --grace-period=0 2>/dev/null || true
    echo "  Download complete (resources kept)."
    exit 0
fi

# Cleanup
echo "  [4/4] Cleaning up k8s resources..."
$K delete pod data-download -n $NS --grace-period=0 2>/dev/null || true
$K delete job prithvi-seasonal-train -n $NS 2>/dev/null || true
$K delete pod data-upload -n $NS --grace-period=0 2>/dev/null || true
$K delete pvc training-data training-checkpoints -n $NS 2>/dev/null || true
$K delete namespace $NS 2>/dev/null || true

echo ""
echo "  All k8s resources cleaned up. Checkpoints saved locally."
