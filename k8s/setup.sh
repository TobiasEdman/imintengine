#!/bin/bash
# ── k8s/setup.sh — Deploy training to ICE Connect Kubernetes ──────────────
#
# Prerequisites:
#   1. Download kubeconfig from https://k8s.ice.ri.se
#   2. Install kubectl: brew install kubectl
#   3. Docker Desktop running (for building image)
#
# Usage:
#   ./k8s/setup.sh              # Full setup: build, push, deploy, upload, train
#   ./k8s/setup.sh --train-only # Just submit training job (data already uploaded)
#
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────────
NAMESPACE="prithvi-training-default"
IMAGE_NAME="imint-engine"
IMAGE_TAG="cuda-seasonal"
REGISTRY="${REGISTRY:-docker.io}"          # Override: REGISTRY=harbor.ice.ri.se
REGISTRY_USER="${REGISTRY_USER:-}"         # Override: REGISTRY_USER=youruser
FULL_IMAGE="${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:${IMAGE_TAG}"
DATA_DIR="data/lulc_seasonal"
KUBECONFIG_PATH="${KUBECONFIG:-$HOME/.kube/config}"

TRAIN_ONLY=false
if [[ "${1:-}" == "--train-only" ]]; then
    TRAIN_ONLY=true
fi

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  ICE Connect — Prithvi Seasonal Training Deployment     ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Check prerequisites ──────────────────────────────────────────
echo "  [0/6] Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "  ERROR: kubectl not found. Install: brew install kubectl"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "  ERROR: docker not found. Install Docker Desktop"; exit 1; }

if [ ! -f "$KUBECONFIG_PATH" ]; then
    echo ""
    echo "  Kubeconfig not found at $KUBECONFIG_PATH"
    echo ""
    echo "  Steps to get it:"
    echo "    1. Go to https://k8s.ice.ri.se"
    echo "    2. Log in to Rancher"
    echo "    3. Click your user icon → 'Download kubeconfig'"
    echo "    4. Save to ~/.kube/config"
    echo ""
    exit 1
fi

export KUBECONFIG="$KUBECONFIG_PATH"
echo "  kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client 2>/dev/null | head -1)"
echo "  kubeconfig: $KUBECONFIG_PATH"

if [ -z "$REGISTRY_USER" ]; then
    echo ""
    echo "  ERROR: Set REGISTRY_USER before running:"
    echo "    export REGISTRY_USER=yourdockerhubuser"
    echo "    # or for ICE Harbor:"
    echo "    export REGISTRY=harbor.ice.ri.se REGISTRY_USER=youruser"
    echo ""
    exit 1
fi

FULL_IMAGE="${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Image: $FULL_IMAGE"
echo ""

if [ "$TRAIN_ONLY" = true ]; then
    echo "  [TRAIN-ONLY] Skipping build/push/upload, submitting job..."
    # Delete old job if exists
    kubectl delete job prithvi-seasonal-train -n "$NAMESPACE" 2>/dev/null || true
    sed "s|IMAGE_PLACEHOLDER|${FULL_IMAGE}|g" k8s/job-train.yaml | kubectl apply -f -
    echo ""
    echo "  Training job submitted!"
    echo "  Logs:    kubectl logs -f job/prithvi-seasonal-train -n $NAMESPACE"
    echo "  Status:  kubectl get pods -n $NAMESPACE"
    exit 0
fi

# ── Step 1: Build Docker image (x86_64 for cluster) ─────────────────────
echo "  [1/6] Building Docker image (linux/amd64)..."
docker build --platform linux/amd64 -f Dockerfile.cuda -t "$FULL_IMAGE" .
echo "  Built: $FULL_IMAGE"

# ── Step 2: Push to registry ─────────────────────────────────────────────
echo ""
echo "  [2/6] Pushing to $REGISTRY..."
docker push "$FULL_IMAGE"
echo "  Pushed."

# ── Step 3: Create namespace + PVCs ──────────────────────────────────────
echo ""
echo "  [3/6] Creating namespace and storage..."
kubectl apply -f k8s/namespace.yaml 2>/dev/null || true

# Set namespace quotas (required by ICE)
echo "  NOTE: Set namespace quotas in Rancher GUI if not done yet."

kubectl apply -f k8s/pvc-data.yaml
kubectl apply -f k8s/pvc-checkpoints.yaml
echo "  PVCs created."

# ── Step 4: Upload data ──────────────────────────────────────────────────
echo ""
echo "  [4/6] Uploading training data..."

# Start uploader pod
kubectl apply -f k8s/job-upload-data.yaml
echo "  Waiting for uploader pod..."
kubectl wait --for=condition=Ready pod/data-upload -n "$NAMESPACE" --timeout=120s

# Create tar of data (faster than kubectl cp for many small files)
echo "  Compressing data..."
DATA_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
echo "  Data size: $DATA_SIZE"

echo "  Uploading (this may take 20-40 minutes)..."
tar czf - -C data lulc_seasonal | \
    kubectl exec -i data-upload -n "$NAMESPACE" -- tar xzf - -C /data/

echo "  Upload complete."

# Clean up uploader pod
kubectl delete pod data-upload -n "$NAMESPACE" --grace-period=0

# ── Step 5: Submit training job ──────────────────────────────────────────
echo ""
echo "  [5/6] Submitting training job..."
sed "s|IMAGE_PLACEHOLDER|${FULL_IMAGE}|g" k8s/job-train.yaml | kubectl apply -f -

# ── Step 6: Show status ─────────────────────────────────────────────────
echo ""
echo "  [6/6] Done!"
echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  Training submitted to ICE Connect k8s cluster          ║"
echo "  ╠══════════════════════════════════════════════════════════╣"
echo "  ║  Image:     $FULL_IMAGE"
echo "  ║  Namespace: $NAMESPACE"
echo "  ║  GPU:       1x A100/H100 (requested)"
echo "  ║  Data:      19-class multitemporal + aux + VPP"
echo "  ╠══════════════════════════════════════════════════════════╣"
echo "  ║  Useful commands:                                       ║"
echo "  ║    kubectl get pods -n $NAMESPACE"
echo "  ║    kubectl logs -f job/prithvi-seasonal-train -n $NAMESPACE"
echo "  ║    kubectl describe job prithvi-seasonal-train -n $NAMESPACE"
echo "  ║    kubectl port-forward pod/<pod> 8000:8000 -n $NAMESPACE"
echo "  ╚══════════════════════════════════════════════════════════╝"
