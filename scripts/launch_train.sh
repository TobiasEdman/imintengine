#!/bin/bash
# scripts/launch_train.sh — Launch a unified-train-* Job from the template.
#
# Usage:
#   RUN_ID=v7-prithvi300 BACKBONE_NAME=prithvi_300m ./scripts/launch_train.sh
#
#   # Override any default:
#   RUN_ID=v7-foo BACKBONE_NAME=prithvi_300m BATCH_SIZE=16 LR=2e-4 \
#     ./scripts/launch_train.sh
#
# Why a wrapper? The template uses envsubst for variable substitution and
# envsubst does NOT implement bash-style defaults (${VAR:-default}). We set
# defaults here, then call envsubst with simple $VAR references in the yaml.

set -euo pipefail

: "${RUN_ID:?RUN_ID is required (kebab-case, e.g. v7-prithvi300)}"
: "${BACKBONE_NAME:?BACKBONE_NAME is required (registry key, e.g. prithvi_300m)}"

# Optional with defaults matching docs/training/hyperparameters.md
: "${IMG_SIZE:=256}"
: "${BATCH_SIZE:=32}"
: "${LR:=3e-4}"
: "${EPOCHS:=10}"

TEMPLATE="$(dirname "$0")/../k8s/unified-train-template.yaml"
if [ ! -f "$TEMPLATE" ]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

echo "=== launch_train.sh ==="
echo "  RUN_ID:        $RUN_ID"
echo "  BACKBONE_NAME: $BACKBONE_NAME"
echo "  IMG_SIZE:      $IMG_SIZE"
echo "  BATCH_SIZE:    $BATCH_SIZE"
echo "  LR:            $LR"
echo "  EPOCHS:        $EPOCHS"
echo ""

export RUN_ID BACKBONE_NAME IMG_SIZE BATCH_SIZE LR EPOCHS
envsubst '$RUN_ID $BACKBONE_NAME $IMG_SIZE $BATCH_SIZE $LR $EPOCHS' \
  < "$TEMPLATE" \
  | kubectl apply -f -
