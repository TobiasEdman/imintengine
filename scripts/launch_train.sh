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
#   # Run on 2080 Ti (smaller pod, FP32, smaller batch):
#   ACCELERATOR=nvidia-gtx-2080ti RUN_ID=v7c-2080-prithvi300 \
#     BACKBONE_NAME=prithvi_300m ./scripts/launch_train.sh
#
# Why a wrapper? The template uses envsubst for variable substitution and
# envsubst does NOT implement bash-style defaults (${VAR:-default}). We set
# defaults here, then call envsubst with simple $VAR references in the yaml.

set -euo pipefail

: "${RUN_ID:?RUN_ID is required (kebab-case, e.g. v7-prithvi300)}"
: "${BACKBONE_NAME:?BACKBONE_NAME is required (registry key, e.g. prithvi_300m)}"

# Accelerator class — drives pod resource sizing + BF16 default.
: "${ACCELERATOR:=nvidia-h100}"

# Per-accelerator defaults. 2080 Ti is Turing — only 11 GB VRAM, no
# native BF16 — so we right-size the pod down and force FP32. H100 keeps
# the v6a-derived defaults: bs=16, lr=2e-4, BF16 on, 48 cpu / 192 Gi.
case "$ACCELERATOR" in
  nvidia-h100)
    : "${BATCH_SIZE:=16}"      # bs=16 + lr=2e-4 = sqrt(16/4) × v6a's 1e-4
    : "${LR:=2e-4}"
    : "${POD_CPU:=48}"
    : "${POD_MEM:=192Gi}"
    : "${DSHM_SIZE:=32Gi}"
    : "${NUM_WORKERS:=16}"
    : "${DISABLE_BF16:=}"      # BF16 on by default on H100
    ;;
  nvidia-gtx-2080ti)
    # 2080 Ti = Turing, 11 GB VRAM. Prithvi-300M FP32 + UPerNet decoder
    # at 4 frames × 256² fits at bs=4. Match v6a hyperparams exactly so
    # this run is a clean v6a replay on a different accelerator.
    : "${BATCH_SIZE:=4}"
    : "${LR:=1e-4}"
    : "${POD_CPU:=8}"
    : "${POD_MEM:=24Gi}"
    : "${DSHM_SIZE:=8Gi}"
    : "${NUM_WORKERS:=4}"
    : "${DISABLE_BF16:=1}"     # 2080 Ti has no native BF16 — force FP32
    ;;
  *)
    echo "Unknown ACCELERATOR=$ACCELERATOR. Valid: nvidia-h100, nvidia-gtx-2080ti" >&2
    exit 1
    ;;
esac

# Run-shape knobs that don't depend on the accelerator.
: "${IMG_SIZE:=256}"
: "${EPOCHS:=10}"

TEMPLATE="$(dirname "$0")/../k8s/unified-train-template.yaml"
if [ ! -f "$TEMPLATE" ]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

echo "=== launch_train.sh ==="
echo "  RUN_ID:        $RUN_ID"
echo "  BACKBONE_NAME: $BACKBONE_NAME"
echo "  ACCELERATOR:   $ACCELERATOR"
echo "  IMG_SIZE:      $IMG_SIZE"
echo "  BATCH_SIZE:    $BATCH_SIZE"
echo "  LR:            $LR"
echo "  EPOCHS:        $EPOCHS"
echo "  DISABLE_BF16:  ${DISABLE_BF16:-(unset → BF16 on)}"
echo "  POD:           cpu=$POD_CPU mem=$POD_MEM dshm=$DSHM_SIZE workers=$NUM_WORKERS"
echo ""

export RUN_ID BACKBONE_NAME ACCELERATOR IMG_SIZE BATCH_SIZE LR EPOCHS DISABLE_BF16
export POD_CPU POD_MEM DSHM_SIZE NUM_WORKERS
envsubst '$RUN_ID $BACKBONE_NAME $ACCELERATOR $IMG_SIZE $BATCH_SIZE $LR $EPOCHS $DISABLE_BF16 $POD_CPU $POD_MEM $DSHM_SIZE $NUM_WORKERS' \
  < "$TEMPLATE" \
  | kubectl apply -f -
