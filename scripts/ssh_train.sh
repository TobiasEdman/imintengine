#!/bin/bash
# ── ssh_train.sh — Launch training on a pre-configured H100 VM ──────────
#
# Quick-launch for when the VM is already set up (venv, deps, data).
# Starts training in a detached tmux session so SSH disconnects are safe.
#
# Usage:
#   ./scripts/ssh_train.sh user@host
#   ./scripts/ssh_train.sh user@host --batch-size 32 --epochs 100
#   ./scripts/ssh_train.sh user@host --evaluate-only
#
set -euo pipefail

VM_HOST="${1:?Usage: $0 <user>@<host> [extra args for train_lulc.py]}"
shift
EXTRA_ARGS="${*:-}"

echo "  Launching training on $VM_HOST..."

ssh "$VM_HOST" "bash -s" <<EOF
cd ~/ImintEngine
source .venv/bin/activate

# Kill existing training session if any
tmux kill-session -t training 2>/dev/null || true

tmux new-session -d -s training \
    "source ~/ImintEngine/.venv/bin/activate && \\
     python3 scripts/train_lulc.py \\
        --data-dir data/lulc_full \\
        --device cuda \\
        --batch-size 16 \\
        --num-workers 8 \\
        --dashboard \\
        $EXTRA_ARGS \\
        2>&1 | tee data/lulc_full/train.log"

echo ""
echo "  Training started in tmux session 'training'."
echo "  Attach:    tmux attach -t training"
echo "  GPU:       nvidia-smi"
echo "  Logs:      tail -f data/lulc_full/train.log"
EOF
