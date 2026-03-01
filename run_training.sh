#!/bin/bash
# run_training.sh — Setup environment and start LULC training
#
# Usage:
#   ./run_training.sh                          # Default: 30 epochs, batch 8, auto device
#   ./run_training.sh --epochs 10 --device mps # Override params
#   ./run_training.sh --evaluate-only          # Only evaluate best checkpoint
#
# Prerequisites:
#   - Python 3.9+ with pip
#   - macOS (MPS) or Linux (CUDA)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data/lulc_full"
VENV_DIR=".venv"

echo "============================================"
echo "  LULC Training Pipeline"
echo "============================================"

# ── Step 1: Python virtual environment ───────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Ensure python3 is available as python inside venv
if [ ! -e "$VENV_DIR/bin/python" ] && [ -e "$VENV_DIR/bin/python3" ]; then
    ln -s python3 "$VENV_DIR/bin/python"
fi

# ── Step 2: Install dependencies ─────────────────────────────
echo ""
echo "  Checking dependencies..."

# Ensure pip is available in the venv
python3 -m ensurepip --upgrade -q 2>/dev/null || true

# Check if torch is installed (key dependency)
if ! python3 -c "import torch" 2>/dev/null; then
    echo "  Installing dependencies (first run)..."
    python3 -m pip install --upgrade pip -q
    python3 -m pip install -r requirements.txt -q
    echo "  Dependencies installed."
else
    echo "  Dependencies OK."
fi

# ── Step 3: Verify data ─────────────────────────────────────
echo ""
TILE_COUNT=$(ls "$DATA_DIR/tiles/"*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "  Data directory: $DATA_DIR"
echo "  Tiles: $TILE_COUNT"

if [ "$TILE_COUNT" -lt 100 ]; then
    echo "  ERROR: Too few tiles ($TILE_COUNT). Run prepare_lulc_data.py first."
    exit 1
fi

# Check split files
for split in train val test; do
    if [ ! -f "$DATA_DIR/split_${split}.txt" ]; then
        echo "  ERROR: Missing $DATA_DIR/split_${split}.txt"
        exit 1
    fi
done

TRAIN_COUNT=$(wc -l < "$DATA_DIR/split_train.txt" | tr -d ' ')
VAL_COUNT=$(wc -l < "$DATA_DIR/split_val.txt" | tr -d ' ')
TEST_COUNT=$(wc -l < "$DATA_DIR/split_test.txt" | tr -d ' ')
echo "  Splits: train=$TRAIN_COUNT, val=$VAL_COUNT, test=$TEST_COUNT"

if [ ! -f "$DATA_DIR/class_stats.json" ]; then
    echo "  WARNING: No class_stats.json — class weights will be uniform"
fi

# ── Step 4: Detect device ────────────────────────────────────
DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")
echo "  Device: $DEVICE"

# ── Step 5: Start HTTP server for dashboard ──────────────────
DASHBOARD_PORT=8050

# Kill any existing dashboard server on this port
lsof -ti :$DASHBOARD_PORT 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

python3 -m http.server $DASHBOARD_PORT --directory "$DATA_DIR" &
HTTP_PID=$!
echo ""
echo "  Dashboard: http://localhost:$DASHBOARD_PORT/training_dashboard.html"

# Cleanup on exit
cleanup() {
    kill $HTTP_PID 2>/dev/null || true
    echo ""
    echo "  Dashboard server stopped."
}
trap cleanup EXIT

# ── Step 6: Launch training ──────────────────────────────────
echo ""
echo "  Starting training..."
echo "============================================"
echo ""

python3 scripts/train_lulc.py \
    --data-dir "$DATA_DIR" \
    --device "$DEVICE" \
    --dashboard \
    --dashboard-port $DASHBOARD_PORT \
    "$@"

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/lulc/best_model.pt"
echo "============================================"
