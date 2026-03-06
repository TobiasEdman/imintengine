#!/bin/bash
# ── setup_h100_vm.sh — Bootstrap ImintEngine on an H100 VM ──────────────
#
# Automates full setup of a fresh GPU VM at ICE Connect (RISE Luleå).
#
# Usage (from local machine):
#   ./scripts/setup_h100_vm.sh user@host
#   ./scripts/setup_h100_vm.sh user@host --transfer-data
#   ./scripts/setup_h100_vm.sh user@host --transfer-data --start-training
#
# Prerequisites:
#   - SSH access to the VM (key-based recommended)
#   - NVIDIA drivers pre-installed on the VM (standard for H100 VMs)
#   - Dataset at data/lulc_full/ on local machine (for --transfer-data)
#
set -euo pipefail

VM_HOST="${1:?Usage: $0 <user>@<host> [--transfer-data] [--start-training]}"
shift

TRANSFER_DATA=false
START_TRAINING=false
REPO_URL="https://github.com/TobiasEdman/imintengine.git"
LOCAL_DATA_DIR="data/lulc_full"
REMOTE_PROJECT_DIR="\$HOME/ImintEngine"
REMOTE_DATA_DIR="\$HOME/ImintEngine/data/lulc_full"

for arg in "$@"; do
    case "$arg" in
        --transfer-data)  TRANSFER_DATA=true ;;
        --start-training) START_TRAINING=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo "  H100 VM Setup — ICE Connect (RISE Luleå)"
echo "============================================"
echo "  Target: $VM_HOST"
echo ""

# ── Step 1: Verify GPU + install system deps ────────────────────────────
echo "  [1/5] Checking GPU and installing system dependencies..."
ssh "$VM_HOST" 'bash -s' <<'REMOTE_DEPS'
set -euo pipefail

# Check NVIDIA driver
if ! nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found. NVIDIA drivers must be pre-installed."
    exit 1
fi
echo "  GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'

# System packages
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git python3.11 python3.11-dev python3.11-venv python3-pip \
    libgdal-dev tmux rsync htop 2>/dev/null || \
sudo apt-get install -y -qq \
    git python3 python3-dev python3-venv python3-pip \
    libgdal-dev tmux rsync htop 2>/dev/null

echo "  System deps OK."
REMOTE_DEPS

# ── Step 2: Clone or update repo ────────────────────────────────────────
echo "  [2/5] Setting up repository..."
ssh "$VM_HOST" "bash -s" <<REMOTE_REPO
set -euo pipefail
if [ -d ~/ImintEngine/.git ]; then
    cd ~/ImintEngine
    git pull --ff-only origin main
    echo "  Repo updated."
else
    git clone "$REPO_URL" ~/ImintEngine
    echo "  Repo cloned."
fi
REMOTE_REPO

# ── Step 3: Python venv + CUDA PyTorch ──────────────────────────────────
echo "  [3/5] Setting up Python environment with CUDA PyTorch..."
ssh "$VM_HOST" 'bash -s' <<'REMOTE_VENV'
set -euo pipefail
cd ~/ImintEngine

# Detect python version
PY=$(command -v python3.11 || command -v python3)

if [ ! -d .venv ]; then
    $PY -m venv .venv
    echo "  Created venv."
fi
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install CUDA-enabled PyTorch from official index
echo "  Installing CUDA PyTorch (this may take a few minutes)..."
pip install torch --index-url https://download.pytorch.org/whl/cu124 -q

# Install remaining requirements
echo "  Installing project requirements..."
pip install -r requirements.txt -q

# Verify CUDA
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  VRAM: {mem_gb:.1f} GB')
else:
    print('  WARNING: CUDA not available — check drivers')
"
REMOTE_VENV

# ── Step 4: Transfer dataset ────────────────────────────────────────────
if [ "$TRANSFER_DATA" = true ]; then
    echo "  [4/5] Transferring dataset ($(du -sh "$LOCAL_DATA_DIR" 2>/dev/null | cut -f1 || echo '~6 GB'))..."
    ssh "$VM_HOST" "mkdir -p ~/ImintEngine/data/lulc_full"
    rsync -avz --progress --compress \
        "$LOCAL_DATA_DIR/" \
        "$VM_HOST:~/ImintEngine/data/lulc_full/"
    echo "  Dataset transferred."
else
    echo "  [4/5] Skipping data transfer (use --transfer-data to enable)."
fi

# ── Step 5: Launch training ─────────────────────────────────────────────
if [ "$START_TRAINING" = true ]; then
    echo "  [5/5] Launching training in tmux session..."
    ssh "$VM_HOST" 'bash -s' <<'REMOTE_TRAIN'
cd ~/ImintEngine
source .venv/bin/activate

# Kill existing training session if any
tmux kill-session -t training 2>/dev/null || true

tmux new-session -d -s training \
    "source ~/ImintEngine/.venv/bin/activate && \
     python3 scripts/train_lulc.py \
        --data-dir data/lulc_full \
        --device cuda \
        --batch-size 16 \
        --num-workers 8 \
        --epochs 50 \
        --enable-all-aux \
        --dashboard \
        2>&1 | tee data/lulc_full/train.log"

echo "  Training started in tmux session 'training'."
REMOTE_TRAIN
else
    echo "  [5/5] Skipping training launch (use --start-training to enable)."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Connect:   ssh $VM_HOST"
echo "  Attach:    ssh $VM_HOST -t 'tmux attach -t training'"
echo "  GPU:       ssh $VM_HOST nvidia-smi"
echo "  Dashboard: ssh -L 8050:localhost:8050 $VM_HOST"
echo "============================================"
