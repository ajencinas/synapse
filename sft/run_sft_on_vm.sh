#!/usr/bin/env bash
# Bootstrap SFT on a fresh Lambda Labs / RunPod GPU VM.
#
# Usage (after `git clone` and `cd` into the repo):
#   bash sft/run_sft_on_vm.sh
#
# What it does:
#   1. Sanity-checks GPU + tools (nvidia-smi, git, python3, rclone).
#   2. Installs torch if missing (sft.py needs ONLY torch).
#   3. Verifies the configured rclone Drive remote exists.
#   4. rclone-pulls the SFT inputs from Drive -> local SSD:
#        tokenizer_out/, manifests/, sft_tokenized/, the single latest pretrain ckpt,
#        and any existing sft_checkpoints/ (for resume across VM rebuilds).
#   5. Sets CHECKPOINT_PUSH_REMOTE so sft.py mirrors checkpoints back to Drive in
#      the background, then runs sft/sft.py.
#
# Override knobs (set before running):
#   LOCAL_DIR        Base dir on local SSD. Auto: /home/ubuntu/synapse_data (Lambda),
#                    /workspace/synapse_data (RunPod), else ./synapse_data.
#   GDRIVE_REMOTE    rclone remote name (default: gdrive).
#   GDRIVE_PATH      Base path on Drive (default: synapse).
#   PRETRAIN_CKPT    Pretrain checkpoint filename (default: synapse_2b_d2560_l28.pth).
#   SKIP_DATA_PULL   If "1", skip step 4 (assume inputs are already local).
set -euo pipefail

# ---------- Auto-detect LOCAL_DIR ----------
if [[ -z "${LOCAL_DIR:-}" ]]; then
    if [[ -d /home/ubuntu ]]; then
        LOCAL_DIR=/home/ubuntu/synapse_data
    elif [[ -d /workspace ]]; then
        LOCAL_DIR=/workspace/synapse_data
    else
        LOCAL_DIR=$(pwd)/synapse_data
    fi
fi
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_PATH="${GDRIVE_PATH:-synapse}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-synapse_2b_d2560_l28.pth}"
SYNAPSE_DIR="${LOCAL_DIR}/synapse"
REMOTE="${GDRIVE_REMOTE}:${GDRIVE_PATH}"

echo "================================================================"
echo "  SynapseGPT SFT bootstrap"
echo "  SYNAPSE_DIR: $SYNAPSE_DIR"
echo "  Drive:       $REMOTE"
echo "================================================================"

# ---------- Step 1: sanity checks ----------
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found - is this a GPU box?"; exit 1; }
command -v git        >/dev/null 2>&1 || { echo "ERROR: git not found"; exit 1; }
command -v python3    >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------- Step 2: deps ----------
echo
echo "[2/5] Checking Python deps (torch)..."
python3 -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "  installing torch..."; pip install -q torch; }

# ---------- Step 3: rclone ----------
echo
echo "[3/5] Verifying rclone remote..."
if ! command -v rclone >/dev/null 2>&1; then
    echo "  rclone not found, installing..."
    curl -fsS https://rclone.org/install.sh | sudo bash
fi
if ! rclone listremotes | grep -q "^${GDRIVE_REMOTE}:$"; then
    echo "ERROR: rclone remote '${GDRIVE_REMOTE}:' not configured. Run 'rclone config'."
    exit 1
fi

# ---------- Step 4: pull inputs ----------
if [[ "${SKIP_DATA_PULL:-0}" != "1" ]]; then
    echo
    echo "[4/5] Pulling SFT inputs from Drive -> $SYNAPSE_DIR ..."
    mkdir -p "$SYNAPSE_DIR/checkpoints"
    rclone copy "$REMOTE/tokenizer_out" "$SYNAPSE_DIR/tokenizer_out" --progress
    rclone copy "$REMOTE/manifests"     "$SYNAPSE_DIR/manifests"     --progress
    rclone copy "$REMOTE/sft_tokenized" "$SYNAPSE_DIR/sft_tokenized" --progress
    # ONLY the latest pretrain checkpoint (not the whole history dir)
    rclone copyto "$REMOTE/checkpoints/$PRETRAIN_CKPT" \
                  "$SYNAPSE_DIR/checkpoints/$PRETRAIN_CKPT" \
                  --checksum --drive-chunk-size=64M --progress
    # existing SFT checkpoints, if any, so we can resume across VM rebuilds
    rclone copy "$REMOTE/sft_checkpoints" "$SYNAPSE_DIR/sft_checkpoints" --progress || true
else
    echo "[4/5] SKIP_DATA_PULL=1 — assuming inputs already at $SYNAPSE_DIR"
fi

# ---------- Step 5: train ----------
echo
echo "[5/5] Starting SFT..."
export SYNAPSE_DIR
export SKIP_DRIVE_MOUNT=1
export CHECKPOINT_NAME="$PRETRAIN_CKPT"
# Background-mirror checkpoints back to Drive so an ephemeral pod can't lose them.
export CHECKPOINT_PUSH_REMOTE="${REMOTE}/sft_checkpoints"
exec python3 sft/sft.py
