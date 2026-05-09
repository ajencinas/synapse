#!/usr/bin/env bash
# Bootstrap pretraining on a fresh Lambda Labs / RunPod GPU VM.
#
# Usage (after `git clone` and `cd` into the repo):
#   bash pretrain/run_on_vm.sh
#
# What it does:
#   1. Sanity-checks GPU + tools.
#   2. Installs Python deps (torch, numpy, tqdm).
#   3. Installs rclone if missing; verifies the configured Drive remote exists.
#   4. rclone copy gdrive:synapse/token_shards_merged -> $LOCAL_DIR (one-time, ~300 GB).
#   5. Pulls existing checkpoint and manifest if any (resume across VM rebuilds).
#   6. Sets env vars and execs python pretrain/train.py.
#
# Override knobs (set before running):
#   LOCAL_DIR        Base dir for staged data and checkpoints on local SSD.
#                    Auto-detected: /home/ubuntu/synapse_data on Lambda,
#                    /workspace/synapse_data on RunPod, ./synapse_data otherwise.
#   GDRIVE_REMOTE    rclone remote name (default: gdrive).
#   GDRIVE_PATH      Base path on Drive (default: synapse).
#   MAX_TOKENS       Token budget passed to train.py.
#   CHECKPOINT_NAME  Checkpoint filename passed to train.py.
#   SKIP_DATA_PULL   If "1", skip step 4 (assume shards are already local).

set -euo pipefail

# ---------- Auto-detect LOCAL_DIR ----------
if [[ -z "${LOCAL_DIR:-}" ]]; then
    if [[ -d /home/ubuntu ]]; then
        LOCAL_DIR=/home/ubuntu/synapse_data           # Lambda Labs default
    elif [[ -d /workspace ]]; then
        LOCAL_DIR=/workspace/synapse_data             # RunPod default
    else
        LOCAL_DIR=$(pwd)/synapse_data
    fi
fi
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_PATH="${GDRIVE_PATH:-synapse}"
SHARD_REMOTE="${GDRIVE_REMOTE}:${GDRIVE_PATH}/token_shards_merged"
CKPT_REMOTE="${GDRIVE_REMOTE}:${GDRIVE_PATH}/checkpoints"
MANIFEST_REMOTE="${GDRIVE_REMOTE}:${GDRIVE_PATH}/manifests"
SHARD_LOCAL="${LOCAL_DIR}/synapse/token_shards_merged"
CKPT_LOCAL="${LOCAL_DIR}/synapse/checkpoints"
MANIFEST_LOCAL="${LOCAL_DIR}/synapse/manifests"

echo "================================================================"
echo "  SynapseGPT pretraining bootstrap"
echo "  LOCAL_DIR: $LOCAL_DIR"
echo "  Drive:     ${GDRIVE_REMOTE}:${GDRIVE_PATH}"
echo "================================================================"

# ---------- Step 1: sanity checks ----------
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found - is this a GPU box?"; exit 1; }
command -v git        >/dev/null 2>&1 || { echo "ERROR: git not found"; exit 1; }
command -v python3    >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------- Step 2: install Python deps ----------
echo
echo "[2/6] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch numpy tqdm
[[ -f requirements.txt ]] && pip install --quiet -r requirements.txt || true

# ---------- Step 3: rclone install + verify remote ----------
echo
echo "[3/6] Verifying rclone..."
if ! command -v rclone >/dev/null 2>&1; then
    echo "  rclone not found, installing..."
    curl -fsS https://rclone.org/install.sh | sudo bash
fi
if ! rclone listremotes | grep -q "^${GDRIVE_REMOTE}:$"; then
    echo "ERROR: rclone remote '${GDRIVE_REMOTE}:' not configured."
    echo
    echo "Run 'rclone config' (interactive) and set up a Google Drive remote"
    echo "named '${GDRIVE_REMOTE}'. Then re-run this script."
    exit 1
fi
echo "  remote '${GDRIVE_REMOTE}:' OK"

# ---------- Step 4: pull merged shards ----------
mkdir -p "${SHARD_LOCAL}" "${CKPT_LOCAL}" "${MANIFEST_LOCAL}"

if [[ "${SKIP_DATA_PULL:-0}" == "1" ]]; then
    echo
    echo "[4/6] Skipping shard pull (SKIP_DATA_PULL=1)"
else
    echo
    echo "[4/6] Pulling merged shards: ${SHARD_REMOTE} -> ${SHARD_LOCAL}"
    echo "      (one-time, expect ~300 GB depending on uploads)"
    rclone copy "${SHARD_REMOTE}" "${SHARD_LOCAL}" \
        --transfers=8 --drive-chunk-size=64M --checksum --progress \
        --stats=10s --stats-one-line
fi

# ---------- Step 5: pull existing checkpoint + manifest (resume support) ----------
echo
echo "[5/6] Pulling existing checkpoint (if any) for resume..."
rclone copy "${CKPT_REMOTE}" "${CKPT_LOCAL}" \
    --include "*.pth" --max-age 30d --transfers=4 --drive-chunk-size=64M \
    --checksum --progress --stats=10s --stats-one-line || true

echo "  Pulling existing manifest (if any)..."
rclone copy "${MANIFEST_REMOTE}" "${MANIFEST_LOCAL}" \
    --include "training_latest.json" --transfers=2 --checksum || true

# ---------- Step 6: exec trainer ----------
echo
echo "[6/6] Starting training..."
echo

export SYNAPSE_DIR="${LOCAL_DIR}/synapse"
export SKIP_DRIVE_MOUNT=1                # not on Colab
export SKIP_STAGE=1                      # data already on local SSD
export CHECKPOINT_PUSH_REMOTE="${CKPT_REMOTE}"

exec python3 pretrain/train.py
