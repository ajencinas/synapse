# SynapseGPT Training on Lambda AI

Run `pretrain/train.py` on a Lambda GPU Cloud VM. Training pulls tokenized shards from
Google Drive, trains a ~2.1B-parameter SynapseGPT model, and pushes checkpoints back to
Drive so they survive ephemeral instances.

---

## Prerequisites

- A [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud) instance (see below).
- `token_shards_merged/`, `tokenizer_out/`, `manifests/`, `checkpoints/` uploaded to
  Google Drive under a shared parent (e.g. `gdrive:synapse/`).
- `rclone` installed and authenticated with that Drive account (done automatically by
  `run_on_vm.sh` if you run `rclone config` first).

---

## GPU Requirements

| Instance       | VRAM   | Cost/hr  | Est. runtime | Notes                        |
|----------------|--------|----------|--------------|------------------------------|
| NVIDIA H200    | 141 GB | ~$2.50   | ~500 hrs     | Recommended balance          |
| NVIDIA B200    | 192 GB | ~$3.00   | ~375 hrs     | Faster, more headroom        |

Minimum VRAM: ~80 GB (model + optimizer + activations for `B=4`, `GA=64`, `seq=2048`).

---

## Quick Start (automated)

```bash
# 1. SSH into your Lambda instance and clone the repo
git clone <your-repo-url> ~/synapse
cd ~/synapse

# 2. One-time: configure rclone for Google Drive
rclone config
#   - Add a new remote named "gdrive"
#   - Authenticate with your Google Drive account (headless auth flow or auth code)

# 3. Bootstrap everything
bash pretrain/run_on_vm.sh
```

`run_on_vm.sh` does all of this automatically:

1. Checks GPU, git, Python3
2. Installs `torch` + `numpy` + `tqdm`
3. Installs rclone (if missing) and verifies the `gdrive:` remote
4. `rclone copy` — pulls ~300 GB of merged shards from Drive to local SSD (one-time)
5. `rclone copy` — pulls latest checkpoint + manifest for resume (if any)
6. Sets `SYNAPSE_DIR`, `SKIP_DRIVE_MOUNT=1`, `SKIP_STAGE=1`, `CHECKPOINT_PUSH_REMOTE`
7. `exec python3 pretrain/train.py`

Env knobs you can set before running:

| Variable         | Default                    | Description                     |
|------------------|----------------------------|---------------------------------|
| `LOCAL_DIR`      | auto-detected              | Base directory for data/ckpts   |
| `GDRIVE_REMOTE`  | `gdrive`                   | rclone remote name              |
| `GDRIVE_PATH`    | `synapse`                  | Base path on Drive              |
| `MAX_TOKENS`     | `42000000000` (42B)        | Token budget                    |
| `CHECKPOINT_NAME`| `synapse_2b_d2560_l28.pth`| Checkpoint filename             |
| `SKIP_DATA_PULL` | (unset)                    | Set to `1` to skip shard pull   |

---

## Manual Setup (step-by-step)

If you prefer to run steps yourself or need to customise:

```bash
# Install dependencies
pip install --upgrade pip
pip install torch numpy tqdm
pip install -r requirements.txt   # openai, python-dotenv, pymupdf

# Install rclone (if needed)
curl -fsS https://rclone.org/install.sh | sudo bash
rclone config                      # one-time: add remote named "gdrive"

# Pull data from Drive to local SSD
LOCAL_DIR=/home/ubuntu/synapse_data
mkdir -p "$LOCAL_DIR/synapse"/{token_shards_merged,checkpoints,manifests}

rclone copy gdrive:synapse/token_shards_merged \
  "$LOCAL_DIR/synapse/token_shards_merged" \
  --transfers=8 --drive-chunk-size=64M --checksum --progress

# Pull existing checkpoint for resume (if any)
rclone copy gdrive:synapse/checkpoints \
  "$LOCAL_DIR/synapse/checkpoints" \
  --include "*.pth" --transfers=4 --drive-chunk-size=64M --progress || true

rclone copy gdrive:synapse/manifests \
  "$LOCAL_DIR/synapse/manifests" \
  --include "training_latest.json" --transfers=2 --checksum || true

# Launch training
export SYNAPSE_DIR="$LOCAL_DIR/synapse"
export SKIP_DRIVE_MOUNT=1
export SKIP_STAGE=1
export CHECKPOINT_PUSH_REMOTE=gdrive:synapse/checkpoints

python pretrain/train.py
```

---

## Environment Variables

All optional — values can be set in the shell before running `train.py`:

| Variable                   | Default                                | Description                                    |
|----------------------------|----------------------------------------|------------------------------------------------|
| `SYNAPSE_DIR`              | `./synapse` (VM)                       | Root dir: must contain `token_shards_merged/`  |
| `CHECKPOINT_NAME`          | `synapse_2b_d2560_l28.pth`             | Checkpoint filename in `checkpoints/`          |
| `MAX_TOKENS`               | `42000000000`                          | Total token budget (42B default)               |
| `EXPECTED_TOK_ID`          | `7a570a7ba9fc7985`                     | Tokenization ID; must match your tokenizer     |
| `SKIP_DRIVE_MOUNT`         | (unset)                                | Set to `1` on VM (skips Colab Drive mount)     |
| `SKIP_STAGE`               | (unset)                                | Set to `1` if data is already on fast storage  |
| `STAGE_DIR`                | auto (Colab: `/content/shards`, VM: "")| Local SSD path for shard staging               |
| `CHECKPOINT_PUSH_REMOTE`   | (unset)                                | rclone path (e.g. `gdrive:synapse/checkpoints`)|

---

## Data Mix Configuration

By default `train.py` trains on **100% Wikipedia**. Before running a real training run,
edit the `DATA_MIX` dict at `pretrain/train.py:192-204` to enable all sources:

```python
DATA_MIX = {
    "data_wikipedia":       {"weight": 0.40, "max_epochs": 4},
    "data_c4":              0.20,
    "data_code":            0.15,
    "data_finemath":        0.10,
    "data_books_gutemberg": {"weight": 0.05, "max_epochs": 20},
    "data_books_faded":     {"weight": 0.03, "max_epochs": 20},
    "data_arxiv":           0.02,
    "data_adult":           {"weight": 0.02, "max_epochs": 20},
    "data_distilled_facts": {"weight": 0.03, "max_epochs": 20},
}
```

Entries with `max_epochs` > 1 allow a source with limited unique shards to repeat and
fill its allocated budget. Weights must sum to 1.00. Sources not found on disk are
silently skipped.

---

## Checkpointing & Resume

Checkpoints use a **v2 format** (dict with schema, model, optimizer, curr_step,
seen_shards, tokenization_id). Every save overwrites the same file; old checkpoints
are auto-archived with a timestamp suffix on startup.

- **Mid-epoch save**: every 2 shards
- **End-of-epoch save**: at epoch boundary
- **Async rclone push**: every save triggers a background rclone copy to Drive
  (configurable via `CHECKPOINT_PUSH_REMOTE`)

Resume is fully idempotent:
1. Checkpoint restored → model weights, optimizer state (warm Adam), step counter,
   and seen-shard list are exact.
2. Already-trained shard-passes are subtracted from the planned selection.
3. Tokenization ID is validated against the checkpoint's manifest — aborts if it
   changed (you must start fresh if you retrain the tokenizer).

If a VM is terminated mid-run: provision a new one, run `run_on_vm.sh`, and training
picks up exactly where it left off.

---

## Monitoring

- **Per-source eval** every 500 optimizer steps: 32 batches per source, reported as
  individual losses plus an unweighted "overall" average.
- **tqdm progress bar** shows: loss, grad norm, LR, step count, elapsed hours,
  latest eval loss.
- **Scrollback logging**: each new shard prints the source domain and the original
  `.txt` filenames that went into it (from `merged_from` in the manifest).
- **Training manifest** saved to `manifests/training_latest.json` at the end,
  with full config, data selection, and eval history.

---

## Cost Estimation

Based on a B200 GPU at ~3.0B tok/hr for a 2B-param model:

| Metric              | Value               |
|---------------------|---------------------|
| Tokens per step     | 524,288             |
| Total tokens        | 42,000,000,000      |
| Estimated steps     | ~80,100             |
| Est. wall time      | ~14-21 days         |
| GPU cost (H200)     | ~$850-$1,250        |
| GPU cost (B200)     | ~$1,000-$1,500      |

Use spot/preemptible instances to reduce cost — the checkpoint system handles
interruptions cleanly.

---

## Dry Run

Validate the full pipeline with a tiny token budget before committing to a long run:

```bash
MAX_TOKENS=100000000 python pretrain/train.py
```

This trains on ~100M tokens (a few shards) and completes in 5-15 minutes. Verify:
- Shard staging completes (print: "Staged N shards")
- Loss decreases (start ~11.0, should drop into single digits)
- Eval runs and reports per-source losses
- Checkpoint saves without errors
- rclone pushback succeeds (check for "pushed" line in output)

---

## Troubleshooting

| Symptom                           | Likely cause & fix                                                      |
|-----------------------------------|-------------------------------------------------------------------------|
| `tokenization_id mismatch`        | Tokenizer was retrained. Clear old checkpoint or update `EXPECTED_TOK_ID` |
| `No usable shards in ...`         | Shard pull incomplete. Run `rclone copy` again or check Drive path      |
| `rclone remote not configured`    | Run `rclone config` and set up a remote named `gdrive`                  |
| Slow shard pull                    | Add `--transfers=16 --drive-chunk-size=128M` to the rclone copy command |
| Training OOM (out of memory)      | Instance VRAM too low. Try H100 (80 GB) in `expandable_segments` mode   |
| Wrong-size shards filtered out    | Partial upload — re-upload the affected shard from tokenizer output     |
| Loss not decreasing               | Check learning rate; verify tokenizer ID matches; inspect shard bytes   |

---

## Recovery

If the token shards on Drive are corrupted or accidentally deleted, recover from cold
backup using:

```bash
python pretrain/recover_from_tar.py
```

This rebuilds merged shards from the 6-part tar archive stored on Drive.
