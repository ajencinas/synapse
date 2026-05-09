# synapse

End-to-end LLM pretraining pipeline: gather text corpora, generate synthetic data, tokenize, and pretrain.

## What's in this GitHub repo vs. what isn't

This repo is a **subset** of the full local working directory. It contains the parts that are stable, reproducible, and reasonable to share publicly. Bulk text corpora, generated data, secrets, and shared utilities live only on the author's machine.

**In GitHub:**

| Path                       | What it is                                                 |
| -------------------------- | ---------------------------------------------------------- |
| `README.md`                | This file                                                  |
| `requirements.txt`         | Python dependencies                                        |
| `download_pretrain_data/`  | Notebooks that download public datasets (C4, Wikipedia, RedPajama, code, FineMath, reasoning) |
| `tokenize/`                | Byte-Level BPE tokenizer training pipeline (Colab + Drive) |
| `pretrain/`                | Pretraining notebook                                       |

**Not in GitHub** (kept local only):

| Path                                | Why excluded                                                  |
| ----------------------------------- | ------------------------------------------------------------- |
| `.env`                              | API keys (OpenRouter, DeepSeek, Zhipu, Inception)             |
| `synapse/`                          | Local Python 3.12 virtualenv                                  |
| `.claude/`                          | Local Claude Code tooling config                              |
| `AGENTS.md`                         | Internal agent notes                                          |
| `common_pretrain_text_processing/`  | Shared text-cleaning + LLM-client utilities (private for now) |
| `download_pretrain_books/`          | Gutenberg / FadedPage / PDF book download + clean pipeline (large text outputs) |
| `download_pretrain_others/`         | Synthetic data generators (large generated corpora)           |

> **Heads-up:** some scripts in this repo import from `common_pretrain_text_processing/`, which is **not** included. Anyone cloning the repo will need to either obtain that package separately or stub it out. The notebooks under `download_pretrain_data/` and `tokenize/` are mostly self-contained.

## Pipeline overview

The full pipeline (across both public and private parts) looks like this:

```
   ┌─────────────────────────┐    ┌─────────────────────────┐
   │ download_pretrain_books │    │ download_pretrain_data  │  ← in GitHub
   │  (Gutenberg, FadedPage, │    │  (C4, Wikipedia,        │
   │   PDFs)        [private]│    │   RedPajama, code, ...) │
   └────────────┬────────────┘    └────────────┬────────────┘
                │                              │
   ┌────────────▼────────────┐                 │
   │ download_pretrain_others│                 │
   │  (synthetic LLM corpora)│                 │
   │                [private]│                 │
   └────────────┬────────────┘                 │
                │                              │
                └──────────────┬───────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ tokenize/  ← in GitHub  │
                  │ (BBPE, 64k vocab,       │
                  │  uint16 shards)         │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ pretrain/  ← in GitHub  │
                  │ (training notebook)     │
                  └─────────────────────────┘
```

## Setup

```bash
python3.12 -m venv synapse
source synapse/bin/activate
pip install -r requirements.txt
```

Create a `.env` at the repo root with API keys (only needed for the data-generation pipelines, most of which live outside this repo):

```
OPENROUTER_API_KEY=...
DEEPSEEK_API_KEY=...
ZHIPU_API=...
INCEPTION_API=...
```

`.env` is gitignored — never commit it.

## What's in each included folder

### `download_pretrain_data/`

Jupyter notebooks that pull public pretraining datasets:

- `download_c4.ipynb` — Common Crawl C4
- `download_wikipedia.ipynb` — Wikipedia
- `download_redpajama.ipynb` — RedPajama
- `download_code.ipynb` — code corpora
- `download_finemath.ipynb` — math
- `download_reasoning.ipynb` — reasoning traces

Each notebook is meant to be run standalone — they typically download into a path on Google Drive (consistent with the `tokenize/` step below).

### `tokenize/`

- `tokenizer_pipeline.ipynb` — Google Colab notebook that mounts Drive at `/content/drive/MyDrive/synapse`, reads source data from `datasets_pretrain/data_*` directories, and trains a Byte-Level BPE tokenizer:
  - vocab 64k
  - 256 special tokens
  - digit-per-token pre-tokenizer
  - writes `uint16` shards plus a manifest back to Drive
- `tokenizer_config.json` — saved tokenizer config

### `pretrain/`

- `train.py` — standalone training script. Reads tokenized shards from `$SYNAPSE_DIR/token_shards_merged/`, writes checkpoints to `$SYNAPSE_DIR/checkpoints/`. Configurable via env vars (see top of file).
- `pre_train_mar23.ipynb` — Colab notebook shim that mounts Drive, downloads `train.py` from this repo, and runs it.
- `run_on_vm.sh` — bash bootstrap for running training on a bare GPU VM (Lambda Labs, RunPod, etc.). See *Running pretraining* below.
- `inspect_shards.ipynb` — utility for tracing which `.txt` source files are inside any given merged shard.
- `recover_from_tar.py` — extract missing merged shards from the cold-backup tar on Drive (used if `token_shards_merged/` gets corrupted).

## Running pretraining

Two supported paths. Pick one.

### A) Google Colab (Pro / Pro+)

1. Open `pretrain/pre_train_mar23.ipynb` in Colab.
2. Connect to a GPU runtime (A100 or "G4" / RTX PRO 6000 Blackwell recommended).
3. Run all cells. The notebook mounts Drive, downloads the latest `train.py` from this repo, stages selected shards to `/content/shards`, and trains.
4. Checkpoints save to `gdrive:synapse/checkpoints/` via the Drive mount. Run resumes automatically across Colab sessions (just re-run the notebook).

Knobs (warmup, learning rate, batch size, checkpoint cadence) all live near the top of `pretrain/train.py`.

### B) Lambda Labs / RunPod (or any bare GPU VM)

One-time setup on a fresh box:

```bash
git clone https://github.com/ajencinas/synapse.git
cd synapse
pip install -r requirements.txt
pip install torch numpy tqdm                       # training deps
curl https://rclone.org/install.sh | sudo bash     # if rclone not present
rclone config                                       # set up "gdrive" remote
```

To start (or resume) training:

```bash
bash pretrain/run_on_vm.sh
```

What the script does:
- Pulls shards from `gdrive:synapse/token_shards_merged/` to local SSD (one-time, ~30 min for 300 GB at typical VM bandwidth).
- Pulls the latest checkpoint and manifest from Drive if they exist (resume).
- Sets `SYNAPSE_DIR`, `SKIP_STAGE=1`, `CHECKPOINT_PUSH_REMOTE=gdrive:synapse/checkpoints`.
- Execs `python pretrain/train.py`.

Each mid-epoch save (every 2 shards by default) is pushed back to Drive in a background thread via `rclone`, so you can destroy the VM at any time and resume on a new one.

Override knobs (set as env vars before running the script):

| Env var | Default | What it does |
|---|---|---|
| `LOCAL_DIR` | auto: `/home/ubuntu/synapse_data` (Lambda) or `/workspace/synapse_data` (RunPod) | Where shards land locally. |
| `GDRIVE_REMOTE` | `gdrive` | Your rclone remote name. |
| `GDRIVE_PATH` | `synapse` | Base path on Drive. |
| `MAX_TOKENS` | (train.py default: 42 B) | Token budget. Set lower for smoke tests. |
| `CHECKPOINT_NAME` | (train.py default) | Checkpoint filename. |
| `SKIP_DATA_PULL` | unset | If `1`, skip the rclone shard copy (assume data is already local). |

## Notes

- No test framework, linter, or formatter is configured.
- The Colab notebook assumes Drive; the VM path uses rclone-pushed Drive sync — both share the same `gdrive:synapse/...` layout.
- If you need the private pipelines (`download_pretrain_books`, `download_pretrain_others`, `common_pretrain_text_processing`), reach out to the author.
