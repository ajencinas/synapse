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

- `pre_train_mar23.ipynb` — pretraining notebook that consumes the tokenized shards produced by `tokenize/`

## Notes

- No test framework, linter, or formatter is configured.
- The included notebooks assume Google Colab + Google Drive for storage.
- If you need the private pipelines (`download_pretrain_books`, `download_pretrain_others`, `common_pretrain_text_processing`), reach out to the author.
