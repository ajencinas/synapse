# SynapseGPT

A **2B-parameter LLM built end to end from scratch**: corpus collection → custom
tokenizer → pretraining → three generations of supervised fine-tuning → an agentic
tool loop (sandboxed **python** + live **web search**) → a served chatbot with
voice mode, plus the full evaluation stack used to gate every release.

## The model

| | |
|---|---|
| Architecture | decoder-only transformer, 2.09B params — d=2560, 28 layers, 20 heads (GQA, 4 KV), SwiGLU, RoPE, RMSNorm, 2048 context |
| Tokenizer | custom byte-level BPE, 64k vocab, digit-per-token, dedicated tool tokens (`<|tool_call|>`, `<|tool_result|>`) |
| Pretraining | ~44B tokens (code / math / arxiv / wikipedia / web / books / synthetic), bf16, single-GPU runs on Colab & rented VMs |
| SFT (v3, current) | 15 sources, 187,989 examples, response-only masked loss, tool-use + tool-abstention + web-search traces |
| Tools | model-emitted JSON tool calls executed by a shared inference loop: sandboxed python (no network, rlimits) and Brave-backed web search |

Measured (v3, held-out data, greedy decoding): the model calls python on 98–100% of
hard math (+10 pts pass@1 over no-tool), **searches on 100% of fact questions with
74.5% accuracy against live web results**, declines tools on prose (0% false calls),
and never emits a tool call without the tool system prompt (0/100).

## Repository map

| Path | What it is |
|---|---|
| `synapse_model.py` | The model (training definition) |
| `download_pretrain_data/` | Notebooks that download the public pretraining corpora |
| `tokenize/` | Tokenizer training pipeline (see its README) |
| `pretrain/` | Pretraining: `train.py`, Colab notebook, VM bootstrap (`README_LAMBDA.md`) |
| `sft/` | **The SFT pipeline** — data builders/generators, tokenization, training, tool runtime & loop, tool eval (see `sft/README.md`) |
| `sparky/` | **Serving & evaluation** — chatbot (web UI, tools, voice), inference model, chat bench, leaderboard harness (see `sparky/README.md`) |
| `tests/` | Unit tests (run with `python -m unittest discover tests`) — data builders, tokenization round-trips, the tool loop against the real tokenizer & sandbox |

Not in the repo (see `.gitignore`): secrets (`.env`), the local venv (`synapse/`),
bulk corpora and generated datasets, model checkpoints, and a few private
data-generation pipelines (`download_pretrain_books/`, `download_pretrain_others/`,
`common_pretrain_text_processing/`). Data and checkpoints live in a Google Drive
layout under `gdrive:synapse/` (datasets, tokenized data, manifests, and versioned
checkpoint archives `sft_checkpoints/v{1,2,3}_*/`, each with a `MODEL_CARD.md`).

## Quickstart

```bash
python3.12 -m venv synapse && source synapse/bin/activate
pip install -r requirements.txt
```

Secrets go in a gitignored `.env` at the repo root (`DEEPSEEK_API_KEY` for
data-generation teachers and the eval judge, `BRAVE_API_KEY` for the search tool,
`NGROK_AUTHTOKEN` for serving; others optional).

Most heavy steps run on a GPU box (Colab or a rented VM) against the Drive layout:

- **Pretrain**: `pretrain/pre_train_mar23.ipynb` (Colab) or `bash pretrain/run_on_vm.sh` (VM — `pretrain/README_LAMBDA.md`)
- **SFT data prep + training**: `sft/sft_data_prep.ipynb`, then `sft/sft_pre_train.ipynb` — details in `sft/README.md`
- **Chat with it**: `sparky/sparky_chatbot_colab.ipynb` — web UI via ngrok, tool calls rendered live, optional hands-free voice mode
- **Evaluate**: `sparky/sft_bench_colab.ipynb` (49-question chat bench, LLM judge), `sparky/eval_tool_use_colab.ipynb` (4-leg tool-behavior eval incl. live search), `sparky/sparky_leaderboard_colab.ipynb` (Open LLM Leaderboard v1 suite)

## Design principles

- **Train == inference, structurally.** One module (`sft/tools_runtime.py`) executes
  tools at data-generation time and at serving time; one chat template
  (`sparky/sparky_chat_template.py`) is byte-identical between the SFT encoder and
  the chatbot; the eval drives the same loop the chatbot serves (`sft/tool_loop.py`).
- **Verified data only.** Generated tool traces are kept only when the final answer
  matches an independent gold answer; synthetic sources re-verify every record with
  an independent checker; failed tool calls are never trained on.
- **Leak-free splits.** Train/val is a global hash of the normalized question, so
  the same question lands on the same side in every source and every re-tokenization.
- **Fail loud.** Mismatched tokenizers, stale data, over-repeated sources, and
  checkpoint key mismatches all abort with an explanation rather than train wrong.
- **Nothing is lost.** Every model generation is archived on Drive with its
  manifests, eval results, and a model card before the next one can overwrite it.
