# sft/ — supervised fine-tuning pipeline

Turns the pretrained checkpoint into the chat + tool-use model. Everything here is
driven by `SYNAPSE_DIR` (the Drive-mirrored data directory); nothing is hardcoded to
one machine. `SFT_DATA_FLOW.md` has the full data-flow diagram (renders on GitHub).

## Pipeline at a glance

```
build/download sources ─→ tokenize (masked ChatML) ─→ consolidate registry ─→ train ─→ eval
   datasets_sft/<name>/     sft_tokenized/<name>/       manifests/…registry     sft.py   eval_tool_use.py
```

### 1. Sources (`datasets_sft/<name>/<name>_raw.jsonl`)

15 sources in v3, three kinds:

- **Downloaded** (`download_sft_data.py`): tulu3, dolly, no_robots, oasst1, metamath,
  opencode, samsum, alpaca_gpt4, nli (SNLI reformatted to yes/no entailment).
- **Generated & verified** (teacher = DeepSeek, rejection-sampled against gold answers):
  - `generate_reasoning_distill.py` — chain-of-thought math, answer-verified
  - `generate_tool_use.py` — agentic traces where the teacher *actually calls* python
    (`--mode python`) or web search (`--mode search`, Brave); a trace is kept only if
    the tools succeeded and the final answer matches gold. `--source-name` separates
    output sources; `--max-searches` hard-caps API spend; resumable via `progress.log`.
  - `generate_tool_problems.py` — Wikidata fact DB (21 families, per-family cap) that
    feeds the search-mode generator with verifiable questions.
  - `generate_creative.py` — creative writing, written and score-gated by the teacher.
- **Synthesized & self-verified** (no API): `build_format_following.py` (exact-format
  instructions, 11 task types, independent checker per task),
  `build_tool_negative.py` (tool prompt + trivial/prose inputs answered *without*
  tools — teaches tool abstention; includes a synthesized trivial-arithmetic tier).

### 2. Tokenization (`tokenize_sft_data.py`)

Encodes ChatML with response-only loss masks. Role markers and tool tokens are fixed
ids verified against the tokenizer fingerprint; assistant `tool_call` JSON is emitted
inline via the single canonical serializer. Train/val is a **global hash split** of
the normalized first user message (same question ⇒ same side, in every source,
forever). Freshness is content-aware (`raw_sha256`) — a rebuilt source can never be
silently skipped. `consolidate_sft_data.py` writes the registry the trainer reads.

### 3. Training (`sft.py`, launched from `sft_pre_train.ipynb`)

Loads the pretrain checkpoint, samples per-batch from `SFT_DATA_MIX` (a load-time
assert caps every source at 1.65× repeats/epoch), token-weighted gradient
accumulation, masked shifted CE, mid-epoch per-source evals. Saves `sft_best.pth`
(model-only) whenever overall val improves — that file, not the endpoint, is the
ship artifact. Checkpoints mirror to Drive in the background; epoch snapshots and
best are pushed durably.

### 4. Tool runtime & loop

- `tools_runtime.py` — the ONE module that executes tools everywhere (data
  generation, eval, chatbot): sandboxed python (no network, CPU/memory rlimits,
  wall-clock kill) and Brave web search (rate-limited, retried). Also owns the
  canonical tool system prompt and tool-call serialization.
- `tool_loop.py` — the inference loop shared by the chatbot and the eval: parse
  `<|tool_call|>` at the token level, execute, feed the result back in the trained
  wire format, repeat (max 5 rounds; malformed JSON becomes an error result the
  model can recover from; identical repeated calls stop the loop).

### 5. Evaluation (`eval_tool_use.py`, run via `sparky/eval_tool_use_colab.ipynb`)

Four legs over held-out (val-side) questions: A tool prompt + hard math (should call
python), B tool prompt + trivial/prose (should NOT call — false-call rate),
C empty prompt + the same math (must never call), D tool prompt + fact questions
(should call search — runs against the **live** Brave API; train == inference).

## Typical run order

1. Build/refresh sources locally (`SYNAPSE_DIR=~/synapse_data python sft/<builder>.py`),
   push with `push_to_drive.py`.
2. Tokenize + consolidate (locally or Colab `sft_data_prep.ipynb`).
3. Archive the previous generation's checkpoints on Drive (the trainer refuses to
   start over an occupied `sft_checkpoints/` root only via its resume logic — move
   old runs to `v{N}_.../` first).
4. Train on Colab (`sft_pre_train.ipynb`), ship `sft_best.pth`.
5. Gate with `eval_tool_use` + `sparky/sft_bench` before serving it.
