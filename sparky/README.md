# sparky/ — serving & evaluation

Everything that runs the trained model: the chatbot, the inference-side model
definition, and the evaluation harnesses that gate each release.

## Serving

- **`sparky_chatbot.py`** — Flask web chat with token streaming (SSE), exposed via
  ngrok. Detects an SFT checkpoint and uses the trained ChatML template; the
  **Tools** toggle switches between the canonical tool system prompt and no prompt
  (the only two distributions the model ever saw). Tool calls run through
  `sft/tool_loop.py` and render live in the UI (🔍 search / 🐍 python blocks with
  running/done state). Extras: a forced-search assist for news/"search for X"
  phrasing the model wasn't trained on, hands-free **🎧 voice mode**
  (speech-to-text in, spoken replies out, auto-relisten), and live checkpoint
  switching. Launch on a GPU box with `sparky_chatbot_colab.ipynb`.
- **`sparky_model.py`** — `SynapseInfer`: KV-cache generation, strict checkpoint
  loading (any missing/unexpected key aborts — never silently random weights).
- **`sparky_chat_template.py`** — the chat wire format, byte-identical to the SFT
  encoder. Prompting the model any other way is out-of-distribution.

## Evaluation

| Harness | Notebook | What it measures |
|---|---|---|
| `test_sft_checkpoint.py` | — (no GPU) | Checkpoint/tokenizer sanity: schema, stage, ids |
| `sft_bench.py` | `sft_bench_colab.ipynb` | 49-question generative chat bench, 23 categories; deterministic graders + DeepSeek judge; `--compare-pretrain` for base-vs-SFT |
| `sft/eval_tool_use.py` | `eval_tool_use_colab.ipynb` | 4-leg tool-behavior eval on held-out data (python calls, tool abstention, prompt contrast, live web search) |
| `sparky_leaderboard.py` | `sparky_leaderboard_colab.ipynb` | Open LLM Leaderboard v1 suite via `lm_eval` (comparable numbers; full runs only) |
| `eval_drive_checkpoints.py` + `eval_dashboard.py` | — | Bulk pretrain-checkpoint evaluation + Streamlit dashboard (`SYNAPSE_EVAL_DIR` sets the work dir) |

## Checkpoints (Drive layout)

`gdrive:synapse/sft_checkpoints/v{N}_<sources>/` — each shipped generation is
archived with `sft_best.pth` (the ship artifact), epoch snapshots, its training
manifest, data registry, eval results, and a `MODEL_CARD.md`. The chatbot serves
the current generation's `sft_best.pth`; older generations remain loadable.
