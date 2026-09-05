# Prompt for Codex: Review `SFT_FINALIZE_PLAN.md`

Please act as a senior ML engineer and review the SFT finalization plan in the same directory: `sft/SFT_FINALIZE_PLAN.md`. The plan describes how to add `reasoning_distill` + `tool_use` data to an already-completed 7-source SFT run, then retrain from the pretrain checkpoint with a new 9-source mix.

Read these files first:
1. `sft/SFT_FINALIZE_PLAN.md` — the plan to review.
2. `sft/SFT_DATA_PLAN.md` — original data pipeline design.
3. `sft/SFT_TRAIN_PLAN.md` — original training design.
4. `sft/TOOL_USE_PLAN.md` and `sft/TOOL_USE_DATAGEN.md` — tool-use design.
5. `sft/sft.py` — the trainer.
6. `sft/tokenize_sft_data.py` — the tokenizer script.
7. `sft/generate_tool_use.py` and `sft/tools_runtime.py` — tool generation/runtime.
8. `sparky/sparky_chat_template.py` and `sparky/sparky_chatbot.py` — inference template and chatbot.

Please review the plan on these dimensions and call out anything you would change, fix, or add. Be specific: cite file/line numbers where possible, and explain the risk if something is wrong.

## 1. Overall strategy
- Is retraining from the pretrain checkpoint (not continuing the existing 7-source SFT checkpoint) the right call? What are the trade-offs?
- Are 2 epochs and the proposed LR schedule still appropriate for the new 9-source mix?
- Is the "prior SFT reference" in the manifest a good idea? Any better way to track catastrophic-forgetting risk?

## 2. Data mix and overfit math
The plan proposes a mix based on actual counts: reasoning_distill ~1.6k (recommended to scale to ~5k), tool_use 4,161 python-mode traces. Verify:
- The per-example draw-rate ceiling (1.65x/epoch) and the arithmetic in §2b/§2e.
- The proposed normalized weights and whether they sum to 1.0.
- Whether scaling reasoning_distill to 5k at ~$3 is worth it.
- Whether tool_use at 4,161 can safely take 0.085 weight (current proposal) or if it should be higher/lower.
- If search/mixed traces are generated later, how should the mix be adjusted?

## 3. Tokenizer changes (Step 2)
The plan asks to modify `sft/tokenize_sft_data.py` to support tool roles and inline `<|tool_call|>` markers. Check:
- Is the proposed `resolve_special_ids` return change correct? It now returns 4 values: `role_id, im_end_id, eot_id, tool_call_id` (with `tool_result_id` embedded in `role_id["tool"]`).
- Does `process()` need to be updated to unpack 4 values and pass `tool_call_id` to `encode_example()`? What exactly should change at lines ~147-163?
- Is `from tools_runtime import dump_tool_call` the right import path for `tokenize_sft_data.py`?
- How should `encode_example` handle `role == "tool"` and assistant messages with a `tool_call` dict field?
- How should `resp_tokens` (short-response filter) count learned tokens for tool traces?
- Is the round-trip verification step sufficient to catch template bugs?

## 4. Tool generation (Step 3)
The existing `tool_use_raw.jsonl` has 4,161 python-mode traces; no meta_raw.json because the run was interrupted. Check:
- Is it safe to use these 4,161 traces as-is, or should the generator be resumed first?
- Are search/mixed traces necessary for the reasoning trigger, or can the first combined run use python-only?
- Any issue with the generator's append-only/resume-by-id behavior if we add search/mixed later?

## 5. Training execution (Step 5)
Check these details in `sft/sft.py`:
- The plan says to rename/move old `sft_latest.pth` and `sft_epoch*.pth` from **both** local and Drive. Is this sufficient to prevent accidental resume? Are there any other resume paths in `sft.py` that could trip up a from-scratch start?
- The mix is hardcoded in `sft.py` line 64-72. Is there a cleaner way to make this configurable for experiments?
- The cache key in `sft.py:189-203` uses per-source file size + mtime. Will adding `reasoning_distill` and `tool_use` correctly invalidate the cache? Any stale-cache risk we missed?
- Per-source val runs every 200 steps with small val sets (e.g., ~83 for tool_use). Is this enough signal, or should eval frequency change?

## 6. Chatbot/tool loop (Steps 6-7)
Check the proposed changes to `sparky_chatbot.py` and `sparky_chat_template.py`:
- Is detecting `<|tool_call|>` by token ID 7 in the generated token sequence the right approach, or would text parsing after decode be better?
- The plan says `SynapseInfer.generate()` rebuilds the KV cache from scratch each call. Is this accurate? What performance impact does it have?
- How should the chatbot handle malformed JSON from the model, or a tool call that keeps repeating?
- Should the `gen_lock` be held across the entire multi-turn tool loop? Any better concurrency model?
- Is importing `CANONICAL_TOOL_SYSTEM` from `generate_tool_use.py` into `sparky_chat_template.py` the best way to keep train==inference system prompts, or should it live in `tools_runtime.py`?
- Any issue with `build_sft_prompt` appending `<|assistant|>` at the end for tool-result turns?

## 7. Eval (Step 8)
The plan proposes a new `sft/eval_tool_use.py` instead of extending `sparky_eval.py`. Check:
- Is `SynapseInfer.from_checkpoint()` the right model interface for the eval harness?
- What metrics should it report beyond pass@1, tool-call rate, avg calls/trace, overflow rate?
- Should we also keep a standard benchmark run (e.g., the existing `sparky_eval.py` presets) after SFT to measure general capability retention?
- How many problems should the tool-eval run on for reliable numbers?

## 8. Risks and missing items
- What risks are missing from §4?
- Are the acceptance criteria in §5 realistic? Specifically, is criterion #3 stated correctly for `tool_use` given the pretrain model never saw tool tokens?
- Is there anything in the build order checklist (§6) that should be reordered, added, or removed?

## Output format
Please produce:
1. A **short verdict**: is the plan ready to execute, or does it need revisions before code is written?
2. A **prioritized list of issues** (high / medium / low) with:
   - The specific claim or step in the plan.
   - Why it is wrong/incomplete/risky.
   - A concrete fix or follow-up question.
3. **Any alternative approaches** you would consider for major decisions (e.g., mix weights, from-scratch vs. continue-train, eval strategy).

Be concise where possible. Focus on things that would actually break the training run or produce a subtly broken model/inference pipeline.
