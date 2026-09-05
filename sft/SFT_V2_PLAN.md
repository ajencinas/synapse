# SFT v2 — clean run from the pretrain checkpoint, with tool use

**Status: SUPERSEDED by `SFT_V2_EXECUTION_PLAN.md` (2026-08-20).** Originally: DRAFT — 2026-07-28. Supersedes `SFT_FINALIZE_PLAN.md`, which was
built on wrong source counts (see §1) and assumed a warm-start from the existing
7-source SFT. The decision now is a **clean run from the final pretrain
checkpoint** with the full source set.

---

## 0. Ground truth (measured 2026-07-28, not assumed)

Every count below was obtained by reading the actual files, because both
`meta_raw.json` files on Drive are wrong.

**Init checkpoint:** `gdrive:synapse/checkpoints/synapse_2b_d2560_l28.pth`
(25 GB, 2026-06-10) — the final pretrain model.

**Tokenized, in `manifests/sft_data_registry.json`** (tok id `7a570a7ba9fc7985`,
block_size 2048):

| source | kept | train | val | p50 len | mean len |
|---|---:|---:|---:|---:|---:|
| metamath | 22,000 | 21,560 | 440 | 214 | 243.6 |
| tulu3 | 15,588 | 15,277 | 311 | 294 | 377.2 |
| dolly | 14,624 | 14,332 | 292 | 110 | 180.5 |
| samsum | 7,998 | 7,839 | 159 | 159 | 187.2 |
| opencode | 5,998 | 5,879 | 119 | 453 | 514.2 |
| alpaca_gpt4 | 5,938 | 5,820 | 118 | 121 | 160.8 |
| oasst1 | 2,249 | 2,205 | 44 | 369 | 424.3 |
| **total** | | **72,912** | **1,483** | | |

**Generated, NOT tokenized:**

| source | raw records | verified |
|---|---:|---|
| `tool_use` (python mode) | **35,274** | 0 dup ids, 0 malformed, all `tool: python`; calls/trace 23,596×1 / 8,135×2 / 3,543×3 |
| `reasoning_distill` | **26,250** | 0 dup ids, 0 malformed, all 2-message user/assistant, mean 589 chars |

> `tool_use/meta_raw.json` reports `kept: 0` (it is the aborted *search* run's
> metadata). `reasoning_distill/meta_raw.json` reports `kept: 1577` (an early
> run; the file was appended to afterwards). **Do not trust either file.**
> Count lines. Re-derive counts from the registry after tokenization.

**Not built:**

| source | est. size | cost | notes |
|---|---:|---|---|
| `no_robots` | ~9,500 | free | wired in `download_sft_data.py:261`, never downloaded |
| `creative` | ~1,200–1,700 | ~$0.30 | generator written + tested, never run. Hard ceiling **2,400** (`build_prompts` enumerates 8 forms × 12 genres × 25 themes and dedupes) |
| `tool_use` search mode | 0 | ~1–2h wall-clock | problem DB **is** built: `facts_problems.jsonl`, 8,000 problems / 12 families |

**Prior run (the regression baseline).** `manifests/sft_training_latest.json`:
7-source mix, 2 epochs, 2,280 steps. Per-source val, baseline → final:
tulu3 2.160→1.745 · metamath 0.897→**0.281** · dolly 2.494→2.003 ·
alpaca_gpt4 1.787→1.429 · samsum 3.348→1.431 · opencode 0.669→**0.260** ·
oasst1 1.842→1.532 · overall 1.885→**1.240**.

Since v2 starts from pretrain, these are a *quality* target, not a resume point:
the v2 run should land at or below them on the 7 base sources.

---

## 1. The two findings that drive the design

### 1a. Math over-concentration is the real risk

`tool_use` and `reasoning_distill` were drawn from the same math problem pool.
Measured overlap on normalized question text:

```
tool_use unique questions        35,274
reasoning_distill unique q'ns    26,250
SHARED question text             22,063   (84.0% of the smaller set)
```

Add metamath (21,560) and math-derived data would be ~57% of an unweighted pool.
The prior run already drove metamath to **0.281** — near its floor, so that
gradient is largely spent. More math buys little and crowds out general
capability. **The mix must actively hold the combined math share down**; this is
the opposite of the scarcity problem `SFT_FINALIZE_PLAN.md` was solving.

### 1b. The overlap is deliberate contrast — and it makes the system prompt load-bearing

The 22k shared problems are not redundant. The two sources differ in exactly one
signal, and each source is internally uniform (verified: 1 distinct system string
each):

| source | system prompt | taught behavior |
|---|---|---|
| `tool_use` | `"You are an expert problem solver with tools. Use \`python\` (sympy available)…"` | emit `<\|tool_call\|>`, read `<\|tool_result\|>`, then answer |
| `reasoning_distill` | `""` (empty) | answer directly with short CoT |

Same question, two strategies, separated cleanly by the system prompt. That is a
well-formed conditioning signal for *when to reach for a tool*.

**The consequence is a correctness blocker.** `improvements.md` #14 already notes
that `sparky_chatbot.py:563,641` always prepends a generic
`DEFAULT_SYSTEM_PROMPT` that matches **neither** training branch. If inference
sends a third, unseen system string, the model is out-of-distribution for both
behaviors and tool-triggering will be unreliable. §4 treats this as a blocker,
not cleanup.

---

## 2. Provisional mix

**These weights are provisional.** Tokenization drops (`too_long` > 2048,
`short_response` < 3 learned tokens) are unknown until Phase 1 runs. Recompute
from `sft_data_registry.json` `train_count`s before editing `sft.py`.

Estimated train counts after tokenization (pool **P ≈ 143,000**):

| source | est. train | w | draws/epoch | per-ex/epoch |
|---|---:|---:|---:|---:|
| tool_use | 34,200 | 0.160 | 22,880 | 0.67× |
| reasoning_distill | 25,700 | 0.120 | 17,160 | 0.67× |
| metamath | 21,560 | 0.140 | 20,020 | 0.93× |
| tulu3 | 15,277 | 0.175 | 25,025 | 1.64× |
| dolly | 14,332 | 0.140 | 20,020 | 1.40× |
| no_robots | 9,000 | 0.094 | 13,442 | 1.49× |
| samsum | 7,839 | 0.050 | 7,150 | 0.91× |
| alpaca_gpt4 | 5,820 | 0.050 | 7,150 | 1.23× |
| opencode | 5,879 | 0.030 | 4,290 | 0.73× |
| oasst1 | 2,205 | 0.025 | 3,575 | 1.62× |
| creative | 1,450 | 0.016 | 2,288 | 1.58× |
| **total** | **~143,000** | **1.000** | | |

```python
SFT_DATA_MIX = {
    "tulu3":             0.175,  # general backbone
    "tool_use":          0.160,  # NEW behavior — largest single new source
    "metamath":          0.140,  # cut from 0.28: near floor (0.281), gradient spent
    "dolly":             0.140,
    "reasoning_distill": 0.120,  # NEW distribution — no-tool CoT contrast
    "no_robots":         0.094,  # prose breadth
    "samsum":            0.050,
    "alpaca_gpt4":       0.050,
    "opencode":          0.030,  # cut from 0.06 per the prose tilt
    "oasst1":            0.025,  # supply-limited; at the 1.65x ceiling
    "creative":          0.016,  # supply-limited; at the ceiling
}
```

**Design rationale:**
- Combined math share (`tool_use` + `reasoning_distill` + `metamath`) = **0.42**,
  down from the ~0.57 an unweighted pool would give.
- metamath cut 0.28 → 0.14. Its loss is near floor; the weight is better spent on
  the two distributions the model has never seen.
- opencode cut 0.06 → 0.03 (also near floor at 0.260) per the reasoning/prose tilt.
- `tool_use` gets the largest new-source weight because it is a **new output
  behavior**, not a new topic — behavioral pattern-learning needs more repetition
  than distributional fit (FireAct / Toolformer).
- **Ceiling rule:** cap per-example draws at ~1.65×/epoch (`w ≤ 1.65·N/P`), the
  watch line `SFT_TRAIN_PLAN.md §2.2` set from oasst1. `tulu3`, `oasst1` and
  `creative` are the binding constraints here; everything else has headroom.

**If `no_robots` and `creative` are skipped:** P drops to ~132,600. Renormalize by
redistributing their 0.110 to tulu3 (+0.03), dolly (+0.03), alpaca_gpt4 (+0.02),
samsum (+0.02) and oasst1 (+0.01), then re-check every source against the ceiling
at the smaller P — the ceilings tighten as P shrinks.

### Epochs, LR, horizon

Clean run from pretrain, so the **original** SFT hyperparameters apply — not the
reduced warm-start LR in the superseded plan:

- **EPOCHS = 2**
- **MAX_LR = 1.5e-5**, **MIN_LR = 1.5e-6**, **WARMUP_STEPS = 100** (as `sft.py` today)
- **Steps/epoch** = ceil(P / 64) ≈ **2,234**; total ≈ **4,469** (vs 2,280 before —
  roughly **2× the prior run's compute**)
- `EVAL_EVERY = 200` → ~22 evals/epoch

---

## 3. Pre-flight fixes (land BEFORE the run)

From `improvements.md`. Ordered by whether they can corrupt this run.

| # | where | why it matters for v2 |
|---|---|---|
| **H2** | `sft/sft.py:527` | Grad accum is a mean-of-means: each microbatch contributes `mean_over_its_response_tokens / GRAD_ACCUM_STEPS`, so short-response microbatches are over-weighted. v2 has **wider length variance than v1** (dolly p50 110 vs opencode p50 453 vs tool_use traces), so the bias is larger. Fix: accumulate `reduction="sum"`, divide once by total non-ignored target tokens. |
| **low 13** | `sft/sft.py:477` | No guard that registry `block_size` ≤ model `BLOCK_SIZE`. Two new sources tokenize in Phase 1 — a wrong `--block-size` should fail upfront, not mid-run inside `apply_rope`. |
| **H3** | `sft/sft.py:596` | Epoch snapshots can silently skip the Drive mirror (a live push thread suppresses the next). A ~2× longer run makes losing an epoch snapshot more expensive. Fix: per-artifact push queue + include newest snapshot in the final sync. |
| **C2** | `pretrain/train.py:827` pattern | Non-atomic `torch.save` in place while a background rclone reads the same path — already produced two torn checkpoints on Drive. Verify `sft.py` does not share the pattern; if it does, tmp + `os.replace()`. |

Deliberately **not** blocking v2:
- **H1** (`\frac` verifier, `generate_reasoning_distill.py:150`) — only affects
  *future* generation runs. It does mean the existing `reasoning_distill` and
  `tool_use` sets are biased against fraction-valued answers; fix it before any
  regeneration, not before training.
- **H5** (`strict=False` hides corruption, `sparky_model.py:233`) — affects eval
  and the chatbot, so fix in Phase 4, before trusting any tool-eval number.

---

## 4. Phases

### Phase 0 — decide scope (blocking, ~0 effort)
1. `no_robots` in or out? (free download; recommend **in** — cheapest counterweight
   against the math tilt)
2. `creative` in or out? (~$0.30; recommend **in**, but run `--limit 200` first —
   it has never produced data despite being committed as "verified")
3. Search-mode tool traces in or out? (~1–2h wall-clock; the fact DB is already
   built). Without them the model only learns calculator behavior, never
   "look it up first" — arguably the more agentic half.

### Phase 1 — finish and tokenize the data
1. `python sft/download_sft_data.py --datasets no_robots` (if in)
2. `python sft/generate_creative.py --limit 200` → inspect → full run (if in)
3. Search traces (if in): `generate_tool_use.py --mode search`, small `--limit`
   first — the last attempt kept **0**, so calibrate before committing hours.
4. **Contamination check** (new, not in any prior plan): `tool_use` and
   `reasoning_distill` share 84% of questions with each other and were drawn from
   an orca/numina pool; metamath is GSM8K/MATH-derived. Check for train/val
   leakage **across** sources before training, or per-source val loss is
   contaminated and the acceptance criteria in §6 mean nothing.
5. `tokenize_sft_data.py --datasets reasoning_distill,tool_use[,no_robots,creative]`
   then `consolidate_sft_data.py`.
6. **Gate:** registry lists every source with real `train_count`/`val_count`;
   check `drops` (esp. `too_long` on tool_use).

### Phase 2 — round-trip verification (gate before training)
The tokenizer already handles tool roles (`tokenize_sft_data.py:104-152`:
`role:"tool"` → id 8, inline `<|tool_call|>` id 7, `dump_tool_call` serializer)
and `sparky_chat_template.py` already renders them. Verify rather than build:
- encode/decode 3–5 real `tool_use` traces;
- labels **cover** assistant content, the inline `<|tool_call|>` span, the
  serialized call JSON, assistant `<|im_end|>`, final EOT;
- labels **never** cover `role:"tool"` content or its `<|im_end|>`;
- `build_sft_prompt` renders byte-identical text to what the tokenizer encoded.

`tests/test_tokenize_sft_tooluse.py` covers most of this on synthetic traces —
extend it to real records from the generated file.

### Phase 3 — trainer changes + the run
1. Apply §3 pre-flight fixes.
2. Set the final `SFT_DATA_MIX` (`sft.py:64-72`) from Phase 1 counts.
3. **Guarantee a clean start from pretrain.** `sft.py` auto-resumes
   `sft_latest.pth` then any `sft_epoch*.pth`. Those files exist on Drive
   (`sft_latest.pth` 25 GB, `sft_epoch1/2.pth` 8.4 GB each) and **will hijack the
   run** — it would resume the old 7-source phase or skip as "epoch 2 complete".
   Move them to `sft_checkpoints/v1_7source/`. Renaming in place is not enough:
   `sft_epoch1_7source.pth` still matches the `sft_epoch*.pth` glob.
4. Record in the manifest: init checkpoint, the v1 per-source finals as
   `results.prior_sft_reference`, and the full mix.
5. Run. Watch per-source val every 200 steps. Expect `tool_use` baseline to be
   **elevated (~3–6)** on the pretrain checkpoint — it has zero exposure to tokens
   7/8. What matters is that it *drops*, not that it matches other sources.

### Phase 4 — inference and eval (the part that makes it agentic)
1. **Move `CANONICAL_TOOL_SYSTEM`** from `generate_tool_use.py:47` into
   `tools_runtime.py`. The chatbot must not import a generator script at runtime.
2. **Fix the system-prompt mismatch (§1b blocker).** The chatbot must send the
   canonical tool system prompt when tools are enabled, and `""` when not — never
   its own generic default. This is what selects between the two trained
   behaviors.
3. **Write the tool loop ONCE** in a shared `sft/tool_loop.py`: parse `<|tool_call|>`
   (token id 7) → execute via `tools_runtime` → append `{"role":"tool"}` →
   regenerate. Max 5 iterations. Handle malformed JSON and repeated identical
   calls. `<|tool_call|>` is **not** a stop token; `sft_stop_token_ids` is unchanged.
4. Wire it into `sparky_chatbot.py` (currently **zero** tool support — `grep -i tool`
   returns nothing). Hold `gen_lock` across the whole loop.
5. `sft/eval_tool_use.py` — imports the same `tool_loop`, reports pass@1,
   tool-call rate, avg calls/trace, malformed-JSON rate, overflow rate. Must be a
   new script: `sparky_eval.py` is a loglikelihood harness on a different model
   interface.
6. Re-run `sparky_eval.py` presets + `sft_bench.py` to confirm general capability
   didn't regress.

---

## 5. Build order checklist

- [ ] **0.** Decide: `no_robots` / `creative` / search traces in or out.
- [ ] **1.** Build any in-scope missing sources.
- [ ] **2.** Cross-source contamination check (train/val leakage).
- [ ] **3.** Tokenize new sources → `consolidate_sft_data.py` → real counts.
- [ ] **4.** Recompute `SFT_DATA_MIX` from `train_count`s; re-check every ceiling.
- [ ] **5.** Round-trip mask verification on real tool traces.
- [ ] **6.** Pre-flight fixes: H2, low-13, H3, C2-check.
- [ ] **7.** Move v1 checkpoints out of the auto-resume glob.
- [ ] **8.** Launch the run; monitor per-source val every 200 steps.
- [ ] **9.** `CANONICAL_TOOL_SYSTEM` → `tools_runtime.py`; fix chatbot system prompt.
- [ ] **10.** Shared `sft/tool_loop.py` → chatbot + `sft/eval_tool_use.py`.
- [ ] **11.** Tool eval + general-capability regression check.

---

## 6. Acceptance criteria

1. `sft_latest.pth` comes from the v2 run: manifest `data_mix` matches the final
   recomputed mix and records the **pretrain** checkpoint as init.
2. Per-source val on the 7 base sources is at or below §0's v1 final column
   (allowing small-source noise). This is a from-pretrain run, so it should be a
   fair comparison — unlike a warm-start.
3. `tool_use` and `reasoning_distill` per-source val are present, non-NaN, and
   drop substantially from their (elevated) pretrain baselines.
4. The chatbot completes a real python-tool turn end to end: user asks a
   non-trivial calculation → model emits `<|tool_call|>{"tool":"python",…}` →
   harness executes → model finalizes with the correct answer.
5. **The contrast holds:** with the tool system prompt the model calls tools; with
   an empty system prompt on the *same question* it answers directly. This is the
   specific thing the 84% overlap was built to teach — measure it explicitly.
6. `sft/eval_tool_use.py` reports pass@1 strictly between 0% and 100%. Either
   extreme means the harness, prompt, or parsing path is broken.
