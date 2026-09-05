# SFT v2 — execution plan

**Status: ACTIVE — 2026-08-20.** Supersedes `SFT_V2_PLAN.md` (DRAFT 2026-07-28),
whose ground truth was independently re-verified and confirmed correct, but whose
design missed four material issues (§2 below). Everything here was measured
against the actual files and the live Drive remote on 2026-08-20 — not assumed.

---

## 0. Verified state (2026-08-20)

**Init checkpoint:** `gdrive:synapse/checkpoints/synapse_2b_d2560_l28.pth`
(25 GB, 2026-06-10) — final pretrain model.

**v1 SFT (the regression baseline):** `gdrive:synapse/sft_checkpoints/` holds
`sft_latest.pth` (25 GB) + `sft_epoch1/2.pth` (8.4 GB each), from the 7-source
run of 2026-06-14. Results: val 1.885 → 1.240; Open LLM v1 avg 37.23
(MMLU 24.2 = chance); sft_bench 24/49 = 49%.

**Data:**

| source | state | count |
|---|---|---:|
| 7 base sources (tulu3…oasst1) | tokenized, in registry | 72,912 train / 1,483 val |
| tool_use (python mode) | raw only, NOT tokenized | 35,274 |
| reasoning_distill | raw only, NOT tokenized | 26,250 |
| no_robots | wired (`download_sft_data.py:261`), never downloaded | ~9,500 |
| creative | generator tested, never run | ≤2,400 |
| facts_problems.jsonl (search DB) | built, never used | 8,000 |

Both `meta_raw.json` files for the new sources are wrong (tool_use `kept: 0`
from the aborted search run; reasoning_distill `kept: 1577` from an early run).
Count lines; trust the registry after tokenization.

**Code, verified fresh:**
- v1 7-source mix still in `sft/sft.py:64`; hyperparams: batch 4 × accum 16,
  EPOCHS 2, MAX_LR 1.5e-5, warmup 100, EVAL_EVERY 200, SAVE_EVERY_STEPS 300,
  DRIVE_EVERY_STEPS 600.
- Auto-resume hijack chain at `sft/sft.py:382-405`: `sft_latest.pth` →
  `sft_epoch*.pth` glob. The v1 files on Drive WILL hijack a v2 run.
- Grad-accum mean-of-means at `sft/sft.py:526` (improvements.md H2).
- Per-source shuffle split at `sft/tokenize_sft_data.py:202-206`.
- tulu3 capped at `max_examples: 18000` (`sft/download_sft_data.py:231`) out of
  a 939k-row mixture.
- `CANONICAL_TOOL_SYSTEM` only in `sft/generate_tool_use.py:47`.
- `sparky/sparky_chatbot.py`: hardcoded generic SYSTEM_PROMPT at 563/641,
  ZERO occurrences of "tool" — no tool support at all.
- GOOD NEWS: `sft.py` checkpoint saves are already atomic
  (`torch.save(tmp)` + `os.replace`, sft.py:310) — pretrain's C2 torn-checkpoint
  bug does NOT apply here. Two torn files on Drive are pretrain-side only.
- Local disk: 24 GB free (94%) — under one full checkpoint. **The run must live
  on Colab/RunPod with `CHECKPOINT_PUSH_REMOTE` set.**

---

## 1. Measured findings that drive this plan

### 1a. The val split is contaminated by construction — needs a fix, not a check
`tokenize_sft_data.py` shuffles and splits each source independently. Measured
cross-source overlap on normalized question text:

```
tool_use ∩ reasoning_distill   22,014 shared questions (84% of the smaller set)
metamath ∩ tool_use             1,055
metamath ∩ reasoning_distill      695
metamath internal dupes         3,518  (22,000 records → 18,482 unique)
union of all three              56,750 unique of 79,848 records
```

With independent splits, ~84% of reasoning_distill's val questions are
guaranteed a train-set twin in tool_use. Every v2 per-source val number would be
fiction. Fix: global deterministic split by hash of normalized question.

### 1b. Zero negative tool examples — "when NOT to call" is unlearnable
All 35,274 traces call python at least once (23,596×1 / 8,135×2 / 3,543×3).
98.9% of tool-calling turns have EMPTY content before the call (no planning
text). 16.8% of calls are trivial arithmetic (`56 / 7`, `32 - 23`). Under the
tool system prompt, the training distribution is "always call python,
immediately" — including, at inference, for prose requests. The prompt-contrast
design (tool prompt vs empty prompt) teaches prompt→behavior memorization, not
tool judgment. Fix: a relabeled `tool_negative` source (§3 Phase 0.3).

### 1c. The measured weaknesses are not math
sft_bench v1 per-category: 100% on arithmetic/coding/commonsense/factual/
multi-step; **0%** on instruction_following (0/4), text_manipulation (0/3),
logic_entailment (0/3), translation (0/2), safety_refusal (0/2),
creative_writing (0/2), summarization (0/1). The 61k new math records don't
touch any of these. The tulu3 cap (18k of 939k) is the free lever: raising it
to 50k directly feeds every 0% category. No source contains dedicated refusal
data at all — it exists only inside tulu3.

### 1d. v1 overfit in epoch 2 — the endpoint is not the best model
Overall val bottomed at step 1400/2280 (1.2340), rose to 1.2402 by the end;
dolly 1.963→2.003 and alpaca_gpt4 1.385→1.429 rose monotonically through
epoch 2. v2 runs ~2× the steps over a ~2.4× pool: one epoch already exceeds
v1's turnover. `sft_latest.pth` is overwritten every 300 steps, so today the
val-optimal model is unrecoverable. Fix: `sft_best.pth` + select on val.

### 1e. Search traces: deferred to v3
Last run kept 0 records. `facts_problems.jsonl` is 93% concentrated in 4 of 12
families (book_author 1,952 / film_director 1,851 / mountain_country 1,828 /
company_founder 1,828; other 8 families ≈ 541). Even a successful run yields a
narrow templated distribution. Not worth calibration hours in v2.

---

## 2. Phases

### Phase 0 — data completion (local, ~half a day, ~$0.30)

0.1 **Uncap tulu3 → 50000** (`download_sft_data.py:231`), then
    `python sft/download_sft_data.py --datasets tulu3,no_robots`.
    Highest-leverage item in the plan (§1c). Expect ~43k tulu3 kept after the
    ~15% drop rate v1 showed.
0.2 **Creative:** DONE 2026-08-24 — 2,299 kept of 2,400 combos (95.8%
    yield, $0.63 total, deepseek-v4-flash writer+judge, min_score 8). All
    prompts unique, 2-turn, system="", max 468 tokens, no meta-openers.
    Pushed to Drive.
0.3 **Build `tool_negative` (new source, zero API cost):** DONE 2026-08-24 via
    `sft/build_tool_negative.py` — 6,029 records on Drive: 2,629 math (all rd
    records unique vs tool_use in tiers ≤2/≤3-digit ints; supply capped the
    planned 4k) + 3,400 prose (1,700 dolly + 1,700 no_robots, compute-request
    and short-answer filtered), all relabeled `system = CANONICAL_TOOL_SYSTEM`,
    answers unchanged. Verified: no tokens 7/8, resp≥3, split-side coherent
    with donors, val share 2.21%. Teaches the §1b decision boundary.
0.4 Search traces: OUT (deferred to v3, §1e).

### Phase 1 — split integrity, then tokenize (blocking, ~2h)

1.1 **Fix the split** at `tokenize_sft_data.py:202-206`: replace the per-source
    shuffle with a global deterministic rule —
    `val if sha256(normalized_question) % 1000 < val_fraction*1000` — so the
    same question lands in the same split in every source. Record the rule in
    the registry.
1.2 **Tokenize** reasoning_distill, tool_use, tool_negative, no_robots,
    creative. Note: 1.1 stamps `split_rule` into meta.json and `is_fresh()`
    requires it, so a plain `--datasets all` run auto-retokenizes ALL
    previously tokenized sources (tulu3 @50k, metamath's 3,518 straddling
    dupes included) — no `--force` needed. Then `consolidate_sft_data.py`
    (fails loud if any source still has the old split).
1.3 **Gate:** registry lists real train/val counts for every source; check
    `too_long` drops on tool_use (char p95 ≈ 1,462, max 6,680 — should clear
    2048 tokens, but verify).

### Phase 2 — round-trip verification (gate, ~1h)

Extend `tests/test_tokenize_sft_tooluse.py` to 5 real records from
`tool_use_raw.jsonl`:
- labels COVER assistant content + `<|tool_call|>` (id 7) + serialized call
  JSON + assistant `<|im_end|>` + final EOT;
- labels NEVER cover `role:"tool"` (id 8) content or its `<|im_end|>`;
- `build_sft_prompt` renders byte-identical to what the tokenizer encoded.
Run the full suite (105 tests green as of 2026-08-20).

### Phase 3 — trainer changes + the run (~6–7h GPU)

3.1 Code fixes in `sft/sft.py`:
    - **H2** (line 526): accumulate `reduction="sum"`, divide once by total
      non-ignored target tokens per optimizer step. Bigger bias in v2 (dolly
      p50 110 vs tool traces >1,000).
    - **low-13** (~line 477): assert registry `block_size` ≤ model
      `BLOCK_SIZE` at startup, fail loud.
    - **H3** (line 261): per-artifact push queue instead of "skip if a push
      thread is alive" — a 2× longer run must not silently miss snapshots.
    - **NEW `sft_best.pth`:** on each eval, if overall val improves, save +
      push a best checkpoint. Ship this, not the epoch-2 endpoint (§1d).
3.2 **Move v1 checkpoints out of the auto-resume glob** (BEFORE launch):
    `rclone move gdrive:synapse/sft_checkpoints gdrive:synapse/sft_checkpoints/v1_7source --include "sft_*.pth"`
    Renaming in place is NOT enough — `sft_epoch1_7source.pth` still matches
    the `sft_epoch*.pth` glob.
3.3 **Mix** — recompute from real Phase-1 registry counts; target shape at
    P ≈ 177k (tulu3 ~43k, tool_negative ~6k), ceiling-checked at that P:

    | source | w | draws/ep | per-ex/ep | rationale |
    |---|---:|---:|---:|---|
    | tulu3 | 0.250 | 44,250 | 1.03× | backbone; only source covering the 0% bench categories |
    | tool_use | 0.140 | 24,780 | 0.72× | new output behavior |
    | dolly | 0.120 | 21,240 | 1.48× | clean short instructions |
    | metamath | 0.100 | 17,700 | 0.82× | at floor (0.281) — cut hard |
    | reasoning_distill | 0.100 | 17,700 | 0.69× | no-tool CoT contrast |
    | no_robots | 0.080 | 14,160 | 1.57× | human prose breadth |
    | tool_negative | 0.050 | 8,850 | 1.48× | the "don't call" signal |
    | samsum | 0.050 | 8,850 | 1.13× | summarization |
    | alpaca_gpt4 | 0.047 | 8,320 | 1.43× | GPT-4 prose |
    | opencode | 0.030 | 5,310 | 0.90× | at floor (0.260); prose tilt |
    | oasst1 | 0.020 | 3,540 | 1.61× | supply ceiling |
    | creative | 0.013 | 2,300 | 1.59× | supply ceiling |

    Shape: general/prose 0.58 · math (tool_use + rd + metamath) 0.34 ·
    tool_negative 0.05 · code 0.03. (Draft plan had math 0.42; unweighted pool
    would be 0.57.) Rules: (1) hard constraint `w ≤ 1.65·N/P` per source —
    oasst1, creative, no_robots bind first and move with P; (2) keep math
    ≤ ~0.35 — metamath/opencode gradient is spent; (3) if creative or
    no_robots are skipped, give their weight to tulu3 (has headroom), not to
    dolly/alpaca (already ~1.45×).
3.4 **Run:** clean start from pretrain. EPOCHS 2, MAX_LR 1.5e-5, MIN_LR
    1.5e-6, warmup 100, eval every 200, on Colab/RunPod with
    `CHECKPOINT_PUSH_REMOTE=gdrive:synapse/sft_checkpoints`. Manifest records
    init checkpoint + final mix + v1 finals as `prior_sft_reference`.
    Expect tool_use baseline val ~3–6 (tokens 7/8 unseen at pretrain) — what
    matters is that it drops. **Final model = `sft_best.pth` by overall val.**

### Phase 4 — make it agentic (~half a day)

4.1 Move `CANONICAL_TOOL_SYSTEM` from `generate_tool_use.py:47` →
    `tools_runtime.py`. The chatbot must not import a generator script.
4.2 New `sft/tool_loop.py` (write ONCE, share): parse `<|tool_call|>` (id 7)
    → execute via `tools_runtime` → append `{"role":"tool"}` → regenerate.
    Max 5 iterations; handle malformed JSON and repeated identical calls.
    `<|tool_call|>` is NOT a stop token; `sft_stop_token_ids` unchanged.
4.3 Wire into `sparky_chatbot.py`: tools ON → canonical tool prompt; tools
    OFF → `""`. Delete the hardcoded string at 563/641 — it is a third
    distribution the model never saw. Hold `gen_lock` across the whole loop.
4.4 New `sft/eval_tool_use.py` (imports the same tool_loop): pass@1,
    tool-call rate, calls/trace, malformed-JSON rate, AND **false-call rate on
    prose prompts under the tool system prompt** — the number `tool_negative`
    exists to move.
4.5 Fix **H5** (`sparky_model.py:233` `strict=False`) before trusting any
    eval number. Re-run `sparky_leaderboard.py` + `sft_bench.py`.

---

## 3. Build-order checklist

- [x] 0.1 tulu3 cap → 50k; download tulu3 + no_robots (Colab, 2026-08-23: 50,000 + 9,482 kept)
- [x] 0.2 creative: `--limit 200` → inspect → full run
- [x] 0.3 build `tool_negative` (relabel, no API)
- [x] 1.1 global hash-based val split in `tokenize_sft_data.py` (done 2026-08-23)
- [x] 1.2 tokenize + consolidate (Colab 2026-08-25: 12 sources, 175,245 train / 3,569 val)
- [x] 1.3 gate passed: tool_use too_long=0; val ~2% everywhere; tulu3 train 39,889
- [x] 2.  round-trip mask tests on 5 real tool traces green (TestRealToolUseRoundTrip)
- [x] 3.1 sft.py: H2 (token-mean accum, verified numerically), low-13 block guard, H3 durable pushes, `sft_best.pth` + 1.65x ceiling assert
- [x] 3.2 v1 checkpoints moved on Drive → `sft_checkpoints/v1_7source/` (2026-08-25)
- [x] 3.3 mix recomputed vs registry (P=175,245): planned weights all clear 1.65x (max oasst1 1.60x); hardcoded in sft.py
- [x] 3.4 run done 2026-08-29 (5,478 steps, 2 epochs). best overall val 1.349 @ step 1200 (44% of epoch 1); endpoint 1.425 — prose sources degrade after 1200 while math/tool improve. Ship `sft_best.pth`.
- [x] sft_bench rerun 2026-08-31 (judge on, EMPTY system prompt = trained distribution): sft_best 28/49, epoch1 30/49, epoch2 28/49 (v1: 24/49) — checkpoints within noise; the seven 0% categories barely move (IF 1–2/4, text 0–1/3, logic 1/3, translation 0/2, refusal 1/2, creative 0–1/2, summ 0/1). Fixes: `DEFAULT_SYSTEM_PROMPT=""`, final-answer-aware numeric grader.
- [x] 4.1 `CANONICAL_TOOL_SYSTEM` → `tools_runtime.py` (generator re-imports; byte-identical to trained records asserted)
- [x] 4.2 `sft/tool_loop.py` (id-level, streaming events, max 5 rounds, malformed→error result, repeat→stop) + `tests/test_tool_loop.py` (9 tests, real tokenizer + sandbox)
- [x] 4.3 chatbot: tools toggle → canonical prompt / empty prompt (hardcoded prompt deleted), tool blocks in UI, full trace kept in history, gen_lock across the loop, `sft_best.pth`; Colab nb unmounts Drive before serving
- [x] 4.4 `sft/eval_tool_use.py` + `sparky/eval_tool_use_colab.ipynb` (legs A/B/C, pass@1, call rate, false-call rate, malformed rate)
- [x] 4.5 H5: `from_checkpoint` raises on missing/unexpected keys (train/infer key sets verified identical, 227/227); CPU end-to-end on real `sft_best.pth`: strict load OK, python call executed, answer returned
- [x] 4.6 tool eval run 2026-09-01 (sft_best, n=100/leg, greedy): **A** call 100%, pass@1 28% (vs 21% same questions with no prompt — paired: tool-only 16 / no-tool-only 9 / both 12 / neither 62); **B** false-call 43% overall but **prose 0/57, trivial-math 43/43** — the prose boundary took perfectly, the trivial-math boundary did not (model calls python on ALL math); **C** call 0/100. Malformed 2-11% = long code truncated at --max-tokens 384 mid-JSON, not a format failure. Acceptance 5 (both-ways contrast) ✓ for prose, ✗ for trivial math; acceptance 6 (0<pass@1<100) ✓.

---

## 4. Acceptance criteria

1. Manifest: pretrain checkpoint as init, final recomputed mix recorded.
2. 7 base sources at/below v1 finals (tulu3 1.745 · metamath 0.281 ·
   dolly 2.003 · alpaca 1.429 · samsum 1.431 · opencode 0.260 · oasst1 1.532),
   allowing small-source noise — measured on the leak-free split.
3. tool_use + reasoning_distill val present, non-NaN, dropping substantially
   from elevated pretrain baselines.
4. Chatbot completes a real python-tool turn end to end.
5. **The contrast holds BOTH ways:** tool prompt + math → calls tools;
   tool prompt + prose → answers directly (false-call rate low);
   empty prompt → never calls. All three legs, same questions.
6. `eval_tool_use.py` pass@1 strictly between 0% and 100%.
7. **The success metric: sft_bench's seven 0% categories move off zero.**
   Val loss and GSM8K are secondary — v1 proved 1.24 val coexists with 49%
   bench and chance-level MMLU.

## 5. Budget & risks

- ~$0.30 API (creative) + one GPU day. Everything in Phases 0–2 is local/free.
- Biggest schedule risk: Phase 0.1 download + retokenize churn.
- Local disk (24 GB free) cannot hold a full checkpoint — never run training
  locally; keep artifacts on Drive via rclone.
- H1 (`\frac` verifier bug, `generate_reasoning_distill.py:150`) stays
  deferred: it biases only FUTURE generation runs; fix before any regeneration.
