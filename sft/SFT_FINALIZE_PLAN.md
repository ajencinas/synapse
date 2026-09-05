# SFT Finalization Plan — tools + reasoning into the train

**Status: DRAFT for review — 2026-06-21.** No code written yet under this plan.

This doc proposes how to close the open work and land a single combined SFT run
that adds `reasoning_distill` + `tool_use` to the existing 7-source mix. It is
the "finalize the train" step requested after the tool-use + reasoning-distill
generators were written.

Everything here is grounded in a scan of the repo + Drive data (rclone). The
arithmetic is computed from **actual on-disk counts**, not the planning docs'
estimates — see §0.

---

## 0. Ground truth at the time of writing (scanned, not assumed)

### 0a. A 7-source SFT run **already completed**
`gdrive:synapse/sft_checkpoints/` contains `sft_epoch1.pth`, `sft_epoch2.pth`,
`sft_latest.pth`. `manifests/sft_training_latest.json`:
- `stage: sft`, `epoch: 2`, `curr_step: 2280`, `total_steps: 2280`
- `data_mix` = the old 7-source mix (tulu3 .30 / metamath .28 / dolly .16 /
  alpaca_gpt4 .08 / samsum .07 / opencode .06 / oasst1 .05)
- **Baseline (pre-SFT) → Final (post-SFT) per-source val loss:**

| source | baseline | final | Δ |
|---|---|---|---|
| tulu3 | 2.160 | 1.745 | -0.42 |
| metamath | 0.897 | **0.281** | -0.62 (near floor) |
| dolly | 2.494 | 2.003 | -0.49 |
| alpaca_gpt4 | 1.787 | 1.429 | -0.36 |
| samsum | 3.348 | 1.431 | -1.92 |
| opencode | 0.669 | **0.260** | -0.41 (near floor) |
| oasst1 | 1.842 | 1.532 | -0.31 |
| **overall** | **1.885** | **1.240** | -0.65 |

**Implication for the new run:** metamath and opencode are already near their
floor on the existing 7-source data — pushing their mix weights *up* further
will not yield more reasoning gain (the gradient is exhausted). The reasoning
lever has to be the **new distributions**: `reasoning_distill` (CoT style the
base never saw) and `tool_use` (a capability the base never had). The new mix
should *not* re-inflate metamath to 0.28 thinking it's the reasoning anchor —
that thesis from `SFT_DATA_PLAN.md §6` is partially spent.

**The existing run is also the regression baseline.** The new combined run's
per-source numbers must beat the **final** column above on the 7 base sources
(not just the pre-SFT baseline) — otherwise adding tools/reasoning caused
catastrophic forgetting of general capability. Record both baselines in the
new run's manifest.

### 0b. `reasoning_distill` — generated, NOT tokenized
`gdrive:synapse/datasets_sft/reasoning_distill/`:
- `meta_raw.json`: processed=3000, **kept=1577**, yield=52.6%, cost=$0.98
- `reasoning_distill_raw.jsonl` present (schema: `{id, system:"", messages:[...]}`,
  roles user/assistant only — sample first line verified, plain CoT, no tools)
- **`gdrive:synapse/sft_tokenized/reasoning_distill/` does NOT exist** — never
  tokenized. The 7-source `sft_data_registry.json` has no `reasoning_distill`
  entry. So it can't be added to `SFT_DATA_MIX` today (`sft.py` line 212/214
  hard-fails on a mix source with no train.jsonl).

### 0c. `tool_use` — clean final file still pending
`gdrive:synapse/datasets_sft/tool_use/` must contain the clean final file:
`tool_use_raw.jsonl`. Dirty scratch outputs (for example old `.warty` files)
are not part of the training source and must stay deleted/ignored.

Implications:
- `tokenize_sft_data.py` will only read `datasets_sft/tool_use/tool_use_raw.jsonl`.
- The final `tool_use` count is unknown until the clean generation run finishes.
- All `tool_use` mix weights in §2 are provisional examples from the old count
  scan. Before editing `SFT_DATA_MIX`, recompute using the final tokenized
  `train_count` from `sft_data_registry.json`.
- Python-mode traces are still the best first target; search/mixed can be added
  after python mode is clean and tokenized.

### 0d. Code gaps (verified by reading the current files, not assumed)
Some TOOL_USE_PLAN items have already landed since the first draft of this doc.
Keep this table current so the build order does not send us to re-implement
work that is already in the repo.

| area | current state | remaining work |
|---|---|---|
| tokenizer tool support | `sft/tokenize_sft_data.py` already maps `role:"tool"` to `<\|tool_result\|>` id 8, resolves `<\|tool_call\|>` id 7, imports `dump_tool_call`, and learns assistant inline tool-call spans. | Do the round-trip mask check on clean `tool_use` examples after generation/tokenization. |
| chat template rendering | `sparky/sparky_chat_template.py` already renders assistant `tool_call` dicts inline and `role:"tool"` as `<\|tool_result\|>`, with explicit role validation. | Add/shared-use the canonical tool system prompt so training and inference condition on the same tool instructions. |
| generator/runtime | `sft/generate_tool_use.py` and `sft/tools_runtime.py` exist. | Generate the clean final `tool_use_raw.jsonl` and then tokenize it. |
| chatbot inference | `sparky/sparky_chatbot.py` is still single-shot around one `model.generate` call. | Add the parse-execute-reinject loop for `<\|tool_call\|>` turns, holding `gen_lock` across the full loop. |
| trainer mix/resume | `sft/sft.py` still has the old 7-source `SFT_DATA_MIX` and auto-resumes `sft_latest.pth` / `sft_epoch*.pth`. | Recompute mix from final tokenized `train_count`s, add new sources, and make warm-start/resume explicit so stale 7-source checkpoints cannot silently drive the run. |
| tool eval | `sparky/sparky_eval.py` is a base-model lm-eval/loglikelihood harness. | Create a new generation/tool-loop eval script, e.g. `sft/eval_tool_use.py`. |

So the remaining work is not "add all tool plumbing from scratch"; it is:
clean data generation, tokenization verification, trainer mix/init handling,
canonical tool-system-prompt wiring, chatbot tool loop, and a tool-loop eval.

---

## 1. The one decision that controls everything else

**Retrain from the pretrain checkpoint with the full 9-source mix, or use the
existing 7-source SFT checkpoint as the initialization and add the new sources?**

Using the existing SFT as the starting point is **not wrong**. It is the
preferred path if the goal is to preserve the already-good chat behavior and add
reasoning/tool behavior cheaply. The tradeoff is that it is no longer a clean
apples-to-apples 9-source run from pretrain: the 7 old sources already received
two epochs of gradient before the new sources appeared.

The preferred path for this repo is therefore:

1. **Warm-start from `sft_epoch2.pth` / `sft_latest.pth`.** Load model weights
   only, use a fresh optimizer, reset `curr_step=0`, compute a new LR horizon
   from the final tokenized train counts, and use a lower LR than the original
   SFT run (start around `5e-6` to `8e-6`). Keep replay from the 7 base sources
   in the mix; do not train only on the two new sources unless this is explicitly
   a narrow repair run.
2. **Clean 9-source run from pretrain.** Keep this as the comparison/fallback if
   warm-start causes base-source regression or fails to learn the new tool
   tokens. It is the cleaner experiment, but not required for the practical goal.

**Important implementation distinction:** warm-start is not resume. If the old
`sft_latest.pth` or any `sft_epoch*.pth` remains where `sft.py` auto-detects it,
the script will resume the old run (or skip because it thinks epoch 2 is already
complete) instead of starting a new phase. Add an explicit model-init path
(for example `SFT_INIT_CKPT=/path/to/old_sft.pth` that loads only `"model"` and
sets `resuming=False`), or move old checkpoints outside the auto-resume glob and
load them through a clearly named warm-start option.

Record the init path in the manifest (`results.init_checkpoint` or similar), so
future comparisons know whether a run was pretrain-init or SFT-warm-start.

---

## 2. The mix — recomputed against actual counts (the core of this doc)

**Current status note:** `tool_use` clean counts are pending. The arithmetic below
shows the intended method and the old scanned-count example; do not execute these
weights literally after regenerating tool data. Recompute from final tokenized
`train_count`s before training.


### 2a. The bug in the §8 mix
`TOOL_USE_PLAN.md §8` proposes:
```
tulu3 0.24 / metamath 0.20 / reasoning_distill 0.14 / tool_use 0.12 /
dolly 0.12 / alpaca_gpt4 0.06 / samsum 0.05 / opencode 0.04 / oasst1 0.03
```
That mix was written when both sources' yields were unknown ("target a few-k
verified traces"). **Actual counts: reasoning_distill = 1,577, tool_use = 4,161.**
Run the per-example draw arithmetic the plan already uses (`SFT_TRAIN_PLAN.md
§2.2`):

- Pool (P) = 72,912 (7 base) + 1,577 (reasoning) + 4,161 (tool_use) = **78,650**
- `reasoning_distill` at §8's **0.14** → draws/epoch = 0.14 × 78,650 ≈ 11,011
  → per-example = 11,011 / 1,577 ≈ **7.0×/epoch** → **14× over 2 epochs**
- `tool_use` at §8's **0.12** → draws/epoch = 0.12 × 78,650 ≈ 9,438
  → per-example = 9,438 / 4,161 ≈ **2.3×/epoch** → **4.5× over 2 epochs**

The plan's own overfit watch-line is **oasst1 at 1.65×/epoch (3.3× over 2
epochs)** — `SFT_TRAIN_PLAN.md §2.2` explicitly says "watch oasst1… lower its
weight if it diverges". 14× is **4.2× past** that watch line. 4.5× is **1.4×
past** it. The §8 mix would overfit both new sources hard.

### 2b. The ceiling rule (formalize what the plan already does for oasst1)
For a source with `N` train examples in a pool of `P` drawn with replacement at
weight `w`: per-example passes/epoch = `w·P / N`. Cap at **≤ 1.65×/epoch** (the
oasst1 watch line) → `w ≤ 1.65 · N / P`. With `P = 78,650`:

| source | N (train) | max w @ 1.65× ceiling | proposed |
|---|---|---|---|
| reasoning_distill | 1,577 | 0.033 | **0.035** (slightly above ceiling, 1.75×/epoch — watch) |
| tool_use (python) | 4,161 | 0.087 | **0.085** (at ceiling 1.62×/epoch — on target) |
| oasst1 | 2,205 | 0.046 | **0.035** (was 0.05 → 1.65×; 0.035 is 1.25×, safer) |

### 2c. Three levers — pick one for reasoning_distill
**Lever A — accept 1,577, cap at w=0.035.** Cheap ($0.98 already spent), low
risk, but reasoning_distill is only ~3.5% of batches. Small reasoning gain.

**Lever B — scale reasoning_distill to ~5k kept (~$3 more).** reasoning_distill
yield was 52.6%. 5k kept → ~9,500 processed → ~$3.10 (at $0.98/3,000).
At N=5,000: max w = 1.65 × 5,000 / 78,650 = 0.105. Then w=0.10 is at the
ceiling (1.57×/epoch) and it's 10% of batches — real reasoning impact.

**Lever C — scale to ~8k kept (~$5 more).** At N=8,000: max w = 0.168.
Then w=0.14 is back to the §8 number. This is the original vision.

**Recommendation: Lever B = scale to ~5k kept ($3).** The base is a STEM
specialist (47% STEM pretrain); reasoning_distill is the highest-fit new data.
The existing run proved metamath (0.281) is near-floor — the *new* CoT
distribution is the only place reasoning gain is still available. $3 is a tiny
cost relative to the compute time.

### 2d. The search/mixed mode question
4,161 traces are all python mode. The `search` mode would capture the
"reasoning trigger" behavior (search the method, then solve). `mixed` mode
would give natural behavior. If these are desired:
1. Run `--mode search` (target ~2-3k kept at Brave's 1 req/s — wall-clock
   dominated by rate limiting, ~1-2 hours for 2k kept at ~40% yield).
2. Run `--mode mixed` (similar cost, both tools available).
3. Add to the pool and recompute the mix weight via the §2b ceiling.

Without search/mixed, the model only learns calculator-using behavior, not
the "search the method" reasoning trigger. The training will still work —
python tool use alone is valuable — but the original TOOL_USE_PLAN §8
envisioned a reasoning-trigger behavior that only search traces provide.

If search/mixed are postponed, the mix shifts: tool_use stays at 4,161 and
w=0.085 (already computed). If search/mixed are added to 6-8k total, the
w goes up proportionally.

### 2e. Concrete proposed mix — "combined run from pretrain"
Assuming Lever B (reasoning scaled to 5k) and tool_use at 4,161 (python only,
no search yet). **If Lever A accepted, change reasoning_distill to 0.035 and
boost tulu3 to 0.305.**

```python
SFT_DATA_MIX = {
    "tulu3":              0.267,  # backbone (15,277 ex)
    "metamath":          0.218,  # near-floor (0.281); distribution anchor
    "reasoning_distill": 0.099,  # IF scaled to ~5k; else 0.035
    "tool_use":          0.084,  # 4,161 ex → 1.62×/epoch at ceiling
    "dolly":             0.139,  # 14,332 ex
    "alpaca_gpt4":       0.069,  # 5,820 ex
    "samsum":            0.050,  # 7,839 ex
    "opencode":          0.040,  # near-floor (0.260)
    "oasst1":            0.035,  # 2,205 ex, down from 0.05
}
# sum = 1.000 (normalized)
```

**If reasoning_distill stays at 1,577 (Lever A):** set w=0.035 (1.75×/epoch,
slightly above ceiling), redistribute the 0.064 to tulu3 (+0.04 → 0.307) and
metamath (+0.024 → 0.244). Run still works; reasoning gain is smaller.

**Per-epoch draw table (P = 78,650):**

| source | N | w | draws/epoch | per-ex/epoch | per-ex/2ep |
|---|---|---|---|---|---|
| tulu3 | 15,277 | 0.267 | 21,020 | 1.38× | 2.75× |
| metamath | 21,560 | 0.218 | 17,146 | 0.80× | 1.59× |
| reasoning_distill | 5,000 | 0.099 | 7,786 | **1.56×** | **3.12×** |
| tool_use | 4,161 | 0.084 | 6,607 | **1.59×** | **3.18×** |
| dolly | 14,332 | 0.139 | 10,890 | 0.76× | 1.52× |
| alpaca_gpt4 | 5,820 | 0.069 | 5,447 | 0.94× | 1.87× |
| samsum | 7,839 | 0.050 | 3,907 | 0.50× | 1.00× |
| opencode | 5,879 | 0.040 | 3,130 | 0.53× | 1.06× |
| oasst1 | 2,205 | 0.035 | 2,733 | 1.24× | 2.48× |
| **total** | **81,073** | **1.000** | **78,650** | | |

Both reasoning_distill and tool_use are at ~1.6×/epoch (≈3.2×/2ep) — right at
the ceiling. If either diverges in per-source val loss, stop after epoch 1.

**If reasoning_distill stays at 1,577 (Lever A):** set w=0.035 (1.75×/epoch,
slightly above ceiling), redistribute the 0.064 to tulu3 (+0.04 → 0.307) and
metamath (+0.024 → 0.244). Run still works; reasoning gain is smaller.

### 2f. Epochs and LR — adjusted for warm-start
The warm-start approach (§1) changes the training parameters from a full from-scratch run:

- **Epochs: 1** (additional pass on top of the existing 2-epoch SFT). The old sources
  already had 2 epochs; one more epoch at lower weight keeps them current without
  overfitting. The new sources get ~1.6×/epoch in practice, or ~1× the entire pool's
  scheduled weight.
- **MAX_LR: 5e-6 to 8e-6** (lower than the original 1.5e-5 — the model is already
  well-fit to the 7 old sources and needs gentle refinement, not large gradient updates).
- **MIN_LR: 1e-6** (same as original, so the cosine decays to a stable floor).
- **Warmup: 50 steps** (reduced from 100 — the loss landscape is already near a minimum).
- **Horizon: total_steps** = steps_per_epoch = 1,229.
**
Steps/epoch = ceiling(P / BATCH_SIZE × GRAD_ACCUM) = ceil(78,650 / 64) ≈ 1,229.

The LR schedule is:
```python
for step in 0..50 (warmup):    lr = MAX_LR × step / 50
for step in 51..1229 (cosine): lr = MIN_LR + 0.5 × (MAX_LR−MIN_LR) × (1 + cos(π × (step−50) / (1229−50)))
```

This matches Tulu 3's approach: continue from the best checkpoint with a reduced LR
and 1–2 additional epochs (ref. Section 4.2 of the Tulu 3 paper).

The original 2-epoch / 1.5e-5 plan and its per-token slowdown considerations are
only relevant if the fallback from‑scratch pretrain run is chosen instead.

### 2g. Research-based refinements (from literature survey)
The literature (Tulu 3, FireAct, Toolformer, Gorilla, OpenHermes) suggests
refinements to the mix and monitoring above:

1. **Tool_use per-example rate should be higher than reasoning_distill.**
   Tool use is a *new output behavior* (the model never emitted `<|tool_call|>`
   in pretraining or the 7-source SFT). Reasoning CoT fits the *existing
   distribution* (the base is 47% STEM). Behavioral pattern-learning needs
   more repetition. The plan's mix gives both ~1.6×/epoch — reasonable but
   suboptimal. If possible, tool_use should be ~2.0×/epoch and
   reasoning_distill ~1.3×/epoch. To achieve this without breaching the
   ceiling for tool_use (N=4,161 → max w=0.087 at 1.65×), either:
   a) Scale tool_use generation to ~5,000 kept (then max w=0.105 → headroom
      for 2.0×/epoch), or
   b) Accept the current mix; tool_use will learn more slowly but still
      converge given diverse task coverage.

2. **Per-sequence loss for short-source monitoring.** The per-token loss
   denominator is small for tool_use traces (~600 tokens vs ~1500 for
   metamath), making per-token val loss noisier. Track per-sequence (raw NLL)
   alongside per-token loss. Flag any source whose val loss exceeds its running
   minimum by +0.03 (absolute) or hasn't improved in 400 steps while others
   continue improving.

3. **Failure-recovery traces improve robustness (FireAct finding).** The
   existing tool_use traces are all successful (rejection-sampled). FireAct
   showed that mixing in 10–20% failure-recovery trajectories reduced
   inference-time errors. Consider a second generation pass that preserves
   failed-then-recovered traces, or accept clean-success-only for a first run
   and add recovery in a follow-up.

4. **Dedicated tool_use validation set.** At `val_fraction=0.02`, tool_use
   with ~4,161 records yields only ~83 val examples — too small for reliable
   per-source loss tracking. Hold out ≥100–200 examples for val, or create a
   fixed eval split from the problem pool (the generator's problem IDs provide
   a natural basis).

5. **The 1.65× ceiling is pragmatic, not standard.** Tulu 3 uses per-source
   weight caps (max fraction of each batch) rather than per-example draw-rate
   ceilings. The two approaches converge at large scale. The 1.65× number
   anchors on oasst1's observed behavior — it's defensible but not a hard
   bound. Adjust it based on observed val loss behavior during warm-start.

---

## 3. Execution order (gated — each step's gate is the next step's input)

Steps 1–5 are the "train + template" path; steps 6–8 are the "eval loop".

### Step 1 — Tokenize `reasoning_distill` (no code change needed)
`reasoning_distill` has roles user/assistant only — the existing
`tokenize_sft_data.py::encode_example` already handles it. Run:
```bash
SYNAPSE_DIR=… python sft/tokenize_sft_data.py --datasets reasoning_distill
SYNAPSE_DIR=… python sft/consolidate_sft_data.py
```
No `--force` needed if this is the first tokenization. The registry rebuild
picks it up. Gate: `sft_tokenized/reasoning_distill/{train,val}.jsonl` exists
and `sft_data_registry.json` has a `reasoning_distill` entry.

### Step 2 — Verify existing tool tokenization support
`tokenize_sft_data.py` already has the required tool path: `role:"tool"` maps
to `<|tool_result|>` id 8, `<|tool_call|>` id 7 is resolved separately, assistant
`tool_call` dicts are serialized with `dump_tool_call`, and tool result content
is masked.

Before tokenizing the clean tool-use source, do a small round-trip check:
- encode/decode 3-5 clean tool traces;
- verify labels cover assistant content, inline `<|tool_call|>` spans, serialized
  tool-call JSON, assistant `<|im_end|>`, and final EOT;
- verify labels never cover `role:"tool"` result content or its `<|im_end|>`.

Gate: the round-trip check passes on a hand-written trace or a few freshly
generated clean `tool_use` examples.

### Step 3 — Generate clean `tool_use` traces (python first; search/mixed optional)
Generate the real final `datasets_sft/tool_use/tool_use_raw.jsonl`. Dirty scratch
files must not be renamed into this path.

Start with python mode because it is deterministic and has no Brave dependency.
**Scale target: ~5,000 kept traces** (the draft mix assumes 4,161; 5,000 gives
enough headroom for tool_use to reach 2.0×/epoch draw rate — see §2g.1):
```bash
SYNAPSE_DIR=... python sft/generate_tool_use.py --mode python --limit 1500   # calibrate
SYNAPSE_DIR=... python sft/generate_tool_use.py --mode python                # full run
```

**Recommended: preserve failure-recovery traces.** The generator currently
rejects traces where the model failed on the first tool call but recovered on
a retry. FireAct found that keeping 10–20% of these improves inference
robustness. Add a `--keep-recovered` flag to `generate_tool_use.py` that saves
traces where `used_tool ≥ 1` and the final answer is verified, regardless of
intermediate failures. Alternatively, accept clean-only for the first run (the
model will still learn basic tool syntax) and add failure-recovery in a
follow-up generation pass.

**Validation split:** after generation, hold out >100 examples from the pool
as a fixed `tool_use` val set, not the random 2% split
(`tokenize_sft_data.py`'s default). The generator's problem IDs provide a
natural partition — e.g., hold out every 50th problem ID. If the val split
is smaller than 100, the per-source val loss signal will be too noisy for
reliable monitoring (§2g.4).
1. Build/fetch the verifiable fact problem DB if using fact lookup.
2. Run `python sft/generate_tool_use.py --mode search ...` with a small `--limit`
   first; Brave is rate-limited around 1 req/s on the free tier.
3. Optionally run `--mode mixed` after search is calibrated.
4. Push the final clean `tool_use` dataset to Drive.
5. Recompute the mix using the final **tokenized train_count**, not raw line
   count, after Step 4.

Gate: `datasets_sft/tool_use/tool_use_raw.jsonl` exists, has the expected schema,
and does not include rejected/scratch data.

### Step 4 — Tokenize `tool_use`
```bash
SYNAPSE_DIR=… python sft/tokenize_sft_data.py --datasets tool_use
SYNAPSE_DIR=… python sft/consolidate_sft_data.py
```
**Important:** `tokenize_sft_data.py` will filter tool_use examples for
`too_long` (>2048) and `short_response` (<3 learned content tokens). The
tool_use traces have ~2 tool calls × ~200 tokens/call + system prompt (~80
tokens) ≈ ~500-700 tokens typical — well within budget. But the system prompt
(`CANONICAL_TOOL_SYSTEM`, ~80 tokens) is stored per-record and is masked, so
it counts toward `block_size` but not `resp_tokens`. No systematic overflow
is expected for python-mode traces, but the final answer is the tokenization meta.
Check `drops`, `train_count`, and `val_count` in the output.

Gate: registry has both new sources; `sft_tokenized/{reasoning_distill,tool_use}/
{train,val}.jsonl` all exist.

### Step 5 — Update `sft.py` + warm-start from the existing SFT
- Recompute the final mix from `sft_data_registry.json` after Steps 1-4. Use
  tokenized `train_count`s for the pool size and per-source ceilings, not raw
  JSONL line counts.
- Edit `sft/sft.py:64-72` to the final normalized mix.
- Add an explicit warm-start path, e.g. `SFT_INIT_CKPT=/path/to/sft_epoch2.pth`,
  that loads only the checkpoint `"model"` weights, creates a fresh optimizer,
  sets `resuming=False`, resets `curr_step=0`, and computes a new LR horizon from
  the new dataset. Start with a lower LR than the original SFT run (`5e-6` to
  `8e-6`) unless calibration says otherwise.
- Prevent accidental auto-resume. `sft.py` checks `sft_latest.pth` first and then
  any `sft_epoch*.pth` snapshots. If old checkpoints are kept as references, move
  them to a subdirectory or rename them so they do **not** match `sft_latest.pth`
  or `sft_epoch*.pth`. Do not rename old snapshots to `sft_epoch1_7source.pth`;
  that still matches the glob.
- Record the existing run's final eval (§0a) and the init checkpoint in the new
  run's manifest, e.g. `results.prior_sft_reference` and
  `results.init_checkpoint`.
- Run the warm-start phase. Monitor per-source val every 200 steps. If
  `reasoning_distill` or `tool_use` loss rises across multiple evals while base
  sources degrade, stop and lower the new-source weights or LR.

### Step 6 — Template/system-prompt plumbing (TOOL_USE_PLAN Phase 1, chatbot side)
`sparky/sparky_chat_template.py` already renders tool calls/results. The missing
piece is the canonical tool system prompt.

- Put the canonical tool prompt in a shared import location used by both data
  generation and inference. Prefer a small protocol module or `tools_runtime.py`
  over importing it from `generate_tool_use.py`; the chatbot should not depend on
  a generator script at runtime.
- Ensure `build_sft_prompt(..., system=TOOL_SYSTEM_PROMPT)` renders byte-identical
  system/tool-call/tool-result text to what `tokenize_sft_data.py` encoded.
- Keep `sft_stop_token_ids` unchanged (`<|im_end|>` + EOT). `<|tool_call|>` is an
  inline marker, not a stop token.
- Round-trip: encode a hand-written tool trace, decode, and confirm it reproduces
  the wire string in `TOOL_USE_PLAN.md §2`.

### Step 7 — Chatbot tool loop (TOOL_USE_PLAN Phase 6)
`sparky/sparky_chatbot.py:638-681`: replace single-shot SFT generation with the
loop from `TOOL_USE_PLAN.md §6`.

The chatbot uses `SynapseInfer.generate()` (from `sparky_model.py:156-204`),
which yields token IDs until a stop token is hit, then breaks. `<|tool_call|>`
(id 7) is NOT a stop token, so the existing SFT stop set remains correct.
Generation continues through the tool-call JSON until `<|im_end|>`.

The loop:
1. Build prompt ending on `<|assistant|>` via `build_sft_prompt` (Step 6), using
   the canonical `TOOL_SYSTEM_PROMPT` as the system message.
2. Encode the prompt, call `model.generate(..., stop_tokens=sft_stop_token_ids(...))`,
   and collect the generated token IDs for this assistant turn.
3. Detect tool calls by token ID 7. Decode the IDs after 7, parse the JSON dict,
   and accept `{"tool":"python","code":...}` or `{"tool":"search","query":...}`.
4. Execute with `tools_runtime.run_python(...)` or `tools_runtime.run_search(...)`,
   truncate with the same budgets used in generation, and append a
   `{"role":"tool","content":...}` message.
5. Append the assistant turn as `{"role":"assistant","content":"...","tool_call":...}`
   so the next prompt re-renders the same wire format.
6. Regenerate from a fresh full prompt. Repeat until no tool call is detected or
   max iterations (5) is hit.
7. On final turn, return assistant content stripped of stop markers.
8. On cap, force a final no-tool answer request and regenerate once.

**KV cache management:** each regeneration has a different prompt, so rebuild the
KV cache from scratch. `SynapseInfer.generate()` already pre-fills the full prompt
on each call.

**Graceful degradation:** if Brave search is unavailable at inference, return the
same sentinel string as the generator and let the model finalize; do not crash.

**`gen_lock`:** hold it across the entire multi-turn tool loop. That is acceptable
for a single-user chatbot; a multi-user service would need more careful scheduling.

### Step 8 — Tool-eval harness (NEW script, NOT sparky_eval.py)
`sparky/sparky_eval.py` is a **base-model lm_eval harness** (loglikelihood
scoring on ARC/MMLU/HellaSwag). It uses `SynapseInfer` from `sparky_model.py`,
not `SynapseGPT`/`model.generate()`. Adding a SFT tool-loop eval to it would
be a mismatch — different model interface, different scoring paradigm.

Create a **new script** `sft/eval_tool_use.py` that:
1. Loads an SFT checkpoint via `SynapseInfer.from_checkpoint()` (from
   `sparky_model.py`) — this is the same interface the chatbot uses for
   generation, not the `SynapseGPT`+forward-pass used by `sft.py`.
2. Loads problem IDs + gold answers from the tool_use generator's problem
   pool (orca/numina, via `generate_reasoning_distill.load_pool()`).
3. For each problem: build the prompt with `build_sft_prompt` (Step 6,
   including `TOOL_SYSTEM_PROMPT`), run the tool loop (Step 7), extract the
   answer via `extract_final_answer()`, compare to gold via `answers_match()`.
4. Reports: pass@1, tool-call rate, avg calls/trace, overflow rate (≥2048).
   Tool calls are detected per-turn via token ID 7 — count how many turns
   contain a `<|tool_call|>` and how many succeed/produce valid JSON.
5. Optionally logs failures to a `.jsonl` for inspection.

This is the only component that can measure whether the SFT model actually
*uses tools correctly*. The per-source val loss measures generalization to
the val distribution, but val examples from tool_use show the *token patterns*,
not whether the model's tool calls would execute correctly.

If Step 6's loop function is extracted to a shared helper (e.g.
`sft/tool_loop.py`), the chatbot and the eval harness use the same code path
— no divergence.

---

## 4. Risks / open items

1. **`sft.py` cache must be deleted or it sees stale data.** The cache
   (`sft_tokenized/_all_sft_cache.pt`) keys on per-source file sizes + mtimes.
   Adding the 2 new sources changes the signature → cache auto-rebuilds. Safe.
   But if you re-tokenize a source in-place (same count, same size), the
   mtime changes and the cache invalidates correctly. Verified by reading
   `sft.py:189-203`. No action needed, noted so it's not mistaken for a bug.

2. **Search/mixed traces are a separate generation run.** Generate clean
   python-mode traces first. If the reasoning trigger behavior is wanted,
   run `--mode search` and `--mode mixed` separately after python mode is sane. Search is rate-limited:
   at 1 req/s, 48 workers, ~40% yield → expect ~1-2 hours for 2k kept traces.
   $0.98/3k reasoning_distill suggests LLM API cost is ~$0.33/k traces;
   search adds Brave API cost negligible at 1 req/s.

3. **2048 overflow on tool_use traces.** The generator pre-filters with
   `trace_max_tok=1984`, and tokenization enforces `block_size=2048` exactly.
   Check the final tokenization `drops` instead of trusting estimates. If
   search/mixed traces are generated, their per-turn length is tighter
   (`SEARCH_RESULT_TOK=128`, `--max-calls=2`).

4. **Catastrophic forgetting of the 7 base sources.** The new run must beat
   §0a's final column on tulu3/dolly/samsum/alpaca_gpt4/oasst1. If after
   epoch 1 the new run's per-source val on these is *worse*, tool_use share
   is too high. Lower tool_use by 0.02–0.03, redistribute to tulu3. The
   per-source val (every 200 steps) is the early signal — don't wait for
   epoch 2.

5. **Baseline eval for `tool_use` on the pretrain checkpoint is misleading.**
   `sft.py:488-491` computes per-source val loss on the freshly loaded pretrain
   checkpoint for all sources. The pretrain model has **zero exposure to tool
   tokens (7/8)** and zero tool-call patterns in its pretraining data. The
   baseline val loss on `tool_use` will be dominated by the model having no
   prior over `<|tool_call|>` / `<|tool_result|>` and JSON patterns. Expect it
   to be noticeably elevated compared to other sources (maybe **3–6**, not
   0.7–3.3), not "near random" (cross-entropy floor for a 64k vocab is ~11).
   What matters is that it **drops significantly** during training (e.g., 5 →
   2), not that it matches the other sources' baselines. Don't let an elevated
   baseline lead to premature panic — the real test is the Step 8 tool-eval
   bench.

6. **Cache invalidation edge case.** `sft.py:189-203` builds the cache
   signature from sources in `SFT_DATA_MIX` only. If a source in the mix
   has its raw data changed on Drive but the tokenized files are unchanged
   (e.g., re-tokenizing with `--force` re-writes files with the SAME content),
   the mtime changes but the data is identical — the cache is unnecessarily
   invalidated. This wastes ~5 seconds to rebuild, not a real problem.
   Contrarily, if the raw data changes and tokenized data is NOT re-written
   (user forgets `--force`), the cache would NOT re-read the tokenized files
   and would serve stale data. The staleness guard in `tokenize_sft_data.py`
   (`is_fresh()`) prevents the second case by detecting mismatched
   `tokenization_id`/`block_size`, but it doesn't detect content-only changes.
   Acceptable risk — the tokenizer is stable and block_size is fixed.

7. **Clean `tool_use_raw.jsonl` is required before tokenization.** The tokenizer
   looks for `datasets_sft/tool_use/tool_use_raw.jsonl` exactly. Scratch or dirty
   files with alternate suffixes are ignored unless they are renamed into that path.
   Do not rename dirty artifacts into the clean filename. Generate the final file,
   then tokenize and use the resulting `train_count`/`val_count` from the registry
   for mix math.

8. **`tool_use` val set may be small/noisy.** At a 0.02 validation split,
   every 1,000 clean traces yields only ~20 validation examples. Per-source val
   loss will be noisier than the larger sources, so use it as a rough divergence
   signal (e.g., loss rising over 2-3 evals), not as a precise metric. The Step 8
   tool-eval bench is the real accuracy measure.

9. **Per-token loss on short sequences (tool_use) is inflated by denominator.**
   Tool_use traces average ~600 tokens vs ~1500 for metamath. Per-token loss
   with `reduction="sum" ÷ count` amplifies variance on short sequences.
   Track per-sequence (raw NLL) for short sources (tool_use, oasst1) as a
   complementary signal. Flag a source if its raw NLL rises >0.03 above its
   running minimum or doesn't improve for 400 steps.

10. **Clean-only tool traces miss failure-recovery behavior.** The rejection
    sampler drops traces where the first tool call fails (even if the model
    recovers and answers correctly). FireAct (arXiv:2310.05915) showed this
    makes the model brittle at inference — it has never seen a tool error.
    Add a `--keep-recovered` pass if inference robustness matters for the
    use case. For a first run, clean-only is acceptable (the model learns
    the tool *protocol*), but add recovered traces in a follow-up.

---

## 5. Summary — minimal acceptance criteria

1. `sft_latest.pth` on Drive is from the **new warm-start 9-source phase**
   (data_mix in its manifest matches the final recomputed mix, and the manifest
   records the SFT checkpoint used for initialization).
2. Per-source val for the **7 base sources** is not materially worse than §0a's
   final column. Lower loss is better, so the target is approximately `<=` the old
   final losses, allowing normal small-source noise rather than requiring every
   source to beat its prior best by a strict margin.
3. `reasoning_distill` and `tool_use` per-source val are present, non-NaN, and
   improve from their phase-start baselines. Treat `tool_use` val as a noisy
   divergence signal, especially if the val split is only ~80 examples; the real
   tool-use capability metric is the Step 8 tool-eval bench.
4. The chatbot (`sparky_chatbot.py`) completes a real `python`-tool turn end to
   end: user asks a non-trivial math question -> model emits
   `<|tool_call|>{"tool":"python",...}` -> harness executes -> model finalizes
   with the correct answer.
5. `sft/eval_tool_use.py` reports pass@1 > 0% and < 100% on the tool-use problem
   pool. Either extreme suggests the harness, prompt, or parsing path is broken.

---

## 6. Build order checklist

- [ ] **1.** Tokenize `reasoning_distill` (no code change) -> registry has it.
- [ ] **2.** Generate clean `tool_use_raw.jsonl` (python first; search/mixed optional).
- [ ] **3.** Verify existing tokenizer tool masking on a hand-written or freshly generated trace.
- [ ] **4.** Tokenize `tool_use` -> registry has both new sources and final train/val counts.
- [ ] **5.** Recompute `SFT_DATA_MIX` from tokenized `train_count`s.
- [ ] **6.** Add explicit SFT warm-start init path, record init/prior eval in manifest, and prevent accidental auto-resume from old `sft_latest.pth` / `sft_epoch*.pth`.
- [ ] **7.** Template/system-prompt plumbing (`sparky_chat_template.py` + shared `TOOL_SYSTEM_PROMPT`) -> round-trip matches `TOOL_USE_PLAN.md §2`.
- [ ] **8.** Chatbot tool loop (`sparky_chatbot.py`) -> §5 criterion 4 holds.
- [ ] **9.** `sft/eval_tool_use.py` tool-eval harness -> §5 criterion 5 holds.
