# Tool-Use SFT Plan

Add **tool use** to the SynapseGPT SFT model: the assistant emits a tool call,
an external harness executes it, the result is injected back, and the assistant
continues until it produces a final answer.

**Decisions locked (2026-06-19):**
- **Tools:** `python` (calculator + sympy + run-code, one unified tool) and
  `search` (web). Narrow/deep beats wide for a 2B model.
- **Search backend:** **Brave Search API** (`BRAVE_API_KEY` already in `.env`),
  live at *both* generation and inference.
- **Trace generation:** **agentic distillation** — the teacher (DeepSeek V4
  Flash) runs a real tool loop, tools actually execute, keep only
  answer-verified traces (reuse the rejection-sampling rigor of
  `generate_reasoning_distill.py`).
- **Reasoning trigger** = a *behavior*, not new machinery: on hard problems the
  assistant first issues a `search` for the *method* ("approach for related-rates
  problems"), then writes a short plan, then solves. Implemented purely as a
  data pattern.

This doc is the design reference; it mirrors the structure of `SFT_DATA_PLAN.md`.

---

## 0. Constraints inherited from the base model

| Property | Value | Implication for tool use |
|---|---|---|
| Context | `BLOCK_SIZE = 2048` (hard) | Tool results MUST be hard-truncated; cap tool calls per trace (~3–4). Long search snippets blow the budget. `tokenize_sft_data.py` drops >2048 (counted as `too_long` in meta, not truly silent) — but generation must still budget conservatively so paid traces aren't discarded downstream. |
| Tokenizer | digit-per-token | Multi-digit arithmetic is expensive *and* the model's weakest skill → `python` is the **highest-value** tool. Big-number tool *outputs* also cost many tokens; truncate. |
| Vocab | 64k BBPE, 256 reserved specials | New tool tokens come from the **reserved pool — no tokenizer retrain**, `tokenization_id` stays `7a570a7ba9fc7985`. |
| Pretrain | zero tool calls ever seen | Behavior is learned entirely in SFT → need meaningful volume (target a few-k verified traces) and a non-trivial `SFT_DATA_MIX` share. |

---

## 1. Tokens — use the DEDICATED tool tokens (already in the tokenizer)

**Phase-0 is done, and the original reserved_3/4 plan was wrong.** The canonical
tokenizer (`tokenizer.json`, fingerprint `7a570a7ba9fc7985`) was **purpose-built
with named tool tokens**. Verified directly from the file:

| String emitted | **Actual id** | Emitted by | Loss |
|---|---|---|---|
| `<\|tool_call\|>` | **7** | assistant (inline) | **learned** |
| `<\|tool_result\|>` | **8** | harness (injected) | **masked** |

Full prefix, for reference: `<|endoftext|>`=0, `<|pad|>`=1, `<|im_start|>`=2,
`<|im_end|>`=3, `<|fim_*|>`=4/5/6, **`<|tool_call|>`=7, `<|tool_result|>`=8**,
`<|reserved_0/1/2|>`=9/10/11 (system/user/assistant), `<|reserved_3/4|>`=12/13
(**unused — do NOT use for tools**).

Consequences:
- The wire string **is** the atomic token name (no "semantic alias" — `<|tool_call|>`
  literally encodes to id 7), so all examples below name the real string.
- Still **zero tokenizer impact / no retrain** — these tokens already exist;
  `tokenization_id` stays `7a570a7ba9fc7985`.
- Phase-0 gate (fail loud) still applies as a regression guard: `resolve_special_ids`
  asserts `token_to_id("<|tool_call|>")==7`, `…("<|tool_result|>")==8`, both atomic
  — refuse to write on any mismatch (a stale tokenizer copy at `trainingdata/`,
  fingerprint `0605002482410ace`, has only 2 specials — never use it).

---

## 2. Wire format (identical at generation, training, inference)

**Canonical serializer (single source of truth — DATAGEN §1d).** Every place that
turns a `tool_call` dict into text uses the *exact same* call, or the tokenization
won't be byte-stable across generator / tokenizer / template / chatbot parser:

```python
def dump_tool_call(tc):  # tc = {"tool": ..., "code"|"query": ...}
    return json.dumps(tc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

`sort_keys=True` fixes key order; `separators=(",",":")` removes the spaces that
plain `json.dumps(sort_keys=True)` would emit. Wire example (note keys are sorted →
`code` before `tool`, no spaces):

```
<|user|>Q<|im_end|>
<|assistant|>reasoning…<|tool_call|>{"code":"print(2847*391)","tool":"python"}<|im_end|>
<|tool_result|>1113177<|im_end|>
<|assistant|>So the product is 1113177. The answer is: 1113177<|im_end|><|endoftext|>
```

- `python` call dict: `{"tool":"python","code":"..."}` → result = captured stdout
  (or last-expression repr), **truncated to PY_RESULT_TOK tokens**.
- `search` call dict: `{"tool":"search","query":"..."}` → result = top-k Brave
  snippets joined, **truncated to SEARCH_RESULT_TOK tokens**.
- Truncation appends ` …[truncated]`. Payloads stay one line.

The `<|im_end|>` after a `<|tool_call|>` span is the only generation stop — the
inference harness inspects the completed assistant segment, and if it contains a
`<|tool_call|>` (id 7), executes and re-injects (see §6). So **the stop-token set
is unchanged** (`<|im_end|>`, `<|endoftext|>`).

---

## 3. `tokenize_sft_data.py` changes (small, surgical)

**The output schema has only three roles: `user`, `assistant`, `tool`.**
`tool_call` is **not** a role — it's an optional **dict field on an assistant
message** (see DATAGEN §6 for why: a separate role would force an `<|im_end|>`
between the reasoning and the call, and `<|im_end|>` is a stop token). So only one
new role token is mapped:

1. Extend the maps — map the **`tool`** role to the dedicated `<|tool_result|>`
   (id 8); the inline `<|tool_call|>` (id 7) is resolved separately, not as a role:
   ```python
   ROLE_TOKEN["tool"] = "<|tool_result|>"       # role string is "tool" → id 8
   EXPECTED_ROLE_ID["tool"] = 8
   # NO ROLE_TOKEN["tool_call"] — it's a dict field, not a role.
   # <|tool_call|> (id 7) is emitted inline by the assistant-encoding step below,
   # never via role_id[role], so it needs no ROLE_TOKEN entry.
   ```
   **Phase-0 fail-loud check lives in `resolve_special_ids` (the resolver that
   already asserts ids 9/10/11 via `EXPECTED_ROLE_ID`).** Extend it to also resolve
   **and assert** `token_to_id("<|tool_call|>") == 7` and
   `token_to_id("<|tool_result|>") == 8` (both atomic), and **return the
   `tool_call_id` (7) alongside `im_end_id`/`eot_id`** so `encode_example` can emit
   it — e.g. change the signature to
   `return role_id, im_end_id, eot_id, tool_call_id`. Without this, `role_id[role]`
   would `KeyError` on a `tool` message at encode time instead of failing loudly at
   startup. The resolver runs once per run before `process(...)`, so it gates the
   whole tokenization (this is DATAGEN §0's Phase-0 gate for the tokenizer path).
2. Learn rule stays `learn = role == "assistant"`. The change is in **how an
   assistant message is encoded** — insert the inline tool-call span before the
   closing `<|im_end|>` (the exact contract pinned in **DATAGEN §6**), using the
   canonical serializer `dump_tool_call` (§2), not bare `json.dumps`:
   ```python
   # assistant: <|assistant|>(mask) + content(LEARN)
   #   + if "tool_call" in m:  <|tool_call|> id7 (LEARN)
   #                           + dump_tool_call(m["tool_call"]) (LEARN)
   #   + <|im_end|> (LEARN)
   # tool:      <|tool_result|> id8 (mask) + content(mask) + <|im_end|> (mask)
   ```
   `dump_tool_call` (canonical serializer) so train/inference serialize
   byte-identically.
3. **`resp_tokens` (short-response filter scalar — NOT the loss-label count):**
   sum the **learned *content*** tokens across the trace — every assistant
   `content` **plus** every serialized `tool_call` string — and **exclude the
   learned stop markers** (`<|im_end|>`, `<|tool_call|>`, final `<|endoftext|>`),
   which are also labels but aren't "response content". `tool`/`tool_result`
   content is masked → excluded. See DATAGEN §5. Drop if `< 3`.
4. Everything else (block-size drop, manifest, staleness guard) is unchanged.
5. **Round-trip check:** decode 3–5 tool traces, confirm the loss mask covers
   *only* assistant content + inline tool-call spans (and their `<|im_end|>`),
   never `tool` (result) turns. This catches the #1 silent template bug.

---

## 4. Chat template (`sparky/sparky_chat_template.py`)

Single source of truth shared by chatbot + eval — must match §2 exactly.

```python
TOOL_CALL_MARKER = "<|tool_call|>"     # id 7 — emitted INLINE in an assistant turn
                                       #        when the msg has a `tool_call` dict field
ROLE_TOOL        = "<|tool_result|>"   # id 8 — role marker for messages with role == "tool"
```

**Naming caution:** the id-8 token marks messages whose **role string is `"tool"`**
(not `"tool_result"`); the constant is named `ROLE_TOOL` to match the role string,
and `TOOL_CALL_MARKER` is deliberately *not* `ROLE_*` because `<|tool_call|>` is an
inline marker, never a role.

**`build_sft_prompt` must validate roles explicitly** (today it falls back to
"assistant" for any non-`user` role, which would render a `tool` message as
`<|assistant|>result<|im_end|>` — wrong). Required logic:
- `role == "user"` → `<|user|>` + content + `<|im_end|>`
- `role == "assistant"` → `<|assistant|>` + content, then **if it has a `tool_call`
  dict**: `TOOL_CALL_MARKER` + `dump_tool_call(tool_call)` (canonical serializer,
  identical to the tokenizer §3), then `<|im_end|>`
- `role == "tool"` → `ROLE_TOOL` + content + `<|im_end|>`
- **any other role → raise** (fail loud, no silent assistant fallback)

`sft_stop_token_ids` unchanged (`<|im_end|>` + EOT). Add a short `TOOL_SYSTEM_PROMPT`
constant (the protocol + the two tools) — reused verbatim by the generator (§5) and
the chatbot (§6) so train/inference distributions match.

---

## 5. Generator — `sft/generate_tool_use.py`

Clone `generate_reasoning_distill.py` (parallel, append-only resumable JSONL,
`progress.log` + `meta_raw.json`, `--limit` calibration, `--budget-usd`,
provider auto-detect) and add the agentic loop.

**Per-problem loop (replaces `solve_one`) — see DATAGEN §2–§3 for the precise
pseudocode.** The teacher uses **OpenAI native function-calling, not text
parsing**; we transcode its structured tool calls into wire format afterward:
1. Send `TOOL_SYSTEM_PROMPT` + problem with `tools=TOOLS`, `tool_choice="auto"`,
   `parallel_tool_calls=False`, `max_tokens=768`.
2. Read `resp.choices[0].message.tool_calls` (structured — **no** `<|tool_call|>`
   string parsing). For the (≤1) call:
   - `python` → execute in the sandbox (§7), capture+truncate output.
   - `search` → rate-limited Brave (§7), join+truncate top-k snippets.
3. Append the OpenAI assistant-with-tool_calls + a `role:"tool"` result to the
   working transcript; **also** append to the Synapse trace as an `assistant`
   message with a `tool_call` dict field + a `role:"tool"` message (DATAGEN §6).
4. Repeat until the teacher replies with no `tool_calls`, or `--max-calls`
   (default 4) is hit.
5. **Rejection sample:** keep the trace iff (a) `answers_match(extract_final_answer(
   final_content), gold)`, (b) it used ≥1 tool (else it's plain CoT →
   `reasoning_distill`'s job), (c) the transcoded trace fits the budget (DATAGEN §7),
   **and (d) no tool result contains a transient-failure sentinel** (e.g.
   `[search unavailable]`). Without (d) a trace can verify via math reasoning
   *despite* a failed search and then train the model on a broken tool call —
   reject it, or divert to a diagnostic file excluded from SFT (DATAGEN §4).
6. Low-temp first, higher-temp retries on failure (same as today).

**Inherited semantics to preserve (verified in `generate_reasoning_distill.py`):**
`--limit` is applied **after** filtering already-done ids, so it means "first N
*remaining* problems", not the first N of the original pool — a resumed run with a
limit processes different problems than the first run. `--budget-usd` trips a stop
flag that prevents *queued* work but lets *in-flight* calls finish, so actual spend
can slightly overshoot the cap. Document both in `generate_tool_use.py` so they
aren't mistaken for bugs.

**Problem pool / driving each tool:**
- `python`: reuse the math pools (`orca`, `numina`) but the system prompt
  instructs "use `python` for any non-trivial arithmetic" → natural calculator
  traces, deterministic + self-verifying.
- `search` + reasoning-trigger: target a **harder math subset** (e.g. NuminaMath
  `math`/competition rows) where method recall genuinely helps and the final
  answer is still verifiable. Verification stays answer-based, so search traces
  are held to the same bar.

**Output:** `$SYNAPSE_DIR/datasets_sft/tool_use/tool_use_raw.jsonl`, schema
`{id, system, messages:[...]}` with roles **`user` / `assistant` / `tool`** —
where an `assistant` message carries an optional `tool_call` **dict field** (not a
role). Exact record shape in DATAGEN §6. Same `push_to_drive.py` flow as the other
sources.

**De-risk order:** ship `python`-only first (deterministic, no network in the
loop), calibrate yield + cost on `--limit`, then enable `search`.

---

## 6. Inference harness (`sparky/sparky_chatbot.py`)

Generation stops being single-shot. New loop:
1. Build prompt ending on `<|assistant|>`; generate to `<|im_end|>`.
2. If the segment contains `<|tool_call|>`: parse JSON, execute (§7), append
   `<|tool_result|>result<|im_end|><|assistant|>`, regenerate.
3. Else: done — strip markers, return the answer.
4. Cap iterations (~5); on cap, force a no-tool finalize turn.

Requires the **same python sandbox + Brave client live at inference** — the model
will emit calls it expects to be answered. Reflect this in `sparky_eval.py` too
so eval exercises the real loop.

---

## 7. Tool implementations (shared lib, used by generator + chatbot)

Put both in one small module (e.g. `sft/tools_runtime.py`) so generation and
inference run *identical* tools.

- **`python`:** subprocess `python -c`, **timeout (~5s), no network, memory cap,
  temp cwd**, capture stdout/stderr, truncate. sympy available (already a dep).
  Never `exec` in-process. Errors are returned as the tool result (the model
  should learn to recover), not raised.
- **`search`:** `GET https://api.search.brave.com/res/v1/web/search?q=...` with
  header `X-Subscription-Token: $BRAVE_API_KEY`; take top-k (~3) title+snippet,
  join, truncate to `SEARCH_RESULT_TOK` (128 — small on purpose, DATAGEN §1b/§7).
  Cache by query during a run.
  **Rate limiting is mandatory (full design in DATAGEN §1c), and the order matters:**
  1. A **process-wide limiter with a single lock-guarded `next_allowed_time`** (or
     equivalent token bucket) gates **all** Brave calls — Brave's free tier is
     ~1 req/s, so 48 workers would 429 every call after the first. Each request
     (including retries) must **reserve a slot before the HTTP call**; a bare
     `Semaphore` bounds *concurrency*, not *rate*, so it alone is insufficient.
     Rate set by the **`--search-rate` CLI knob (default 1.0 req/s)**; raise for a
     paid key. Same limiter object reused at inference.
  2. On **429/5xx → exponential backoff and retry** (transient, not a hard error).
  3. **Only after retries are exhausted** return `"[search unavailable]"`. Such a
     trace still *completes*, but is then **rejected** by §5(d) so the model never
     trains on a failed tool call (don't rely on the answer happening to be wrong).

---

## 8. Training mix (`sft/sft.py`)

Add the new source (and `reasoning_distill`, now that it's generated) and carve
share from the general/reasoning backbone. Starting point — tune after a val run:

```python
SFT_DATA_MIX = {
    "tulu3":             0.24,
    "metamath":          0.20,
    "reasoning_distill": 0.14,   # newly generated CoT
    "tool_use":          0.12,   # NEW
    "dolly":             0.12,
    "alpaca_gpt4":       0.06,
    "samsum":            0.05,
    "opencode":          0.04,
    "oasst1":            0.03,
}
```

Weighted sampling means the collected pool size doesn't matter — these are
per-batch proportions. Watch the `tool_use` and `reasoning_distill` per-source val
loss separately (the harness already evals per source).

**Order this AFTER tokenization, not before.** `sft.py` hard-fails
(`raise SystemExit("[{name}] has 0 train examples …")`) if any source in
`SFT_DATA_MIX` lacks a `train.jsonl`. So adding `reasoning_distill` / `tool_use`
to the mix **before** both have been tokenized + consolidated will break trainer
startup. Generate → `tokenize_sft_data.py --datasets tool_use` →
`consolidate_sft_data.py` **first**, then edit the mix.

---

## 9. Phasing

| Phase | Work | Gate |
|---|---|---|
| 0 | ✅ **Done** — verified `<\|tool_call\|>`=7, `<\|tool_result\|>`=8 atomic in the canonical tokenizer (fp `7a570a7ba9fc7985`). Remaining: bake the assertion into `resolve_special_ids` as a regression guard | resolver asserts 7/8 |
| 1 | Template plumbing: `tokenize_sft_data.py` + `sparky_chat_template.py` | round-trip mask check on a hand-written tool trace |
| 2 | `tools_runtime.py` (python sandbox + Brave) | unit-exercise both tools standalone |
| 3 | `generate_tool_use.py` — **python only**, `--limit` calibration | yield% + cost projection sane |
| 4 | Enable `search`; full run; `push_to_drive.py` | traces verify, fit 2048. **Note:** the search run is Brave-rate-limited (`--search-rate`, ~1 req/s free tier), so its wall-clock is dominated by search and is **much slower** than the local-only `python` run — size/parallelize accordingly (DATAGEN §1c). |
| 5 | `tokenize_sft_data.py --datasets tool_use` + add to `SFT_DATA_MIX`; train | per-source val loss drops |
| 6 | Tool loop in `sparky_chatbot.py` + `sparky_eval.py` | chatbot completes a real python + search turn |

---

## 10. Open questions / risks

- **Yield unknown.** Agentic traces fail more ways than plain CoT (bad tool JSON,
  tool error, overflow). Calibrate cost on `--limit` before the full run.
- **2048 overflow** is the main silent killer — multi-call search traces are the
  worst offenders. Truncate aggressively; consider dropping any trace whose
  *result* turns dominate the budget.
- **Search non-determinism:** Brave results drift over time, so traces aren't
  reproducible. Acceptable (we keep the captured snippet in the trace), but means
  re-running the generator yields different (still verified) data.
- **Inference dependency:** the chatbot now needs network (Brave) + a sandbox at
  runtime. Degrade gracefully when search is unavailable.
