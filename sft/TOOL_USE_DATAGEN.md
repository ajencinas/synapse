# Tool-Use Trace Generation — Detailed Design

How `sft/generate_tool_use.py` produces verified agentic tool-use traces for SFT.
Companion to `TOOL_USE_PLAN.md` (the overall plan); this doc is **data generation
only**.

**Core idea:** we do **not** make the teacher emit our literal wire format
(`<|tool_call|>…`) — that's fragile. The teacher uses the provider's **native
function-calling**, the tools **actually execute**, and we **transcode** the
resulting conversation into Synapse format afterward. This decouples teacher
quirks from our on-disk format.

**Method decisions (locked):** tools = `python` + `search`; search backend =
**Brave** (`BRAVE_API_KEY`, verified live HTTP 200); generation = **agentic
distillation** with rejection sampling on the verified final answer.

---

## 0. Tokens — VERIFIED (Phase 0 done): use the dedicated tool tokens

The canonical tokenizer (`tokenizer.json`, fingerprint **`7a570a7ba9fc7985`** — the
exact `tokenization_id` the pipeline enforces) was **purpose-built with named tool
tokens.** Verified directly from the file:

- **`<|tool_call|>` → id 7**, **`<|tool_result|>` → id 8**, both atomic (length-1).
- Surrounding layout: `<|endoftext|>`=0 … `<|im_end|>`=3 … `<|fim_*|>`=4/5/6,
  **tool tokens 7/8**, then `<|reserved_0/1/2|>`=9/10/11 (system/user/assistant),
  `<|reserved_3/4|>`=12/13 **unused**. 256 named+reserved specials total.

**So the original `reserved_3/4 → 12/13` plan was wrong** — emit the dedicated
`<|tool_call|>`(7) / `<|tool_result|>`(8). The wire string equals the atomic token
name (no semantic alias). A stale copy at `trainingdata/tokenizer_out`
(fp `0605002482410ace`, only 2 specials) exists — never use it.

Phase-0 gate now serves as a **regression guard** baked into `resolve_special_ids`
(fail loud, mirrors the existing 9/10/11 check): assert
`token_to_id("<|tool_call|>") == 7`, `…("<|tool_result|>") == 8`, both length-1.
Anything else → stop.

---

## 1. Shared utilities this doc must define (not hand-wave)

These are used identically at generation **and** inference, so they live in
`sft/tools_runtime.py` and are pinned here, not left as "~200 tokens".

### 1a. `synapse_token_len(text) -> int`
The model is **digit-per-token**, so char/word estimates are wrong (a 5-digit
number = 5 tokens). All length budgeting uses the **real Synapse tokenizer**
(`tokenizer_out/tokenizer.json`, `add_special_tokens=False`). The generator loads
it once at startup. If it's genuinely unavailable in the run environment, **fail
loud** — do not silently fall back to char counts (that reintroduces the
mismatch). The tokenizer is small and already on Drive next to the data.

### 1b. `truncate_tokens(text, max_tok) -> str`
Encode with the Synapse tokenizer, keep the first `max_tok` ids, decode, append
`" …[truncated]"`. Budgets:
- `python` result → **`PY_RESULT_TOK = 200`**
- `search` result → **`SEARCH_RESULT_TOK = 128`** (deliberately small: search results
  are the main 2048-overflow risk, and the model needs the *gist* of the method, not
  full snippets — raise only if calibration §7 shows headroom).

Same function, same budget, at gen and inference — this is what makes "identical
truncation" real instead of aspirational.

### 1c. Brave rate limiting — mandatory, not optional
The generator clones `generate_reasoning_distill.py`'s **48 ThreadPoolExecutor
workers**, but Brave's free tier is **~1 request/second**. 48 concurrent searches
would 429 every call after the first; the `[search unavailable]` fallback would
then fail rejection sampling on essentially **every** search trace — `--mode
search` would yield ~0%. So a **process-wide rate limiter gates all Brave calls**:

- **Precise contract — a single process-wide lock-guarded `next_allowed_time`** (or
  equivalent token bucket), NOT a bare semaphore. A `Semaphore` caps *concurrency*,
  not *rate*; per-worker `time.sleep` doesn't bound the *aggregate* rate. Every
  Brave request — **including each retry** — acquires the lock, waits until
  `now >= next_allowed_time`, sets `next_allowed_time = max(now, next_allowed_time)
  + 1/search_rate`, releases, *then* issues the HTTP call. Default
  **`--search-rate 1.0` req/s** (CLI knob; raise for a paid key).
- **429 / 5xx → exponential backoff and retry** (up to 4 tries, like the existing
  API-error loop), **distinct** from the `[search unavailable]` fallback. Return
  `[search unavailable]` only after retries are exhausted.
- **A trace whose result contains `[search unavailable]` is rejected** (§4), so a
  transient failure can't slip into SFT even if the final answer happens to verify.
- `python` calls are unaffected (local, no network), so this throttles only the
  `search`/`mixed` modes. At 1 req/s, search-mode wall-clock is Brave-bound, not
  teacher-bound — size `--limit` calibration accordingly.

The same limiter object is reused at inference (a lone chatbot rarely trips it, but
the code path is identical to generation).

### 1d. `dump_tool_call(tc) -> str` — the canonical serializer (byte-stable)
One serializer, used **everywhere** a `tool_call` dict becomes text — generator
trace-writer, `tokenize_sft_data.encode_example`, `build_sft_prompt`, and the
chatbot's tool-call parser tests. If any serialize differently, the tokenization
isn't byte-identical and train ≠ inference:
```python
def dump_tool_call(tc):
    return json.dumps(tc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```
- `sort_keys=True` — fixes key order (`{"code":...,"tool":...}`, not call-site order).
- `separators=(",",":")` — **drops the spaces** plain `json.dumps(sort_keys=True)`
  emits (`", "` / `": "`), which would differ from compact callers.
- `ensure_ascii=False` — keep unicode literal (digit-per-token already taxes the
  budget; `\uXXXX` escapes inflate it).

All pseudocode below that shows `json.dumps(tc)` means **`dump_tool_call(tc)`**.

---

## 2. Teacher protocol: native function-calling

DeepSeek V4 Flash speaks the OpenAI function-calling API. Two tool schemas; the
API handles the protocol:

```python
TOOLS = [
  {"type":"function","function":{
    "name":"python",
    "description":"Run Python (sympy available) for any non-trivial arithmetic/algebra. Returns stdout.",
    "parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}},
  {"type":"function","function":{
    "name":"search",
    "description":"Web search. Use to recall the general METHOD for a hard problem before solving.",
    "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
]
```

Call with `tool_choice="auto"`, `parallel_tool_calls=False` (≤1 call/turn →
simple wire format), and **`max_tokens=768` per turn** (a single assistant turn
must not eat the whole 2048 budget — the missing guard the old `solve_one` had via
`max_tokens=1024`).

**System prompt ends with the answer convention** so the verifier can parse it:
the `TOOL_SYSTEM_PROMPT` instructs the teacher to finish with a line
`The answer is: <answer>` — identical to `generate_reasoning_distill.py`'s
`SYSTEM_PROMPT`. Without this, the final answer may be embedded after a tool call
in a form `extract_final_answer` can't read (critique #3).

---

## 3. The agentic loop (replaces `solve_one`)

Uses the **real** verifier functions from `generate_reasoning_distill.py`
(`extract_final_answer`, `answers_match`) — no fictional `verify()` (critique #3).

```python
msgs = [system, {role:user, content:problem}]      # OpenAI working transcript
trace_msgs = [{role:"user", content:problem}]       # Synapse trace being built
used_tool = 0
learned_tok = 0                                     # running count of LEARNED tokens (§5)

wire_tok = synapse_token_len_of(system) + MARKERS_OVERHEAD   # running total incl. markers

for step in range(MAX_CALLS + 1):                   # MAX_CALLS: python=3, search/mixed=2
    resp = client.chat.completions.create(
        model=model, messages=msgs, tools=TOOLS,
        tool_choice="auto", parallel_tool_calls=False,
        temperature=0.2, max_tokens=768)            # higher-temp retries on failure, as today
    m = resp.choices[0].message

    if not m.tool_calls:                            # final-answer turn
        if synapse_token_len(m.content) > FINAL_ANSWER_TOK:   # = 768; the MAX_SOLUTION_CHARS analogue
            return None
        wire_tok += synapse_token_len(m.content) + 2          # + <|im_end|> + <|endoftext|>
        if wire_tok > TRACE_MAX_TOK:                          # final overflow check, incl. EOT
            return None
        trace_msgs.append({role:"assistant", content:m.content})
        learned_tok += synapse_token_len(m.content)
        ok = answers_match(extract_final_answer(m.content), gold)
        return (trace_msgs, used_tool, learned_tok, ok)

    call = m.tool_calls[0]
    try:
        args = json.loads(call.function.arguments)  # bad JSON -> reject this trace
    except Exception:
        return None
    if call.function.name == "python":
        result = truncate_tokens(run_python(args["code"]), PY_RESULT_TOK)
    else:
        result = truncate_tokens(run_search(args["query"]), SEARCH_RESULT_TOK)
        if "[search unavailable]" in result:        # §4(d): never train on a failed tool
            return None

    tc = {"tool":call.function.name, **args}
    # incremental budget check AFTER each pair — bail before paying for more calls
    wire_tok += (synapse_token_len(m.content or "") + synapse_token_len(dump_tool_call(tc))
                 + synapse_token_len(result) + 3)   # +3 ≈ <|tool_call|>,<|im_end|>,<|tool_result|>,<|im_end|>
    if wire_tok > TRACE_MAX_TOK:
        return None
    msgs.append(m); msgs.append({role:"tool", tool_call_id:call.id, content:result})
    trace_msgs.append({role:"assistant", content:m.content or "", tool_call:tc})
    trace_msgs.append({role:"tool", content:result})
    learned_tok += synapse_token_len(m.content or "") + synapse_token_len(dump_tool_call(tc))
    used_tool += 1

return None                                         # ran out of calls, no final answer -> reject
```

Three length guards now exist: per-turn `max_tokens=768` (API side), an
**incremental `wire_tok > TRACE_MAX_TOK` check after every assistant/tool pair**
(bails before paying for further calls), and the final-answer `FINAL_ANSWER_TOK=768`
reject. The final check **includes the trailing `<|endoftext|>`** (critiques #4,
#5, #8, B5).

---

## 4. Rejection sampling (the quality bar)

Keep a trace **only if all four**:
1. `answers_match(extract_final_answer(final_content), gold)` is true,
2. `used_tool >= 1` (else it's plain CoT → belongs to `reasoning_distill`),
3. the **whole transcoded trace** fits the budget (§7),
4. **no tool result contains a transient-failure sentinel** (`[search unavailable]`).
   Without this, a trace can verify via math reasoning *despite* a failed search and
   then teach the model on a broken tool call. The loop above rejects these inline;
   alternatively divert them to a `tool_use_diagnostic.jsonl` (kept out of SFT) for
   debugging search-failure rates rather than discarding silently.

Same low-temp-first, higher-temp-retry-on-failure as `generate_reasoning_distill.py`.

---

## 5. `resp_tokens` / short-response filter — exact rule (critique #5)

`tokenize_sft_data.py`'s `short_response` filter keys on a per-example scalar.
**This scalar is a content-size heuristic, NOT the count of learned loss labels**
(the labels also include the learned stop markers `<|im_end|>`, `<|tool_call|>`,
final `<|endoftext|>` — those are excluded here by design). Define it as the **sum
of learned *content* tokens across the whole trace**: every assistant `content`
**plus** every serialized `tool_call` string (via `dump_tool_call`, §1d).
`tool`/`tool_result` content is masked → excluded. The generator tracks this as
`learned_tok` and stores it; the tokenizer recomputes the same sum with the same
serializer. Drop if `learned_tok < 3` — same threshold as today, summed over turns
instead of one response.

---

## 6. Output schema + the interface contracts it imposes

### Record (one line of `datasets_sft/tool_use/tool_use_raw.jsonl`)
```json
{"id":"py_3af1…","mode":"python","system":"<TOOL_SYSTEM_PROMPT>",
 "messages":[
   {"role":"user","content":"…"},
   {"role":"assistant","content":"Let me compute the product.",
    "tool_call":{"tool":"python","code":"print(2847*391)"}},
   {"role":"tool","content":"1113177"},
   {"role":"assistant","content":"So the product is 1113177. The answer is: 1113177"}
]}
```

### `system` field decision (critique #6)
We **write the canonical `TOOL_SYSTEM_PROMPT` into every record's `system`**, and
the chatbot/eval use the *identical* constant at inference. Rationale: the model
**must** condition on the tool list to use tools, so it has to be in the encoded
sequence; storing it per-record keeps `tokenize_sft_data.py` unchanged (it already
encodes `system`) and guarantees train==inference. Cost: those (masked) tokens
count against the 2048 budget per example, so **`TOOL_SYSTEM_PROMPT` is kept
short (~80 tokens)** and lives as one constant in `sparky_chat_template.py`,
imported by both the generator and the chatbot. (Contrast: other sources store
`system:""`.)

### `encode_example` contract (critique #11 — specify, don't switch roles)
The `tool_call` **dict field on an assistant message** is the deliberate choice
(a separate `tool_call` *role* would force an `<|im_end|>` between reasoning and
the call — see §6 "subtlety" below — breaking inference). Phase-1
`tokenize_sft_data.py` must therefore read three things, not just `role`/`content`:
- `{role:"assistant"}` → `<|assistant|>`(mask) + content(**learn**) +
  **if `"tool_call"` in msg** → `<|tool_call|>` **id 7**(**learn**) +
  `dump_tool_call(msg["tool_call"])`(**learn**) → then **one** `<|im_end|>`(**learn**).
- `{role:"tool"}` → `<|tool_result|>` **id 8**(mask) + content(mask) + `<|im_end|>`(mask).
- `{role:"user"}` unchanged. **Any other role → raise** (fail loud; today's encoder
  would `KeyError`, and the resolver must already have mapped only user/assistant/
  system/tool — §3).

Use the canonical `dump_tool_call` (§1d) so train and inference serialize the dict
byte-for-byte identically.

### `build_sft_prompt` contract (critique #7)
Even though template code is Phase 1, the interface is fixed **here** because this
schema is its input: at inference `build_sft_prompt` must **validate roles
explicitly** (only `user`/`assistant`/`tool`; raise on anything else — today it
silently treats every non-`user` role as assistant, which would mis-render a
`tool` message) and render an assistant message's optional `tool_call` dict with
the *same* `<|tool_call|>` + `dump_tool_call(...)` rule, injected results as
`<|tool_result|>…<|im_end|>`. Round-trip: encode a generated trace, decode, and
confirm it reproduces the wire string below.

### The one subtlety (why a dict field, not a role)
Reasoning and the call it leads to must share **one** trailing `<|im_end|>`:
```
<|assistant|>reasoning…<|tool_call|>{"code":"…","tool":"python"}<|im_end|><|tool_result|>1113177<|im_end|><|assistant|>… The answer is: …<|im_end|><|endoftext|>
```
A separate role would insert `<|im_end|>` after the reasoning; since `<|im_end|>`
is a stop token, the model would stop before ever emitting the call.

---

## 7. Length budgeting & overflow — measured, not guessed (critiques #2, #9)

There is **no hardcoded 1800**. Instead:

1. The generator computes each kept trace's **exact** total length with
   `synapse_token_len` over the fully transcoded wire sequence (system + all
   turns + markers).
2. It **drops** any trace whose total `> TRACE_MAX_TOK` and counts the drop.
   `TRACE_MAX_TOK` defaults to `block_size - 64 = 1984` (margin for the trailing
   `<|endoftext|>` + any inference-time system delta), overridable by flag.
3. **Calibration reports the real distribution, broken down by mode and call
   count.** `meta_raw.json` records `total_len` and `learned_tok` percentiles
   (p50/p95/p99/max, reuse `pcts()`) **and an `overflow` drop count split by
   `mode` and by number of tool calls** — because overflow concentrates in
   multi-call `search` traces, an aggregate number hides it. If `search` p95
   approaches `TRACE_MAX_TOK`, lower `SEARCH_RESULT_TOK` further or drop
   `--max-calls` to 2 **before** the full (paid) run. The margin is chosen from
   this data, not asserted up front.
4. **Defaults are pre-tightened for search** (so the first calibration isn't pure
   overflow): `SEARCH_RESULT_TOK=128`, `FINAL_ANSWER_TOK=768`, and `--max-calls`
   defaults to 3 but **2 for `search`/`mixed`**. The incremental check in §3 bails
   mid-loop, so an overflowing search trace doesn't burn all 2–3 paid calls before
   being dropped.

---

## 8. Two modes drive the tool mix (and the reasoning trigger)

`--mode` swaps `TOOL_SYSTEM_PROMPT` so we **control** the behavior collected:

| Mode | System-prompt gist | Pool | Produces |
|---|---|---|---|
| `python` | "Use `python` for any non-trivial arithmetic/algebra." | `orca` + `numina` | calculator traces — deterministic, high yield |
| `search` | "Hard problem: first `search` the general method, state a short plan, then solve." | harder `numina` rows | **the reasoning trigger** |
| `mixed` | both tools available, model decides | mixed | natural blend |

All modes still finish with `The answer is: X` and are verified against gold.
Composite `id = f"{mode}_{qid}"` so one problem can appear in two modes without a
resume collision. All append to one `tool_use_raw.jsonl` with a `mode` tag.

---

## 9. Inherited from `generate_reasoning_distill.py` (cloned)

Parallel `ThreadPoolExecutor`; append-only `tool_use_raw.jsonl` + `progress.log`
(durable resume; **the source generator's docstring says `checkpoint.json` but the
code uses `progress.log`** — the latter is correct, don't copy the stale docstring);
`meta_raw.json`; provider auto-detect. New knobs: `--mode {python,search,mixed}`,
`--max-calls` (3; **2 for search/mixed**), `--trace-max-tok` (1984), `--search-rate`
(1.0 req/s — Brave limiter, §1c).

**Inherited semantics — replicate exactly, document so they read as intentional
(critique #10 / SF8):**
- `--limit N` is applied **after** filtering already-done ids, so it means "first N
  *remaining*", not the first N of the original pool — a resumed run with `--limit`
  processes *different* problems than the first run. It processes those N and prints
  a yield/cost **projection at the end**; it does **not** stop early mid-run.
- `--budget-usd` trips a stop flag that blocks *queued* work but lets *in-flight*
  calls finish, so actual spend can overshoot the cap slightly.

---

## 10. Files & order

1. `sft/tools_runtime.py` — `run_python`, `run_search` (rate-limited, §1c),
   `synapse_token_len`, `truncate_tokens`, the shared Brave limiter (shared with
   inference).
2. **Phase-0 regression assert** (Phase 0 itself is done — §0): bake
   `token_to_id("<|tool_call|>")==7` / `…("<|tool_result|>")==8` (atomic) into
   `resolve_special_ids` so a future tokenizer swap fails loud.
3. `sft/generate_tool_use.py` — cloned from `generate_reasoning_distill.py` + the
   agentic loop.
4. Calibration: `SYNAPSE_DIR=… python sft/generate_tool_use.py --mode python --limit 1500`
   → read `total_len`/`learned_tok` percentiles + yield/cost before any full run.
5. Full runs per mode → `push_to_drive.py`.

Tokenization (the `encode_example` change in §6), `SFT_DATA_MIX`, and the
inference loop are tracked in `TOOL_USE_PLAN.md` Phases 1, 5, 6 — but their
**interfaces are pinned here** so the generator isn't writing into a vacuum.
