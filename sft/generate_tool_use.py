#!/usr/bin/env python3
"""Agentic tool-use trace generator for SynapseGPT SFT.

The teacher (DeepSeek V4 Flash) solves math problems with two REAL tools exposed via
native function-calling — `python` (sft/tools_runtime.run_python) and `search`
(Brave). The generator executes each tool call for real, feeds the result back, and
keeps a trace ONLY if (a) the final answer verifies against gold, (b) >=1 tool was
used, (c) the transcoded trace fits the 2048 budget, and (d) no result contains the
`[search unavailable]` sentinel. Verified traces are transcoded into the Synapse
wire schema and appended to datasets_sft/tool_use/tool_use_raw.jsonl.

Design refs: sft/TOOL_USE_PLAN.md, sft/TOOL_USE_DATAGEN.md. Reuses the pool/verify/
provider plumbing of generate_reasoning_distill.py and the tools of tools_runtime.py.

Modes (`--mode`):
  python  — teacher steered to compute with `python` (only the python tool exposed;
            zero Brave spend). Deterministic, high yield.
  search  — teacher steered to first `search` the METHOD, then solve (search+python
            exposed). This is the "reasoning trigger".
  mixed   — both tools, model decides.

System prompt: a single CANONICAL prompt is stored in every record and used at
inference (train==inference). A transient per-mode STEERING line is sent to the
teacher to induce the behavior but is NOT stored — standard distillation (we induce
in the teacher, the student learns it from the canonical prompt).

Usage:
  SYNAPSE_DIR=/path python sft/generate_tool_use.py --mode python --limit 1500   # calibrate
  SYNAPSE_DIR=/path python sft/generate_tool_use.py --mode python                # full
Env: DEEPSEEK_API_KEY or OPENROUTER_API_KEY; BRAVE_API_KEY (search/mixed modes).
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_reasoning_distill as grd          # pool, verify, providers, pricing
import tools_runtime as tr

# ---------------------------------------------------------------------------
# Prompts + tool schemas
# ---------------------------------------------------------------------------
CANONICAL_TOOL_SYSTEM = tr.CANONICAL_TOOL_SYSTEM   # single source: tools_runtime.py
MODE_STEER = {
    "python": "Prefer the `python` tool for all arithmetic and algebra; never compute "
              "multi-digit results by hand.",
    "search": "First use `search` to recall the general method for this type of problem, "
              "briefly state your plan, then solve (use `python` for any computation).",
    "mixed": "",
}

PY_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python (sympy available) for any non-trivial arithmetic/algebra. Returns stdout.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search",
    "description": "Web search. Use to recall the general METHOD for a hard problem before solving.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
TOOLS_FOR_MODE = {"python": [PY_TOOL], "search": [SEARCH_TOOL, PY_TOOL],
                  "mixed": [PY_TOOL, SEARCH_TOOL]}

# Budget defaults (DATAGEN §1b/§7). search/mixed cap calls tighter (overflow risk).
DEFAULT_MAX_CALLS = {"python": 3, "search": 2, "mixed": 2}
FINAL_ANSWER_TOK = 768
# Rough per-turn marker overhead (role markers + <|im_end|>); the exact gate is
# tokenize_sft_data.py — this is a conservative pre-filter.
_BASE_OVERHEAD = 5          # system + final <|endoftext|> markers
_PAIR_MARKERS = 4           # <|tool_call|>, <|im_end|>, <|tool_result|>, <|im_end|>
_FINAL_MARKERS = 2          # <|im_end|> + <|endoftext|> on the closing assistant turn


def _system_for(mode):
    steer = MODE_STEER.get(mode, "")
    return (CANONICAL_TOOL_SYSTEM + "\n" + steer) if steer else CANONICAL_TOOL_SYSTEM


def _create(client, model, msgs, tools, temperature, max_tokens=2048, api_retries=4,
            tool_choice="auto"):
    # DeepSeek V4 Flash is a REASONING model: it emits long reasoning_content that
    # counts against max_tokens. 768 was too low — hard problems exhausted it on
    # reasoning and returned EMPTY content (finish_reason=length). 2048 gives the
    # final answer room. (reasoning_content is not stored in the trace.)
    """One teacher turn with transient-error backoff. Returns (resp, ok)."""
    for attempt in range(api_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=msgs, tools=tools, tool_choice=tool_choice,
                parallel_tool_calls=False, temperature=temperature, max_tokens=max_tokens)
            return resp, True
        except Exception:
            if attempt == api_retries - 1:
                return None, False
            time.sleep(2 ** attempt)
    return None, False


def _usage(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0
    return (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)


class SearchBudget:
    """Thread-safe hard cap on Brave requests for a whole run ($50-cap enforcement:
    --budget-usd only caps TEACHER spend). take() -> False once exhausted."""
    def __init__(self, n):
        import threading as _t
        self.left, self._lock = n, _t.Lock()
    def take(self):
        with self._lock:
            if self.left <= 0:
                return False
            self.left -= 1
            return True


def run_agentic(client, model, question, gold, mode, *, max_calls, trace_max_tok,
                limiter=None, brave_key=None, temperature=0.2, force_first_tool=None,
                search_budget=None):
    """Drive one problem through the tool loop. Returns a dict:
    {trace, used_tool, in_tok, out_tok, status}. status: ok|wrong|long|overflow|
    badjson|badtool|searchfail|api|maxcalls. 'trace' is the Synapse-schema message
    list (only meaningful when status=='ok').

    force_first_tool: name of a tool to FORCE on the first turn (then auto). Used for
    fact-lookup: the teacher knows many facts, so we force a `search` call up front to
    induce the look-it-up-then-answer behavior (verified against the known gold)."""
    tools = TOOLS_FOR_MODE[mode]
    sys_text = _system_for(mode)
    if force_first_tool:
        # DeepSeek thinking mode 400s on a forced tool_choice ("Thinking mode does
        # not support this tool_choice"), so "forcing" is now steering text +
        # rejection sampling (solve_problem keeps only traces with used_tool >= 1).
        sys_text += f"\nYou MUST call the `{force_first_tool}` tool before answering."
    msgs = [{"role": "system", "content": sys_text},
            {"role": "user", "content": question}]
    trace = [{"role": "user", "content": question}]
    used = 0
    in_tok = out_tok = 0
    # budget is measured against the STORED record (canonical system, not steering)
    wire_tok = (tr.synapse_token_len(CANONICAL_TOOL_SYSTEM)
                + tr.synapse_token_len(question) + _BASE_OVERHEAD)

    def done(status, final=""):
        return {"trace": trace, "used_tool": used, "in_tok": in_tok,
                "out_tok": out_tok, "status": status, "final": final}

    for step in range(max_calls + 1):
        resp, ok = _create(client, model, msgs, tools, temperature, tool_choice="auto")
        if not ok:
            return done("api")
        it, ot = _usage(resp)
        in_tok += it
        out_tok += ot
        m = resp.choices[0].message

        if not getattr(m, "tool_calls", None):          # final-answer turn
            content = m.content or ""
            if tr.synapse_token_len(content) > FINAL_ANSWER_TOK:
                return done("long")
            wire_tok += tr.synapse_token_len(content) + _FINAL_MARKERS
            if wire_tok > trace_max_tok:
                return done("overflow")
            trace.append({"role": "assistant", "content": content})
            verified = grd.answers_match(grd.extract_final_answer(content), gold)
            return done("ok" if verified else "wrong", final=content)

        if used >= max_calls:          # already used the budget; this turn wanted more
            return done("maxcalls")
        call = m.tool_calls[0]
        name = call.function.name
        try:
            cargs = json.loads(call.function.arguments)
        except (ValueError, TypeError):
            return done("badjson")

        if name == "python":
            raw = tr.run_python(cargs.get("code", ""))
            result = tr.truncate_tokens(raw, tr.PY_RESULT_TOK)
            tc = {"tool": "python", "code": cargs.get("code", "")}
        elif name == "search":
            if search_budget is not None and not search_budget.take():
                return done("searchbudget")   # Brave cap hit — stop this trace, resumable
            raw = tr.run_search(cargs.get("query", ""), limiter=limiter, api_key=brave_key)
            if tr.SEARCH_UNAVAILABLE in raw:            # §4(d): never train on a failed tool
                return done("searchfail")
            result = tr.truncate_tokens(raw, tr.SEARCH_RESULT_TOK)
            tc = {"tool": "search", "query": cargs.get("query", "")}
        else:
            return done("badtool")

        wire_tok += (tr.synapse_token_len(m.content or "") + tr.synapse_token_len(tr.dump_tool_call(tc))
                     + tr.synapse_token_len(result) + _PAIR_MARKERS)
        if wire_tok > trace_max_tok:
            return done("overflow")

        # OpenAI working transcript (only the call we actually answer)
        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
            {"id": call.id, "type": "function",
             "function": {"name": name, "arguments": call.function.arguments}}]})
        msgs.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # Synapse trace
        trace.append({"role": "assistant", "content": m.content or "", "tool_call": tc})
        trace.append({"role": "tool", "content": result})
        used += 1

    return done("maxcalls")


def load_problems(path):
    """Load a custom problem DB (JSONL of {id, question, gold, ...}); skip bad lines."""
    pool = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            q, g = (d.get("question") or "").strip(), str(d.get("gold") or "").strip()
            if not (q and g):
                continue
            pool.append({"id": d.get("id") or f"prob_{abs(hash(q))%10**12}",
                         "question": q, "gold": g})
    return pool


def solve_problem(client, model, problem, mode, args, limiter, brave_key, force_first_tool=None,
                  search_budget=None):
    """Whole-problem attempt with low-temp-first, higher-temp retry. Returns
    (record_or_None, transient_bool, in_tok, out_tok, status). status is the reason
    it wasn't kept ('no_tool' = verified but used no tool), or 'ok' when kept."""
    max_calls = args.max_calls if args.max_calls is not None else DEFAULT_MAX_CALLS[mode]
    in_tok = out_tok = 0
    last_status = "?"
    last_final = ""
    for attempt in range(args.retries + 1):
        temp = 0.2 if attempt == 0 else 0.7
        r = run_agentic(client, model, problem["question"], problem["gold"], mode,
                        max_calls=max_calls, trace_max_tok=args.trace_max_tok,
                        limiter=limiter, brave_key=brave_key, temperature=temp,
                        force_first_tool=force_first_tool, search_budget=search_budget)
        if r["status"] == "searchbudget":
            return None, True, in_tok + r["in_tok"], out_tok + r["out_tok"], "transient", None
        in_tok += r["in_tok"]
        out_tok += r["out_tok"]
        if r["status"] == "ok" and r["used_tool"] >= 1:
            rec = {"id": problem["id"], "mode": mode, "system": CANONICAL_TOOL_SYSTEM,
                   "messages": r["trace"]}
            return rec, False, in_tok, out_tok, "ok", None
        # distinguish "verified but didn't call a tool" from a real failure
        last_status = "no_tool" if r["status"] == "ok" else r["status"]
        last_final = r.get("final", "") or last_final
    # pure API failure before any tokens -> transient (retry next run)
    if last_status == "api" and in_tok == 0 and out_tok == 0:
        return None, True, in_tok, out_tok, "transient", None
    debug = {"id": problem["id"], "status": last_status, "gold": problem["gold"],
             "final": last_final, "extracted": grd.extract_final_answer(last_final)}
    return None, False, in_tok, out_tok, last_status, debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="python", choices=["python", "search", "mixed"])
    ap.add_argument("--sources", default="orca,numina", help="HF pools (ignored if --problems)")
    ap.add_argument("--source-name", default="tool_use",
                    help="output source dir/file stem under datasets_sft/ (v3 search "
                         "runs use tool_search so v2's tool_use file is never touched)")
    ap.add_argument("--problems", default=None,
                    help="JSONL of custom {id,question,gold} (e.g. generate_tool_problems.py output); "
                         "bypasses the HF sources")
    ap.add_argument("--force-search", dest="force_search", action="store_true", default=None,
                    help="force a search call on the first turn (default: on for --mode search)")
    ap.add_argument("--no-force-search", dest="force_search", action="store_false")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-calls", type=int, default=None, help="override per-mode default")
    ap.add_argument("--trace-max-tok", type=int, default=1984)
    ap.add_argument("--debug-rejects", type=int, default=0,
                    help="dump the first N rejected {gold,final,extracted,status} to "
                         "rejects_debug.jsonl (diagnose 'wrong' yield)")
    ap.add_argument("--search-rate", type=float, default=1.0)
    ap.add_argument("--max-searches", type=int, default=None,
                    help="HARD cap on Brave requests this run (in-process enforcement "
                         "of the $50 plan cap); run stops taking new searches when hit "
                         "(resumable via progress.log)")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--provider", default="auto", choices=["auto", "openrouter", "deepseek"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-decontaminate", dest="decontaminate", action="store_false", default=True)
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR") or grd.default_synapse_dir())
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    except ImportError:
        pass

    provider = args.provider
    if provider == "auto":
        if os.environ.get("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        elif os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            raise SystemExit("no API key — set DEEPSEEK_API_KEY or OPENROUTER_API_KEY")
    cfg = grd.PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"{cfg['key_env']} not set for provider={provider}")
    model = args.model or cfg["model"]
    brave_key = os.environ.get("BRAVE_API_KEY")
    if args.mode in ("search", "mixed") and not brave_key:
        raise SystemExit(f"--mode {args.mode} needs BRAVE_API_KEY")
    tr.verify_tool_tokens()        # fail loud before producing data
    print(f"[provider] {provider} model={model} mode={args.mode}")

    out_dir = os.path.join(args.synapse_dir, "datasets_sft", args.source_name)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{args.source_name}_raw.jsonl")
    prog_path = os.path.join(out_dir, "progress.log")
    meta_path = os.path.join(out_dir, "meta_raw.json")

    done = set()
    for p in (prog_path, raw_path):
        if os.path.exists(p):
            with open(p) as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        done.add(json.loads(ln).get("id") if p == raw_path else ln)
                    except json.JSONDecodeError:
                        pass
    done.discard(None)
    if done:
        print(f"[resume] {len(done):,} already processed")

    if args.problems:                           # custom DB (already has unique ids)
        pool = load_problems(args.problems)
        print(f"[problems] {len(pool):,} from {args.problems}")
    else:                                        # HF pools, mode-scoped ids
        pool = grd.load_pool(args.sources.split(","), args.decontaminate)
        for p in pool:
            p["id"] = f"{args.mode}_{p['id']}"   # one problem can appear in both modes
    full = len(pool)
    pool = [p for p in pool if p["id"] not in done]
    if args.limit:
        pool = pool[:args.limit]

    # Force a search call up front for fact-lookup (teacher knows many facts → would
    # answer without searching). Default ON for search mode; override with the flags.
    want_force = args.force_search if args.force_search is not None else args.mode == "search"
    search_available = any(t["function"]["name"] == "search" for t in TOOLS_FOR_MODE[args.mode])
    force_first = "search" if (want_force and search_available) else None
    if want_force and not search_available:
        print(f"[warn] --force-search ignored: search tool not exposed in --mode {args.mode}")
    print(f"[run] {len(pool):,} problems workers={args.workers} force_search={force_first is not None}")

    client = grd.make_client(api_key, cfg["base_url"])
    limiter = tr.RateLimiter(args.search_rate)
    search_budget = SearchBudget(args.max_searches) if args.max_searches else None
    lock = threading.Lock()
    stop = threading.Event()
    stats = {"processed": 0, "kept": 0, "in_tok": 0, "out_tok": 0, "reasons": {}}
    raw_f = open(raw_path, "a")
    prog_f = open(prog_path, "a")
    dbg_f = open(os.path.join(out_dir, "rejects_debug.jsonl"), "w") if args.debug_rejects else None
    dbg_n = [0]
    t0 = time.time()

    def handle(problem):
        if stop.is_set():
            return
        rec, transient, it, ot, status, debug = solve_problem(
            client, model, problem, args.mode, args, limiter, brave_key,
            force_first_tool=force_first, search_budget=search_budget)
        if search_budget is not None and search_budget.left <= 0 and not stop.is_set():
            stop.set()
            print(f"  [searches] --max-searches exhausted — stopping (resumable)")
        with lock:
            stats["in_tok"] += it
            stats["out_tok"] += ot
            if args.budget_usd:
                cur = stats["in_tok"] * grd.PRICE_IN_PER_TOK + stats["out_tok"] * grd.PRICE_OUT_PER_TOK
                if cur >= args.budget_usd and not stop.is_set():
                    stop.set()
                    print(f"  [budget] ${cur:.2f} ≥ ${args.budget_usd:.0f} — stopping (resumable)")
            if transient:
                return
            stats["reasons"][status] = stats["reasons"].get(status, 0) + 1
            if dbg_f and debug and dbg_n[0] < args.debug_rejects:
                dbg_f.write(json.dumps(debug, ensure_ascii=False) + "\n")
                dbg_f.flush()
                dbg_n[0] += 1
            if rec:
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw_f.flush()
                stats["kept"] += 1
            prog_f.write(problem["id"] + "\n")
            prog_f.flush()
            done.add(problem["id"])
            stats["processed"] += 1
            if stats["processed"] % 200 == 0:
                _progress(stats, t0)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(handle, p) for p in pool]):
            fut.result()   # fail LOUD: a worker exception must kill the run, not
                           # vanish into a "kept 0/0" success (v3 calibration bug)
    raw_f.close()
    prog_f.close()
    if dbg_f:
        dbg_f.close()
        print(f"[debug] wrote {dbg_n[0]} rejected samples -> {os.path.join(out_dir, 'rejects_debug.jsonl')}")
    _progress(stats, t0)

    cost = stats["in_tok"] * grd.PRICE_IN_PER_TOK + stats["out_tok"] * grd.PRICE_OUT_PER_TOK
    yield_pct = 100 * stats["kept"] / max(1, stats["processed"])
    json.dump({"source": f"{args.source_name} ({args.mode})", "model": model,
               "processed": stats["processed"], "kept": stats["kept"],
               "yield_pct": round(yield_pct, 1), "cost_usd": round(cost, 2),
               "in_tok": stats["in_tok"], "out_tok": stats["out_tok"],
               "reject_reasons": stats["reasons"]},
              open(meta_path, "w"), indent=2)
    print(f"\n[done] kept {stats['kept']:,}/{stats['processed']:,} "
          f"({yield_pct:.1f}%) cost=${cost:.2f}")
    print(f"[reasons] {dict(sorted(stats['reasons'].items(), key=lambda kv: -kv[1]))}")
    if args.limit:
        per = cost / max(1, stats["processed"])
        print(f"[projection] full pool {full:,} → ~{int(yield_pct/100*full):,} traces, ~${per*full:.0f}")


def _progress(stats, t0):
    dt = time.time() - t0
    cost = stats["in_tok"] * grd.PRICE_IN_PER_TOK + stats["out_tok"] * grd.PRICE_OUT_PER_TOK
    top = dict(sorted(stats.get("reasons", {}).items(), key=lambda kv: -kv[1])[:4])
    print(f"  processed={stats['processed']:,} kept={stats['kept']:,} "
          f"({100*stats['kept']/max(1,stats['processed']):.0f}%) "
          f"{stats['processed']/max(1e-9,dt):.1f}/s cost=${cost:.2f} drops={top}")


if __name__ == "__main__":
    main()
