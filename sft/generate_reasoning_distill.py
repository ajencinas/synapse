#!/usr/bin/env python3
"""Reasoning-distillation data generator.

Generates verified, concise chain-of-thought math solutions with DeepSeek V4 Flash
(via OpenRouter), keeps only traces whose final answer matches the gold answer
(rejection sampling), and writes them in the SFT {system, messages} format that
tokenize_sft_data.py consumes (source name: reasoning_distill).

Why single-problem PARALLEL calls (not bundled):
  - Cost is per-token; bundling N problems saves only the tiny shared system
    prompt, not the solution tokens. V4-Flash (~$0.10/$0.20 per M tok) makes the
    full run ~$30 — cost isn't the bottleneck, wall-clock is, and that's solved by
    concurrency.
  - Bundling degrades reasoning quality (split attention), makes per-problem
    answer parsing/verification fragile, risks max_tokens truncation, and loses
    clean per-problem retry/resume. Trace quality is the whole point, so we don't.

Resumable: appends to the raw JSONL and records processed problem ids in
checkpoint.json — safe to Ctrl+C / re-run.

Usage:
  # 1) calibration slice — prints yield%% + projected full cost/time, then stop
  SYNAPSE_DIR=/path python sft/generate_reasoning_distill.py --limit 3000
  # 2) full run
  SYNAPSE_DIR=/path python sft/generate_reasoning_distill.py

Env: OPENROUTER_API_KEY (in env or repo .env).
Deps: openai, datasets, sympy, python-dotenv, tqdm.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# DeepSeek V4 Flash via either provider (both OpenAI-SDK compatible). Auto-detected
# from whichever API key is present; override with --provider / --model.
PROVIDERS = {
    "openrouter": {"key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
                   "model": "deepseek/deepseek-v4-flash"},
    "deepseek":   {"key_env": "DEEPSEEK_API_KEY",   "base_url": "https://api.deepseek.com",
                   "model": "deepseek-v4-flash"},
}
PRICE_IN_PER_TOK = 0.098 / 1_000_000      # $/input token  (V4-Flash, ~same both providers)
PRICE_OUT_PER_TOK = 0.196 / 1_000_000     # $/output token

SYSTEM_PROMPT = (
    "You are an expert math tutor. Solve the problem with clear, CONCISE "
    "step-by-step reasoning — show the key steps only, no rambling. Then end your "
    "answer with a final line in exactly this format:\n"
    "The answer is: <answer>\n"
    "where <answer> is the final value only (a number or simplified expression)."
)
MAX_SOLUTION_CHARS = 4000   # cheap pre-filter; tokenize_sft_data enforces 2048 exactly


# ----------------------------------------------------------------------------
# Problem sources: each yields (question, gold_answer)
# ----------------------------------------------------------------------------
# NuminaMath subsets to KEEP (2B-learnable, concise). Drop gsm8k (eval integrity),
# olympiads/aops/amc (too hard, solutions overflow 2048 and don't transfer).
NUMINA_KEEP = {"orca_math", "cn_k12", "synthetic_math", "math"}


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def adapter_orca(row):
    q = (row.get("question") or "").strip()
    gold = extract_final_answer(row.get("answer") or "")
    return (q, gold) if (q and gold) else None


def adapter_numina(row):
    if row.get("source") not in NUMINA_KEEP:
        return None
    q = (row.get("problem") or "").strip()
    gold = extract_boxed(row.get("solution") or "") or extract_final_answer(row.get("solution") or "")
    return (q, gold) if (q and gold) else None


SOURCES = {
    "orca":   {"hf_path": "microsoft/orca-math-word-problems-200k", "split": "train",
               "adapter": adapter_orca, "max": 120000},
    "numina": {"hf_path": "AI-MO/NuminaMath-CoT", "split": "train",
               "adapter": adapter_numina, "max": 130000},
}


# ----------------------------------------------------------------------------
# Answer extraction + verification
# ----------------------------------------------------------------------------

def extract_boxed(text):
    """Return the content of the LAST \\boxed{...} (brace-balanced), or ''."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    i = idx + len("\\boxed{")
    depth, out = 1, []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_answer(text):
    """Pull the final answer from a solution string. Priority:
    'The answer is: X' -> last \\boxed{} -> last number."""
    if not text:
        return ""
    m = list(re.finditer(r"answer\s*is\s*:?\s*(.+)", text, flags=re.IGNORECASE))
    if m:
        cand = m[-1].group(1).strip().rstrip(".").strip()
        if cand:
            return cand
    b = extract_boxed(text)
    if b:
        return b
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else ""


def normalize_answer(s):
    s = s.strip()
    # normalize unicode spaces (nbsp / narrow-nbsp / thin space) so unit-strip + the
    # space-removal below behave (e.g. "3 m").
    s = re.sub(r"[\u00a0\u2009\u202f\u2007\u2008]", " ", s)
    # strip LaTeX wrappers / formatting noise (incl. \( \) \[ \] math delimiters)
    s = s.replace("\\boxed{", "").rstrip("}")
    for tok in ["\\$", "$", "\\!", "\\,", "\\;", "\\left", "\\right", "%", "\\%",
                "\\(", "\\)", "\\[", "\\]"]:
        s = s.replace(tok, "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = s.replace("\\dfrac", "").replace("\\tfrac", "")
    # strip currency prefix ("Rs. 1500", "$5") and a trailing UNIT word after a
    # number ("15.5 kmph", "3 m", "80 days") — recovers correct answers the verifier
    # otherwise rejects. Curated unit list so algebraic vars (x, n) are NOT stripped.
    s = re.sub(r"(?i)^\s*(?:rs\.?|usd|inr|\$|€|£)\s*", "", s)
    s = re.sub(r"(?i)(?<=\d)\s*(?:kmph|km/?h|km|cm|mm|kg|mg|ml|litres?|liters?|"
               r"metres?|meters?|m|kg|g|l|days?|hours?|hrs?|minutes?|mins?|"
               r"seconds?|secs?|years?|yrs?|months?|weeks?|degrees?|°|"
               r"dollars?|cents?|rupees?|units?|people|students?)\.?$", "", s)
    s = s.replace(",", "").replace(" ", "").strip().rstrip(".")
    return s


def answers_match(pred, gold):
    if not pred or not gold:
        return False
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    # numeric
    try:
        if abs(float(p) - float(g)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # symbolic (sympy) — guarded; never let it crash a worker
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
        if sympy.simplify(parse_expr(p) - parse_expr(g)) == 0:
            return True
    except Exception:
        pass
    return False


# ----------------------------------------------------------------------------
# Load + dedup + (optional) decontam
# ----------------------------------------------------------------------------

def norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (s or "").lower())).strip()


def qid(question):
    return hashlib.sha1(norm_text(question).encode()).hexdigest()[:16]


def build_gsm8k_guard(n=8):
    from datasets import load_dataset
    print("[decontam] building 8-gram guard from GSM8K test")
    grams = set()
    for row in load_dataset("openai/gsm8k", "main", split="test"):
        toks = norm_text(row["question"]).split()
        grams |= {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}
    print(f"[decontam] {len(grams):,} guard 8-grams")
    return grams


def is_contaminated(question, guard, n=8):
    toks = norm_text(question).split()
    grams = {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}
    return bool(grams & guard)


def load_pool(which, decontaminate, seed=42):
    from datasets import load_dataset
    import random
    guard = build_gsm8k_guard() if decontaminate else None
    pool, seen = [], set()
    drops = {"adapter": 0, "dup": 0, "contaminated": 0}
    for name in which:
        spec = SOURCES[name]
        print(f"[load] {name}: {spec['hf_path']} ({spec['split']})")
        ds = load_dataset(spec["hf_path"], split=spec["split"])
        kept = 0
        for row in ds:
            norm = spec["adapter"](row)
            if norm is None:
                drops["adapter"] += 1
                continue
            q, gold = norm
            i = qid(q)
            if i in seen:
                drops["dup"] += 1
                continue
            if guard is not None and is_contaminated(q, guard):
                drops["contaminated"] += 1
                continue
            seen.add(i)
            pool.append({"id": i, "question": q, "gold": gold, "source": name})
            kept += 1
            if kept >= spec["max"]:
                break
        print(f"[load] {name}: kept {kept:,}")
    random.Random(seed).shuffle(pool)
    print(f"[load] pool={len(pool):,}  drops={drops}")
    return pool


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def make_client(api_key, base_url):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def solve_one(client, model, problem, retries):
    """Returns (solution_text, ok, in_tok, out_tok). Tries low temp first, then
    higher-temp retries only if the answer is wrong (cheap yield boost)."""
    in_tok = out_tok = 0
    for attempt in range(retries + 1):
        temp = 0.2 if attempt == 0 else 0.7
        for api_try in range(4):  # transient-error backoff
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=temp, max_tokens=1024,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": problem["question"]}],
                )
                break
            except Exception as e:
                if api_try == 3:
                    return None, False, in_tok, out_tok  # give up (transient) -> not 'done'
                time.sleep(2 ** api_try)
        usage = getattr(resp, "usage", None)
        if usage:
            in_tok += usage.prompt_tokens or 0
            out_tok += usage.completion_tokens or 0
        sol = (resp.choices[0].message.content or "").strip()
        if sol and len(sol) <= MAX_SOLUTION_CHARS and \
                answers_match(extract_final_answer(sol), problem["gold"]):
            return sol, True, in_tok, out_tok
    return None, False, in_tok, out_tok  # exhausted -> wrong/too-long -> 'done' (skip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="orca,numina")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N pool problems (calibration)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--provider", default="auto", choices=["auto", "openrouter", "deepseek"],
                    help="auto = pick whichever API key is present")
    ap.add_argument("--model", default=None, help="override the provider's default model id")
    ap.add_argument("--retries", type=int, default=1,
                    help="extra higher-temp attempts when the answer is wrong")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="hard stop once estimated spend reaches this (e.g. 30)")
    ap.add_argument("--no-decontaminate", dest="decontaminate", action="store_false",
                    default=True)
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR") or default_synapse_dir())
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    except ImportError:
        pass

    # Resolve provider from whichever key is present (or --provider override).
    provider = args.provider
    if provider == "auto":
        # Prefer DeepSeek direct when both keys are present (cheaper / lower latency).
        if os.environ.get("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        elif os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            raise SystemExit("no API key found — set DEEPSEEK_API_KEY or OPENROUTER_API_KEY (env or repo .env)")
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"{cfg['key_env']} not set (env or repo .env) for provider={provider}")
    model = args.model or cfg["model"]
    print(f"[provider] {provider}  model={model}  base={cfg['base_url']}")

    out_dir = os.path.join(args.synapse_dir, "datasets_sft", "reasoning_distill")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "reasoning_distill_raw.jsonl")
    prog_path = os.path.join(out_dir, "progress.log")   # append-only, one id/line (durable)
    meta_path = os.path.join(out_dir, "meta_raw.json")

    # Resume durably: a problem is "done" if its id is in progress.log (logged the
    # instant it was processed) OR already present in the raw file. So a kill at any
    # point (e.g. out of credits) never re-spends on finished problems and never
    # writes a duplicate kept trace.
    done = set()
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            done |= {ln.strip() for ln in f if ln.strip()}
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln).get("id"))
                except json.JSONDecodeError:
                    pass
    done.discard(None)
    if done:
        print(f"[resume] {len(done):,} problems already processed — skipping them")

    pool = load_pool(args.sources.split(","), args.decontaminate)
    full_pool_size = len(pool)
    pool = [p for p in pool if p["id"] not in done]
    if args.limit:
        pool = pool[:args.limit]
    print(f"[run] processing {len(pool):,} problems  workers={args.workers} model={model}")

    client = make_client(api_key, cfg["base_url"])
    lock = threading.Lock()
    stop = threading.Event()       # tripped when --budget-usd is reached
    stats = {"processed": 0, "kept": 0, "in_tok": 0, "out_tok": 0}
    raw_f = open(raw_path, "a")     # append → never clobbers earlier work
    prog_f = open(prog_path, "a")
    t0 = time.time()

    def handle(problem):
        if stop.is_set():          # budget hit → queued tasks become no-ops (left undone)
            return
        sol, ok, it, ot = solve_one(client, model, problem, args.retries)
        transient = (sol is None and not ok and it == 0 and ot == 0)
        with lock:
            stats["in_tok"] += it
            stats["out_tok"] += ot
            if args.budget_usd:
                cur = stats["in_tok"] * PRICE_IN_PER_TOK + stats["out_tok"] * PRICE_OUT_PER_TOK
                if cur >= args.budget_usd and not stop.is_set():
                    stop.set()
                    print(f"  [budget] reached ${cur:.2f} ≥ ${args.budget_usd:.0f} "
                          f"— finishing in-flight, skipping the rest (resumable)")
            if transient:
                return  # API failed before any tokens → leave undone, retry next run
            # write the KEPT trace first (the valuable artifact), then mark done —
            # both flushed immediately so a crash loses at most the in-flight item.
            if ok:
                rec = {"id": problem["id"], "system": "", "messages": [
                    {"role": "user", "content": problem["question"]},
                    {"role": "assistant", "content": sol}]}
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw_f.flush()
                stats["kept"] += 1
            prog_f.write(problem["id"] + "\n")
            prog_f.flush()
            done.add(problem["id"])
            stats["processed"] += 1
            if stats["processed"] % 500 == 0:
                _progress(stats, t0)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(handle, p) for p in pool]
        for _ in as_completed(futs):
            pass

    raw_f.close()
    prog_f.close()
    _progress(stats, t0)

    cost = stats["in_tok"] * PRICE_IN_PER_TOK + stats["out_tok"] * PRICE_OUT_PER_TOK
    yield_pct = 100 * stats["kept"] / max(1, stats["processed"])
    meta = {
        "source": f"reasoning_distill (DeepSeek V4 Flash via {provider})",
        "model": model, "processed": stats["processed"], "kept": stats["kept"],
        "yield_pct": round(yield_pct, 1), "cost_usd": round(cost, 2),
        "in_tok": stats["in_tok"], "out_tok": stats["out_tok"],
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"\n[done] kept {stats['kept']:,}/{stats['processed']:,} "
          f"({yield_pct:.1f}% yield)  cost=${cost:.2f}")
    if args.limit:  # calibration projection
        per = cost / max(1, stats["processed"])
        proj_cost = per * full_pool_size
        proj_keep = int(yield_pct / 100 * full_pool_size)
        print(f"[projection] full pool {full_pool_size:,} → ~{proj_keep:,} traces, "
              f"~${proj_cost:.0f}. Re-run without --limit for the full pass.")


def _progress(stats, t0):
    dt = time.time() - t0
    rate = stats["processed"] / max(1e-9, dt)
    cost = stats["in_tok"] * PRICE_IN_PER_TOK + stats["out_tok"] * PRICE_OUT_PER_TOK
    print(f"  processed={stats['processed']:,} kept={stats['kept']:,} "
          f"({100*stats['kept']/max(1,stats['processed']):.0f}%) "
          f"{rate:.1f}/s cost=${cost:.2f}")


if __name__ == "__main__":
    main()
