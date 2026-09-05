#!/usr/bin/env python3
"""Generative bench for the Synapse SFT (chat) model.

Unlike sparky_eval.py (loglikelihood scoring for the base model), this asks the
SFT model real questions, reads its generated answers, and grades them. Prompts
use the exact trained chat template (sparky_chat_template.build_sft_prompt), so
the bench tests what the chatbot actually serves.

Grading is LLM-judge primary (OpenAI): every answer is scored against a rubric.
Deterministic graders (numeric/keywords/regex/contains/refusal) always run too
and are recorded for cross-check; if the judge is unavailable they become the
verdict.

Usage:
  # SFT, judged by DeepSeek V4 (default; needs DEEPSEEK_API_KEY in env or sparky/.env)
  python sft_bench.py --ckpt sparky_data/sft_latest.pth \\
                      --tokenizer sparky_data/tokenizer.json
  # ...or judge with OpenAI:  --judge-provider openai   (needs OPENAI_API_KEY)

  # Offline (deterministic graders only)
  python sft_bench.py --ckpt sparky_data/sft_latest.pth \\
                      --tokenizer sparky_data/tokenizer.json --no-judge

  # Side-by-side vs the pretrain base model
  python sft_bench.py --ckpt sparky_data/sft_latest.pth \\
                      --tokenizer sparky_data/tokenizer.json \\
                      --compare-pretrain sparky_data/synapse_2b_d2560_l28.pth

  # Quick smoke test without loading the model
  python sft_bench.py --ckpt x --tokenizer sparky_data/tokenizer.json --dry-run --limit 5
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
from tokenizers import Tokenizer

from sparky_model import SynapseInfer, BLOCK_SIZE
from sparky_chat_template import (
    build_sft_prompt, sft_stop_token_ids, DEFAULT_SYSTEM_PROMPT,
)

REFUSAL_PHRASES = [
    "can't", "cannot", "can not", "won't", "will not", "not able",
    "unable", "i'm sorry", "i am sorry", "i won't", "i will not",
    "not appropriate", "cannot help", "can't help", "i cannot assist",
    "i can't assist", "not something i can", "i must decline",
]


# ── env / tokenizer ──────────────────────────────────────────────────────────

def load_env_value(key):
    """Look up `key` in the process env, then sparky/.env."""
    if os.environ.get(key):
        return os.environ[key]
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == key:
                        return v.strip().strip('"').strip("'")
    return None


def load_tokenizer(path):
    tok = Tokenizer.from_file(path)
    eot_id = tok.token_to_id("<|endoftext|>")
    if eot_id is None:
        eot_id = 0
    return tok, eot_id


def load_bench(path, limit=None):
    items = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[bench] bad JSON on line {ln} of {path}: {e}")
    if limit:
        items = items[:limit]
    return items


def item_messages(item):
    """Return the chat messages for an item (single-turn or multi-turn)."""
    if item.get("messages"):
        return item["messages"]
    return [{"role": "user", "content": item.get("question", "")}]


def render_for_judge(item):
    """Human-readable rendering of the prompt for the judge."""
    msgs = item_messages(item)
    if len(msgs) == 1:
        return msgs[0]["content"]
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)


# ── generation ───────────────────────────────────────────────────────────────

def generate_reply(model, tok, eot_id, device, item, system, gen_cfg):
    msgs = item_messages(item)
    prompt = build_sft_prompt(msgs, system)
    input_ids = tok.encode(prompt, add_special_tokens=False).ids
    max_tokens = item.get("max_tokens", gen_cfg["max_tokens"])
    max_prompt = max(BLOCK_SIZE - max_tokens, 1)
    if len(input_ids) > max_prompt:
        input_ids = input_ids[-max_prompt:]
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    stops = sft_stop_token_ids(tok, eot_id)
    out_ids = []
    for tid in model.generate(idx, max_tokens,
                              temperature=gen_cfg["temperature"],
                              top_k=gen_cfg["top_k"],
                              top_p=gen_cfg["top_p"],
                              repetition_penalty=gen_cfg["repetition_penalty"],
                              eot_id=eot_id, stop_tokens=stops):
        out_ids.append(tid)
    return tok.decode(out_ids).strip()


# ── deterministic graders ────────────────────────────────────────────────────

_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}


def _numbers(text):
    """Numeric values in the text: digit forms plus spelled-out integers."""
    vals = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]
    for w, v in _WORD_NUMS.items():
        if re.search(rf"\b{w}\b", text, re.IGNORECASE):
            vals.append(float(v))
    return vals


def grade_deterministic(item, response):
    """Returns True/False, or None if this item has no deterministic grader."""
    g = item.get("grader")
    r = response.strip()
    rl = r.lower()
    cs = item.get("case_sensitive", False)  # honored by contains/keywords
    hay = r if cs else rl
    if g == "numeric":
        try:
            exp = float(str(item["expected"]))
        except (TypeError, ValueError):
            return None
        # If the model committed to a final answer line (the trained metamath /
        # tool_use convention), grade ONLY that — otherwise a wrong final answer
        # passes because the right number appears mid-derivation.
        m = re.findall(r"the answer is:?\s*(.+)", r, re.IGNORECASE)
        scope = m[-1] if m else r
        return any(abs(n - exp) < 1e-6 for n in _numbers(scope))
    if g == "contains":
        needle = str(item["expected"]) if cs else str(item["expected"]).lower()
        return needle in hay
    if g == "keywords":
        kws = [str(k) if cs else str(k).lower() for k in item["expected"]]
        if item.get("match") == "any":
            return any(k in hay for k in kws)
        return all(k in hay for k in kws)
    if g == "regex":
        return re.search(item["expected"], r) is not None
    if g == "refusal":
        return any(p in rl for p in REFUSAL_PHRASES)
    return None  # judge-only or unknown


# ── LLM judge ────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict but fair grader for a small language model's answers. "
    "Given a question, a grading rubric, and the model's answer, decide whether "
    "the answer satisfies the rubric. Be lenient about formatting and verbosity "
    "but strict about correctness. Respond ONLY with JSON: "
    '{"pass": true|false, "score": 0.0-1.0, "reason": "<one short sentence>"}.'
)


# LLM-judge providers (all OpenAI-SDK compatible). DeepSeek V4 is a strong, cheap
# grader; deepseek-v4-flash is the chat (non-thinking) successor to deepseek-chat.
JUDGE_PROVIDERS = {
    "deepseek": {"key": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com",
                 "model": "deepseek-v4-flash"},
    "openai":   {"key": "OPENAI_API_KEY",   "base_url": None, "model": "gpt-4o-mini"},
}


def make_judge(provider, judge_model=None):
    cfg = JUDGE_PROVIDERS.get(provider)
    if cfg is None:
        print(f"[judge] unknown provider {provider!r} (choices: {list(JUDGE_PROVIDERS)}) — judge disabled.")
        return None
    key = load_env_value(cfg["key"])
    if not key:
        print(f"[judge] {cfg['key']} not found (env or sparky/.env) — judge disabled.")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[judge] `openai` not installed (pip install openai) — judge disabled.")
        return None
    client = OpenAI(api_key=key, base_url=cfg["base_url"]) if cfg["base_url"] else OpenAI(api_key=key)
    model = judge_model or cfg["model"]
    print(f"[judge] enabled: provider={provider} model={model}")
    return {"client": client, "model": model}


def judge_answer(judge, item, response):
    ref = item.get("expected")
    user = (
        f"QUESTION / CONVERSATION:\n{render_for_judge(item)}\n\n"
        f"GRADING RUBRIC:\n{item.get('rubric', '(none provided)')}\n"
        + (f"\nREFERENCE ANSWER: {ref}\n" if ref is not None else "")
        + f"\nMODEL ANSWER:\n{response or '(empty)'}\n\n"
        "Grade the model answer per the rubric."
    )
    try:
        resp = judge["client"].chat.completions.create(
            model=judge["model"],
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": user}],
        )
        verdict = json.loads(resp.choices[0].message.content)
        return {"pass": bool(verdict.get("pass")),
                "score": float(verdict.get("score", 1.0 if verdict.get("pass") else 0.0)),
                "reason": str(verdict.get("reason", ""))}
    except Exception as e:
        return {"pass": None, "score": None, "reason": f"judge error: {e}"}


# ── run one model over the bench ─────────────────────────────────────────────

def run_model(label, ckpt, tok, eot_id, device, items, system, gen_cfg,
              judge, no_compile):
    print(f"\n[{label}] loading {ckpt} ({os.path.getsize(ckpt)/1e9:.2f} GB)")
    model, info = SynapseInfer.from_checkpoint(ckpt, device=device, no_compile=no_compile)
    is_sft = bool(info.get("is_sft")) if info else False
    if not is_sft:
        print(f"[{label}] WARNING: checkpoint is_sft=False — chat template may be "
              f"out-of-distribution for a pretrain-only model.")
    records = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        resp = generate_reply(model, tok, eot_id, device, item, system, gen_cfg)
        det = grade_deterministic(item, resp)
        rec = {"id": item["id"], "category": item.get("category", "uncat"),
               "prompt": render_for_judge(item), "response": resp,
               "deterministic": det, "judge": None}
        if judge is not None:
            rec["judge"] = judge_answer(judge, item, resp)
        # Final verdict: judge wins when available, else deterministic.
        if rec["judge"] and rec["judge"]["pass"] is not None:
            rec["verdict"] = rec["judge"]["pass"]
        else:
            rec["verdict"] = det
        records.append(rec)
        mark = {True: "✓", False: "✗", None: "?"}[rec["verdict"]]
        print(f"  [{i:>2}/{len(items)}] {mark} {item['id']:<14} "
              f"{(resp[:60] + '…') if len(resp) > 60 else resp!r}")
    elapsed = time.time() - t0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, elapsed, info


def summarize(records):
    by_cat = defaultdict(lambda: [0, 0, 0])  # [pass, total_graded, ungraded]
    overall = [0, 0, 0]
    for r in records:
        cat = by_cat[r["category"]]
        if r["verdict"] is None:
            cat[2] += 1; overall[2] += 1
        else:
            cat[1] += 1; overall[1] += 1
            if r["verdict"]:
                cat[0] += 1; overall[0] += 1
    return by_cat, overall


def print_report(label, records):
    by_cat, overall = summarize(records)
    print(f"\n{'='*56}\n  {label} — results by category\n{'='*56}")
    print(f"  {'category':<22}{'pass':>6}{'total':>7}{'rate':>8}")
    print(f"  {'-'*43}")
    for cat in sorted(by_cat):
        p, t, u = by_cat[cat]
        rate = f"{100*p/t:.0f}%" if t else "n/a"
        extra = f"  (+{u} ungraded)" if u else ""
        print(f"  {cat:<22}{p:>6}{t:>7}{rate:>8}{extra}")
    p, t, u = overall
    rate = f"{100*p/t:.0f}%" if t else "n/a"
    print(f"  {'-'*43}")
    print(f"  {'OVERALL':<22}{p:>6}{t:>7}{rate:>8}" + (f"  (+{u} ungraded)" if u else ""))


def write_outputs(out_dir, tag, payload, models):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{tag}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    md_path = os.path.join(out_dir, f"{tag}.md")
    with open(md_path, "w") as f:
        f.write(f"# SFT bench — {tag}\n\n")
        for label, recs in models.items():
            by_cat, overall = summarize(recs)
            f.write(f"## {label}\n\n| category | pass | total | rate |\n|---|---|---|---|\n")
            for cat in sorted(by_cat):
                p, t, u = by_cat[cat]
                f.write(f"| {cat} | {p} | {t} | {100*p/t:.0f}% |\n" if t else
                        f"| {cat} | - | 0 | n/a |\n")
            p, t, u = overall
            f.write(f"| **OVERALL** | **{p}** | **{t}** | **{100*p/t:.0f}%** |\n\n"
                    if t else "| **OVERALL** | - | 0 | n/a |\n\n")
        f.write("## Transcripts\n\n")
        for label, recs in models.items():
            f.write(f"### {label}\n\n")
            for r in recs:
                mark = {True: "✓", False: "✗", None: "?"}[r["verdict"]]
                f.write(f"**{mark} {r['id']}** ({r['category']})\n\n")
                f.write(f"> {r['prompt']}\n\n")
                f.write(f"```\n{r['response']}\n```\n\n")
                if r.get("judge"):
                    f.write(f"_judge: {r['judge'].get('reason','')}_\n\n")
    print(f"\n[bench] wrote {json_path}\n[bench] wrote {md_path}")


def main():
    ap = argparse.ArgumentParser(description="Synapse SFT generative bench")
    ap.add_argument("--ckpt", required=True, help="SFT checkpoint .pth")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--bench", default=os.path.join("bench", "sft_bench.jsonl"))
    ap.add_argument("--compare-pretrain", default=None,
                    help="Also run this base checkpoint for side-by-side comparison")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT,
                    help="System prompt ('' to disable)")
    # judge
    ap.add_argument("--no-judge", action="store_true", help="Disable LLM judge")
    ap.add_argument("--judge-provider", default="deepseek", choices=list(JUDGE_PROVIDERS),
                    help="LLM judge provider (default: deepseek)")
    ap.add_argument("--judge-model", default=None,
                    help="override judge model id (default per provider)")
    # generation (greedy by default for reproducibility)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=1, help="1 = greedy/deterministic")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--output-dir", default=os.path.join("results", "sft_bench"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate bench + show prompts without loading the model")
    args = ap.parse_args()

    items = load_bench(args.bench, args.limit)
    system = args.system or None
    gen_cfg = {"max_tokens": args.max_tokens, "temperature": args.temperature,
               "top_k": args.top_k, "top_p": args.top_p,
               "repetition_penalty": args.repetition_penalty}

    print("=" * 56)
    print("  SYNAPSE SFT BENCH")
    print("=" * 56)
    print(f"  bench:    {args.bench} ({len(items)} questions)")
    cats = sorted({it.get('category', 'uncat') for it in items})
    print(f"  topics:   {len(cats)} — {', '.join(cats)}")
    print(f"  decoding: top_k={args.top_k} temp={args.temperature} "
          f"top_p={args.top_p} rep={args.repetition_penalty} max_tokens={args.max_tokens}")

    if args.dry_run:
        tok, _ = load_tokenizer(args.tokenizer)
        for it in items:
            prompt = build_sft_prompt(item_messages(it), system)
            ids = tok.encode(prompt, add_special_tokens=False).ids
            print(f"\n--- {it['id']} ({it.get('category')}) | {len(ids)} tok | "
                  f"grader={it.get('grader')} ---")
            print(repr(prompt[:200]))
        print("\n[dry-run] bench parsed and prompts built OK.")
        return

    if not torch.cuda.is_available() and args.device == "cuda":
        sys.exit("[error] CUDA requested but not available (use --device cpu, slow)")
    device = torch.device(args.device)
    tok, eot_id = load_tokenizer(args.tokenizer)

    judge = None if args.no_judge else make_judge(args.judge_provider, args.judge_model)
    print(f"  judge:    {judge['model'] if judge else 'OFF (deterministic graders)'}")

    models = {}
    timings = {}
    sft_recs, sft_t, sft_info = run_model("SFT", args.ckpt, tok, eot_id, device,
                                          items, system, gen_cfg, judge, args.no_compile)
    models["SFT"] = sft_recs
    timings["SFT"] = sft_t
    if args.compare_pretrain:
        pt_recs, pt_t, _ = run_model("PRETRAIN", args.compare_pretrain, tok, eot_id,
                                     device, items, system, gen_cfg, judge, args.no_compile)
        models["PRETRAIN"] = pt_recs
        timings["PRETRAIN"] = pt_t

    for label, recs in models.items():
        print_report(label, recs)
    if "PRETRAIN" in models:
        sp = summarize(models["SFT"])[1]
        pp = summarize(models["PRETRAIN"])[1]
        sr = 100*sp[0]/sp[1] if sp[1] else 0
        pr = 100*pp[0]/pp[1] if pp[1] else 0
        print(f"\n  SFT lift: {pr:.0f}% (pretrain) → {sr:.0f}% (SFT)  "
              f"[{sr-pr:+.0f} pts]")

    step = (sft_info or {}).get("step")
    tag = f"step{step}" if step else f"run{int(time.time())}"
    payload = {"config": vars(args), "topics": cats,
               "timings_s": {k: round(v, 1) for k, v in timings.items()},
               "results": models}
    write_outputs(args.output_dir, tag, payload, models)


if __name__ == "__main__":
    main()
