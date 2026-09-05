#!/usr/bin/env python3
"""Tool-use eval for the SFT model — SFT v2 Phase 4.4.

Drives the SAME loop the chatbot serves (sft/tool_loop.py) over three legs built
from the VAL side of the global hash split (never-trained questions):

  A  tool prompt + hard math      (tool_use val)       -> should CALL python; pass@1 vs gold
  B  tool prompt + trivial/prose  (tool_negative val)  -> should answer DIRECTLY
                                                          (false-call rate: the number tool_negative exists to move)
  C  empty prompt + the leg-A math                     -> must NEVER call (the prompt->behavior contrast)
  D  tool prompt + fact questions (tool_search val)    -> should CALL search; pass@1 vs gold
     (v3+ only: needs datasets_sft/tool_search/ and a live BRAVE_API_KEY at eval
      time — train == inference. Skipped with a notice when either is missing.)

Per leg: call rate, calls/trace, malformed-JSON rate, loop status mix, and pass@1
where a gold answer exists (tool_use and the math-donor tool_negatives carry a
verified "The answer is: X" line; prose negatives are judged on call rate only).

Decoding is greedy (top_k=1) so numbers are reproducible run to run.

Usage (GPU box / Colab):
  SYNAPSE_DIR=/content/drive/MyDrive/synapse python sft/eval_tool_use.py \\
      --ckpt $SYNAPSE_DIR/sft_checkpoints/sft_best.pth --n 100 --output results.json
  Local CPU smoke test:  --device cpu --n 3 --no-compile
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "sparky"))

import torch                                                    # noqa: E402
from tokenizers import Tokenizer                                # noqa: E402

import tools_runtime as tr                                      # noqa: E402
import tool_loop as tl                                          # noqa: E402
from tokenize_sft_data import split_bucket                      # noqa: E402
from generate_reasoning_distill import extract_final_answer, answers_match  # noqa: E402
from sparky_chat_template import sft_stop_token_ids             # noqa: E402

VAL_FRACTION = 0.02            # must match the tokenization run (registry split rule)


def load_val(path, n, val_fraction):
    """Val-side records of a raw source, deterministic order by sha256(id)."""
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — pull datasets_sft/ from Drive")
    thr = int(round(val_fraction * 1000))
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if split_bucket(ex) < thr:
                rows.append(ex)
    if not rows:
        raise SystemExit(f"{path}: no val-side records (split rule mismatch?)")
    rows.sort(key=lambda r: hashlib.sha256(str(r["id"]).encode()).hexdigest())
    return rows[:n]


def gold_of(ex):
    """Verified final answer from the record's last assistant turn, or None."""
    last = ex["messages"][-1]
    if last["role"] != "assistant":
        return None
    if "the answer is" not in last["content"].lower():
        return None
    return extract_final_answer(last["content"]) or None


def run_leg(name, records, system, gen, tok, max_prompt, log):
    items, t0 = [], time.time()
    for i, ex in enumerate(records):
        q = ex["messages"][0]["content"]
        assert ex["messages"][0]["role"] == "user"
        calls, results, malformed = [], [], 0
        for ev in tl.run_tool_loop(gen, tok, [{"role": "user", "content": q}], system,
                                   max_prompt_tokens=max_prompt):
            if ev["type"] == "tool_call":
                calls.append(ev["call"])
                if ev["tool"] is None:
                    malformed += 1
            elif ev["type"] == "tool_result":
                results.append(ev["text"])
            elif ev["type"] == "final":
                final = ev
        answer_turns = [m for m in final["messages"] if m["role"] == "assistant"]
        text = answer_turns[-1]["content"] if answer_turns else ""
        gold = gold_of(ex)
        pred = extract_final_answer(text) if gold else None
        item = {"id": ex["id"], "question": q, "gold": gold, "pred": pred,
                "pass": (answers_match(pred, gold) if gold else None),
                "n_calls": len(calls), "malformed": malformed, "status": final["status"],
                "calls": calls, "results": results, "final": text,
                "trace": final["messages"]}
        items.append(item)
        if log:
            mark = "✓" if item["pass"] else ("✗" if item["pass"] is False else "·")
            print(f"  [{name} {i+1:>3}/{len(records)}] {mark} calls={len(calls)} "
                  f"{final['status']:10s} {q[:60]!r} -> {text[-50:]!r}", flush=True)
    n = len(items)
    graded = [x for x in items if x["pass"] is not None]
    m = {
        "n": n,
        "call_rate": sum(x["n_calls"] > 0 for x in items) / n,
        "calls_by_tool": dict(Counter(c.get("tool") or "malformed"
                                      for x in items for c in x["calls"])),
        "calls_per_trace": sum(x["n_calls"] for x in items) / n,
        "malformed_rate": sum(x["malformed"] for x in items) / max(1, sum(x["n_calls"] for x in items)),
        "status": dict(Counter(x["status"] for x in items)),
        "graded": len(graded),
        "pass_at_1": (sum(x["pass"] for x in graded) / len(graded)) if graded else None,
        "seconds": round(time.time() - t0, 1),
    }
    return m, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None, help="default: $SYNAPSE_DIR/tokenizer_out/tokenizer.json")
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR"))
    ap.add_argument("--n", type=int, default=100, help="records per leg")
    ap.add_argument("--legs", default="A,B,C,D")
    ap.add_argument("--max-tokens", type=int, default=384, help="per generation round")
    ap.add_argument("--max-rounds", type=int, default=tl.MAX_ROUNDS)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if not args.synapse_dir:
        raise SystemExit("set SYNAPSE_DIR (refusing to guess — ./synapse is the venv)")
    tok_path = args.tokenizer or os.path.join(args.synapse_dir, "tokenizer_out", "tokenizer.json")
    os.environ.setdefault("SYNAPSE_TOKENIZER", tok_path)      # tools_runtime.truncate_tokens
    base = os.path.join(args.synapse_dir, "datasets_sft")

    tok = Tokenizer.from_file(tok_path)
    tr.verify_tool_tokens(tok)
    eot_id = tok.token_to_id("<|endoftext|>") or 0
    stops = sft_stop_token_ids(tok, eot_id)

    # Fail fast on leg-D requirements BEFORE the 8 GB model load.
    requested_early = [L.strip() for L in args.legs.split(",") if L.strip()]
    if "D" in requested_early and os.path.exists(os.path.join(base, "tool_search", "tool_search_raw.jsonl")) \
            and not os.environ.get("BRAVE_API_KEY"):
        raise SystemExit("leg D needs BRAVE_API_KEY at eval time (or drop D from --legs)")

    from sparky_model import SynapseInfer, BLOCK_SIZE
    print(f"[model] loading {args.ckpt}")
    model, info = SynapseInfer.from_checkpoint(args.ckpt, device=args.device, no_compile=args.no_compile)
    if not info.get("is_sft"):
        raise SystemExit("checkpoint is not an SFT checkpoint (stage != 'sft') — tool eval is meaningless")
    max_prompt = BLOCK_SIZE - args.max_tokens

    @torch.no_grad()
    def gen(prompt_ids):
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        yield from model.generate(idx, args.max_tokens, temperature=1.0, top_k=1, top_p=1.0,
                                  repetition_penalty=1.0, eot_id=eot_id, stop_tokens=stops)

    math = load_val(os.path.join(base, "tool_use", "tool_use_raw.jsonl"), args.n, args.val_fraction)
    neg = load_val(os.path.join(base, "tool_negative", "tool_negative_raw.jsonl"), args.n, args.val_fraction)
    legs = {
        "A": ("tool prompt + hard math (tool_use val)", math, tr.CANONICAL_TOOL_SYSTEM),
        "B": ("tool prompt + trivial/prose (tool_negative val)", neg, tr.CANONICAL_TOOL_SYSTEM),
        "C": ("EMPTY prompt + the same hard math", math, ""),
    }
    requested = [L.strip() for L in args.legs.split(",") if L.strip()]
    search_raw = os.path.join(base, "tool_search", "tool_search_raw.jsonl")
    if "D" in requested:
        # Leg D is v3+: fact questions the model was trained to SEARCH for. Needs the
        # source on disk AND a live key (search really runs — train == inference).
        if not os.path.exists(search_raw):
            print("[legs] D skipped: no datasets_sft/tool_search/ (v2 model has no search training)")
            requested.remove("D")
        else:
            legs["D"] = ("tool prompt + fact questions (tool_search val)",
                         load_val(search_raw, args.n, args.val_fraction),
                         tr.CANONICAL_TOOL_SYSTEM)
    out = {"config": vars(args), "ckpt_step": info.get("step"), "legs": {}}
    for L in requested:
        title, recs, system = legs[L]
        print(f"\n=== leg {L}: {title} — {len(recs)} records ===")
        metrics, items = run_leg(L, recs, system, gen, tok, max_prompt, not args.quiet)
        out["legs"][L] = {"title": title, "metrics": metrics, "items": items}
        print(json.dumps(metrics, indent=None))

    print("\n" + "=" * 64 + "\n  TOOL-USE EVAL SUMMARY\n" + "=" * 64)
    print(f"  {'leg':4s} {'call%':>6s} {'calls/tr':>8s} {'malf%':>6s} {'pass@1':>7s}  expectation")
    expect = {"A": "call% HIGH, pass@1 in (0,100)", "B": "call% LOW  (false-call rate)",
              "C": "call% == 0 (never calls without the prompt)",
              "D": "search-call% HIGH, pass@1 in (0,100)"}
    for L, leg in out["legs"].items():
        m = leg["metrics"]
        p = f"{100*m['pass_at_1']:.0f}%" if m["pass_at_1"] is not None else "  n/a"
        print(f"  {L:4s} {100*m['call_rate']:5.0f}% {m['calls_per_trace']:8.2f} "
              f"{100*m['malformed_rate']:5.0f}% {p:>7s}  {expect[L]}")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\n[eval] wrote {args.output}")


if __name__ == "__main__":
    main()
