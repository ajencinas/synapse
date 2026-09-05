#!/usr/bin/env python3
"""Comparable benchmark numbers for the Synapse BASE model.

Reproduces the Open LLM Leaderboard v1 suite — the most widely-cited setup for
comparing small base models — using each task's CANONICAL few-shot count and
metric. This is the apples-to-apples report; sparky_eval.py's single global
--num_fewshot cannot reproduce these (each task needs a different shot count).

Suite (Open LLM Leaderboard v1):
    ARC-Challenge   25-shot   acc_norm
    HellaSwag       10-shot   acc_norm
    MMLU             5-shot   acc
    TruthfulQA-MC2   0-shot   acc (mc2)
    Winogrande       5-shot   acc
    GSM8K            5-shot   exact_match (strict)
    Average = mean of the six (leaderboard convention)

For PUBLISHABLE numbers run the FULL sets (no --limit). --limit gives a fast,
NON-comparable estimate. Always report the lm_eval version (printed below);
numbers shift slightly across harness versions.

Usage:
  # Full comparable run (slow — minutes to hours on one GPU)
  python sparky_leaderboard.py --ckpt sparky_data/synapse_2b_d2560_l28.pth \\
                               --tokenizer sparky_data/tokenizer.json \\
                               --output results/leaderboard/synapse_2b.json

  # Fast smoke estimate (NOT comparable)
  python sparky_leaderboard.py --ckpt ... --tokenizer ... --limit 50

  # Subset of tasks
  python sparky_leaderboard.py --ckpt ... --tokenizer ... --tasks mmlu,gsm8k
"""
import argparse
import gc
import json
import os
import sys
import time

# Reduce fragmentation OOMs on small cards — must be set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from lm_eval import simple_evaluate

from sparky_eval import SynapseEvalLM, get_gpu_info, human_readable_gb

try:
    import importlib.metadata as _md
    LM_EVAL_VERSION = _md.version("lm_eval")
except Exception:
    LM_EVAL_VERSION = "unknown"


# Open LLM Leaderboard v1 canonical settings.
LEADERBOARD = [
    {"task": "arc_challenge",  "shots": 25, "metric": "acc_norm,none",          "name": "ARC-Challenge"},
    {"task": "hellaswag",      "shots": 10, "metric": "acc_norm,none",          "name": "HellaSwag"},
    {"task": "mmlu",           "shots": 5,  "metric": "acc,none",               "name": "MMLU"},
    {"task": "truthfulqa_mc2", "shots": 0,  "metric": "acc,none",               "name": "TruthfulQA-MC2"},
    {"task": "winogrande",     "shots": 5,  "metric": "acc,none",               "name": "Winogrande"},
    {"task": "gsm8k",          "shots": 5,  "metric": "exact_match,strict-match", "name": "GSM8K"},
]


def extract_metric(results, task, metric_key):
    """Pull a metric value out of an lm_eval results dict, with fallbacks."""
    block = results.get("results", {}).get(task)
    if not block:
        return None
    if metric_key in block:
        return block[metric_key]
    # fallback: match on the metric name before the comma
    want = metric_key.split(",")[0]
    for k, v in block.items():
        if isinstance(v, (int, float)) and k.split(",")[0] == want:
            return v
    # last resort: first acc-like float
    for k, v in block.items():
        if isinstance(v, (int, float)) and not k.endswith("_stderr"):
            return v
    return None


def main():
    ap = argparse.ArgumentParser(description="Synapse Open LLM Leaderboard v1 runner")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated subset of leaderboard tasks (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-task example cap — FAST ESTIMATE ONLY, not comparable")
    ap.add_argument("--device", default="cuda")
    # Compile is OFF by default: torch.compile(mode="reduce-overhead") reuses
    # CUDA-graph buffers and crashes on the generative task (GSM8K) with the
    # KV-cache model ("accessing tensor output of CUDAGraphs ... overwritten").
    # It also gives ~no speedup for lm_eval's variable shapes. Opt in with --compile.
    ap.add_argument("--no-compile", action="store_true",
                    help="(default behavior) skip torch.compile")
    ap.add_argument("--compile", action="store_true",
                    help="opt into torch.compile (may crash on GSM8K — not recommended)")
    ap.add_argument("--max-batch-tokens", type=int, default=2048,
                    help="Lower this if you hit CUDA OOM (default 2048 for ~16GB cards)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    suite = LEADERBOARD
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",")}
        suite = [e for e in LEADERBOARD if e["task"] in wanted]
        if not suite:
            sys.exit(f"[error] no leaderboard tasks matched {sorted(wanted)}")

    gpu_name, vram_mb = get_gpu_info(torch.device(args.device))
    print("=" * 64)
    print("  SYNAPSE — Open LLM Leaderboard v1 (comparable)")
    print("=" * 64)
    print(f"  lm_eval:  {LM_EVAL_VERSION}")
    print(f"  GPU:      {gpu_name or 'N/A'} ({vram_mb or 0:.0f} MB)")
    print(f"  ckpt:     {args.ckpt}")
    print(f"  tasks:    {', '.join(e['task'] for e in suite)}")
    if args.limit:
        print(f"  limit:    {args.limit}  ⚠️  ESTIMATE ONLY — NOT comparable")
    print()

    print(f"[lb] loading model from {args.ckpt}")
    t0 = time.time()
    lm = SynapseEvalLM(args.ckpt, args.tokenizer, device=args.device,
                       no_compile=(args.no_compile or not args.compile),
                       max_batch_tokens=args.max_batch_tokens)
    print(f"[lb] loaded in {time.time()-t0:.1f}s | "
          f"VRAM {human_readable_gb(torch.cuda.memory_allocated() if args.device=='cuda' else 0)}")

    rows = []
    raw = {}
    grand_t0 = time.time()
    for entry in suite:
        task, shots = entry["task"], entry["shots"]
        print(f"\n[lb] {entry['name']}  ({shots}-shot)  ...")
        t0 = time.time()
        res = simple_evaluate(
            model=lm, tasks=[task], num_fewshot=shots, limit=args.limit,
            bootstrap_iters=0, log_samples=False, confirm_run_unsafe_code=True,
        )
        dt = time.time() - t0
        val = extract_metric(res, task, entry["metric"]) if res else None
        rows.append({**entry, "value": val, "seconds": round(dt, 1)})
        raw[task] = res.get("results", {}).get(task) if res else None
        pct = f"{100*val:.2f}%" if isinstance(val, (int, float)) else "ERR"
        print(f"[lb] {entry['name']}: {pct}  ({dt:.0f}s)")
        # Release per-task memory so log-likelihood batches don't starve the
        # next task's generation (matters on small GPUs).
        del res
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    scored = [r["value"] for r in rows if isinstance(r["value"], (int, float))]
    avg = sum(scored) / len(scored) if scored else None

    print("\n" + "=" * 64)
    print("  RESULTS — Open LLM Leaderboard v1")
    print("=" * 64)
    print(f"  {'Task':<18}{'Shots':>6}{'Metric':>14}{'Score':>10}")
    print(f"  {'-'*48}")
    for r in rows:
        metric_disp = r["metric"].split(",")[0]
        pct = f"{100*r['value']:.2f}" if isinstance(r["value"], (int, float)) else "ERR"
        print(f"  {r['name']:<18}{r['shots']:>6}{metric_disp:>14}{pct:>10}")
    print(f"  {'-'*48}")
    if avg is not None:
        print(f"  {'AVERAGE':<18}{'':>6}{'':>14}{100*avg:>10.2f}")
    print(f"\n[lb] total time {time.time()-grand_t0:.0f}s")
    if args.limit:
        print("[lb] ⚠️  --limit was set: these numbers are NOT comparable to "
              "published results. Re-run without --limit for the real numbers.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out = {
            "suite": "open_llm_leaderboard_v1",
            "lm_eval_version": LM_EVAL_VERSION,
            "comparable": args.limit is None,
            "config": {"ckpt": args.ckpt, "limit": args.limit, "gpu": gpu_name,
                       "checkpoint_step": (lm._info or {}).get("step")},
            "scores": {r["name"]: (round(100*r["value"], 2)
                                   if isinstance(r["value"], (int, float)) else None)
                       for r in rows},
            "shots": {r["name"]: r["shots"] for r in rows},
            "metrics": {r["name"]: r["metric"].split(",")[0] for r in rows},
            "average": round(100*avg, 2) if avg is not None else None,
            "raw": raw,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[lb] saved {args.output}")


if __name__ == "__main__":
    main()
