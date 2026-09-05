#!/usr/bin/env python3
"""Zero-shot benchmark evaluation for SynapseGPT (base model) via lm_eval.

All tasks use log-likelihood scoring, which is the correct method for a
pre-trained base model (no instruction/chat tuning).

Presets (use --preset instead of --tasks for convenience):
  quick      ~5-10 min   anli_r1, boolq, piqa, sciq, openbookqa
  standard   ~20-30 min  quick + hellaswag, arc_easy, arc_challenge,
                         winogrande, copa
  full       ~1-2 hrs    standard + mmlu, wikitext

Usage:
  # Quick sanity check
  python sparky_eval.py --ckpt sparky_data/synapse_2b_d2560_l28.pth \\
                        --tokenizer sparky_data/tokenizer.json \\
                        --preset quick --limit 50

  # Full evaluation suite
  python sparky_eval.py --ckpt sparky_data/synapse_2b_d2560_l28.pth \\
                        --tokenizer sparky_data/tokenizer.json \\
                        --preset full --output results/synapse_full.json

  # Custom tasks
  python sparky_eval.py --ckpt ... --tokenizer ... \\
                        --tasks arc_easy,arc_challenge,hellaswag

Pre-trained model benchmark rationale:
  - All tasks are scored via loglikelihood (not generation)
  - MMLU, ARC, HellaSwag, etc. are standard for base-model comparison
  - Generation-heavy tasks (GSM8K, HumanEval) are excluded —
    base models without instruction tuning cannot follow prompt formats
"""

import argparse
import json
import os
import sys
import time
import math
from collections import defaultdict
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval import simple_evaluate
from lm_eval.utils import make_table

from sparky_model import SynapseInfer, BLOCK_SIZE, VOCAB_SIZE
from tokenizers import Tokenizer


def load_tokenizer(path):
    tok = Tokenizer.from_file(path)
    eot_id = tok.token_to_id("<|endoftext|>") or 0
    return tok, eot_id


BUCKET_SIZE = 64
DEFAULT_BATCH_TOKENS = 4096  # safe for 16 GB VRAM (model ~4.2 GB + overhead)


TASK_PRESETS = {
    "quick": [
        "anli_r1",
        "boolq",
        "piqa",
        "sciq",
        "openbookqa",
    ],
    "standard": [
        "anli_r1",
        "boolq",
        "piqa",
        "sciq",
        "openbookqa",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "copa",
    ],
    "full": [
        "anli_r1",
        "boolq",
        "piqa",
        "sciq",
        "openbookqa",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "copa",
        "mmlu",
        "wikitext",
    ],
}


class SynapseEvalLM(LM):
    def __init__(self, ckpt_path, tokenizer_path, device="cuda", no_compile=False,
                 max_batch_tokens=DEFAULT_BATCH_TOKENS):
        super().__init__()
        self._device = torch.device(device)
        self.tok, self.eot_id = load_tokenizer(tokenizer_path)
        self.model, self._info = SynapseInfer.from_checkpoint(
            ckpt_path, device=device, no_compile=no_compile
        )
        self.model.eval()
        self._max_length = BLOCK_SIZE
        self._max_batch_tokens = max_batch_tokens
        self._info = self._info or {}

    @property
    def eot_token_id(self):
        return self.eot_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    @property
    def batch_size(self):
        return 1

    def loglikelihood(self, requests, disable_tqdm=False):
        tokenized = []
        for req in requests:
            ctx_str, cont_str = req.args
            ctx_ids = self.tok.encode(ctx_str).ids
            cont_ids = self.tok.encode(cont_str).ids
            if not ctx_ids:
                ctx_ids = [self.eot_id]
            if len(cont_ids) >= self._max_length:
                cont_ids = cont_ids[-(self._max_length - 1):]
            max_ctx_len = self._max_length - len(cont_ids)
            ctx_ids = ctx_ids[-max_ctx_len:]
            seq = ctx_ids + cont_ids
            tokenized.append((seq, len(ctx_ids), cont_ids))

        buckets = defaultdict(list)
        for i, (seq, _, _) in enumerate(tokenized):
            b = (len(seq) // BUCKET_SIZE) * BUCKET_SIZE
            buckets[b].append(i)

        results = [None] * len(requests)
        for bucket_min, indices in buckets.items():
            batch_n_tokens = 0
            batch_idxs = []
            for idx in indices:
                seq_len = len(tokenized[idx][0])
                if batch_n_tokens + seq_len > self._max_batch_tokens and batch_idxs:
                    self._eval_batch(tokenized, batch_idxs, results)
                    batch_idxs = []
                    batch_n_tokens = 0
                batch_idxs.append(idx)
                batch_n_tokens += seq_len
            if batch_idxs:
                self._eval_batch(tokenized, batch_idxs, results)
        return results

    def _eval_batch(self, tokenized, indices, results):
        max_len = max(len(tokenized[i][0]) for i in indices)
        B = len(indices)
        inp = torch.full((B, max_len), 0, dtype=torch.long, device=self._device)
        ctx_lens = []
        cont_ids_list = []
        for bpos, idx in enumerate(indices):
            seq, ctx_len, cont_ids = tokenized[idx]
            inp[bpos, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            ctx_lens.append(ctx_len)
            cont_ids_list.append(cont_ids)

        with torch.no_grad():
            logits, _ = self.model(inp, kv_cache=None, pos_offset=0)
            log_probs = F.log_softmax(logits, dim=-1)

        for bpos, idx in enumerate(indices):
            cs = ctx_lens[bpos]
            cont_ids = cont_ids_list[bpos]
            total_logprob = 0.0
            is_greedy = True
            logits_b = logits[bpos]
            for i, tid in enumerate(cont_ids):
                pred_pos = cs + i - 1
                if pred_pos < 0 or pred_pos >= max_len:
                    break
                total_logprob += log_probs[bpos, pred_pos, tid].item()
                if logits_b[pred_pos].argmax().item() != tid:
                    is_greedy = False
            results[idx] = (total_logprob, is_greedy)

        del inp, logits, log_probs
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def generate_until(self, requests, disable_tqdm=False):
        results = []
        iterator = tqdm(requests, desc="generate_until", disable=disable_tqdm, leave=False)
        for req in iterator:
            context_str, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen_tokens = gen_kwargs.get("max_gen_tokens", 256)
            ctx_ids = self.tok.encode(context_str).ids
            if len(ctx_ids) > self._max_length - max_gen_tokens:
                ctx_ids = ctx_ids[-(self._max_length - max_gen_tokens):]
            inp = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)
            generated = self._greedy_generate(inp, max_gen_tokens, until)
            results.append(generated)
        return results

    @torch.no_grad()
    def _greedy_generate(self, idx, max_new_tokens, until):
        kv_cache = None
        pos_offset = 0
        logits, kv_cache = self.model(idx, kv_cache, pos_offset)
        pos_offset = (pos_offset + idx.shape[1]) % BLOCK_SIZE
        output_ids = []
        for _ in range(max_new_tokens):
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            output_ids.append(next_token)
            text_so_far = self.tok.decode(output_ids)
            for stop in until:
                if stop in text_so_far:
                    idx_stop = text_so_far.index(stop)
                    return text_so_far[:idx_stop]
            idx_next = torch.tensor([[next_token]], dtype=torch.long, device=self._device)
            logits, kv_cache = self.model(idx_next, kv_cache, pos_offset)
            pos_offset = (pos_offset + 1) % BLOCK_SIZE
        return self.tok.decode(output_ids)

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        iterator = tqdm(requests, desc="loglikelihood_rolling", disable=disable_tqdm, leave=False)
        for req in iterator:
            (string,) = req.args
            ids = self.tok.encode(string).ids
            total_logprob = 0.0
            if len(ids) < 2:
                results.append(0.0)
                continue
            if len(ids) <= self._max_length:
                inp = torch.tensor([ids], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    logits, _ = self.model(inp)
                    log_probs = F.log_softmax(logits[0], dim=-1)
                for i in range(len(ids) - 1):
                    total_logprob += log_probs[i, ids[i + 1]].item()
            else:
                stride = self._max_length // 2
                target_start = 1
                while target_start < len(ids):
                    target_end = min(target_start + stride, len(ids))
                    context_start = max(0, target_end - self._max_length)
                    chunk = ids[context_start:target_end]
                    if len(chunk) < 2:
                        target_start = target_end
                        continue
                    inp = torch.tensor([chunk], dtype=torch.long, device=self._device)
                    with torch.no_grad():
                        logits, _ = self.model(inp)
                        log_probs = F.log_softmax(logits[0], dim=-1)
                    first_target = max(1, target_start - context_start)
                    for pos in range(first_target, len(chunk)):
                        total_logprob += log_probs[pos - 1, chunk[pos]].item()
                    target_start = target_end
            results.append(total_logprob)
        return results


def get_gpu_info(device):
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return name, total_mb
    return None, None


def human_readable_gb(value_bytes):
    return f"{value_bytes / 1e9:.2f} GB"


def print_checkpoint_info(info):
    if not info:
        return
    if info.get("step") is not None:
        print(f"  Checkpoint step: {info['step']}")
    if "eval_history" in info and info["eval_history"]:
        last = info["eval_history"][-1]
        print(f"  Final training eval loss: {last.get('overall') or last['loss']:.4f} @ step {last['step']}")
    params = info.get("total_params") or info.get("n_params")
    if params:
        print(f"  Parameters: {params / 1e9:.2f}B")


def main():
    parser = argparse.ArgumentParser(
        description="SynapseGPT base-model zero-shot evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets:\n  quick    " + ", ".join(TASK_PRESETS["quick"])
        + "\n  standard " + ", ".join(TASK_PRESETS["standard"])
        + "\n  full     " + ", ".join(TASK_PRESETS["full"]),
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--preset", choices=["quick", "standard", "full"],
                        help="Predefined task set (overrides --tasks)")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated task names (ignored if --preset is set)")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples (0 for zero-shot)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (for quick smoke tests)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (avoids recompilation for dynamic shapes)")
    parser.add_argument("--max-batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS,
                        help=f"Max tokens per batch (controls VRAM, default {DEFAULT_BATCH_TOKENS})")
    parser.add_argument("--output", default=None,
                        help="Save results JSON to path")
    args = parser.parse_args()

    if args.preset:
        task_list = TASK_PRESETS[args.preset]
    elif args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",")]
    else:
        parser.error("either --preset or --tasks is required")

    gpu_name, vram_mb = get_gpu_info(torch.device(args.device))

    print("=" * 60)
    print("SYNAPSE EVALUATION")
    print("=" * 60)
    print(f"  GPU:      {gpu_name or 'N/A'} ({vram_mb or 0:.0f} MB)")
    print(f"  Phase:    {args.preset or 'custom'}")
    print(f"  Tasks:    {', '.join(task_list)}")
    print(f"  Few-shot: {args.num_fewshot}")
    print(f"  Limit:    {args.limit or 'none (full)'}")
    print(f"  Batch:    {args.max_batch_tokens} tokens max")
    print(f"  Output:   {args.output or '(console only)'}")
    print()

    print(f"[eval] loading model from {args.ckpt}")
    t0 = time.time()
    lm = SynapseEvalLM(args.ckpt, args.tokenizer,
                       device=args.device, no_compile=args.no_compile,
                       max_batch_tokens=args.max_batch_tokens)
    load_time = time.time() - t0
    vram_used = torch.cuda.memory_allocated() if args.device == "cuda" else 0
    print(f"[eval] model loaded in {load_time:.1f}s")
    print(f"[eval] VRAM used: {human_readable_gb(vram_used)}")
    print_checkpoint_info(lm._info)

    print(f"\n[eval] running {len(task_list)} task(s) ...")
    t0 = time.time()
    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        bootstrap_iters=0,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )
    elapsed = time.time() - t0

    if results is None:
        print("[eval] no results returned (task loading failed?)")
        sys.exit(1)

    table = make_table(results)
    print()
    print(table)
    print(f"\n[eval] completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out = {
            "config": {
                "ckpt": args.ckpt,
                "preset": args.preset,
                "tasks": task_list,
                "num_fewshot": args.num_fewshot,
                "limit": args.limit,
                "max_batch_tokens": args.max_batch_tokens,
                "gpu": gpu_name,
                "vram_mb": vram_mb,
                "checkpoint_size_bytes": os.path.getsize(args.ckpt),
                "checkpoint_step": lm._info.get("step"),
                "checkpoint_last_eval_loss": lm._info.get("last_eval_loss"),
                "load_time_s": round(load_time, 1),
                "eval_time_s": round(elapsed, 1),
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[eval] results saved to {args.output}")


if __name__ == "__main__":
    main()
