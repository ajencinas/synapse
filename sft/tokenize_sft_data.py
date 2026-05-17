#!/usr/bin/env python3
"""Tokenize raw SFT JSONL → ChatML-formatted token IDs with prefix_len mask.

Usage:
  SYNAPSE_DIR=/path/to/synapse python sft/tokenize_sft_data.py \
      --datasets alpaca --block-size 2048 --val-fraction 0.02

Reads:  $SYNAPSE_DIR/datasets_sft/<name>/<name>_raw.jsonl
        $SYNAPSE_DIR/tokenizer_out/tokenizer.json
        $SYNAPSE_DIR/manifests/tokenization_latest.json  (for tokenization_id check)

Writes: $SYNAPSE_DIR/sft_tokenized/<name>/train.jsonl   {"input_ids":[...],"prefix_len":N}
        $SYNAPSE_DIR/sft_tokenized/<name>/val.jsonl
        $SYNAPSE_DIR/sft_tokenized/<name>/meta.json
        $SYNAPSE_DIR/sft_tokenized/tokenization_id.txt
        $SYNAPSE_DIR/manifests/sft_tokenization_latest.json

Refuses to run if the tokenizer's tokenization_id doesn't match pretrain's
(prevents accidentally SFT-ing with a tokenizer the model never saw).

ChatML boundary chosen so BPE cannot merge across prompt/response:
  prompt   = "<|im_start|>user\\n{instruction}\\n\\n{input}\\n<|im_end|>\\n<|im_start|>assistant"
  response = "\\n{output}<|im_end|><|endoftext|>"
The prompt ends on an atomic special token; the response carries the
following newline as its first byte.
"""
import argparse
import hashlib
import json
import os
import random
import statistics

from tokenizers import Tokenizer


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def tokenizer_id(tokenizer_path):
    h = hashlib.sha256()
    with open(tokenizer_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_prompt(instruction, inp):
    if inp:
        body = f"{instruction}\n\n{inp}"
    else:
        body = instruction
    return (
        f"<|im_start|>user\n{body}\n<|im_end|>\n"
        f"<|im_start|>assistant"
    )


def build_response(output):
    return f"\n{output}<|im_end|><|endoftext|>"


def encode_one(tokenizer, instruction, inp, output, block_size):
    prompt = build_prompt(instruction, inp)
    response = build_response(output)
    prompt_ids = tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(response).ids
    full_ids = prompt_ids + response_ids
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "BPE merged across prompt/response boundary — template bug. "
            "Prompt should end on an atomic special token."
        )
    return full_ids, len(prompt_ids), len(response_ids)


def process(name, raw_dir, out_dir, tokenizer, block_size, val_fraction, seed):
    raw_path = os.path.join(raw_dir, f"{name}_raw.jsonl")
    if not os.path.exists(raw_path):
        raise SystemExit(f"[{name}] missing raw file: {raw_path} — run download_sft_data.py first")

    os.makedirs(out_dir, exist_ok=True)
    with open(raw_path) as f:
        raw_rows = [json.loads(line) for line in f]

    kept = []
    drops = {"short_response": 0, "prompt_too_long": 0, "truncation_lost_eos": 0}
    prefix_lens, response_lens = [], []
    for row in raw_rows:
        full_ids, prefix_len, response_len = encode_one(
            tokenizer, row["instruction"], row["input"], row["output"], block_size,
        )
        if response_len < 3:
            drops["short_response"] += 1
            continue
        if prefix_len >= block_size:
            drops["prompt_too_long"] += 1
            continue
        if len(full_ids) > block_size + 1:
            # Truncate from the response tail; if that strips <|endoftext|>, drop the example.
            full_ids = full_ids[: block_size + 1]
            if full_ids[-1] != tokenizer.token_to_id("<|endoftext|>"):
                drops["truncation_lost_eos"] += 1
                continue
        kept.append({"input_ids": full_ids, "prefix_len": prefix_len})
        prefix_lens.append(prefix_len)
        response_lens.append(response_len)

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * val_fraction))
    val = kept[:n_val]
    train = kept[n_val:]

    def write_jsonl(path, rows):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)

    def pcts(xs):
        if not xs:
            return {}
        xs_s = sorted(xs)
        n = len(xs_s)
        return {
            "p50": xs_s[n // 2],
            "p95": xs_s[min(n - 1, int(0.95 * n))],
            "p99": xs_s[min(n - 1, int(0.99 * n))],
            "max": xs_s[-1],
            "mean": round(statistics.mean(xs_s), 1),
        }

    meta = {
        "dataset": name,
        "raw_count": len(raw_rows),
        "kept": len(kept),
        "train_count": len(train),
        "val_count": len(val),
        "drops": drops,
        "block_size": block_size,
        "val_fraction": val_fraction,
        "seed": seed,
        "prefix_len_stats": pcts(prefix_lens),
        "response_len_stats": pcts(response_lens),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{name}] kept {len(kept):,} (train={len(train):,}, val={len(val):,}) "
          f"drops={drops} prefix_p95={meta['prefix_len_stats'].get('p95')} "
          f"response_p95={meta['response_len_stats'].get('p95')}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated names, or 'all'")
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR") or default_synapse_dir())
    args = ap.parse_args()

    syn = args.synapse_dir
    raw_base = os.path.join(syn, "datasets_sft")
    out_base = os.path.join(syn, "sft_tokenized")
    manifest_dir = os.path.join(syn, "manifests")
    tokenizer_path = os.path.join(syn, "tokenizer_out", "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        raise SystemExit(f"tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tok_id = tokenizer_id(tokenizer_path)

    pretrain_manifest = os.path.join(manifest_dir, "tokenization_latest.json")
    if not os.path.exists(pretrain_manifest):
        raise SystemExit(
            f"pretrain tokenization manifest missing: {pretrain_manifest} "
            f"— can't verify tokenization_id match"
        )
    with open(pretrain_manifest) as f:
        pretrain_meta = json.load(f)
    pretrain_tok_id = pretrain_meta["tokenization_id"]
    if tok_id != pretrain_tok_id:
        raise SystemExit(
            f"tokenization_id mismatch:\n"
            f"  tokenizer.json fingerprint: {tok_id}\n"
            f"  pretrain manifest:          {pretrain_tok_id}\n"
            f"refusing to write — the model was trained on a different tokenizer"
        )

    if args.datasets == "all":
        names = sorted(
            n for n in os.listdir(raw_base)
            if os.path.isdir(os.path.join(raw_base, n))
        )
    else:
        names = args.datasets.split(",")

    os.makedirs(out_base, exist_ok=True)
    metas = {}
    for name in names:
        out_dir = os.path.join(out_base, name)
        metas[name] = process(
            name=name,
            raw_dir=os.path.join(raw_base, name),
            out_dir=out_dir,
            tokenizer=tokenizer,
            block_size=args.block_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

    with open(os.path.join(out_base, "tokenization_id.txt"), "w") as f:
        f.write(tok_id + "\n")

    os.makedirs(manifest_dir, exist_ok=True)
    sft_manifest = {
        "stage": "sft_tokenization",
        "tokenization_id": tok_id,
        "block_size": args.block_size,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "datasets": metas,
    }
    manifest_path = os.path.join(manifest_dir, "sft_tokenization_latest.json")
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sft_manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
