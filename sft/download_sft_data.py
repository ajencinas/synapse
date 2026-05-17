#!/usr/bin/env python3
"""Download SFT datasets from HuggingFace → normalized raw JSONL.

Usage:
  SYNAPSE_DIR=/path/to/synapse python sft/download_sft_data.py --datasets alpaca

Output layout (under $SYNAPSE_DIR/datasets_sft/<name>/):
  <name>_raw.jsonl   one {"instruction","input","output"} per line
  meta_raw.json      HF revision, raw/kept counts, drop reasons

Idempotent: skips a dataset if its <name>_raw.jsonl already exists.
"""
import argparse
import json
import os
import sys

from datasets import load_dataset


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def adapter_alpaca(row):
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    if not instr or not out:
        return None
    return {"instruction": instr, "input": inp, "output": out}


DATASETS = {
    "alpaca": {
        "hf_path": "tatsu-lab/alpaca",
        "hf_revision": "main",
        "split": "train",
        "adapter": adapter_alpaca,
    },
}


def process(name, spec, out_dir):
    raw_path = os.path.join(out_dir, f"{name}_raw.jsonl")
    meta_path = os.path.join(out_dir, "meta_raw.json")
    if os.path.exists(raw_path):
        print(f"[{name}] skip — {raw_path} already exists")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"[{name}] loading {spec['hf_path']} @ {spec['hf_revision']} ({spec['split']})")
    ds = load_dataset(spec["hf_path"], revision=spec["hf_revision"], split=spec["split"])

    seen = set()
    kept = []
    drops = {"adapter_rejected": 0, "too_short": 0, "duplicate": 0}
    for row in ds:
        norm = spec["adapter"](row)
        if norm is None:
            drops["adapter_rejected"] += 1
            continue
        total_chars = len(norm["instruction"]) + len(norm["input"]) + len(norm["output"])
        if total_chars < 10:
            drops["too_short"] += 1
            continue
        key = (norm["instruction"], norm["input"], norm["output"])
        if key in seen:
            drops["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(norm)

    tmp_path = raw_path + ".tmp"
    with open(tmp_path, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp_path, raw_path)

    meta = {
        "dataset": name,
        "hf_path": spec["hf_path"],
        "hf_revision": spec["hf_revision"],
        "split": spec["split"],
        "raw_count": len(ds),
        "kept": len(kept),
        "drops": drops,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{name}] wrote {len(kept):,} examples → {raw_path}")
    print(f"[{name}] drops: {drops}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="comma-separated dataset names, or 'all'")
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR") or default_synapse_dir())
    args = ap.parse_args()

    names = list(DATASETS.keys()) if args.datasets == "all" else args.datasets.split(",")
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}. known: {list(DATASETS)}")

    base = os.path.join(args.synapse_dir, "datasets_sft")
    print(f"output base: {base}")
    for name in names:
        process(name, DATASETS[name], os.path.join(base, name))


if __name__ == "__main__":
    main()
