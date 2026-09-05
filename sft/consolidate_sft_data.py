#!/usr/bin/env python3
"""Consolidate tokenized SFT sources → manifests/sft_data_registry.json.

Scans every $SYNAPSE_DIR/sft_tokenized/<name>/ directory directly (so the
registry always reflects what's actually on disk, regardless of incremental
tokenization), and writes a single registry analogous to pretrain's
shard_manifest.json.

Per-source val examples stay in their own sft_tokenized/<name>/val.jsonl; the
directory name IS the source tag, recorded as `val_path` so training can track
per-source response-token val loss.

Fail-loud: aborts if sources disagree on tokenization_id or block_size.

Usage:
  SYNAPSE_DIR=/path/to/synapse python sft/consolidate_sft_data.py
"""
import argparse
import json
import os


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synapse-dir",
                    default=os.environ.get("SYNAPSE_DIR") or default_synapse_dir())
    args = ap.parse_args()

    syn = args.synapse_dir
    tok_base = os.path.join(syn, "sft_tokenized")
    raw_base = os.path.join(syn, "datasets_sft")
    manifest_dir = os.path.join(syn, "manifests")
    if not os.path.isdir(tok_base):
        raise SystemExit(f"no tokenized data at {tok_base} — run tokenize_sft_data.py first")

    names = sorted(n for n in os.listdir(tok_base)
                   if os.path.isdir(os.path.join(tok_base, n)))
    if not names:
        raise SystemExit(f"no tokenized sources under {tok_base}")

    datasets = {}
    tok_ids, block_sizes, split_rules = set(), set(), set()
    for name in names:
        meta_path = os.path.join(tok_base, name, "meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit(f"[{name}] missing meta.json — retokenize this source")
        with open(meta_path) as f:
            meta = json.load(f)
        tok_ids.add(meta.get("tokenization_id"))
        block_sizes.add(meta.get("block_size"))
        split_rules.add(meta.get("split_rule"))

        source = None
        raw_meta_path = os.path.join(raw_base, name, "meta_raw.json")
        if os.path.exists(raw_meta_path):
            with open(raw_meta_path) as f:
                source = json.load(f).get("hf_path")

        datasets[name] = {
            "source": source,
            "kept": meta.get("kept"),
            "train_count": meta.get("train_count"),
            "val_count": meta.get("val_count"),
            "val_path": os.path.join("sft_tokenized", name, "val.jsonl"),
            "drops": meta.get("drops"),
            "stats": {
                "total_len": meta.get("total_len_stats"),
                "response_len": meta.get("response_len_stats"),
            },
        }

    if len(tok_ids) != 1:
        raise SystemExit(f"sources disagree on tokenization_id: {tok_ids} — retokenize all")
    if len(block_sizes) != 1:
        raise SystemExit(f"sources disagree on block_size: {block_sizes} — retokenize all")
    if len(split_rules) != 1 or None in split_rules:
        raise SystemExit(
            f"sources disagree on train/val split rule: {split_rules} — a None "
            f"means a source was tokenized before the global hash split "
            f"(SFT v2 Phase 1.1); retokenize it (tokenize_sft_data.py will "
            f"auto-detect it as stale)")

    registry = {
        "tokenization_id": next(iter(tok_ids)),
        "block_size": next(iter(block_sizes)),
        "split_rule": next(iter(split_rules)),
        "datasets": datasets,
        "total_train": sum(d["train_count"] or 0 for d in datasets.values()),
        "total_val": sum(d["val_count"] or 0 for d in datasets.values()),
    }

    os.makedirs(manifest_dir, exist_ok=True)
    out_path = os.path.join(manifest_dir, "sft_data_registry.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, out_path)

    print(f"registry: {out_path}")
    print(f"sources: {len(datasets)}  "
          f"total_train={registry['total_train']:,}  total_val={registry['total_val']:,}")
    for name, d in datasets.items():
        print(f"  {name:14s} train={d['train_count']:>7,} val={d['val_count']:>5,} "
              f"resp_p95={(d['stats']['response_len'] or {}).get('p95')}")


if __name__ == "__main__":
    main()
