#!/usr/bin/env python3
"""
Order-preserving streaming merger for tokenizer .bin shards.

Walks shards in their original manifest order, groups by domain (parent dir of
the source file), and concatenates consecutive shards into ~target-size buckets.
Source order is preserved within each domain — the only operation is "cut here,
start a new shard." No reordering, no bin-packing.

Why: the upstream tokenizer writes one shard per source .txt, and the source
files vary wildly in size (a 5MB shard sits next to a 800MB one). Per-shard
training iteration then mis-weights data mix and produces poor intra-shard
shuffles. Merging gives uniform shards without retokenizing.

Output dir mirrors the input layout (shard_*.bin + shard_manifest.json +
meta.json + tokenization_id.txt) so the trainer can point at it transparently.

Usage:
  python merge_shards.py                         # in: $TOKENIZER_SHARD_DIR -> <in>_merged
  python merge_shards.py --target-bytes 200000000
  python merge_shards.py --in /mnt/ssd/token_shards --out /mnt/ssd/token_shards_merged
  python merge_shards.py --dry-run               # print plan only
"""

import os
import sys
import json
import shutil
import hashlib
import argparse

DEFAULT_TARGET_BYTES = 100 * 1024 * 1024   # 100 MB
COPY_CHUNK_BYTES     = 4 * 1024 * 1024      # 4 MB stream copy

DTYPE_BYTES = {"uint8": 1, "uint16": 2, "uint32": 4, "int8": 1, "int16": 2, "int32": 4}

# ---------- Helpers ----------
def parent_dir_name(path):
    return os.path.basename(os.path.dirname(path))


def fingerprint(target_bytes, source_entries):
    h = hashlib.sha256()
    h.update(str(target_bytes).encode())
    for s in source_entries:
        h.update(b"|")
        h.update((s.get("source") or "").encode())
        h.update(b"|")
        h.update(str(s.get("source_size", "")).encode())
        h.update(b"|")
        h.update((s.get("source_hash") or "").encode())
        h.update(b"|")
        h.update(str(s.get("tokens", "")).encode())
    return h.hexdigest()[:16]


def plan_merge(input_manifest, target_bytes, dtype_bytes):
    """
    Group shards by domain, preserve manifest order, bucket by accumulated bytes.
    Cut policy: flush after first shard that pushes bucket_bytes >= target.
    Returns list of {"domain", "sources": [shard entries], "bytes": int}.
    """
    by_domain = {}
    domain_order = []
    for s in input_manifest.get("shards", []):
        dom = parent_dir_name(s["source"])
        if dom not in by_domain:
            by_domain[dom] = []
            domain_order.append(dom)
        by_domain[dom].append(s)

    plan = []
    for dom in domain_order:
        bucket = []
        bucket_bytes = 0
        for s in by_domain[dom]:
            shard_bytes = s["tokens"] * dtype_bytes
            bucket.append(s)
            bucket_bytes += shard_bytes
            if bucket_bytes >= target_bytes:
                plan.append({"domain": dom, "sources": bucket, "bytes": bucket_bytes})
                bucket = []
                bucket_bytes = 0
        if bucket:
            plan.append({"domain": dom, "sources": bucket, "bytes": bucket_bytes})
    return plan


def print_plan_summary(plan, target_bytes):
    by_dom = {}
    for p in plan:
        by_dom.setdefault(p["domain"], []).append(p)
    print(f"Plan: {len(plan)} merged shards (target {target_bytes / 1024 / 1024:.0f} MB each)")
    for dom, ps in by_dom.items():
        sizes_mb = sorted(p["bytes"] / 1024 / 1024 for p in ps)
        toks = sum(s["tokens"] for p in ps for s in p["sources"])
        srcs = sum(len(p["sources"]) for p in ps)
        print(
            f"  {dom}: {len(ps)} merged shards from {srcs} sources, "
            f"{toks:,} tokens | "
            f"min={sizes_mb[0]:.1f}MB p50={sizes_mb[len(sizes_mb) // 2]:.1f}MB "
            f"max={sizes_mb[-1]:.1f}MB"
        )


def merge_shards(input_dir, output_dir, target_bytes, dry_run=False, force=False):
    """
    Stream-merge .bin shards from input_dir into ~target_bytes buckets in output_dir.
    Returns the new shard_manifest dict (or None if dry_run / nothing to do).
    """
    in_manifest_path = os.path.join(input_dir, "shard_manifest.json")
    in_meta_path     = os.path.join(input_dir, "meta.json")
    in_tokid_path    = os.path.join(input_dir, "tokenization_id.txt")

    if not os.path.exists(in_manifest_path):
        sys.exit(f"No shard_manifest.json at {input_dir}")
    if not os.path.exists(in_meta_path):
        sys.exit(f"No meta.json at {input_dir}")

    with open(in_manifest_path) as f:
        in_manifest = json.load(f)
    with open(in_meta_path) as f:
        in_meta = json.load(f)

    dtype_name = in_meta.get("shard_dtype", "uint16")
    if dtype_name not in DTYPE_BYTES:
        sys.exit(f"Unknown shard_dtype {dtype_name!r}; add it to DTYPE_BYTES.")
    dtype_bytes = DTYPE_BYTES[dtype_name]

    print(f"Input:  {len(in_manifest.get('shards', []))} shards at {input_dir} ({dtype_name})")
    print(f"Output: {output_dir}")

    plan = plan_merge(in_manifest, target_bytes, dtype_bytes)
    print_plan_summary(plan, target_bytes)

    if dry_run:
        print("DRY RUN: no files written.")
        return None

    fp = fingerprint(target_bytes, in_manifest.get("shards", []))
    out_manifest_path = os.path.join(output_dir, "shard_manifest.json")
    if not force and os.path.exists(out_manifest_path):
        try:
            with open(out_manifest_path) as f:
                existing = json.load(f)
            if existing.get("merge_fingerprint") == fp:
                print("Output already matches input fingerprint — nothing to do.")
                return existing
        except Exception:
            pass

    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".bin"):
            os.remove(os.path.join(output_dir, f))

    out_shards = []
    for idx, p in enumerate(plan):
        shard_name = f"shard_{idx:05d}.bin"
        out_path = os.path.join(output_dir, shard_name)
        expected_bytes = 0
        total_tokens = 0
        total_docs = 0
        with open(out_path, "wb") as fout:
            for s in p["sources"]:
                src_bin = os.path.join(input_dir, s["shard"])
                if not os.path.exists(src_bin):
                    sys.exit(f"Missing input shard: {src_bin}")
                with open(src_bin, "rb") as fin:
                    shutil.copyfileobj(fin, fout, length=COPY_CHUNK_BYTES)
                expected_bytes += s["tokens"] * dtype_bytes
                total_tokens   += s["tokens"]
                total_docs     += s.get("documents", 0)
        actual_bytes = os.path.getsize(out_path)
        if actual_bytes != expected_bytes:
            sys.exit(
                f"Byte mismatch for {shard_name}: "
                f"expected {expected_bytes} (sum of input shard sizes), got {actual_bytes}"
            )

        # Synthetic source path lets the trainer keep using
        # `os.path.basename(os.path.dirname(source))` to recover the domain.
        synthetic_source = os.path.join(
            os.path.dirname(p["sources"][0]["source"]),
            f"_merged_{idx:05d}",
        )
        out_shards.append({
            "shard": shard_name,
            "source": synthetic_source,
            "domain": p["domain"],
            "tokens": total_tokens,
            "documents": total_docs,
            "shard_mb": round(actual_bytes / 1024 / 1024, 2),
            "merged_from": [
                {"shard": s["shard"], "source": s["source"], "tokens": s["tokens"]}
                for s in p["sources"]
            ],
        })
        print(
            f"  [{idx + 1}/{len(plan)}] {shard_name} ({p['domain']}): "
            f"{total_tokens:,} tokens, {actual_bytes / 1024 / 1024:.1f} MB, "
            f"{len(p['sources'])} sources"
        )

    out_manifest = {
        "tokenization_id":     in_manifest.get("tokenization_id"),
        "merged_from":         input_dir,
        "merge_target_bytes":  target_bytes,
        "merge_fingerprint":   fp,
        "shards":              out_shards,
    }
    tmp = out_manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out_manifest, f, indent=2)
    os.replace(tmp, out_manifest_path)

    out_meta = {
        "num_tokens":         sum(s["tokens"] for s in out_shards),
        "num_shards":         len(out_shards),
        "vocab_size":         in_meta["vocab_size"],
        "eot_id":             in_meta["eot_id"],
        "pad_id":             in_meta["pad_id"],
        "tokenizer":          in_meta.get("tokenizer", "byte-level-bpe-digits"),
        "shard_dtype":        dtype_name,
        "tokenization_id":    in_meta.get("tokenization_id"),
        "merged_from":        input_dir,
        "merge_target_bytes": target_bytes,
    }
    out_meta_path = os.path.join(output_dir, "meta.json")
    tmp = out_meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out_meta, f, indent=2)
    os.replace(tmp, out_meta_path)

    if os.path.exists(in_tokid_path):
        shutil.copyfile(in_tokid_path, os.path.join(output_dir, "tokenization_id.txt"))

    print(
        f"\nMerged: {len(out_shards)} shards, "
        f"{sum(s['tokens'] for s in out_shards):,} tokens -> {output_dir}"
    )
    return out_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Order-preserving streaming merger for tokenizer .bin shards."
    )
    parser.add_argument(
        "--in", dest="in_dir",
        default=os.environ.get("TOKENIZER_SHARD_DIR", "/mnt/ssd/token_shards"),
        help="Input shard dir (default: $TOKENIZER_SHARD_DIR or /mnt/ssd/token_shards)",
    )
    parser.add_argument(
        "--out", dest="out_dir", default=None,
        help="Output dir (default: <in>_merged)",
    )
    parser.add_argument(
        "--target-bytes", type=int, default=DEFAULT_TARGET_BYTES,
        help=f"Target merged shard size in bytes (default: {DEFAULT_TARGET_BYTES} = 100 MB)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't write")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild merged shards even if fingerprint matches")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.in_dir.rstrip("/") + "_merged")
    merge_shards(args.in_dir, out_dir, args.target_bytes,
                 dry_run=args.dry_run, force=args.force)
