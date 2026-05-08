#!/usr/bin/env python3
"""
Triple-check tokenizer outputs before merging or training.

Runs four independent pre-merge sanity checks against a shard dir:
  1. Size:          shard_bytes == tokens * dtype_bytes for every entry
  2. JSON:          manifest, meta, tokenization_id are well-formed and consistent
  3. Tokenization:  random N-shard sample, first 10 shard tokens re-derive
                    from re-tokenizing the source's first non-empty document
  4. Merge-ready:   shard names unique, sources unique, files present, byte
                    alignment correct, every source has a domain (parent dir)

Exits 0 if all checks pass. Exits 1 if anything fails. Print a per-domain
summary plus first 5 examples of each failure category.

Stdlib only except for the `tokenizers` library, which is needed for check 3
(skipped with a warning if not installed or tokenizer.json missing).

Usage:
  python validate_shards.py                           # all defaults
  python validate_shards.py --sample 5000             # bigger spot-check
  python validate_shards.py --tokenizer /path/to/tokenizer.json
  python validate_shards.py --dir /mnt/ssd/token_shards
"""

import os
import sys
import json
import struct
import random
import argparse
from collections import defaultdict

DTYPE_BYTES = {"uint8": 1, "uint16": 2, "uint32": 4, "int8": 1, "int16": 2, "int32": 4}
STRUCT_FMT  = {1: "B", 2: "H", 4: "I"}


# ---------- Helpers ----------
def _domain(source_path: str) -> str:
    return os.path.basename(os.path.dirname(source_path or ""))


def _read_first_tokens(bin_path: str, n: int, dtype_bytes: int):
    n_bytes = n * dtype_bytes
    with open(bin_path, "rb") as f:
        data = f.read(n_bytes)
    if len(data) < n_bytes:
        return None
    return list(struct.unpack(f"<{n}{STRUCT_FMT[dtype_bytes]}", data))


# ---------- Check 1: Size ----------
def check_size(shards, shard_dir, dtype_bytes):
    issues = []
    domain_stats = defaultdict(lambda: {"ok": 0, "mismatch": 0, "missing": 0})
    for s in shards:
        dom = _domain(s.get("source", ""))
        bin_path = os.path.join(shard_dir, s.get("shard", ""))
        expected = s.get("tokens", 0) * dtype_bytes
        if not os.path.exists(bin_path):
            domain_stats[dom]["missing"] += 1
            issues.append((s.get("shard"), s.get("source"), "MISSING", expected, -1))
            continue
        actual = os.path.getsize(bin_path)
        if actual == expected:
            domain_stats[dom]["ok"] += 1
        else:
            domain_stats[dom]["mismatch"] += 1
            issues.append((s.get("shard"), s.get("source"), "MISMATCH", expected, actual))
    return issues, domain_stats


# ---------- Check 2: JSON structure ----------
def check_json(manifest, meta, tok_id_file, manifest_path, meta_path, tok_id_path):
    issues = []
    if not isinstance(manifest, dict):
        return [f"manifest is not a JSON object ({manifest_path})"]
    for k in ("tokenization_id", "shards"):
        if k not in manifest:
            issues.append(f"manifest missing top-level key '{k}'")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        issues.append("manifest['shards'] is not a list")
        return issues

    required = {
        "shard":       (str,),
        "source":      (str,),
        "tokens":      (int,),
        "source_size": (int,),
    }
    for i, s in enumerate(shards):
        if not isinstance(s, dict):
            issues.append(f"shards[{i}] is not a dict")
            continue
        for field, types in required.items():
            if field not in s:
                issues.append(f"shards[{i}] ({s.get('shard','?')}) missing field '{field}'")
                continue
            v = s[field]
            if v is None:
                issues.append(f"shards[{i}] ({s.get('shard','?')}) field '{field}' is null")
                continue
            if not isinstance(v, types):
                issues.append(f"shards[{i}] ({s.get('shard','?')}) field '{field}' "
                              f"is {type(v).__name__}, expected {types[0].__name__}")
        if "tokens" in s and isinstance(s["tokens"], int) and s["tokens"] < 0:
            issues.append(f"shards[{i}] tokens={s['tokens']} is negative")
        if "source_size" in s and isinstance(s["source_size"], int) and s["source_size"] < 0:
            issues.append(f"shards[{i}] source_size={s['source_size']} is negative")

    if meta is None:
        issues.append(f"meta.json missing or invalid at {meta_path}")
    else:
        for k in ("num_tokens", "num_shards", "vocab_size", "shard_dtype"):
            if k not in meta:
                issues.append(f"meta missing key '{k}'")
        if "num_shards" in meta and meta["num_shards"] != len(shards):
            issues.append(
                f"meta.num_shards ({meta['num_shards']}) != len(manifest.shards) ({len(shards)})"
            )
        if "num_tokens" in meta:
            mfst_tokens = sum(s.get("tokens", 0) for s in shards if isinstance(s, dict))
            if meta["num_tokens"] != mfst_tokens:
                issues.append(
                    f"meta.num_tokens ({meta['num_tokens']:,}) != sum(manifest tokens) ({mfst_tokens:,})"
                )

    if tok_id_file is None:
        issues.append(f"tokenization_id.txt missing at {tok_id_path}")
    else:
        if "tokenization_id" in manifest and tok_id_file != manifest["tokenization_id"]:
            issues.append(
                f"tokenization_id.txt ({tok_id_file}) != manifest.tokenization_id "
                f"({manifest['tokenization_id']})"
            )
        if meta and "tokenization_id" in meta and tok_id_file != meta["tokenization_id"]:
            issues.append(
                f"tokenization_id.txt ({tok_id_file}) != meta.tokenization_id "
                f"({meta['tokenization_id']})"
            )

    return issues


# ---------- Check 3: tokenization spot-check ----------
def load_tokenizer(path):
    if not os.path.exists(path):
        return None, f"tokenizer.json not found at {path}"
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None, "Python 'tokenizers' library not installed"
    try:
        return Tokenizer.from_file(path), None
    except Exception as e:
        return None, f"failed to load tokenizer.json: {e}"


def check_tokenization_sample(shards, shard_dir, tokenizer, dtype_bytes, eot_token,
                              n_sample, seed):
    rng = random.Random(seed)
    pool = [s for s in shards
            if os.path.exists(os.path.join(shard_dir, s.get("shard", "")))
               and os.path.exists(s.get("source", ""))]
    sample = rng.sample(pool, min(n_sample, len(pool)))

    SAMPLE_TOKENS = 10
    SOURCE_PREFIX_CHARS = 2000
    failures = []
    skipped = 0

    for s in sample:
        bin_path = os.path.join(shard_dir, s["shard"])
        src_path = s["source"]

        shard_tokens = _read_first_tokens(bin_path, SAMPLE_TOKENS, dtype_bytes)
        if shard_tokens is None:
            failures.append({
                "shard": s["shard"], "source": src_path,
                "reason": f"shard smaller than {SAMPLE_TOKENS} tokens",
            })
            continue

        try:
            with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(SOURCE_PREFIX_CHARS)
        except Exception as e:
            skipped += 1
            continue

        # Mirror _tokenize_one in run_tokenizer.py: split on EOT, strip,
        # take first non-empty document.
        first_doc = None
        for doc in text.split(eot_token):
            doc = doc.strip()
            if doc:
                first_doc = doc[:500]   # plenty for 10 tokens
                break
        if first_doc is None:
            skipped += 1
            continue

        encoded = tokenizer.encode(first_doc)
        expected = list(encoded.ids[:SAMPLE_TOKENS])
        cmp_len  = min(len(expected), SAMPLE_TOKENS)
        if expected[:cmp_len] == shard_tokens[:cmp_len]:
            continue

        failures.append({
            "shard": s["shard"], "source": src_path,
            "reason": (f"first {cmp_len} tokens of shard {shard_tokens[:cmp_len]} "
                       f"!= tokenize(source[:500])[:{cmp_len}] {expected[:cmp_len]}"),
        })

    return {
        "sampled":  len(sample),
        "pool":     len(pool),
        "failed":   len(failures),
        "skipped":  skipped,
        "failures": failures,
    }


# ---------- Check 4: merge-ready structure ----------
def check_merge_structure(manifest, shard_dir, dtype_bytes):
    issues = []
    shards = manifest.get("shards", [])

    # Shard name uniqueness
    name_counts = defaultdict(list)
    for s in shards:
        name_counts[s.get("shard")].append(s.get("source"))
    name_dups = {n: srcs for n, srcs in name_counts.items() if len(srcs) > 1}
    if name_dups:
        examples = list(name_dups.items())[:3]
        issues.append(
            f"{len(name_dups)} shard name(s) appear in multiple manifest entries. "
            f"Examples: {[(n, len(s)) for n, s in examples]}"
        )

    # Source uniqueness
    src_counts = defaultdict(int)
    for s in shards:
        src_counts[s.get("source")] += 1
    src_dups = [n for n, c in src_counts.items() if c > 1]
    if src_dups:
        issues.append(
            f"{len(src_dups)} source(s) appear in multiple manifest entries. "
            f"Examples: {src_dups[:3]}"
        )

    # Domain attribution
    no_domain = [s.get("source") for s in shards if not _domain(s.get("source", ""))]
    if no_domain:
        issues.append(
            f"{len(no_domain)} source(s) have no parent dir — merger can't determine domain. "
            f"Examples: {no_domain[:3]}"
        )

    # Files present + alignment
    missing  = []
    misalign = []
    for s in shards:
        bin_path = os.path.join(shard_dir, s.get("shard", ""))
        if not os.path.exists(bin_path):
            missing.append(s.get("shard"))
            continue
        size = os.path.getsize(bin_path)
        if size % dtype_bytes != 0:
            misalign.append((s.get("shard"), size))
    if missing:
        issues.append(f"{len(missing)} shard .bin files missing. Examples: {missing[:3]}")
    if misalign:
        issues.append(
            f"{len(misalign)} shard .bin files are not aligned to dtype ({dtype_bytes} bytes). "
            f"Examples: {misalign[:3]}"
        )

    # Orphans on disk (not in manifest) — these would also break the merger
    referenced = {s.get("shard") for s in shards}
    on_disk    = {f for f in os.listdir(shard_dir)
                  if f.startswith("shard_") and f.endswith(".bin")
                     and os.path.isfile(os.path.join(shard_dir, f))}
    orphans    = sorted(on_disk - referenced)
    if orphans:
        issues.append(
            f"{len(orphans)} orphan .bin file(s) on disk not in manifest. "
            f"Examples: {orphans[:3]}"
        )

    return issues


# ---------- Driver ----------
def main():
    parser = argparse.ArgumentParser(description="Triple-check shard dir before merging.")
    parser.add_argument("--dir",
                        default=os.environ.get("TOKENIZER_SHARD_DIR", "/mnt/ssd/token_shards"),
                        help="Shard dir (default: $TOKENIZER_SHARD_DIR or /mnt/ssd/token_shards)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json (default: $TOKENIZER_OUT_DIR/tokenizer.json)")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Random sample size for tokenization spot-check (default 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Sample seed")
    parser.add_argument("--eot-token", default="<|endoftext|>",
                        help="EOT separator string in source files (default: <|endoftext|>)")
    args = parser.parse_args()

    shard_dir      = args.dir
    tokenizer_path = args.tokenizer or os.path.join(
        os.environ.get("TOKENIZER_OUT_DIR", "/mnt/ssd/tokenizer_out"), "tokenizer.json"
    )
    manifest_path = os.path.join(shard_dir, "shard_manifest.json")
    meta_path     = os.path.join(shard_dir, "meta.json")
    tok_id_path   = os.path.join(shard_dir, "tokenization_id.txt")

    if not os.path.exists(manifest_path):
        sys.exit(f"No shard_manifest.json at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  WARNING: meta.json could not be parsed: {e}")
    tok_id_file = None
    if os.path.exists(tok_id_path):
        with open(tok_id_path) as f:
            tok_id_file = f.read().strip()

    dtype_name = (meta or {}).get("shard_dtype", "uint16")
    if dtype_name not in DTYPE_BYTES:
        sys.exit(f"Unknown shard_dtype {dtype_name!r}; add it to DTYPE_BYTES.")
    dtype_bytes = DTYPE_BYTES[dtype_name]

    shards = manifest.get("shards", [])
    print(f"Shard dir:  {shard_dir}")
    print(f"Tokenizer:  {tokenizer_path}")
    print(f"Manifest:   {len(shards)} entries (dtype={dtype_name}, {dtype_bytes} bytes/token)")
    print()

    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"[1/4] CHECK SIZE: shard_bytes == tokens * {dtype_bytes}")
    print("=" * 72)
    issues_1, dom_stats_1 = check_size(shards, shard_dir, dtype_bytes)
    print(f"{'domain':25s}  {'ok':>7s}  {'mismatch':>8s}  {'missing':>8s}")
    for dom, st in sorted(dom_stats_1.items()):
        print(f"{dom:25s}  {st['ok']:>7d}  {st['mismatch']:>8d}  {st['missing']:>8d}")
    if issues_1:
        print(f"\n  FAIL: {len(issues_1)} entries with size issues. First 5:")
        for shard, src, status, exp, act in issues_1[:5]:
            act_str = "MISSING" if act < 0 else f"{act:,}"
            print(f"    {status:8s}  {shard:20s}  exp={exp:>14,}  act={act_str:>14s}  ({src})")
    else:
        print(f"\n  PASS: all {len(shards)} shards size-match the manifest")
    print()

    # ------------------------------------------------------------------
    print("=" * 72)
    print("[2/4] CHECK JSON: manifest / meta / tokenization_id well-formed and consistent")
    print("=" * 72)
    issues_2 = check_json(manifest, meta, tok_id_file, manifest_path, meta_path, tok_id_path)
    if issues_2:
        print(f"  FAIL: {len(issues_2)} issues:")
        for issue in issues_2[:20]:
            print(f"    - {issue}")
        if len(issues_2) > 20:
            print(f"    ... and {len(issues_2) - 20} more")
    else:
        print("  PASS: all required fields present, types correct, IDs consistent")
    print()

    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"[3/4] CHECK TOKENIZATION: re-encode {args.sample} random source prefixes, "
          f"compare first 10 tokens")
    print("=" * 72)
    tokenizer, err = load_tokenizer(tokenizer_path)
    result_3 = None
    if tokenizer is None:
        print(f"  SKIPPED: {err}")
    else:
        result_3 = check_tokenization_sample(shards, shard_dir, tokenizer, dtype_bytes,
                                             args.eot_token, args.sample, args.seed)
        print(f"  Pool:    {result_3['pool']:,} entries with both shard and source on disk")
        print(f"  Sampled: {result_3['sampled']:,}")
        print(f"  Skipped: {result_3['skipped']:,} (source unreadable / empty after EOT-strip)")
        if result_3["failed"] == 0:
            print(f"  PASS: all sampled shards' first 10 tokens re-derive from source")
        else:
            print(f"  FAIL: {result_3['failed']:,}/{result_3['sampled']:,} mismatched. First 5:")
            for f in result_3["failures"][:5]:
                print(f"    {f['shard']}  ({f['source']})")
                print(f"      {f['reason']}")
    print()

    # ------------------------------------------------------------------
    print("=" * 72)
    print("[4/4] CHECK MERGE-READY: uniqueness, alignment, domains, files present")
    print("=" * 72)
    issues_4 = check_merge_structure(manifest, shard_dir, dtype_bytes)
    if issues_4:
        print(f"  FAIL: {len(issues_4)} issue(s):")
        for issue in issues_4:
            print(f"    - {issue}")
    else:
        print("  PASS: all shard names unique, all sources unique, all aligned, "
              "no orphans, all files present")
    print()

    # ------------------------------------------------------------------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    failures = []
    if issues_1:
        failures.append(f"[1] size:           {len(issues_1)} mismatched/missing entries")
    if issues_2:
        failures.append(f"[2] JSON:           {len(issues_2)} structural issues")
    if result_3 and result_3["failed"] > 0:
        failures.append(f"[3] tokenization:   {result_3['failed']}/{result_3['sampled']} samples failed")
    if issues_4:
        failures.append(f"[4] merge-ready:    {len(issues_4)} structural issues")

    if not failures:
        print("  ALL CHECKS PASSED — safe to merge")
        sys.exit(0)
    else:
        print("  FAILURES:")
        for f in failures:
            print(f"    {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
