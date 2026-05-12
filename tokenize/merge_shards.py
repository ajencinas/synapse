#!/usr/bin/env python3
"""
Incremental, order-preserving merger for tokenizer .bin shards.

Walks shards in their manifest order (which is append-stable: sorted by
shard_NNNNN index, so new files always land at the tail), groups by domain
(parent dir of the source file), and concatenates consecutive shards into
~target-size buckets. Source order is preserved within each domain — the only
operation is "cut here, start a new shard." No reordering, no bin-packing.

Why incremental: pretrain checkpoints record `seen_shards` by the merged shard
filename (`shard_NNNNN.bin`). A wipe-and-rebuild reassigns indexes, so a shard
the trainer logged as "seen" now points at completely different bytes — silent
checkpoint corruption. Incremental merge keeps every existing merged .bin
byte-identical and only appends new shards with indexes after the current max.

Invariants:
  - Any merged_from prefix that already exists is held immutable: same name,
    same bytes. No "extend the last under-target shard" — that would invalidate
    seen_shards too.
  - Net-new input shards (per domain) bucket into new merged shards with
    indexes max(existing) + 1, +2, ... The cut policy is the same as the
    initial build: flush after first shard that pushes bucket_bytes >= target.
  - Divergence between the existing merged manifest and the current input
    manifest (reorder, remove, retokenize) is a hard error. The user explicitly
    chooses --rebuild, which wipes and re-merges from scratch (and is documented
    to invalidate any pretrain checkpoint's seen_shards).

Why a domain might end up with multiple under-target shards: when you add a
small batch of new sources to a domain whose last merged shard is already
under-target, we *don't* extend the old shard (see invariant above). The new
content opens a fresh bucket, which is also potentially under-target until
more sources arrive. This is acceptable — the trainer is robust to small
shards; it just costs a bit of storage.

Output dir mirrors the input layout (shard_*.bin + shard_manifest.json +
meta.json + tokenization_id.txt) so the trainer can point at it transparently.

Usage:
  python merge_shards.py                         # in: $TOKENIZER_SHARD_DIR -> <in>_merged, incremental
  python merge_shards.py --target-bytes 200000000
  python merge_shards.py --in /mnt/ssd/token_shards --out /mnt/ssd/token_shards_merged
  python merge_shards.py --dry-run               # print plan only
  python merge_shards.py --rebuild               # wipe output dir and re-merge from scratch
                                                 # WARNING: invalidates pretrain seen_shards
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


def _idx_from_shard_name(name):
    """Parse the integer index from a 'shard_NNNNN.bin' filename. Returns -1
    on any parse failure so callers can use max(..., default=-1) safely."""
    try:
        stem = name[len("shard_"):-len(".bin")]
        return int(stem)
    except (ValueError, IndexError, TypeError):
        return -1


def fingerprint(target_bytes, source_entries):
    """Stable hash over the inputs that determine merge output. Used only as a
    fast-path no-op check ("inputs identical to last merge — nothing to do")."""
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


def _bucket_for_domain(remaining_inputs, domain, target_bytes, dtype_bytes):
    """Bucket a per-domain input queue into merged-shard plan entries using
    the >= target_bytes cut policy. The last bucket may be under target."""
    plan = []
    bucket = []
    bucket_bytes = 0
    for s in remaining_inputs:
        shard_bytes = s["tokens"] * dtype_bytes
        bucket.append(s)
        bucket_bytes += shard_bytes
        if bucket_bytes >= target_bytes:
            plan.append({"domain": domain, "sources": bucket, "bytes": bucket_bytes})
            bucket = []
            bucket_bytes = 0
    if bucket:
        plan.append({"domain": domain, "sources": bucket, "bytes": bucket_bytes})
    return plan


def _group_by_domain(shard_entries):
    """Group a list of shard manifest entries by domain. Preserves the input
    order both for domain emergence and within-domain ordering."""
    by_domain = {}
    domain_order = []
    for s in shard_entries:
        # Input shard entries store full source paths; existing merged shard
        # entries store an explicit 'domain' field. Prefer 'domain' when set
        # to avoid path-prefix surprises across mounts.
        dom = s.get("domain") or parent_dir_name(s.get("source", ""))
        if dom not in by_domain:
            by_domain[dom] = []
            domain_order.append(dom)
        by_domain[dom].append(s)
    return by_domain, domain_order


# ---------- Incremental planning ----------
def _plan_incremental(in_manifest, existing_merged, target_bytes, dtype_bytes):
    """Return (kept_merged, new_plan, kept_indexes).

    kept_merged:  list of existing merged-shard manifest entries to preserve
                  on disk byte-for-byte. Order matches the existing manifest.
    new_plan:     list of bucket dicts {"domain", "sources", "bytes"} for
                  shards that need to be written.
    kept_indexes: list of shard_idx integers parsed from kept_merged names —
                  used to pick the next index for new shards.

    Raises SystemExit with a user-actionable message on any divergence between
    the existing merged manifest and the current inputs.
    """
    input_by_domain, input_domain_order = _group_by_domain(in_manifest.get("shards", []))
    existing_by_domain, _ = _group_by_domain(existing_merged.get("shards", []))

    # Fail-loud: a domain present in existing merged but absent from current
    # inputs means source data was removed. Incremental can't reconcile this.
    input_domain_set = set(input_by_domain.keys())
    for dom in existing_by_domain:
        if dom not in input_domain_set:
            sys.exit(
                f"Incremental merge: existing merged dir has shards for domain {dom!r}, "
                f"but no inputs reference that domain anymore. Source data appears to "
                f"have been removed.\n"
                f"  Pass --rebuild to wipe and re-merge from current inputs.\n"
                f"  WARNING: --rebuild invalidates seen_shards in any pretrain checkpoint."
            )

    kept_merged = []
    kept_indexes = []
    new_plan = []

    for dom in input_domain_order:
        input_queue = input_by_domain[dom]
        existing_for_dom = existing_by_domain.get(dom, [])

        input_idx = 0
        for ms in existing_for_dom:
            # Pop the merged_from prefix off the input queue and verify each
            # entry matches by shard name AND token count. Token count guards
            # against retokenization that kept the same shard name.
            expected_inputs = ms.get("merged_from", [])
            if not expected_inputs:
                sys.exit(
                    f"Incremental merge: existing merged shard {ms.get('shard')!r} has "
                    f"no merged_from record — can't verify what input shards it covers.\n"
                    f"  Pass --rebuild to wipe and re-merge from current inputs.\n"
                    f"  WARNING: --rebuild invalidates seen_shards in any pretrain checkpoint."
                )
            for expected in expected_inputs:
                if input_idx >= len(input_queue):
                    sys.exit(
                        f"Incremental merge: existing merged shard {ms.get('shard')!r} "
                        f"(domain={dom}) expects input {expected.get('shard')!r} but the "
                        f"input queue for {dom} is exhausted. A source file was likely "
                        f"removed from the input manifest.\n"
                        f"  Pass --rebuild to wipe and re-merge from current inputs.\n"
                        f"  WARNING: --rebuild invalidates seen_shards in any pretrain checkpoint."
                    )
                actual = input_queue[input_idx]
                exp_name, exp_tok = expected.get("shard"), expected.get("tokens")
                act_name, act_tok = actual.get("shard"), actual.get("tokens")
                if act_name != exp_name or act_tok != exp_tok:
                    sys.exit(
                        f"Incremental merge: existing merged shard {ms.get('shard')!r} "
                        f"(domain={dom}) expects input {exp_name!r} (tokens={exp_tok}) at "
                        f"position {input_idx} of the {dom} queue, but found {act_name!r} "
                        f"(tokens={act_tok}). Inputs have been reordered, removed, or "
                        f"retokenized after the last merge.\n"
                        f"  Pass --rebuild to wipe and re-merge from current inputs.\n"
                        f"  WARNING: --rebuild invalidates seen_shards in any pretrain checkpoint."
                    )
                input_idx += 1

            kept_merged.append(ms)
            idx = _idx_from_shard_name(ms.get("shard", ""))
            if idx >= 0:
                kept_indexes.append(idx)

        remaining = input_queue[input_idx:]
        if remaining:
            new_plan.extend(_bucket_for_domain(remaining, dom, target_bytes, dtype_bytes))

    return kept_merged, new_plan, kept_indexes


def _plan_full_rebuild(in_manifest, target_bytes, dtype_bytes):
    """Full rebuild plan: bucket every input shard from scratch. Used by
    --rebuild and by the first-ever merge (when there's no existing manifest)."""
    by_domain, domain_order = _group_by_domain(in_manifest.get("shards", []))
    plan = []
    for dom in domain_order:
        plan.extend(_bucket_for_domain(by_domain[dom], dom, target_bytes, dtype_bytes))
    return plan


# ---------- Output writing ----------
def _write_one_shard(input_dir, output_dir, shard_name, bucket, dtype_bytes):
    """Stream-copy bucket sources into output_dir/shard_name. Returns the
    merged-shard manifest entry. Verifies byte count on the way out — the
    only failure mode is a copy that produced wrong-size output, which would
    indicate a disk problem (truncated reads, full disk, etc.)."""
    out_path = os.path.join(output_dir, shard_name)
    expected_bytes = 0
    total_tokens = 0
    total_docs = 0
    with open(out_path, "wb") as fout:
        for s in bucket["sources"]:
            src_bin = os.path.join(input_dir, s["shard"])
            if not os.path.exists(src_bin):
                sys.exit(
                    f"Missing input shard: {src_bin}\n"
                    f"  The manifest references it but it's not on disk. Restore the "
                    f"input shards before running merge."
                )
            with open(src_bin, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=COPY_CHUNK_BYTES)
            expected_bytes += s["tokens"] * dtype_bytes
            total_tokens   += s["tokens"]
            total_docs     += s.get("documents", 0)

    actual_bytes = os.path.getsize(out_path)
    if actual_bytes != expected_bytes:
        sys.exit(
            f"Byte mismatch for {shard_name}: expected {expected_bytes} "
            f"(sum of input shard sizes), got {actual_bytes}. Disk full or corrupt?"
        )

    # Synthetic source path lets the trainer recover the domain via the same
    # `os.path.basename(os.path.dirname(source))` trick it uses for input shards.
    synthetic_source = os.path.join(
        os.path.dirname(bucket["sources"][0]["source"]),
        f"_merged_{shard_name[len('shard_'):-len('.bin')]}",
    )
    return {
        "shard": shard_name,
        "source": synthetic_source,
        "domain": bucket["domain"],
        "tokens": total_tokens,
        "documents": total_docs,
        "shard_mb": round(actual_bytes / 1024 / 1024, 2),
        "merged_from": [
            {"shard": s["shard"], "source": s["source"], "tokens": s["tokens"]}
            for s in bucket["sources"]
        ],
    }


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ---------- Plan summary ----------
def _print_plan_summary(kept, new_plan, target_bytes, rebuild):
    if rebuild:
        total = sum(p["bytes"] for p in new_plan)
        print(f"REBUILD plan: {len(new_plan)} merged shards "
              f"(target {target_bytes / 1024 / 1024:.0f} MB each), "
              f"{total / 1024 / 1024 / 1024:.2f} GB total")
        by_dom = {}
        for p in new_plan:
            by_dom.setdefault(p["domain"], []).append(p)
        for dom, ps in by_dom.items():
            sizes_mb = sorted(p["bytes"] / 1024 / 1024 for p in ps)
            toks = sum(s["tokens"] for p in ps for s in p["sources"])
            srcs = sum(len(p["sources"]) for p in ps)
            print(
                f"  {dom}: {len(ps)} shards from {srcs} sources, "
                f"{toks:,} tokens | "
                f"min={sizes_mb[0]:.1f}MB p50={sizes_mb[len(sizes_mb) // 2]:.1f}MB "
                f"max={sizes_mb[-1]:.1f}MB"
            )
        return

    print(f"Incremental plan (target {target_bytes / 1024 / 1024:.0f} MB):")
    print(f"  Keep {len(kept)} existing merged shards (byte-identical on disk)")
    print(f"  Add  {len(new_plan)} new merged shards")
    if new_plan:
        by_dom = {}
        for p in new_plan:
            by_dom.setdefault(p["domain"], []).append(p)
        for dom, ps in by_dom.items():
            sizes_mb = sorted(p["bytes"] / 1024 / 1024 for p in ps)
            toks = sum(s["tokens"] for p in ps for s in p["sources"])
            srcs = sum(len(p["sources"]) for p in ps)
            print(
                f"    +{dom}: {len(ps)} new shards from {srcs} new sources, "
                f"+{toks:,} tokens | "
                f"min={sizes_mb[0]:.1f}MB max={sizes_mb[-1]:.1f}MB"
            )


# ---------- Main entry ----------
def merge_shards(input_dir, output_dir, target_bytes, dry_run=False, rebuild=False):
    """Incrementally extend (or fully rebuild with rebuild=True) the merged
    shard dir. Returns the new shard_manifest dict, or None if dry_run."""
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

    out_manifest_path = os.path.join(output_dir, "shard_manifest.json")
    existing_merged = None
    if not rebuild and os.path.exists(out_manifest_path):
        try:
            with open(out_manifest_path) as f:
                existing_merged = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            sys.exit(
                f"Failed to read existing merged manifest at {out_manifest_path}: {e}\n"
                f"  Pass --rebuild to wipe and re-merge from scratch.\n"
                f"  WARNING: --rebuild invalidates seen_shards in any pretrain checkpoint."
            )

    # Compatibility checks for incremental mode. These guard the invariants
    # the trainer's seen_shards logic relies on: same tokenizer, same bucket
    # size. Any mismatch is fail-loud — user must explicitly opt into --rebuild.
    if existing_merged is not None:
        existing_tok_id = existing_merged.get("tokenization_id")
        input_tok_id = in_manifest.get("tokenization_id")
        if existing_tok_id and input_tok_id and existing_tok_id != input_tok_id:
            sys.exit(
                f"tokenization_id mismatch: existing merged dir was built from tokenizer "
                f"{existing_tok_id!r}, but current input manifest is {input_tok_id!r}. "
                f"Existing merged shards contain different tokens — cannot incrementally "
                f"extend.\n"
                f"  Pass --rebuild to wipe and re-merge.\n"
                f"  WARNING: --rebuild invalidates seen_shards in any pretrain checkpoint."
            )
        existing_target = existing_merged.get("merge_target_bytes")
        if existing_target is not None and existing_target != target_bytes:
            sys.exit(
                f"merge_target_bytes mismatch: existing merged dir uses {existing_target}, "
                f"current run requests {target_bytes}. Cannot incrementally extend with a "
                f"different bucket size — existing shard boundaries wouldn't align.\n"
                f"  Pass --rebuild to wipe and re-merge at the new size.\n"
                f"  WARNING: --rebuild invalidates seen_shards in any pretrain checkpoint."
            )

    # Fingerprint fast-path: input identical to last merge -> no-op.
    fp = fingerprint(target_bytes, in_manifest.get("shards", []))
    if existing_merged is not None and existing_merged.get("merge_fingerprint") == fp:
        print("Existing merged dir already matches input fingerprint — nothing to do.")
        return existing_merged

    # ---- Plan ----
    if rebuild or existing_merged is None:
        if rebuild and existing_merged is not None:
            print("REBUILD: existing merged dir will be wiped and re-merged from scratch.")
            print("WARNING: this invalidates seen_shards in any pretrain checkpoint that")
            print("         was trained against the existing merged shard layout.")
        kept_merged = []
        new_plan = _plan_full_rebuild(in_manifest, target_bytes, dtype_bytes)
        next_idx = 0
    else:
        kept_merged, new_plan, kept_indexes = _plan_incremental(
            in_manifest, existing_merged, target_bytes, dtype_bytes
        )
        next_idx = max(kept_indexes, default=-1) + 1

    _print_plan_summary(kept_merged, new_plan, target_bytes, rebuild=rebuild)

    if dry_run:
        print("DRY RUN: no files written.")
        return None

    # ---- Write ----
    os.makedirs(output_dir, exist_ok=True)

    if rebuild:
        # Full wipe of .bin files. Manifest/meta are overwritten below.
        for f in os.listdir(output_dir):
            if f.endswith(".bin"):
                os.remove(os.path.join(output_dir, f))

    new_out_shards = []
    for i, bucket in enumerate(new_plan):
        shard_name = f"shard_{next_idx:05d}.bin"
        next_idx += 1
        entry = _write_one_shard(input_dir, output_dir, shard_name, bucket, dtype_bytes)
        new_out_shards.append(entry)
        prefix = "  +" if not rebuild else f"  [{i + 1}/{len(new_plan)}]"
        print(
            f"{prefix} {shard_name} ({bucket['domain']}): "
            f"{entry['tokens']:,} tokens, {entry['shard_mb']:.1f} MB, "
            f"{len(bucket['sources'])} sources"
        )

    if not rebuild and not new_out_shards:
        # Incremental run with no new inputs: nothing on disk changed. The
        # fingerprint short-circuit above would normally catch this, but if
        # the existing manifest is missing a fingerprint field we land here.
        # Touch the manifest to record the new fingerprint and return.
        out_manifest = {
            **(existing_merged or {}),
            "merge_fingerprint": fp,
        }
        _atomic_write_json(out_manifest_path, out_manifest)
        print("No new shards; refreshed merged manifest fingerprint.")
        return out_manifest

    # ---- Manifest + meta ----
    all_shards = kept_merged + new_out_shards
    out_manifest = {
        "tokenization_id":     in_manifest.get("tokenization_id"),
        "merged_from":         input_dir,
        "merge_target_bytes":  target_bytes,
        "merge_fingerprint":   fp,
        "shards":              all_shards,
    }
    _atomic_write_json(out_manifest_path, out_manifest)

    out_meta = {
        "num_tokens":         sum(s["tokens"] for s in all_shards),
        "num_shards":         len(all_shards),
        "vocab_size":         in_meta["vocab_size"],
        "eot_id":             in_meta["eot_id"],
        "pad_id":             in_meta["pad_id"],
        "tokenizer":          in_meta.get("tokenizer", "byte-level-bpe-digits"),
        "shard_dtype":        dtype_name,
        "tokenization_id":    in_meta.get("tokenization_id"),
        "merged_from":        input_dir,
        "merge_target_bytes": target_bytes,
    }
    _atomic_write_json(os.path.join(output_dir, "meta.json"), out_meta)

    if os.path.exists(in_tokid_path):
        shutil.copyfile(in_tokid_path, os.path.join(output_dir, "tokenization_id.txt"))

    print(
        f"\nMerged: {len(all_shards)} shards total "
        f"({len(kept_merged)} kept + {len(new_out_shards)} added), "
        f"{sum(s['tokens'] for s in all_shards):,} tokens -> {output_dir}"
    )
    return out_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incremental, order-preserving merger for tokenizer .bin shards."
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
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe output dir and re-merge from scratch. WARNING: invalidates "
             "seen_shards in any pretrain checkpoint trained against the existing "
             "merged shard layout."
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.in_dir.rstrip("/") + "_merged")
    merge_shards(args.in_dir, out_dir, args.target_bytes,
                 dry_run=args.dry_run, rebuild=args.rebuild)
