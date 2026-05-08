#!/usr/bin/env python3
"""
Repair shard_manifest.json by resolving duplicate shard-name mappings.

Root cause this fixes
---------------------
run_tokenizer.py historically assigned new shard names with
  shard_idx = len(shard_manifest["shards"])
which underestimates the next safe index whenever the manifest had ever lost
entries (crashed run, manual edit, repair). New sources reused names of
existing on-disk shards and OVERWROTE them. The original source's data is gone
from disk, but its manifest entry sticks around — so the manifest now has
multiple sources mapped to the same shard_NNNNN.bin, with only one of them
matching the actual file content.

What this script does
---------------------
Groups manifest entries by shard name. Per group:
  - If the on-disk file's size matches exactly one entry's tokens*dtype_bytes,
    that entry survived — keep it, drop the rest as duplicate losers.
  - If no entry matches (file truncated/corrupt for everyone), drop all and
    quarantine the file.
  - If the file is missing entirely, drop all entries (nothing to recover).

The dropped sources are NOT lost forever — rerun
  python tokenize/run_tokenizer.py --no-train
afterward and the tokenizer will see them as missing and re-tokenize ONLY
those, assigning fresh non-colliding shard names (provided you also have the
shard_idx=max+1 fix in run_tokenizer.py).

Default mode is DRY-RUN. Pass --apply to mutate.

Usage:
  python repair_shards.py                           # dry-run, summary
  python repair_shards.py --apply                   # repair + quarantine bad files
  python repair_shards.py --apply --delete-bad      # delete instead of quarantine
  python repair_shards.py --apply --quarantine-orphans  # also handle stray .bin files
"""

import os
import sys
import json
import shutil
import argparse
from collections import defaultdict


def atomic_dump_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Repair shard_manifest.json — duplicate-aware.")
    parser.add_argument("--dir", default=os.environ.get("TOKENIZER_SHARD_DIR", "/mnt/ssd/token_shards"),
                        help="Shard dir (default: $TOKENIZER_SHARD_DIR or /mnt/ssd/token_shards)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually mutate. Without this flag, only print a plan.")
    parser.add_argument("--delete-bad", action="store_true",
                        help="Delete bad .bin files instead of moving to _quarantine/")
    parser.add_argument("--quarantine-orphans", action="store_true",
                        help="Also handle .bin files present on disk but absent from manifest")
    parser.add_argument("--show-collisions", type=int, default=20,
                        help="How many shard-name collisions to print in detail (default 20)")
    args = parser.parse_args()

    shard_dir     = args.dir
    manifest_path = os.path.join(shard_dir, "shard_manifest.json")
    meta_path     = os.path.join(shard_dir, "meta.json")
    quarantine    = os.path.join(shard_dir, "_quarantine")

    if not os.path.exists(manifest_path):
        sys.exit(f"No shard_manifest.json at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    dtype_bytes = 2  # uint16 default
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            dtype_bytes = {"uint8": 1, "uint16": 2, "uint32": 4}.get(
                meta.get("shard_dtype", "uint16"), 2)
        except Exception:
            pass

    shards = manifest.get("shards", [])
    print(f"Shard dir: {shard_dir}")
    print(f"Manifest:  {len(shards)} entries (dtype={dtype_bytes} bytes/token)")

    # Group entries by shard name. The healthy invariant is one entry per name;
    # multi-entry groups are duplicates from the shard_idx-reuse bug.
    by_shard = defaultdict(list)
    for s in shards:
        by_shard[s["shard"]].append(s)

    n_collisions = sum(1 for g in by_shard.values() if len(g) > 1)
    n_collision_entries = sum(len(g) for g in by_shard.values() if len(g) > 1)
    print(f"Distinct shard names:  {len(by_shard)}")
    print(f"Collisions (duplicate shard names): {n_collisions} groups, "
          f"{n_collision_entries} entries involved")
    print()

    keep_entries  = []
    drop_dup_loser = []     # duplicate, file matches a different entry
    drop_corrupt   = []     # no entry matches the on-disk file
    drop_missing   = []     # file not on disk at all
    bins_to_remove = []     # bin files we can safely remove (no surviving entry)

    domain_summary = defaultdict(lambda: {"keep": 0, "dup_loser": 0, "corrupt": 0, "missing": 0})
    collision_examples = []

    def _dom(s):
        return os.path.basename(os.path.dirname(s.get("source", "")))

    for shard_name, group in by_shard.items():
        bin_path = os.path.join(shard_dir, shard_name)

        if not os.path.exists(bin_path):
            for e in group:
                drop_missing.append(e)
                domain_summary[_dom(e)]["missing"] += 1
            continue

        actual = os.path.getsize(bin_path)
        matching = [e for e in group if e.get("tokens", 0) * dtype_bytes == actual]

        if matching:
            winner = matching[0]
            keep_entries.append(winner)
            domain_summary[_dom(winner)]["keep"] += 1
            losers = [e for e in group if e is not winner]
            for e in losers:
                drop_dup_loser.append(e)
                domain_summary[_dom(e)]["dup_loser"] += 1
            if losers and len(collision_examples) < args.show_collisions:
                collision_examples.append({
                    "shard": shard_name,
                    "actual_bytes": actual,
                    "winner": (_dom(winner), os.path.basename(winner["source"]),
                               winner.get("tokens", 0)),
                    "losers": [(_dom(e), os.path.basename(e["source"]), e.get("tokens", 0))
                               for e in losers],
                })
        else:
            for e in group:
                drop_corrupt.append(e)
                domain_summary[_dom(e)]["corrupt"] += 1
            bins_to_remove.append(bin_path)

    # Also find orphan .bin files (on disk but no manifest entry).
    on_disk_bins = sorted(
        f for f in os.listdir(shard_dir)
        if f.endswith(".bin") and os.path.isfile(os.path.join(shard_dir, f))
    )
    referenced = set(by_shard.keys())
    orphan_bins = [f for f in on_disk_bins if f not in referenced]

    # ----- Per-domain summary -----
    print(f"{'domain':25s}  {'keep':>6s}  {'dup_loser':>10s}  {'corrupt':>8s}  {'missing':>8s}  {'re-tokenize?':>14s}")
    total = {"keep": 0, "dup_loser": 0, "corrupt": 0, "missing": 0}
    for dom, st in sorted(domain_summary.items()):
        re_tok = st["dup_loser"] + st["corrupt"] + st["missing"]
        print(f"{dom:25s}  {st['keep']:>6d}  {st['dup_loser']:>10d}  "
              f"{st['corrupt']:>8d}  {st['missing']:>8d}  {re_tok:>14d}")
        for k in total:
            total[k] += st[k]
    re_tok_total = total["dup_loser"] + total["corrupt"] + total["missing"]
    print(f"{'TOTAL':25s}  {total['keep']:>6d}  {total['dup_loser']:>10d}  "
          f"{total['corrupt']:>8d}  {total['missing']:>8d}  {re_tok_total:>14d}")
    print()
    print(f"Manifest entries to keep:               {len(keep_entries)}")
    print(f"  dropped (duplicate loser):            {len(drop_dup_loser)}  (data overwritten by another source)")
    print(f"  dropped (corrupt — no entry matches): {len(drop_corrupt)}")
    print(f"  dropped (file missing):               {len(drop_missing)}")
    print(f"Bad .bin files to {'delete' if args.delete_bad else 'quarantine'}:        "
          f"{len(bins_to_remove)}  (safe — no surviving entry references them)")
    print(f"Orphan .bin files on disk:              {len(orphan_bins)}"
          f"  ({'will quarantine' if args.quarantine_orphans else 'will leave alone'})")

    # ----- Collision examples -----
    if collision_examples:
        print()
        print(f"Collision examples (up to {args.show_collisions}, file size determines winner):")
        for c in collision_examples:
            print(f"  {c['shard']}  ({c['actual_bytes']:,} bytes on disk)")
            wd, wf, wt = c["winner"]
            print(f"    keep:  [{wd}] {wf}  ({wt:,} tokens)")
            for ld, lf, lt in c["losers"]:
                print(f"    drop:  [{ld}] {lf}  ({lt:,} tokens) — overwritten")

    if not args.apply:
        print()
        print("DRY RUN — pass --apply to mutate manifest and quarantine bad files.")
        print("After --apply, rerun:  python tokenize/run_tokenizer.py --no-train")
        print("(only the dropped sources get re-tokenized; the keepers are skipped on size match)")
        return

    # ----- APPLY -----
    if not args.delete_bad and (bins_to_remove or (args.quarantine_orphans and orphan_bins)):
        os.makedirs(quarantine, exist_ok=True)

    moved = removed = 0
    for path in bins_to_remove:
        if not os.path.exists(path):
            continue
        if args.delete_bad:
            os.remove(path)
            removed += 1
        else:
            shutil.move(path, os.path.join(quarantine, os.path.basename(path)))
            moved += 1

    orphan_moved = orphan_removed = 0
    if args.quarantine_orphans:
        for name in orphan_bins:
            src = os.path.join(shard_dir, name)
            if not os.path.exists(src):
                continue
            if args.delete_bad:
                os.remove(src)
                orphan_removed += 1
            else:
                shutil.move(src, os.path.join(quarantine, name))
                orphan_moved += 1

    new_manifest = dict(manifest)
    new_manifest["shards"] = keep_entries
    atomic_dump_json(manifest_path, new_manifest)

    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
        meta["num_tokens"] = sum(s.get("tokens", 0) for s in keep_entries)
        meta["num_shards"] = len(keep_entries)
        atomic_dump_json(meta_path, meta)

    print()
    print(f"  Manifest rewritten: {len(keep_entries)} entries kept, "
          f"{len(drop_dup_loser) + len(drop_corrupt) + len(drop_missing)} dropped")
    print(f"  Bad .bins:          {removed} deleted, {moved} quarantined -> {quarantine}")
    if args.quarantine_orphans:
        print(f"  Orphan .bins:       {orphan_removed} deleted, {orphan_moved} quarantined")
    print()
    print("Next step:  python tokenize/run_tokenizer.py --no-train")
    print("            (will re-tokenize only the dropped sources, with non-colliding shard names)")


if __name__ == "__main__":
    main()
