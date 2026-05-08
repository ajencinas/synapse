#!/usr/bin/env python3
"""
Diagnostic: compare shard_manifest.json claims vs actual .bin file sizes on disk.

The merger trusts the manifest's `tokens` field as the authoritative count for
each input shard, then verifies the merged output's size matches the sum. If
the on-disk .bin files contain more (or less) data than the manifest claims,
the merger refuses to write a corrupted result. This script tells you which
shards are inconsistent so you can decide between re-tokenizing the affected
sources or repairing the manifest from disk.

Usage:
  python check_shard_manifest.py                              # default /mnt/ssd/token_shards
  python check_shard_manifest.py --dir /path/to/token_shards
  python check_shard_manifest.py --domain data_adult          # filter
  python check_shard_manifest.py --bad-only                   # only inconsistent ones
"""

import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compare shard_manifest.json vs actual .bin sizes.")
    parser.add_argument("--dir", default=os.environ.get("TOKENIZER_SHARD_DIR", "/mnt/ssd/token_shards"),
                        help="Shard dir to inspect (default: $TOKENIZER_SHARD_DIR or /mnt/ssd/token_shards)")
    parser.add_argument("--domain", default=None,
                        help="Only check shards whose source path contains this substring (e.g. data_adult)")
    parser.add_argument("--bad-only", action="store_true",
                        help="Only print shards with size mismatch or missing file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap detail rows printed (summary still totals everything)")
    args = parser.parse_args()

    manifest_path = os.path.join(args.dir, "shard_manifest.json")
    meta_path     = os.path.join(args.dir, "meta.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"No shard_manifest.json at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    dtype_bytes = 2  # uint16 default
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            dt = meta.get("shard_dtype", "uint16")
            dtype_bytes = {"uint8": 1, "uint16": 2, "uint32": 4}.get(dt, 2)
        except Exception:
            pass

    shards = manifest.get("shards", [])
    if args.domain:
        shards = [s for s in shards if args.domain in s.get("source", "")]

    by_domain = {}
    for s in shards:
        dom = os.path.basename(os.path.dirname(s.get("source", "")))
        by_domain.setdefault(dom, {"ok": 0, "mismatch": 0, "missing": 0,
                                   "expected_bytes": 0, "actual_bytes": 0})

    print(f"Shard dir: {args.dir}")
    print(f"Checking  {len(shards)} manifest entries (dtype={dtype_bytes} bytes/token)")
    print()

    rows = []
    for s in shards:
        dom = os.path.basename(os.path.dirname(s.get("source", "")))
        path = os.path.join(args.dir, s["shard"])
        expected = s.get("tokens", 0) * dtype_bytes
        if not os.path.exists(path):
            status, actual = "MISSING", -1
            by_domain[dom]["missing"] += 1
        else:
            actual = os.path.getsize(path)
            if actual == expected:
                status = "OK"
                by_domain[dom]["ok"] += 1
            else:
                status = "MISMATCH"
                by_domain[dom]["mismatch"] += 1
        by_domain[dom]["expected_bytes"] += max(0, expected)
        by_domain[dom]["actual_bytes"]   += max(0, actual)

        if args.bad_only and status == "OK":
            continue
        ratio = (actual / expected) if expected and actual >= 0 else 0
        rows.append((status, s["shard"], dom, s.get("tokens", 0), expected, actual, ratio,
                     os.path.basename(s.get("source", ""))))

    # Per-domain summary
    print(f"{'domain':25s}  {'ok':>6s}  {'mismatch':>8s}  {'missing':>7s}  "
          f"{'expected_GB':>12s}  {'actual_GB':>12s}  {'delta_GB':>10s}")
    total = {"ok": 0, "mismatch": 0, "missing": 0, "expected_bytes": 0, "actual_bytes": 0}
    for dom, st in sorted(by_domain.items()):
        delta = st["actual_bytes"] - st["expected_bytes"]
        print(f"{dom:25s}  {st['ok']:>6d}  {st['mismatch']:>8d}  {st['missing']:>7d}  "
              f"{st['expected_bytes']/1e9:>12.3f}  {st['actual_bytes']/1e9:>12.3f}  "
              f"{delta/1e9:>+10.3f}")
        for k in total:
            total[k] += st[k]
    delta = total["actual_bytes"] - total["expected_bytes"]
    print(f"{'TOTAL':25s}  {total['ok']:>6d}  {total['mismatch']:>8d}  {total['missing']:>7d}  "
          f"{total['expected_bytes']/1e9:>12.3f}  {total['actual_bytes']/1e9:>12.3f}  "
          f"{delta/1e9:>+10.3f}")

    if args.bad_only and not rows:
        print("\nNo bad rows. Manifest is consistent with disk.")
        return

    # Detail rows
    print()
    print(f"{'status':10s}  {'shard':20s}  {'domain':22s}  {'mfst_tokens':>14s}  "
          f"{'exp_bytes':>14s}  {'act_bytes':>14s}  {'ratio':>7s}  source")
    if args.limit:
        rows = rows[: args.limit]
    for status, shard, dom, tokens, expected, actual, ratio, src in rows:
        actual_str = "MISSING" if actual < 0 else f"{actual:,}"
        ratio_str  = "-" if actual < 0 or expected == 0 else f"{ratio:.2f}x"
        print(f"{status:10s}  {shard:20s}  {dom:22s}  {tokens:>14,}  "
              f"{expected:>14,}  {actual_str:>14s}  {ratio_str:>7s}  {src}")


if __name__ == "__main__":
    main()
