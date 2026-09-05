#!/usr/bin/env python3
"""recover_from_tar.py — rebuild `token_shards_merged/` on Drive from the cold-backup tar.

Designed to run on a non-Colab box (your home machine, a Lambda VM, etc.) where
Colab's Drive FUSE constraints don't apply. Streams the 6-part tar from Drive
via rclone (one part on local disk at a time), rebuilds the missing merged
shards on the fly, and uploads each one back to Drive as soon as it completes.

Inputs the script reads from Drive (under `<remote>:<drive-path>/`):
  - token_shards_merged/shard_manifest.json   (defines the merged shards + their merged_from)
  - token_shards_merged/meta.json             (gives shard dtype)
  - token_shards/token_shards.tar.part-{aa..af}

Outputs the script writes to Drive:
  - token_shards_merged/shard_*.bin           (one per repaired merged shard)

Local disk required (rough): one tar part (~50 GB) plus the in-progress merged
shards. If tar member order tracks merged_from order, the in-progress set stays
small (a handful of files); worst case it can grow to the full repair size
(~256 GB of sparse files). Use a partition with sparse-file support (ext4/xfs/btrfs).

Resumable: re-running the script lists existing merged shards on Drive and
skips them. Interrupted partial files (.tmp) are recreated fresh.

Requires rclone configured with a remote pointing at the Drive root, identical
to the one `tokenize/upload_to_drive.py` uses.

Usage:
  python recover_from_tar.py
  python recover_from_tar.py --dry-run
  python recover_from_tar.py --limit 100               # repair first 100 missing
  python recover_from_tar.py --remote gdrive --drive-path synapse
  GDRIVE_REMOTE=gdrive python recover_from_tar.py

Recommended: run under `nohup`, `screen`, or `tmux` — full pass is multi-hour.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time

DTYPE_BYTES   = {"uint8": 1, "uint16": 2, "uint32": 4, "int8": 1, "int16": 2, "int32": 4}
COPY_CHUNK    = 4 * 1024 * 1024
PART_SUFFIXES = ["aa", "ab", "ac", "ad", "ae", "af"]


def run(cmd, check=True, **kwargs):
    """subprocess.run with a sane default for echoing failures."""
    return subprocess.run(cmd, check=check, **kwargs)


def rclone_lsf(remote, include_glob):
    out = subprocess.run(
        ["rclone", "lsf", remote, "--include", include_glob],
        capture_output=True, text=True, check=True,
    ).stdout
    return [l for l in out.strip().split("\n") if l]


def rclone_copyto(src, dst):
    run([
        "rclone", "copyto", src, dst,
        "--checksum",
        "--drive-chunk-size=64M",
        "--retries=5",
        "--low-level-retries=20",
    ])


def assert_rclone_remote(remote_name):
    out = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, check=True).stdout
    if f"{remote_name}:" not in out:
        sys.exit(
            f"rclone remote '{remote_name}' not configured. Run `rclone config`, "
            f"or set GDRIVE_REMOTE to the right remote name."
        )


def fmt_gb(b):
    return f"{b / 1e9:.2f} GB"


def fmt_mb(b):
    return f"{b / 1e6:.1f} MB"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--remote",     default=os.environ.get("GDRIVE_REMOTE", "gdrive"))
    parser.add_argument("--drive-path", default=os.environ.get("GDRIVE_PATH",   "synapse"))
    parser.add_argument("--work-dir",   default=os.environ.get("LOCAL_WORK",    "./synapse_recover"),
                        help="Local scratch directory. Pick a disk with ~70+ GB free "
                             "(one tar part is ~50 GB plus working merged shards). "
                             "Avoid /tmp — often tmpfs/RAM-backed. Env: LOCAL_WORK.")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Plan and exit; don't download/extract/upload.")
    parser.add_argument("--limit",      type=int, default=0,
                        help="Process only the first N missing merged shards (0 = all).")
    parser.add_argument("--keep-local", action="store_true",
                        help="Don't delete merged shards locally after upload.")
    args = parser.parse_args()

    assert_rclone_remote(args.remote)

    remote_root   = f"{args.remote}:{args.drive_path}"
    tars_remote   = f"{remote_root}/token_shards"
    merged_remote = f"{remote_root}/token_shards_merged"
    work          = os.path.abspath(args.work_dir)
    os.makedirs(work, exist_ok=True)
    work_free = shutil.disk_usage(work).free
    work_total = shutil.disk_usage(work).total

    print(f"Remote root:   {remote_root}")
    print(f"Tars remote:   {tars_remote}")
    print(f"Merged remote: {merged_remote}")
    print(f"Work dir:      {work}")
    print(f"  filesystem:  {fmt_gb(work_free)} free of {fmt_gb(work_total)}")

    # ---------------- 1. Manifest + meta ----------------
    manifest_local = os.path.join(work, "shard_manifest.json")
    meta_local     = os.path.join(work, "meta.json")
    print("\nFetching merged manifest + meta from Drive...")
    rclone_copyto(f"{merged_remote}/shard_manifest.json", manifest_local)
    rclone_copyto(f"{merged_remote}/meta.json",           meta_local)

    with open(manifest_local) as f:
        manifest = json.load(f)
    with open(meta_local) as f:
        meta = json.load(f)

    dtype_name = meta.get("shard_dtype", "uint16")
    if dtype_name not in DTYPE_BYTES:
        sys.exit(f"Unknown shard_dtype {dtype_name!r} in meta.json")
    dtype_bytes = DTYPE_BYTES[dtype_name]

    expected = {s["shard"]: s for s in manifest["shards"]}
    print(f"Merged manifest: {len(expected)} shards, dtype={dtype_name} ({dtype_bytes}B/token)")

    # ---------------- 2. Compute missing ----------------
    print("\nListing existing merged shards on Drive...")
    present = set(rclone_lsf(merged_remote, "shard_*.bin"))
    missing = sorted(set(expected) - present)
    print(f"Present: {len(present)}  Missing: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    if args.limit > 0 and args.limit < len(missing):
        missing = missing[: args.limit]
        print(f"(Processing first {len(missing)} due to --limit)")

    # ---------------- 3. Build dependency graph ----------------
    # deps[small_shard_basename] = [(merged_name, offset_in_merged, expected_size)]
    # state[merged_name] = {expected_bytes, remaining: set, tmp_path}
    deps  = {}
    state = {}
    for name in missing:
        sources = expected[name].get("merged_from")
        if not sources:
            sys.exit(f"{name} has no merged_from in manifest — cannot reconstruct.")
        offset = 0
        rem = set()
        for s in sources:
            base = os.path.basename(s["shard"])
            sz = s["tokens"] * dtype_bytes
            deps.setdefault(base, []).append((name, offset, sz))
            offset += sz
            rem.add(base)
        state[name] = {"expected_bytes": offset, "remaining": rem}

    total_repair = sum(st["expected_bytes"] for st in state.values())
    print(f"Need {len(deps)} distinct small shards; producing {len(missing)} merged shards "
          f"({fmt_gb(total_repair)})")

    free = shutil.disk_usage(work).free
    print(f"Work dir free space: {fmt_gb(free)}")
    if free < 70 * 1024**3:
        print(f"WARNING: less than 70 GB free in {work}. One tar part is ~50 GB; "
              f"in-progress merged shards add up to ~{fmt_gb(total_repair)} (sparse, but worst-case real).")

    if args.dry_run:
        print("\nDRY RUN — stopping before download/extract.")
        return

    # ---------------- 4. Pre-allocate merged shard temp files ----------------
    merged_tmp_dir = os.path.join(work, "merged_in_progress")
    os.makedirs(merged_tmp_dir, exist_ok=True)
    for name, st in state.items():
        tmp = os.path.join(merged_tmp_dir, name + ".tmp")
        with open(tmp, "wb") as f:
            f.truncate(st["expected_bytes"])
        st["tmp_path"] = tmp

    # ---------------- 5. Stream tar through a pipe; producer thread downloads parts ----------------
    parts = [f"token_shards.tar.part-{x}" for x in PART_SUFFIXES]
    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, "rb")
    w_file = os.fdopen(w_fd, "wb")

    producer_error = []

    def producer():
        """Download each tar part in turn, stream to pipe, delete, advance."""
        try:
            for p in parts:
                local = os.path.join(work, p)
                if os.path.exists(local):
                    print(f"  (reusing existing {p}, {fmt_gb(os.path.getsize(local))})")
                else:
                    print(f"  Downloading {p}...")
                    t = time.time()
                    rclone_copyto(f"{tars_remote}/{p}", local)
                    sz = os.path.getsize(local)
                    dt = time.time() - t
                    print(f"    {p} -> {fmt_gb(sz)} in {dt/60:.1f} min ({sz/1e6/max(dt,1):.1f} MB/s)")
                with open(local, "rb") as f:
                    while True:
                        buf = f.read(COPY_CHUNK)
                        if not buf:
                            break
                        w_file.write(buf)
                os.remove(local)
                print(f"    consumed and deleted {p}")
            w_file.close()
        except Exception as e:
            producer_error.append(e)
            try:
                w_file.close()
            except Exception:
                pass

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    # ---------------- 6. Consume tar; route members; finalize+upload merged shards ----------------
    print("\nProcessing tar members...")
    tar = tarfile.open(fileobj=r_file, mode="r|")
    completed     = 0
    bytes_in      = 0
    bytes_up      = 0
    members_seen  = 0
    t0 = time.time()
    last_report = t0

    try:
        for ti in tar:
            members_seen += 1
            if not ti.isfile():
                continue
            base = os.path.basename(ti.name)
            writes = deps.get(base)
            if writes is None:
                # Not needed; tarfile streaming will skip member data automatically as we iterate.
                continue
            # Sanity-check size against manifest
            for w_name, off, sz in writes:
                if sz != ti.size:
                    sys.exit(
                        f"size mismatch for {ti.name}: tar has {ti.size}B, manifest expects {sz}B "
                        f"(for {w_name} @ offset {off})"
                    )
            data = tar.extractfile(ti).read()
            bytes_in += len(data)

            for w_name, off, _ in writes:
                st = state[w_name]
                with open(st["tmp_path"], "r+b") as f:
                    f.seek(off)
                    f.write(data)
                st["remaining"].discard(base)
                if not st["remaining"]:
                    actual = os.path.getsize(st["tmp_path"])
                    if actual != st["expected_bytes"]:
                        sys.exit(f"{w_name}: byte mismatch {actual} vs {st['expected_bytes']}")
                    final_local = os.path.join(merged_tmp_dir, w_name)
                    os.replace(st["tmp_path"], final_local)
                    print(f"  Uploading {w_name} ({fmt_mb(actual)})...")
                    rclone_copyto(final_local, f"{merged_remote}/{w_name}")
                    bytes_up += actual
                    if not args.keep_local:
                        os.remove(final_local)
                    completed += 1
                    elapsed = time.time() - t0
                    rate_in = bytes_in / 1e6 / max(elapsed, 1)
                    rate_up = bytes_up / 1e6 / max(elapsed, 1)
                    print(f"    [{completed}/{len(missing)}] uploaded "
                          f"(avg in={rate_in:.1f} MB/s, up={rate_up:.1f} MB/s)")

            del data  # free memory promptly

            now = time.time()
            if now - last_report > 60 and completed == 0:
                print(f"  ... still streaming, members_seen={members_seen}, "
                      f"in={fmt_gb(bytes_in)}, "
                      f"in-progress merged={sum(1 for st in state.values() if st['remaining'])}")
                last_report = now
    finally:
        try:
            tar.close()
        except Exception:
            pass
        try:
            r_file.close()
        except Exception:
            pass
        prod_thread.join(timeout=300)

    if producer_error:
        sys.exit(f"Producer error: {producer_error[0]}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Repaired {completed}/{len(missing)} merged shards.")

    # ---------------- 7. Sanity check: clean up leftover .tmp; verify remote ----------------
    leftover = []
    for name, st in state.items():
        if os.path.exists(st["tmp_path"]):
            leftover.append(name)
            os.remove(st["tmp_path"])
    if leftover:
        print(f"WARNING: {len(leftover)} merged shard(s) had unfinished sources — not uploaded.")
        print(f"  first 5: {leftover[:5]}")
        print(f"  Their sources were not all present in the tar. Either the tar is incomplete "
              f"or the merged manifest references shards from a different tokenize run.")

    print("\nVerifying remote state...")
    present_after = set(rclone_lsf(merged_remote, "shard_*.bin"))
    still_missing = sorted(set(expected) - present_after)
    print(f"Remote merged shards now: {len(present_after)} / {len(expected)} expected")
    if still_missing:
        print(f"Still missing: {len(still_missing)} (first 5: {still_missing[:5]})")
        sys.exit(1)
    print("All merged shards present on Drive.")


if __name__ == "__main__":
    main()
