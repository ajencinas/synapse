#!/usr/bin/env python3
"""
Archive the unmerged token_shards dir into ONE tar(.gz) and upload to Drive.

Why: uploading 20K+ small .bin files individually to Google Drive hits per-
file metadata overhead and rate limits, so a "for f in files: rclone copy" loop
crawls. Bundling into a single archive sidesteps that — Drive only has to
create one file no matter how many shards you have.

Two upload paths
----------------
1. **Local-first** (default): build the archive locally, then rclone copy.
   Resumable on upload failure (rclone matches size+modtime). Needs free local
   disk space equal to the compressed archive size.

2. **Streaming + chunked** (--stream): pipe `tar | split | rclone rcat`. No
   local intermediate file. Each chunk uploads as its own file on Drive
   (token_shards.tar.gz.part-aa, .part-ab, ...) so a network blip only loses
   the in-flight chunk, not the entire archive. Recommended for >100 GB
   sources.

Archive contents (relative to the source dir's parent):
  token_shards/shard_*.bin
  token_shards/shard_manifest.json
  token_shards/meta.json
  token_shards/tokenization_id.txt

Excluded by default: _quarantine/  (stale debris from past repairs)

To restore later on a fresh VM
------------------------------
Local-first archive:
  rclone copy gdrive:synapse/token_shards.tar.gz /mnt/ssd/
  cd /mnt/ssd && tar xzf token_shards.tar.gz

Chunked archive:
  rclone copy gdrive:synapse/ /mnt/ssd/restore --include='token_shards.tar.gz.part-*'
  cd /mnt/ssd && cat restore/token_shards.tar.gz.part-* | tar xzf -

Usage:
  python archive_unmerged.py                            # local-first, tar.gz
  python archive_unmerged.py --stream                   # one-step streaming, 50GB chunks
  python archive_unmerged.py --stream --no-compress     # RECOMMENDED for token .bin (incompressible)
  python archive_unmerged.py --stream --chunk-size 100G # bigger chunks
  python archive_unmerged.py --stream --rclone-verbose  # rclone rcat -v for early progress
  python archive_unmerged.py --no-compress              # plain tar (faster, bigger)
  python archive_unmerged.py --no-upload                # build archive only (local-first)
  python archive_unmerged.py --skip-archive             # upload existing archive (local-first)
  python archive_unmerged.py --keep-archive             # retain local copy after upload
  python archive_unmerged.py --gdrive gdrive:other/path # custom Drive destination dir

Why --no-compress for token shards
----------------------------------
uint16 token .bin files are essentially random — gzip saves ~5% but is single-
threaded and CPU-bound at ~30-80 MB/s, which becomes the pipeline bottleneck
and delays the first visible upload progress by minutes. Plain tar streams at
disk-read speed (300+ MB/s) and rclone reports stats almost immediately.
"""

import os
import sys
import time
import shlex
import shutil
import subprocess
import argparse


DEFAULT_SHARD_DIR  = os.environ.get("TOKENIZER_SHARD_DIR", "/mnt/ssd/token_shards")
DEFAULT_GDRIVE     = "gdrive:synapse"
DEFAULT_CHUNK_SIZE = "50G"   # default streaming chunk size when --stream is set


def run(cmd, desc=""):
    """Run a command with stdout/stderr inheriting the terminal so progress is live."""
    print(f"  {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"  ERROR ({desc}): exit code {result.returncode}")
    return result


def check_rclone_remote(remote_uri: str):
    remote_name = remote_uri.split(":", 1)[0]
    r = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("rclone not installed or unreachable. Install rclone first.")
    if remote_name + ":" not in r.stdout:
        sys.exit(
            f"rclone remote '{remote_name}' not configured. "
            f"Run: rclone config (or set GDRIVE_REMOTE)."
        )


def stream_chunked(src_dir: str, gdrive_dir: str, chunk_size: str,
                   excludes: list, compress: bool, rclone_verbose: bool):
    """One-step pipeline: tar | split | rclone rcat — no local intermediate.

    Each chunk uploads as <archive_basename>.part-{aa,ab,ac,...} via a
    separate rclone rcat invocation, so a failure during chunk N doesn't
    waste the work for chunks 0..N-1. set -o pipefail makes upstream tar
    or split failures propagate as nonzero exit.
    """
    src_basename     = os.path.basename(src_dir)
    src_parent       = os.path.dirname(src_dir)
    archive_basename = src_basename + (".tar.gz" if compress else ".tar")

    tar_args = ["tar", "--create"]
    if compress:
        tar_args.append("--gzip")
    tar_args += ["--directory", src_parent]
    for ex in excludes:
        tar_args.append(f"--exclude={src_basename}/{ex}")
    tar_args += ["--totals", "--file=-", src_basename]
    tar_cmd = " ".join(shlex.quote(a) for a in tar_args)

    # split's --filter is invoked per-chunk; $FILE is the would-be filename.
    # Single-quoting the filter prevents the outer shell from expanding $FILE
    # before split sees it.
    verbose_flag = "-v " if rclone_verbose else ""
    rcat_filter = (
        f"rclone rcat {verbose_flag}"
        f"--stats=10s --stats-one-line "
        f"--drive-chunk-size=64M "
        f"{shlex.quote(gdrive_dir.rstrip('/'))}/$FILE"
    )
    split_cmd = (
        f"split --bytes={shlex.quote(chunk_size)} "
        f"--filter={shlex.quote(rcat_filter)} "
        f"- {shlex.quote(archive_basename + '.part-')}"
    )

    pipeline = f"set -o pipefail; {tar_cmd} | {split_cmd}"
    print(f"\n  Pipeline:")
    print(f"    {tar_cmd}")
    print(f"    | {split_cmd}")
    print(f"  Destination: {gdrive_dir.rstrip('/')}/{archive_basename}.part-*")
    print()
    t0 = time.time()
    result = subprocess.run(pipeline, shell=True, executable="/bin/bash")
    if result.returncode != 0:
        sys.exit(f"  ERROR: streaming pipeline exit code {result.returncode}")
    elapsed = time.time() - t0
    print(f"\n  Streaming upload done in {elapsed:.1f}s")
    return archive_basename


def main():
    parser = argparse.ArgumentParser(
        description="Archive the unmerged token_shards dir into one tar.gz and upload to Drive."
    )
    parser.add_argument("--dir", default=DEFAULT_SHARD_DIR,
                        help=f"Source dir to archive (default: {DEFAULT_SHARD_DIR})")
    parser.add_argument("--out", default=None,
                        help="Output archive path (default: <SHARD_DIR>/../<basename>.tar.gz)")
    parser.add_argument("--gdrive", default=DEFAULT_GDRIVE,
                        help=f"Drive destination dir (default: {DEFAULT_GDRIVE})")
    parser.add_argument("--no-compress", action="store_true",
                        help="Plain tar, no gzip (faster, larger)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Build the archive but skip the upload step")
    parser.add_argument("--skip-archive", action="store_true",
                        help="Skip building; upload an existing archive at --out")
    parser.add_argument("--keep-archive", action="store_true",
                        help="Don't delete the local archive after a successful upload")
    parser.add_argument("--exclude", action="append", default=["_quarantine"],
                        help="Subpaths to exclude from the archive (default: _quarantine)")
    parser.add_argument("--stream", action="store_true",
                        help="One-step streaming upload: tar | split | rclone rcat. "
                             "No local intermediate file. Recommended for >100 GB.")
    parser.add_argument("--chunk-size", default=DEFAULT_CHUNK_SIZE,
                        help=f"Chunk size for --stream mode (default: {DEFAULT_CHUNK_SIZE}). "
                             f"Bigger = fewer files, smaller = more retry granularity.")
    parser.add_argument("--rclone-verbose", action="store_true",
                        help="Pass -v to rclone rcat so it logs early ('Transferring...') "
                             "instead of waiting for the first 10s stats tick.")
    args = parser.parse_args()

    # Mutually-exclusive guards on streaming-mode flags
    if args.stream:
        if args.skip_archive:
            sys.exit("--stream and --skip-archive are mutually exclusive")
        if args.keep_archive:
            sys.exit("--stream produces no local archive; --keep-archive doesn't apply")
        if args.no_upload:
            sys.exit("--stream uploads as it builds; --no-upload doesn't apply. "
                     "Use the local-first flow without --stream.")

    src_dir = os.path.abspath(args.dir.rstrip("/"))
    if not os.path.isdir(src_dir):
        sys.exit(f"Source dir not found: {src_dir}")

    src_basename = os.path.basename(src_dir)
    src_parent   = os.path.dirname(src_dir)
    ext          = ".tar" if args.no_compress else ".tar.gz"
    out_path     = args.out or os.path.join(src_parent, src_basename + ext)
    gdrive_dir   = args.gdrive.rstrip("/")
    gdrive_path  = f"{gdrive_dir}/{os.path.basename(out_path)}"

    print(f"Source dir:  {src_dir}")
    if args.stream:
        print(f"Mode:        STREAM (chunked, no local file)")
        print(f"Chunk size:  {args.chunk_size}")
        print(f"Drive dest:  {gdrive_dir}/{src_basename + ext}.part-*")
    else:
        print(f"Archive:     {out_path}")
        print(f"Drive dest:  {gdrive_path}")
    print(f"Exclude:     {args.exclude}")
    print()

    # ----- STREAMING MODE -----
    if args.stream:
        check_rclone_remote(gdrive_dir)
        if not args.no_compress:
            print("  NOTE: gzip on uint16 token .bin saves ~5% but is single-threaded "
                  "and CPU-bound (~30-80 MB/s).\n        For faster runs and earlier "
                  "progress visibility, pass --no-compress.\n")
        archive_basename = stream_chunked(
            src_dir, gdrive_dir, args.chunk_size,
            args.exclude, compress=not args.no_compress,
            rclone_verbose=args.rclone_verbose,
        )
        print()
        print("=" * 60)
        print(f"DONE. Chunks on Drive: {gdrive_dir}/{archive_basename}.part-*")
        print("=" * 60)
        print("\nTo restore on a fresh VM:")
        print(f"  mkdir -p /mnt/ssd/restore && rclone copy {gdrive_dir}/ /mnt/ssd/restore "
              f"--include='{archive_basename}.part-*'")
        x_flag = "xzf" if not args.no_compress else "xf"
        print(f"  cd /mnt/ssd && cat restore/{archive_basename}.part-* | tar {x_flag} -")
        return

    # ----- BUILD ARCHIVE -----
    if not args.skip_archive:
        # Quick survey of what we're about to archive.
        total_files = 0
        total_bytes = 0
        for f in os.listdir(src_dir):
            if f in args.exclude:
                continue
            p = os.path.join(src_dir, f)
            if os.path.isfile(p):
                total_files += 1
                total_bytes += os.path.getsize(p)
            elif os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        total_files += 1
                        total_bytes += os.path.getsize(os.path.join(root, fn))
        print(f"Will archive {total_files:,} files ({total_bytes / 1024 / 1024 / 1024:.2f} GB raw)")

        # Disk-space sanity check on the archive's destination filesystem.
        free_bytes = shutil.disk_usage(os.path.dirname(out_path)).free
        # Conservative: assume tar+gz reduces by 0% in worst case (no compression).
        if free_bytes < total_bytes + 1024 * 1024 * 1024:
            print(f"  WARNING: free space at {os.path.dirname(out_path)} is "
                  f"{free_bytes / 1024 / 1024 / 1024:.1f} GB; archive may not fit. "
                  f"Pass --out to a roomier dir, or use --no-compress on a different volume.")

        if os.path.exists(out_path):
            print(f"  WARNING: {out_path} exists; removing before rebuild")
            os.remove(out_path)

        # tar -c [-z] -f <out> -C <parent> <basename> [--exclude=...]
        tar_cmd = ["tar", "--create"]
        if not args.no_compress:
            tar_cmd.append("--gzip")
        tar_cmd += ["--file", out_path, "--directory", src_parent]
        for excl in args.exclude:
            tar_cmd.append(f"--exclude={src_basename}/{excl}")
        tar_cmd += ["--totals", src_basename]

        print(f"\n  Creating archive (tar{'.gz' if not args.no_compress else ''})...")
        t0 = time.time()
        run(tar_cmd, "tar")
        elapsed = time.time() - t0
        archive_size = os.path.getsize(out_path)
        ratio = archive_size / total_bytes if total_bytes else 1.0
        print(f"  Done in {elapsed:.1f}s — {archive_size / 1024 / 1024 / 1024:.2f} GB "
              f"(ratio {ratio:.2f}, "
              f"{(total_bytes - archive_size) / 1024 / 1024 / 1024:.2f} GB saved by compression)")
    else:
        if not os.path.exists(out_path):
            sys.exit(f"--skip-archive requires {out_path} to exist")
        print(f"  Using existing archive: {out_path} "
              f"({os.path.getsize(out_path) / 1024 / 1024 / 1024:.2f} GB)")

    if args.no_upload:
        print(f"\nUpload skipped (--no-upload). Archive at: {out_path}")
        return

    # ----- UPLOAD -----
    check_rclone_remote(gdrive_dir)

    print(f"\n  Uploading to {gdrive_path}...  (rclone --progress, live)")
    rclone_cmd = [
        "rclone", "copyto", out_path, gdrive_path,
        "--progress",
        "--stats=5s", "--stats-one-line",
        "--drive-chunk-size=64M",
        "--transfers=1", "--checkers=1",   # one big file — parallelism doesn't help
    ]
    t0 = time.time()
    run(rclone_cmd, "rclone upload")
    print(f"  Upload done in {time.time() - t0:.1f}s")

    # ----- CLEANUP -----
    if not args.keep_archive:
        os.remove(out_path)
        print(f"\n  Removed local archive {out_path} (pass --keep-archive to retain)")
    else:
        print(f"\n  Local archive retained at {out_path}")

    print()
    print("=" * 60)
    print(f"DONE. Archive on Drive: {gdrive_path}")
    print("=" * 60)
    print("\nTo restore on a fresh VM:")
    print(f"  rclone copy {gdrive_path} /mnt/ssd/")
    extract_flag = "xzf" if not args.no_compress else "xf"
    print(f"  cd /mnt/ssd && tar {extract_flag} {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
