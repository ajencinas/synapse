#!/usr/bin/env python3
"""
ONE-OFF: merge token shards locally, then full-upload tokenization artifacts to Drive.

Use this from the VM when you want to (re)populate the merged-shards dir on Drive
quickly, without sync_from_drive.py's manifest-diff machinery. Designed for the
flow where you've manually emptied gdrive:synapse/token_shards_merged/ and want
to push everything fresh.

What gets uploaded (full content, no diffing):
  - All .bin files in <SHARD_DIR>_merged
  - shard_manifest.json, meta.json, tokenization_id.txt   (merged-dir sidecars)
  - tokenizer.json, tokenizer_eval.json                    (tokenizer artifacts)
  - manifests/tokenization_latest.json                     (run manifest)

What is NOT uploaded:
  - Unmerged token_shards/ (handled by sync_from_drive.py)
  - bpe_subset.txt, bpe_eval_sample.txt (BPE training corpora — large + rebuildable)
  - Anything in pretrain/ or training run state

Destination on Drive:
  gdrive:synapse/token_shards_merged/                      (merged shards + sidecars)
  gdrive:synapse/datasets_pretrain/tokenizer_out/          (tokenizer.json + eval)
  gdrive:synapse/manifests/                                (tokenization_latest.json)

Usage:
  python merge_and_upload.py                # merge then upload, target=100MB
  python merge_and_upload.py --no-merge     # already merged, just upload
  python merge_and_upload.py --no-upload    # merge only, skip upload
  python merge_and_upload.py --target-bytes 209715200    # 200 MB shards
  python merge_and_upload.py --force        # rebuild merged shards even if fingerprint matches

Delete this file once the initial Drive population is done — sync_from_drive.py
handles incremental updates from then on.
"""

import os
import sys
import subprocess
import argparse

# Reuse env vars from run_tokenizer.py / sync_from_drive.py
SHARD_DIR = os.environ.get("TOKENIZER_SHARD_DIR",    "/mnt/ssd/token_shards")
TOK_DIR   = os.environ.get("TOKENIZER_OUT_DIR",      "/mnt/ssd/tokenizer_out")
MAN_DIR   = os.environ.get("TOKENIZER_MANIFEST_DIR", "/mnt/ssd/manifests")

GDRIVE_REMOTE = os.environ.get("GDRIVE_REMOTE",        "gdrive")
GDRIVE_PATH   = os.environ.get("GDRIVE_PATH",          "synapse")
SHARD_NAME    = os.environ.get("SHARD_DIR_NAME",       "token_shards")
TOK_NAME      = os.environ.get("TOKENIZER_DIR_NAME",   "tokenizer_out")
MAN_NAME      = os.environ.get("MANIFEST_DIR_NAME",    "manifests")
DATA_NAME     = os.environ.get("DATA_DIR_NAME",        "datasets_pretrain")

DEFAULT_TARGET_BYTES = 100 * 1024 * 1024   # 100 MB

GDRIVE_MERGED = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{SHARD_NAME}_merged"
GDRIVE_TOK    = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{DATA_NAME}/{TOK_NAME}"
GDRIVE_MAN    = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{MAN_NAME}"


def run_rclone_streaming(args, desc=""):
    """Run rclone with stdout/stderr inheriting the terminal so --progress and
    --stats render live. The previous sync script captured output, so the user
    saw nothing for many minutes per file — that's the bug we're avoiding."""
    cmd = ["rclone"] + args
    if not any(a.startswith("--stats") for a in args):
        cmd += ["--stats=5s", "--stats-one-line"]
    print(f"  rclone {' '.join(args[:4])}...  (live progress below)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR ({desc}): rclone exited code {result.returncode}")
    return result


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge shards then full-upload tokenization artifacts to Drive."
    )
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES,
                        help=f"Merge target shard size (default {DEFAULT_TARGET_BYTES} = 100 MB)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merge; assumes <SHARD_DIR>_merged already exists")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip upload; only run merge")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild merged shards even if fingerprint matches")
    args = parser.parse_args()

    merged_dir = SHARD_DIR.rstrip("/") + "_merged"

    # ---- 1. MERGE ----
    if not args.no_merge:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from merge_shards import merge_shards
        section(f"STEP 1: MERGE  {SHARD_DIR}  ->  {merged_dir}")
        merge_shards(SHARD_DIR, merged_dir, args.target_bytes, force=args.force)
    else:
        print("Merge skipped (--no-merge)")

    if not os.path.isdir(merged_dir):
        sys.exit(f"Merged dir not found: {merged_dir}")

    if args.no_upload:
        print(f"\nUpload skipped (--no-upload). Merged dir at: {merged_dir}")
        return

    # ---- 2. VALIDATE rclone ----
    r = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    if r.returncode != 0 or GDRIVE_REMOTE + ":" not in r.stdout:
        sys.exit(
            f"rclone remote '{GDRIVE_REMOTE}' not configured.\n"
            f"  Configure with: rclone config\n"
            f"  Or set GDRIVE_REMOTE env var."
        )

    # ---- 3. UPLOAD MERGED DIR ----
    # All .bin + sidecars. --include filters keep stray .tmp files out and
    # ensure rclone doesn't push anything unintended (e.g. earlier debugging).
    section(f"STEP 2: UPLOAD MERGED DIR  ->  {GDRIVE_MERGED}")
    bin_files = sorted(f for f in os.listdir(merged_dir) if f.endswith(".bin"))
    bin_bytes = sum(os.path.getsize(os.path.join(merged_dir, f)) for f in bin_files)
    print(f"  Source:  {merged_dir}")
    print(f"  Files:   {len(bin_files)} .bin shards ({bin_bytes / 1024 / 1024 / 1024:.2f} GB)")
    print(f"           + sidecars (shard_manifest.json, meta.json, tokenization_id.txt)")
    run_rclone_streaming(
        [
            "copy", merged_dir, GDRIVE_MERGED,
            "--include=*.bin",
            "--include=shard_manifest.json",
            "--include=meta.json",
            "--include=tokenization_id.txt",
            "--progress",
            "--transfers=8",
            "--drive-chunk-size=64M",
        ],
        "upload merged shards",
    )

    # ---- 4. UPLOAD TOKENIZER ARTIFACTS ----
    if os.path.isdir(TOK_DIR):
        section(f"STEP 3: UPLOAD TOKENIZER  ->  {GDRIVE_TOK}")
        # Tokenizer.json + eval report. Skip bpe_subset.txt and bpe_eval_sample.txt
        # which are large training corpora rebuildable from source data.
        run_rclone_streaming(
            [
                "copy", TOK_DIR, GDRIVE_TOK,
                "--include=tokenizer.json",
                "--include=tokenizer_eval.json",
                "--progress",
                "--transfers=4",
            ],
            "upload tokenizer artifacts",
        )
    else:
        print(f"\n  Skipping tokenizer dir (not found): {TOK_DIR}")

    # ---- 5. UPLOAD TOKENIZATION RUN MANIFEST ----
    tok_manifest = os.path.join(MAN_DIR, "tokenization_latest.json")
    if os.path.isfile(tok_manifest):
        section(f"STEP 4: UPLOAD RUN MANIFEST  ->  {GDRIVE_MAN}")
        run_rclone_streaming(
            [
                "copy", MAN_DIR, GDRIVE_MAN,
                "--include=tokenization_latest.json",
                "--progress",
            ],
            "upload tokenization manifest",
        )
    else:
        print(f"\n  Skipping run manifest (not found): {tok_manifest}")

    section("DONE")
    print(f"  Merged dir: {merged_dir}")
    print(f"  Drive:      {GDRIVE_MERGED}")


if __name__ == "__main__":
    main()
