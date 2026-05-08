#!/usr/bin/env python3
"""
Upload tokenizer outputs from local SSD to Google Drive.

Workflow this script supports: tokenize on a non-Colab machine, then push the
results to Drive so the Colab pretraining notebook can read them. Uses rclone
under the hood; requires an rclone remote pointing at the Drive root.

What gets uploaded (each step skips if the local dir is missing):
  - token_shards_merged/  -> trainer input (~1 GB shards), incremental
  - tokenizer_out/        -> tokenizer.json + eval reports (top-level on Drive)
  - manifests/            -> reproducibility manifests

What is NOT uploaded:
  - token_shards/         -> cold backup. Lives on Drive as a 6-part tar
                             archive; do not overwrite it from here. Re-archive
                             manually if you need to refresh it.
  - datasets_pretrain/    -> source data is not produced by tokenization.

Usage:
  python upload_to_drive.py             # upload everything that exists locally
  python upload_to_drive.py --dry-run   # show plan, don't transfer

Paths (override via env vars):
  GDRIVE_REMOTE        — rclone remote name (default: gdrive)
  GDRIVE_PATH          — base path on Drive (default: synapse)
  LOCAL_PATH           — base path on local SSD (default: /mnt/ssd)
  SHARD_DIR_NAME       — merged-shard dir name (default: token_shards_merged)
  TOKENIZER_DIR_NAME   — tokenizer dir name   (default: tokenizer_out)
  MANIFEST_DIR_NAME    — manifest dir name    (default: manifests)
"""

import os
import sys
import json
import subprocess
import argparse

GDRIVE_REMOTE = os.environ.get("GDRIVE_REMOTE", "gdrive")
GDRIVE_PATH   = os.environ.get("GDRIVE_PATH", "synapse")
LOCAL_PATH    = os.environ.get("LOCAL_PATH", "/mnt/ssd")
MERGED_NAME   = os.environ.get("SHARD_DIR_NAME", "token_shards_merged")
TOK_NAME      = os.environ.get("TOKENIZER_DIR_NAME", "tokenizer_out")
MAN_NAME      = os.environ.get("MANIFEST_DIR_NAME", "manifests")

GDRIVE_MERGED = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{MERGED_NAME}"
GDRIVE_TOK    = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{TOK_NAME}"
GDRIVE_MAN    = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{MAN_NAME}"

LOCAL_MERGED  = os.path.join(LOCAL_PATH, MERGED_NAME)
LOCAL_TOK     = os.path.join(LOCAL_PATH, TOK_NAME)
LOCAL_MAN     = os.path.join(LOCAL_PATH, MAN_NAME)


def run_rclone_streaming(args, desc=""):
    """Stream rclone stdout/stderr live so --progress and --stats render."""
    cmd = ["rclone"] + args
    if not any(a.startswith("--stats") for a in args):
        cmd += ["--stats=5s", "--stats-one-line"]
    print(f"  rclone {' '.join(args[:4])}...  (live progress below)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"rclone failed for {desc!r} (exit {result.returncode})")
    return result


def assert_tokenization_id(local_dir):
    """Fail loud if a shard dir's meta.json and tokenization_id.txt disagree."""
    meta_path  = os.path.join(local_dir, "meta.json")
    tokid_path = os.path.join(local_dir, "tokenization_id.txt")
    if not (os.path.exists(meta_path) and os.path.exists(tokid_path)):
        return
    with open(meta_path) as f:
        meta = json.load(f)
    meta_id  = meta.get("tokenization_id")
    tokid_id = open(tokid_path).read().strip()
    if meta_id != tokid_id:
        raise RuntimeError(
            f"tokenization_id mismatch in {local_dir}: "
            f"meta.json={meta_id!r}, tokenization_id.txt={tokid_id!r}"
        )


def upload_dir(local, remote, label, dry_run, transfers=8):
    if not os.path.isdir(local):
        print(f"  [skip] {label}: {local} does not exist")
        return
    print(f"\n  Uploading {label}: {local} -> {remote}")
    args = ["copy", local, remote, "--progress",
            f"--transfers={transfers}", "--drive-chunk-size=64M",
            "--checksum"]
    if dry_run:
        args.append("--dry-run")
    run_rclone_streaming(args, label)


def upload_outputs(dry_run=False):
    print(f"{'='*60}")
    print(f"UPLOAD: Local -> Drive ({'DRY RUN' if dry_run else 'live'})")
    print(f"  Drive root: {GDRIVE_REMOTE}:{GDRIVE_PATH}")
    print(f"  Local root: {LOCAL_PATH}")
    print(f"{'='*60}")

    assert_tokenization_id(LOCAL_MERGED)

    upload_dir(LOCAL_MERGED, GDRIVE_MERGED, "merged shards (trainer input)", dry_run)
    upload_dir(LOCAL_TOK,    GDRIVE_TOK,    "tokenizer artifact",            dry_run, transfers=4)
    upload_dir(LOCAL_MAN,    GDRIVE_MAN,    "manifests",                     dry_run, transfers=4)

    print("\n  Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload tokenizer outputs to Google Drive")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without transferring")
    args = parser.parse_args()

    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    if GDRIVE_REMOTE + ":" not in result.stdout:
        print(f"ERROR: rclone remote '{GDRIVE_REMOTE}' not found.")
        print(f"  Configure it first: rclone config")
        print(f"  Or set GDRIVE_REMOTE env var to your remote name.")
        sys.exit(1)

    upload_outputs(dry_run=args.dry_run)
    print("\nDone.")
