#!/usr/bin/env python3
"""
Smart sync between Google Drive and local SSD for the tokenizer pipeline.
Copies only new/changed files from Drive to local, and uploads only new
outputs from local back to Drive.

Uses rclone under the hood. Requires rclone configured with a remote named
'gdrive' pointing to your Google Drive.

Usage:
  python sync_from_drive.py --download         # copy source data Drive -> local
  python sync_from_drive.py --upload           # copy shards local -> Drive
  python sync_from_drive.py --download --upload  # both

Paths (override via env vars):
  GDRIVE_REMOTE  — rclone remote name (default: gdrive)
  GDRIVE_PATH    — base path on Drive (default: synapse)
  LOCAL_PATH     — base path on local SSD (default: /mnt/ssd)
  SHARD_DIR      — shard output dir name (default: token_shards)
  TOKENIZER_DIR  — tokenizer output dir name (default: tokenizer_out)
  MANIFEST_DIR   — manifest dir name (default: manifests)
"""

import os
import sys
import json
import subprocess
import argparse
import hashlib

GDRIVE_REMOTE = os.environ.get("GDRIVE_REMOTE", "gdrive")
GDRIVE_PATH   = os.environ.get("GDRIVE_PATH", "synapse")
LOCAL_PATH    = os.environ.get("LOCAL_PATH", "/mnt/ssd")
SHARD_NAME    = os.environ.get("SHARD_DIR_NAME", "token_shards")
TOK_NAME      = os.environ.get("TOKENIZER_DIR_NAME", "tokenizer_out")
MAN_NAME      = os.environ.get("MANIFEST_DIR_NAME", "manifests")
DATA_NAME     = os.environ.get("DATA_DIR_NAME", "datasets_pretrain")

# Equivalent to the Colab paths:
#   Drive: gdrive:synapse/datasets_pretrain  →  local: /mnt/ssd/datasets_pretrain
#   Drive: gdrive:synapse/token_shards       →  local: /mnt/ssd/token_shards
#   Drive: gdrive:synapse/manifests          →  local: /mnt/ssd/manifests

GDRIVE_DATA          = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{DATA_NAME}"
GDRIVE_SHARDS        = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{SHARD_NAME}"
GDRIVE_SHARDS_MERGED = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{SHARD_NAME}_merged"
GDRIVE_TOK           = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{DATA_NAME}/{TOK_NAME}"
GDRIVE_MAN           = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{MAN_NAME}"

LOCAL_DATA          = os.path.join(LOCAL_PATH, DATA_NAME)
LOCAL_SHARDS        = os.path.join(LOCAL_PATH, SHARD_NAME)
LOCAL_SHARDS_MERGED = os.path.join(LOCAL_PATH, SHARD_NAME + "_merged")
LOCAL_TOK           = os.path.join(LOCAL_PATH, DATA_NAME, TOK_NAME)
LOCAL_MAN           = os.path.join(LOCAL_PATH, MAN_NAME)

# shard_manifest.json on local and Drive — used to skip already-copied shards
LOCAL_MANIFEST_FILE = os.path.join(LOCAL_SHARDS, "shard_manifest.json")
DRIVE_MANIFEST_FILE = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/{SHARD_NAME}/shard_manifest.json"
TEMP_MANIFEST_DIR   = "/tmp/synapse_sync_manifests"


def run_rclone(args: list, desc: str = ""):
    """Run rclone with output captured. Use for non-transfer commands (listremotes,
    manifest fetches) where the caller wants the stdout/stderr text back."""
    cmd = ["rclone"] + args
    print(f"  rclone {' '.join(args[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR ({desc}): {result.stderr.strip()}")
    return result


def run_rclone_streaming(args: list, desc: str = ""):
    """Run rclone with stdout/stderr streamed to the terminal so --progress and
    --stats render live. Use for all transfer commands (copy/sync). Returns the
    completed CompletedProcess (with empty stdout/stderr — output already shown)."""
    cmd = ["rclone"] + args
    # Force frequent stats so the user sees ongoing progress without --progress
    # taking over the terminal completely (rclone defaults to 1m stats interval).
    if not any(a.startswith("--stats") for a in args):
        cmd += ["--stats=5s", "--stats-one-line"]
    print(f"  rclone {' '.join(args[:4])}...  (live progress below)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR ({desc}): rclone exited with code {result.returncode}")
    return result


def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_drive_manifest():
    """Download the Drive shard_manifest.json to temp to check what's already sharded."""
    os.makedirs(TEMP_MANIFEST_DIR, exist_ok=True)
    local_copy = os.path.join(TEMP_MANIFEST_DIR, "shard_manifest.json")
    result = run_rclone(
        ["copy", DRIVE_MANIFEST_FILE, TEMP_MANIFEST_DIR, "--verbose"],
        "download Drive manifest",
    )
    if result.returncode != 0 or not os.path.exists(local_copy):
        return None
    with open(local_copy) as f:
        return json.load(f)


def download_source_data():
    """Smart download: rclone copy only new/changed files."""
    print(f"\n{'='*60}")
    print(f"DOWNLOAD: Drive -> Local")
    print(f"  From: {GDRIVE_DATA}")
    print(f"  To:   {LOCAL_DATA}")
    print(f"{'='*60}")

    os.makedirs(LOCAL_DATA, exist_ok=True)
    os.makedirs(LOCAL_SHARDS, exist_ok=True)
    os.makedirs(LOCAL_TOK, exist_ok=True)
    os.makedirs(LOCAL_MAN, exist_ok=True)

    # Download everything in datasets_pretrain (source .txt files + tokenizer_out)
    run_rclone_streaming(
        [
            "copy", GDRIVE_DATA, LOCAL_DATA,
            "--progress", "--transfers=8", "--drive-chunk-size=64M",
            "--checksum",  # use md5 not modtime
            "--exclude=*/tokenizer_out/**",
        ],
        "download source data",
    )

    # Download tokenizer separately so we can exclude from above cleanly
    run_rclone_streaming(
        [
            "copy", GDRIVE_TOK, LOCAL_TOK,
            "--progress", "--transfers=4",
        ],
        "download tokenizer output",
    )

    # Download existing shards/manifests so tokenizer can skip already-tokenized files
    run_rclone_streaming(
        [
            "copy", GDRIVE_SHARDS, LOCAL_SHARDS,
            "--progress", "--transfers=8",
        ],
        "download existing shards",
    )
    run_rclone_streaming(
        [
            "copy", GDRIVE_MAN, LOCAL_MAN,
            "--progress", "--transfers=4",
        ],
        "download manifests",
    )

    print("\n  Download complete.")


def upload_outputs():
    """Upload shards + manifests + tokenizer back to Drive."""
    print(f"\n{'='*60}")
    print(f"UPLOAD: Local -> Drive")
    print(f"  From: {LOCAL_SHARDS}")
    print(f"  To:   {GDRIVE_SHARDS}")
    print(f"{'='*60}")

    # Check if Drive already has a shard_manifest.json — if so, only upload
    # shards that are missing or changed.
    drive_manifest = load_drive_manifest()

    if drive_manifest and os.path.exists(LOCAL_MANIFEST_FILE):
        with open(LOCAL_MANIFEST_FILE) as f:
            local_manifest = json.load(f)

        local_shards = {s["shard"]: s for s in local_manifest.get("shards", [])}
        drive_shards = {s["shard"]: s for s in drive_manifest.get("shards", [])}

        # Find shards that exist locally but not on Drive (or have changed)
        to_upload = []
        for shard_name, entry in local_shards.items():
            if shard_name not in drive_shards:
                to_upload.append(shard_name)
            elif entry.get("source_hash") != drive_shards[shard_name].get("source_hash"):
                to_upload.append(shard_name)

        print(f"  Local shards:  {len(local_shards)}")
        print(f"  Drive shards:  {len(drive_shards)}")
        print(f"  To upload:     {len(to_upload)}")

        if to_upload:
            # Single rclone call with --files-from so the user sees aggregate
            # live progress (bytes transferred / ETA / per-file bar) and uploads
            # run in parallel via --transfers. The previous one-rclone-per-shard
            # loop was both slow (process spawn overhead) and silent (output
            # captured) so the user couldn't tell anything was happening.
            os.makedirs(TEMP_MANIFEST_DIR, exist_ok=True)
            files_from_path = os.path.join(TEMP_MANIFEST_DIR, "upload_shards.txt")
            with open(files_from_path, "w") as f:
                for shard_name in to_upload:
                    local_path = os.path.join(LOCAL_SHARDS, shard_name)
                    if os.path.exists(local_path):
                        f.write(shard_name + "\n")
            print(f"\n  Uploading {len(to_upload)} shards (aggregate progress)...")
            result = run_rclone_streaming(
                [
                    "copy", LOCAL_SHARDS, GDRIVE_SHARDS,
                    f"--files-from={files_from_path}",
                    "--progress", "--transfers=8", "--drive-chunk-size=64M",
                ],
                "upload changed shards",
            )
            if result.returncode != 0:
                print(f"    WARNING: shard upload returned non-zero ({result.returncode})")

        # Always upload the manifest/meta files so Drive is in sync.
        sidecar_files = [
            f for f in ["shard_manifest.json", "meta.json", "tokenization_id.txt"]
            if os.path.exists(os.path.join(LOCAL_SHARDS, f))
        ]
        if sidecar_files:
            print(f"\n  Uploading manifest sidecars ({', '.join(sidecar_files)})...")
            os.makedirs(TEMP_MANIFEST_DIR, exist_ok=True)
            sidecar_files_from = os.path.join(TEMP_MANIFEST_DIR, "upload_sidecars.txt")
            with open(sidecar_files_from, "w") as f:
                for name in sidecar_files:
                    f.write(name + "\n")
            run_rclone_streaming(
                [
                    "copy", LOCAL_SHARDS, GDRIVE_SHARDS,
                    f"--files-from={sidecar_files_from}",
                    "--progress",
                ],
                "upload manifest sidecars",
            )
    else:
        # No Drive manifest — upload everything
        print("  No existing Drive manifest found. Uploading all shards...")
        run_rclone_streaming(
            [
                "copy", LOCAL_SHARDS, GDRIVE_SHARDS,
                "--progress", "--transfers=8", "--drive-chunk-size=64M",
            ],
            "upload all shards",
        )

    # Upload tokenizer output and manifests
    print("\n  Uploading tokenizer output...")
    run_rclone_streaming(
        [
            "copy", LOCAL_TOK, GDRIVE_TOK,
            "--progress", "--transfers=4",
        ],
        "upload tokenizer output",
    )

    print("  Uploading manifests...")
    run_rclone_streaming(
        [
            "copy", LOCAL_MAN, GDRIVE_MAN,
            "--progress", "--transfers=4",
        ],
        "upload manifests",
    )

    # Mirror the merged shard dir if it exists. Merged shards are uniform-size
    # buckets produced by run_tokenizer.py's step 4 (or merge_shards.py).
    if os.path.isdir(LOCAL_SHARDS_MERGED):
        print(f"\n  Uploading merged shards: {LOCAL_SHARDS_MERGED} -> {GDRIVE_SHARDS_MERGED}")
        run_rclone_streaming(
            [
                "copy", LOCAL_SHARDS_MERGED, GDRIVE_SHARDS_MERGED,
                "--progress", "--transfers=8", "--drive-chunk-size=64M",
            ],
            "upload merged shards",
        )

    print("\n  Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync tokenizer data between Drive and local SSD")
    parser.add_argument("--download", action="store_true", help="Download source data from Drive")
    parser.add_argument("--upload", action="store_true", help="Upload shards/outputs to Drive")
    args = parser.parse_args()

    if not args.download and not args.upload:
        parser.print_help()
        sys.exit(1)

    # Validate rclone configured
    result = subprocess.run(
        ["rclone", "listremotes"],
        capture_output=True, text=True,
    )
    if GDRIVE_REMOTE + ":" not in result.stdout:
        print(f"ERROR: rclone remote '{GDRIVE_REMOTE}' not found.")
        print(f"  Configure it first: rclone config")
        print(f"  Or set GDRIVE_REMOTE env var to your remote name.")
        sys.exit(1)

    if args.download:
        download_source_data()
    if args.upload:
        upload_outputs()

    print("\nDone.")
