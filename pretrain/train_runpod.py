#!/usr/bin/env python3
"""SynapseGPT pretraining on RunPod — pulls only selected shards from
Google Drive via rclone, trains locally, pushes checkpoints back.

Usage (on RunPod):
  1. rclone config  (set up remote named "gdrive")
  2. python3 pretrain/train_runpod.py

Environment variables (all optional):
  GDRIVE_REMOTE       rclone remote name (default: gdrive)
  GDRIVE_PATH         Base path on Drive (default: synapse)
  LOCAL_DIR           Local working directory (default: /workspace/synapse_data)
  MAX_TOKENS          Token budget (default: 5_000_000_000 ≈ ~10 GB)
  CHECKPOINT_NAME     Checkpoint filename
  SKIP_DATA_PULL      If "1", skip pulling shards (use existing local)
"""
import os
import sys
import time
import json
import random
import subprocess

GDRIVE_REMOTE = os.environ.get("GDRIVE_REMOTE", "gdrive")
GDRIVE_PATH = os.environ.get("GDRIVE_PATH", "synapse")
LOCAL_DIR = os.environ.get("LOCAL_DIR", "/workspace/synapse_data")
DEFAULT_TOKEN_BUDGET = 5_000_000_000
TOKEN_BUDGET = int(os.environ.get("MAX_TOKENS", os.environ.get("TOKEN_BUDGET", str(DEFAULT_TOKEN_BUDGET))))
SKIP_DATA_PULL = os.environ.get("SKIP_DATA_PULL", "0") == "1"

SHARD_REMOTE = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/token_shards_merged"
CKPT_REMOTE = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/checkpoints"
MANIFEST_REMOTE = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/manifests"

SYNAPSE_LOCAL = os.path.join(LOCAL_DIR, "synapse")
SHARD_LOCAL = os.path.join(SYNAPSE_LOCAL, "token_shards_merged")
CKPT_LOCAL = os.path.join(SYNAPSE_LOCAL, "checkpoints")
MANIFEST_LOCAL = os.path.join(SYNAPSE_LOCAL, "manifests")


def run(cmd, desc=None, check=True, capture=False):
    if desc:
        print(f"  {desc}...")
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=3600)
    except FileNotFoundError:
        print(f"ERROR: command not found: {cmd[0]}")
        sys.exit(1)
    if check and r.returncode != 0:
        print(f"ERROR: {' '.join(cmd)} failed (code {r.returncode})")
        if capture:
            print(r.stderr[:500])
        sys.exit(1)
    return r


def main():
    print("=" * 60)
    print("  SynapseGPT — RunPod Launcher")
    print(f"  Remote: {GDRIVE_REMOTE}:{GDRIVE_PATH}")
    print(f"  Local:  {LOCAL_DIR}")
    print(f"  Budget: {TOKEN_BUDGET:,} tokens")
    print("=" * 60)

    # ── 1. Check rclone remote ──
    r = run(["rclone", "listremotes"], capture=True)
    if f"{GDRIVE_REMOTE}:" not in r.stdout:
        print(f"ERROR: rclone remote '{GDRIVE_REMOTE}:' not configured.")
        print("Run 'rclone config' to set it up, then re-run.")
        sys.exit(1)
    print(f"  remote '{GDRIVE_REMOTE}:' OK\n")

    os.makedirs(SHARD_LOCAL, exist_ok=True)
    os.makedirs(CKPT_LOCAL, exist_ok=True)
    os.makedirs(MANIFEST_LOCAL, exist_ok=True)

    # ── 2. Pull metadata (manifest, meta, tokid) ──
    for f in ("meta.json", "shard_manifest.json", "tokenization_id.txt"):
        dst = os.path.join(SHARD_LOCAL, f)
        if not os.path.exists(dst):
            run(["rclone", "copyto",
                 f"{SHARD_REMOTE}/{f}", dst,
                 "--checksum"],
                desc=f"pulling {f}")

    # ── 3. Pull existing checkpoint + manifest for resume (skip if remote dir empty) ──
    ckpt_exists = subprocess.run(
        ["rclone", "ls", CKPT_REMOTE, "--max-depth", "1"],
        capture_output=True, text=True, timeout=30
    )
    if ckpt_exists.returncode == 0 and ckpt_exists.stdout.strip():
        run(["rclone", "copy", CKPT_REMOTE, CKPT_LOCAL,
             "--include", "*.pth", "--max-age", "30d",
             "--transfers", "1", "--drive-chunk-size", "64M", "--checksum"],
            desc="pulling existing checkpoint", check=False)
    else:
        print("  no existing checkpoint found on Drive")

    run(["rclone", "copy", MANIFEST_REMOTE, MANIFEST_LOCAL,
         "--include", "training_latest.json",
         "--transfers", "1", "--checksum"],
        desc="pulling existing manifest", check=False)

    # ── 4. Select shards within budget ──
    if SKIP_DATA_PULL:
        print("[data] SKIP_DATA_PULL=1 — skipping shard pull, using existing local")
    else:
        with open(os.path.join(SHARD_LOCAL, "shard_manifest.json")) as f:
            manifest = json.load(f)

        shards = manifest["shards"]
        random.seed(42)
        random.shuffle(shards)

        selected = []
        running = 0
        for s in shards:
            if running >= TOKEN_BUDGET:
                break
            selected.append(s)
            running += s["tokens"]

        print(f"\n[data] Selected {len(selected)} shards ({running:,} tokens) "
              f"from {len(shards)} available")

        # ── 5. Pull only selected shards via rclone copy ──
        # Write a temp file list for rclone's --files-from
        list_path = os.path.join(LOCAL_DIR, "selected_shards.txt")
        with open(list_path, "w") as f:
            for s in selected:
                f.write(s["shard"] + "\n")

        print(f"[data] Pulling {len(selected)} shards to {SHARD_LOCAL}...")
        t0 = time.time()
        run(["rclone", "copy", SHARD_REMOTE, SHARD_LOCAL,
             "--files-from", list_path,
             "--transfers", "8", "--drive-chunk-size", "64M",
             "--checksum", "--progress"],
            desc="downloading selected shards")
        print(f"[data] Done in {time.time()-t0:.0f}s")
        os.remove(list_path)

    # ── 6. Launch train.py ──
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    if not os.path.exists(train_script):
        train_script = os.path.join(os.getcwd(), "pretrain", "train.py")

    env = os.environ.copy()
    env["SYNAPSE_DIR"] = SYNAPSE_LOCAL
    env["SKIP_DRIVE_MOUNT"] = "1"
    env["SKIP_STAGE"] = "1"
    env["CHECKPOINT_PUSH_REMOTE"] = CKPT_REMOTE
    env["MAX_TOKENS"] = str(TOKEN_BUDGET)

    print(f"\n[{int(time.time())}] Launching train.py...")
    print(f"  SYNAPSE_DIR={env['SYNAPSE_DIR']}")
    print(f"  CHECKPOINT_PUSH_REMOTE={env['CHECKPOINT_PUSH_REMOTE']}")
    print(f"  MAX_TOKENS={env['MAX_TOKENS']}")
    print()

    os.execve(sys.executable, [sys.executable, train_script], env)


if __name__ == "__main__":
    main()
