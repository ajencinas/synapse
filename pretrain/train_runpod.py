#!/usr/bin/env python3
"""SynapseGPT pretraining on RunPod — mounts Google Drive via rclone FUSE,
stages only the selected shards to local SSD, runs train.py, pushes checkpoints back.

Usage (on RunPod):
  1. Configure rclone remote named "gdrive" pointing at your Google Drive.
  2. Ensure shards exist at gdrive:synapse/token_shards_merged/.
  3. python3 pretrain/train_runpod.py

Environment variables (all optional):
  GDRIVE_REMOTE       rclone remote name (default: gdrive)
  GDRIVE_PATH         Base path on Drive (default: synapse)
  LOCAL_DIR           Local working directory (default: /workspace/synapse_data)
  MOUNT_DIR           rclone mount point (default: LOCAL_DIR/gdrive)
  MAX_TOKENS          Token budget passed to train.py (default: 5_000_000_000)
  CHECKPOINT_NAME     Checkpoint filename passed to train.py
  SKIP_MOUNT          If "1", skip rclone mount (assume already mounted)
  TOKEN_BUDGET        Same as MAX_TOKENS (fallback)
"""
import os
import sys
import time
import subprocess
import argparse

# ── Config ──────────────────────────────────────────────────────────────────
GDRIVE_REMOTE = os.environ.get("GDRIVE_REMOTE", "gdrive")
GDRIVE_PATH = os.environ.get("GDRIVE_PATH", "synapse")
LOCAL_DIR = os.environ.get("LOCAL_DIR", "/workspace/synapse_data")
MOUNT_DIR = os.environ.get("MOUNT_DIR", os.path.join(LOCAL_DIR, "gdrive"))
SKIP_MOUNT = os.environ.get("SKIP_MOUNT", "0") == "1"

# Token budget default: 5B (~10 GB of shards) — just enough for a meaningful test run.
DEFAULT_TOKEN_BUDGET = 5_000_000_000

SHARD_REMOTE = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/token_shards_merged"
CKPT_REMOTE = f"{GDRIVE_REMOTE}:{GDRIVE_PATH}/checkpoints"

SYNAPSE_LOCAL = os.path.join(LOCAL_DIR, "synapse")
SHARD_LOCAL = os.path.join(SYNAPSE_LOCAL, "token_shards_merged")
CKPT_LOCAL = os.path.join(SYNAPSE_LOCAL, "checkpoints")
MANIFEST_LOCAL = os.path.join(SYNAPSE_LOCAL, "manifests")


def run(cmd, desc=None, check=True, capture=False, timeout=None):
    if desc:
        print(f"  {desc}...")
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    else:
        r = subprocess.run(cmd, timeout=timeout)
    if check and r.returncode != 0:
        print(f"ERROR: {' '.join(cmd)} failed (code {r.returncode})")
        if capture:
            print(r.stderr[:500])
        sys.exit(1)
    return r


def check_rclone_remote():
    r = run(["rclone", "listremotes"], capture=True)
    if f"{GDRIVE_REMOTE}:" not in r.stdout:
        print(f"ERROR: rclone remote '{GDRIVE_REMOTE}:' not configured.")
        print("Run 'rclone config' to set it up, then re-run.")
        sys.exit(1)
    print(f"  remote '{GDRIVE_REMOTE}:' OK")


def ensure_rclone():
    r = subprocess.run(["which", "rclone"], capture_output=True)
    if r.returncode != 0:
        print("  rclone not found, installing...")
        run(["curl", "-fsS", "https://rclone.org/install.sh"], desc="downloading rclone installer")
        run(["sudo", "bash"], desc="installing rclone")


def mount_drive():
    if SKIP_MOUNT:
        print("[mount] SKIP_MOUNT=1 — assuming already mounted")
        return
    os.makedirs(MOUNT_DIR, exist_ok=True)

    # Check if already mounted
    r = subprocess.run(["mountpoint", "-q", MOUNT_DIR], capture_output=True)
    if r.returncode == 0:
        print(f"[mount] Already mounted at {MOUNT_DIR}")
        return

    print(f"[mount] Mounting {GDRIVE_REMOTE}:{GDRIVE_PATH} -> {MOUNT_DIR}")
    run([
        "rclone", "mount",
        f"{GDRIVE_REMOTE}:{GDRIVE_PATH}",
        MOUNT_DIR,
        "--daemon",
        "--vfs-cache-mode", "writes",
        "--dir-cache-time", "1h",
        "--poll-interval", "15s",
        "--log-level", "ERROR",
    ], desc="mounting Google Drive via rclone FUSE")
    # Wait for mount to be ready
    for i in range(15):
        time.sleep(1)
        r = subprocess.run(["mountpoint", "-q", MOUNT_DIR], capture_output=True)
        if r.returncode == 0:
            print(f"  mounted OK at {MOUNT_DIR}")
            return
    print("ERROR: mount did not become ready within 15s")
    sys.exit(1)


def pull_metadata():
    """Pull meta.json, shard_manifest.json, tokenization_id.txt locally.
    train.py needs these to read the manifest and select shards.
    The actual .bin shards are read on-demand from the FUSE mount."""
    os.makedirs(SHARD_LOCAL, exist_ok=True)
    for fname in ("meta.json", "shard_manifest.json", "tokenization_id.txt"):
        src = os.path.join(MOUNT_DIR, "token_shards_merged", fname)
        dst = os.path.join(SHARD_LOCAL, fname)
        if not os.path.exists(dst):
            print(f"  pulling {fname}...")
            run(["cp", src, dst])


def pull_checkpoint_and_manifest():
    os.makedirs(CKPT_LOCAL, exist_ok=True)
    os.makedirs(MANIFEST_LOCAL, exist_ok=True)
    run(["rclone", "copy", CKPT_REMOTE, CKPT_LOCAL,
         "--include", "*.pth", "--max-age", "30d",
         "--transfers", "1", "--drive-chunk-size", "64M",
         "--checksum"],
        desc="pulling existing checkpoint for resume",
        check=False)
    run(["rclone", "copy", CKPT_REMOTE.replace("/checkpoints", "/manifests"),
         MANIFEST_LOCAL,
         "--include", "training_latest.json",
         "--transfers", "1", "--checksum"],
        desc="pulling existing manifest",
        check=False)


def stage_selected_shards():
    """Read the manifest from the mount, select shards for the token budget,
    copy only those shards to local SSD. This avoids pulling 300 GB."""
    import json

    manifest_path = os.path.join(MOUNT_DIR, "token_shards_merged", "shard_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(SHARD_LOCAL, "shard_manifest.json")

    with open(manifest_path) as f:
        manifest = json.load(f)

    shards = manifest["shards"]
    token_budget = int(os.environ.get("MAX_TOKENS", os.environ.get("TOKEN_BUDGET", str(DEFAULT_TOKEN_BUDGET))))

    # Sort deterministic shuffle, then take shards up to budget
    import random
    random.seed(42)
    random.shuffle(shards)

    selected = []
    running = 0
    for s in shards:
        if running >= token_budget:
            break
        selected.append(s)
        running += s["tokens"]

    print(f"\n[stage] Selected {len(selected)} shards (~{running:,} tokens) "
          f"from {len(shards)} available ({len(shards) - len(selected)} skipped)")
    print(f"[stage] Copying {len(selected)} shards to local SSD...")

    os.makedirs(SHARD_LOCAL, exist_ok=True)
    t0 = time.time()
    copied = 0
    for i, s in enumerate(selected, 1):
        name = s["shard"]
        src = os.path.join(MOUNT_DIR, "token_shards_merged", name)
        dst = os.path.join(SHARD_LOCAL, name)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            run(["cp", src, dst], check=False)
            copied += 1
        if i % 20 == 0 or i == len(selected):
            elapsed = max(time.time() - t0, 0.01)
            print(f"    {i}/{len(selected)} | {elapsed:.0f}s")

    print(f"[stage] Done. {copied} shards copied in {time.time()-t0:.0f}s.")


def main():
    print("=" * 60)
    print("  SynapseGPT — RunPod Launcher")
    print(f"  Remote: {GDRIVE_REMOTE}:{GDRIVE_PATH}")
    print(f"  Local:  {LOCAL_DIR}")
    print(f"  Mount:  {MOUNT_DIR}")
    print("=" * 60)

    ensure_rclone()
    check_rclone_remote()
    mount_drive()
    pull_metadata()
    pull_checkpoint_and_manifest()
    stage_selected_shards()

    # ── Launch train.py ────────────────────────────────────────────────
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    if not os.path.exists(train_script):
        train_script = os.path.join(os.getcwd(), "pretrain", "train.py")
    if not os.path.exists(train_script):
        print(f"ERROR: train.py not found. Run from repo root.")
        sys.exit(1)

    env = os.environ.copy()
    env["SYNAPSE_DIR"] = SYNAPSE_LOCAL
    env["SKIP_DRIVE_MOUNT"] = "1"
    env["SKIP_STAGE"] = "1"  # we already staged selected shards locally
    env["CHECKPOINT_PUSH_REMOTE"] = CKPT_REMOTE
    if "MAX_TOKENS" not in env and "TOKEN_BUDGET" in env:
        env["MAX_TOKENS"] = env["TOKEN_BUDGET"]

    print(f"\n[{int(time.time())}] Launching train.py...")
    print(f"  SYNAPSE_DIR={env['SYNAPSE_DIR']}")
    print(f"  CHECKPOINT_PUSH_REMOTE={env['CHECKPOINT_PUSH_REMOTE']}")
    print(f"  MAX_TOKENS={env.get('MAX_TOKENS', 'default')}")
    print()

    os.execve(sys.executable, [sys.executable, train_script], env)


if __name__ == "__main__":
    main()
