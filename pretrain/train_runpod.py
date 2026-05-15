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
from collections import defaultdict

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

    os.makedirs(LOCAL_DIR, exist_ok=True)
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
        lines = [l.strip().split()[-1] for l in ckpt_exists.stdout.strip().split("\n") if l.strip()]
        pth_files = sorted([l for l in lines if l.endswith(".pth")])
        if pth_files:
            # Prefer main checkpoint, else pick latest alphabetically (newest archive)
            ckpt_name = os.environ.get("CHECKPOINT_NAME", "synapse_2b_d2560_l28.pth")
            pick = ckpt_name if ckpt_name in pth_files else pth_files[-1]
            run(["rclone", "copy",
                 CKPT_REMOTE, CKPT_LOCAL,
                 "--include", pick,
                 "--transfers", "1", "--drive-chunk-size", "64M",
                 "--checksum", "--progress"],
                desc=f"pulling checkpoint ({pick})", check=False)
        else:
            print("  no .pth checkpoint found on Drive")
    else:
        print("  no existing checkpoint found on Drive")

    mf_exists = subprocess.run(
        ["rclone", "ls", MANIFEST_REMOTE, "--max-depth", "1"],
        capture_output=True, text=True, timeout=30
    )
    if mf_exists.returncode == 0 and mf_exists.stdout.strip():
        run(["rclone", "copy", MANIFEST_REMOTE, MANIFEST_LOCAL,
             "--include", "training_latest.json",
             "--transfers", "1", "--checksum", "--progress"],
            desc="pulling existing manifest", check=False)
    else:
        print("  no existing manifest found on Drive")

    # ── 4. Verify Python deps ──
    missing_deps = []
    for mod_name in ("torch", "numpy", "tqdm"):
        try:
            __import__(mod_name)
        except ImportError:
            missing_deps.append(mod_name)
    if missing_deps:
        print(f"ERROR: missing Python packages: {', '.join(missing_deps)}")
        print("Run: pip install torch numpy tqdm")
        sys.exit(1)
    print("  Python deps: OK")

    # ── 5. Select shards matching train.py's DATA_MIX ──
    if SKIP_DATA_PULL:
        print("[data] SKIP_DATA_PULL=1 — skipping shard pull, using existing local")
    else:
        manifest_path = os.path.join(SHARD_LOCAL, "shard_manifest.json")
        if not os.path.exists(manifest_path):
            print(f"ERROR: {manifest_path} not found.")
            print(f"  Check that GDRIVE_PATH={GDRIVE_PATH!r} is correct")
            print(f"  and that token_shards_merged/ exists on Drive.")
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)

        all_shards = manifest["shards"]

        # Group by source (same logic as train.py: get_source_name)
        def get_source_name(shard_entry):
            source = shard_entry.get("source", "")
            parts = source.replace("\\", "/").split("/")
            for p in parts:
                if p.startswith("data_"):
                    return p
            return "other"

        shards_by_source = defaultdict(list)
        for s in all_shards:
            shards_by_source[get_source_name(s)].append(s)

        random.seed(42)
        selected_shards = []
        DATA_MIX = {
            "data_c4":               0.42,
            "data_code":             0.20,
            "data_arxiv":            0.10,
            "data_finemath":         0.10,
            "data_wikipedia":        0.10,
            "data_books_gutemberg":  0.02,
            "data_math_operations":  0.02,
            "data_distilled_facts":  0.01,
            "data_books_faded":      0.01,
            "data_math_text":        0.01,
            "data_adult":            0.01,
        }
        budget = TOKEN_BUDGET
        for source, weight in DATA_MIX.items():
            source_budget = int(budget * weight)
            available = shards_by_source.get(source, [])
            if not available:
                print(f"  WARNING: {source} not found in shards, skipping")
                continue
            random.shuffle(available)
            running = 0
            for s in available:
                if running >= source_budget:
                    break
                selected_shards.append(s)
                running += s["tokens"]

        selected_tokens = sum(s["tokens"] for s in selected_shards)
        print(f"\n[data] Selected {len(selected_shards)} shards ({selected_tokens:,} tokens)")

        # ── 6. Pull selected shards via rclone copy ──
        list_path = os.path.join(LOCAL_DIR, "selected_shards.txt")
        with open(list_path, "w") as f:
            for s in selected_shards:
                f.write(s["shard"] + "\n")

        print(f"[data] Pulling {len(selected_shards)} shards to {SHARD_LOCAL}...")
        t0 = time.time()
        run(["rclone", "copy", SHARD_REMOTE, SHARD_LOCAL,
             "--files-from", list_path,
             "--transfers", "8", "--drive-chunk-size", "64M",
             "--checksum", "--progress"],
            desc="downloading selected shards")
        print(f"[data] Done in {time.time()-t0:.0f}s")
        os.remove(list_path)

    # ── 7. Launch train.py ──
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    if not os.path.exists(train_script):
        train_script = os.path.join(os.getcwd(), "pretrain", "train.py")
    if not os.path.exists(train_script):
        print("ERROR: train.py not found. Run this script from the synapse repo root.")
        sys.exit(1)

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
