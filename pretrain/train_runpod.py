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
  MAX_TOKENS          Selection budget pre seen-subtraction (default:
                      70_000_000_000). Sized so the FRESH remainder exceeds the
                      ~25B needed to reach train.py's LR_HORIZON_STEPS (120k);
                      the launcher then pulls only that fresh, trimmed set.
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
DEFAULT_TOKEN_BUDGET = 70_000_000_000
TOKEN_BUDGET = int(os.environ.get("MAX_TOKENS", os.environ.get("TOKEN_BUDGET", str(DEFAULT_TOKEN_BUDGET))))
SKIP_DATA_PULL = os.environ.get("SKIP_DATA_PULL", "0") == "1"
SKIP_RESUME = os.environ.get("SKIP_RESUME", "0") == "1"
SKIP_RESUME = os.environ.get("SKIP_RESUME", "0") == "1"

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
    try:
        ckpt_exists = subprocess.run(
            ["rclone", "ls", CKPT_REMOTE, "--max-depth", "1"],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("  WARN: rclone ls (checkpoints) timed out; assuming none and continuing")
        ckpt_exists = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
    if SKIP_RESUME:
        print("[resume] SKIP_RESUME=1 — skipping checkpoint pull (starting from scratch)")
        # Wipe local checkpoint dir to ensure fresh start
        import shutil
        if os.path.isdir(CKPT_LOCAL):
            for f in os.listdir(CKPT_LOCAL):
                if f.endswith(".pth"):
                    os.remove(os.path.join(CKPT_LOCAL, f))
    elif ckpt_exists.returncode == 0 and ckpt_exists.stdout.strip():
        lines = [l.strip().split()[-1] for l in ckpt_exists.stdout.strip().split("\n") if l.strip()]
        pth_files = sorted([l for l in lines if l.endswith(".pth")])
        if pth_files:
            # Prefer main checkpoint, else pick latest alphabetically (newest archive)
            ckpt_name = os.environ.get("CHECKPOINT_NAME", "synapse_2b_d2560_l28.pth")
            pick = ckpt_name if ckpt_name in pth_files else pth_files[-1]
            run(["rclone", "copy",
                 CKPT_REMOTE, CKPT_LOCAL,
                 "--include", pick,
                 "--transfers", "1",
                 "--multi-thread-streams", "8",
                 "--multi-thread-cutoff", "128M",
                 "--drive-chunk-size", "128M",
                 "--progress", "--stats", "10s", "--stats-one-line"],
                desc=f"pulling checkpoint ({pick})", check=False)
        else:
            print("  no .pth checkpoint found on Drive")
    else:
        print("  no existing checkpoint found on Drive")

    try:
        mf_exists = subprocess.run(
            ["rclone", "ls", MANIFEST_REMOTE, "--max-depth", "1"],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("  WARN: rclone ls (manifests) timed out; assuming none and continuing")
        mf_exists = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
    if mf_exists.returncode == 0 and mf_exists.stdout.strip():
        # Pull both training_latest.json and the eval pin (if present on Drive).
        # The eval pin tells train.py which shards make up the held-out set;
        # the launcher needs to know that list so it pulls those shards too.
        run(["rclone", "copy", MANIFEST_REMOTE, MANIFEST_LOCAL,
             "--include", "training_latest.json",
             "--include", "eval_shards.json",
             "--transfers", "2", "--checksum", "--progress"],
            desc="pulling existing manifest + eval pin", check=False)
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

    # ── Read seen_shards (multiset) + step from the pulled checkpoint so the
    # selection below can drop already-trained shards and pull ONLY fresh ones.
    # mmap=True keeps the 25 GB of weights off the heap; we only touch the list.
    seen_passes = []
    saved_curr_step = 0
    ckpt_name = os.environ.get("CHECKPOINT_NAME", "synapse_2b_d2560_l28.pth")
    ckpt_path = os.path.join(CKPT_LOCAL, ckpt_name)
    if not SKIP_RESUME and os.path.exists(ckpt_path):
        try:
            import torch
            _ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
            if isinstance(_ck, dict) and _ck.get("schema") == "v2":
                seen_passes = list(_ck.get("seen_shards", []))
                saved_curr_step = int(_ck.get("curr_step", 0))
            del _ck
            print(f"  checkpoint: curr_step={saved_curr_step:,}, seen_shards={len(seen_passes)}")
        except Exception as e:
            print(f"  WARN: could not read seen_shards ({e}); pulling full plan")

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
        # MUST stay in sync with pretrain/train.py:DATA_MIX. Until we consolidate
        # into a shared module, edit both files together. Mix sums to 1.000.
        DATA_MIX = {
            "data_code":                   0.15,
            "data_finemath":               0.21,
            "data_arxiv":                  0.21,
            "data_wikipedia":              0.13,
            "data_fineweb":                0.20,
            "data_math_operations_cot_v2": 0.025,
            "data_books_gutemberg":        0.02,
            "data_math_operations":        0.005,
            "data_code_math":              0.005,
            "data_distilled_facts":        0.01,
            "data_books_faded":            0.01,
            "data_math_text":              0.005,
            "data_math_text_v2":           0.01,
            "data_adult":                  0.01,
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
        print(f"\n[data] Selected {len(selected_shards)} shard-passes ({selected_tokens:,} tokens)")

        # Drop already-trained passes (count-based; matches train.py section 8) so
        # only fresh shards get pulled.
        if seen_passes:
            seen_counts = defaultdict(int)
            for n in seen_passes:
                seen_counts[n] += 1
            before = len(selected_shards)
            remaining = []
            for s in selected_shards:
                if seen_counts[s["shard"]] > 0:
                    seen_counts[s["shard"]] -= 1
                else:
                    remaining.append(s)
            selected_shards = remaining
            print(f"[data] Skipping {before - len(selected_shards)} already-trained "
                  f"shard-passes ({len(selected_shards)} fresh remaining)")

        # Trim to the tokens needed to reach the LR horizon (+5% margin) so the
        # pull is ~what we train, not the whole plan. These constants MUST match
        # pretrain/train.py (LR_HORIZON_STEPS and BATCH_SIZE*GRAD_ACCUM_STEPS*
        # BLOCK_SIZE). train.py re-applies the same trim, so pulling a hair extra
        # is harmless; pulling too little would stop the run short of the horizon.
        LR_HORIZON_STEPS = 120_000
        TOKENS_PER_STEP = 4 * 64 * 2048
        needed = max(0, int((LR_HORIZON_STEPS - saved_curr_step) * TOKENS_PER_STEP * 1.05))
        fresh_tok = sum(s["tokens"] for s in selected_shards)
        if needed == 0:
            print(f"[data] Already at/past LR horizon ({saved_curr_step:,} >= "
                  f"{LR_HORIZON_STEPS:,}); nothing fresh to pull.")
            selected_shards = []
        elif fresh_tok > needed:
            random.shuffle(selected_shards)
            trimmed, acc = [], 0
            for s in selected_shards:
                trimmed.append(s)
                acc += s["tokens"]
                if acc >= needed:
                    break
            print(f"[data] Trimmed to {len(trimmed)} fresh shards (~{acc/1e9:.2f}B tok) "
                  f"to reach step {LR_HORIZON_STEPS:,} from {saved_curr_step:,}")
            selected_shards = trimmed
        else:
            print(f"[data] Fresh remainder {fresh_tok/1e9:.2f}B < needed "
                  f"{needed/1e9:.2f}B; pulling all fresh (may stop short of "
                  f"{LR_HORIZON_STEPS:,} — raise MAX_TOKENS).")

        # Augment with pinned eval shards (if a pin exists) so train.py finds
        # them locally. The pin lives in MANIFEST_LOCAL/eval_shards.json after
        # the step-3 manifest pull above.
        pin_path = os.path.join(MANIFEST_LOCAL, "eval_shards.json")
        selected_names = {s["shard"] for s in selected_shards}
        if os.path.exists(pin_path):
            with open(pin_path) as f:
                pin = json.load(f)
            pinned_eval = [n for n in pin["shards"] if n not in selected_names]
            if pinned_eval:
                print(f"[data] Pin adds {len(pinned_eval)} eval shard(s) to pull list")
                # Pull as raw filename entries that rclone --files-from can use.
                extra_entries = [{"shard": n, "tokens": 0} for n in pinned_eval]
                selected_shards = selected_shards + extra_entries

        # ── 6. Pull selected shards via rclone copy ──
        list_path = os.path.join(LOCAL_DIR, "selected_shards.txt")
        with open(list_path, "w") as f:
            for s in selected_shards:
                f.write(s["shard"] + "\n")

        print(f"[data] Pulling {len(selected_shards)} shards to {SHARD_LOCAL}...")
        t0 = time.time()
        run(["rclone", "copy", SHARD_REMOTE, SHARD_LOCAL,
             "--files-from", list_path,
             "--transfers", "16",
             "--drive-chunk-size", "128M",
             "--checkers", "16",
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
