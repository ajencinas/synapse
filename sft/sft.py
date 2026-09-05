#!/usr/bin/env python3
"""SynapseGPT supervised fine-tuning (SFT).

Continues the pretrained SynapseGPT checkpoint on the ~73k ChatML dataset built
by the SFT data pipeline. Implements the spec in sft/SFT_TRAIN_PLAN.md:
  - loads pretrain weights only (fresh optimizer), infers vocab from the ckpt
  - 4-way tokenization_id guard (fail loud on mismatch)
  - weighted per-source sampling (the §6 mix), dynamic padding
  - SHIFTED masked cross-entropy (loss only on assistant response tokens)
  - per-source validation loss + a pre-SFT baseline eval
  - per-epoch model-only snapshots + a full "latest" checkpoint for resume

Run (env-configurable, all optional):
  SYNAPSE_DIR=/path/to/synapse python sft/sft.py

Env vars:
  SYNAPSE_DIR    base dir (default: Drive on Colab, ./synapse otherwise)
  SFT_COMPILE    "1" (default) to torch.compile; "0" for eager
  SFT_CACHE      "1" (default) to cache parsed examples to a .pt
  SKIP_DRIVE_MOUNT  "1" to skip Colab Drive mount
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import re
import glob
import json
import math
import time
import shutil
import hashlib
import logging
import threading
import subprocess

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# --- shared model (single source of truth; see sft/SFT_TRAIN_PLAN.md §0.1) ---
# synapse_model.py lives at the repo root; add both the script dir and the repo
# root to sys.path so this works whether run from a clone (VM) or a flat copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from synapse_model import SynapseGPT, maybe_mount_drive, default_synapse_dir, BLOCK_SIZE

torch._logging.set_logs(inductor=logging.WARNING, dynamic=logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


# ==================== CONFIGURATION ====================

EXPECTED_TOK_ID = os.environ.get("EXPECTED_TOK_ID", "7a570a7ba9fc7985")

# Special token IDs (must match the tokenizer / data pipeline).
IGNORE_INDEX = -100
PAD_ID = 0  # <|endoftext|>; right-padding + IGNORE labels make the value irrelevant

# -- Data mix (per-batch sampling proportions; SFT v3 — see
#    SFT_V3_EXECUTION_PLAN.md. v2's 39% math/tool degraded prose after step 1200;
#    v3 tilts prose and adds tool_search / format_following / nli. Ns below are
#    pre-tokenization estimates (P~188k); the load-time 1.65x ceiling assert is
#    the real gate) --
SFT_DATA_MIX = {
    "tulu3":             0.340,  # broad instructions (N~39.9k; ~1.60x — ceiling-bound)
    "dolly":             0.110,  # human QA/prose (N~14.3k; ~1.45x)
    "no_robots":         0.080,  # human prose (N~9.2k; ~1.63x — ceiling-bound)
    "tool_use":          0.075,  # python tool traces (N~34.6k; ~0.41x)
    "tool_negative":     0.060,  # tool prompt -> direct answer, incl. 2k synth trivial (N~7.9k; ~1.43x)
    "metamath":          0.050,  # math CoT (N~21.5k; ~0.44x)
    "reasoning_distill": 0.050,  # verified teacher CoT (N~25.8k; ~0.37x)
    "samsum":            0.045,  # summarization (N~7.9k; ~1.08x)
    "alpaca_gpt4":       0.045,  # GPT-4 prose (N~5.8k; ~1.45x)
    "format_following":  0.040,  # NEW: exact-format instructions (N~4.9k; ~1.53x)
    "opencode":          0.025,  # code, at floor (N~5.9k; ~0.80x)
    "tool_search":       0.025,  # NEW: search tool traces (N~2.9k; ~1.61x — ceiling-bound)
    "nli":               0.020,  # NEW: yes/no entailment (N~2.9k; ~1.28x)
    "oasst1":            0.018,  # conversational (N~2.2k; ~1.54x)
    "creative":          0.017,  # creative writing (N~2.3k; ~1.41x)
}
MAX_EPOCH_REPEAT = 1.65          # hard cap: no example drawn >1.65x/epoch on avg

# -- Training hyperparameters (SFT-specific; ~10-20x gentler than pretrain) --
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 16          # effective batch = 64 sequences
EPOCHS           = 1   # v3: v2's val bottomed at 44% of epoch 1; epoch 2 only
                       # degraded prose. sft_best still catches the true minimum.
MAX_LR           = 1.5e-5
MIN_LR           = 1.5e-6
WARMUP_STEPS     = 100
WEIGHT_DECAY     = 0.01
BETAS            = (0.9, 0.95)
GRAD_CLIP        = 1.0
EVAL_EVERY       = 200         # optimizer steps between per-source evals
SAVE_EVERY_STEPS = int(os.environ.get("SFT_SAVE_EVERY", 300))   # local full ckpt (fast)
DRIVE_EVERY_STEPS = int(os.environ.get("SFT_DRIVE_EVERY", 600))  # background push full->Drive
EVAL_BATCH_SIZE  = 8
SEED             = 1337

USE_COMPILE = os.environ.get("SFT_COMPILE", "1") == "1"
USE_CACHE   = os.environ.get("SFT_CACHE", "1") == "1"
# rclone remote for durable checkpoint pushes on VM/RunPod (no Drive FUSE there),
# e.g. "gdrive:synapse/sft_checkpoints". Unset on Colab (Drive is FUSE-mounted).
PUSH_REMOTE = os.environ.get("CHECKPOINT_PUSH_REMOTE", "").rstrip("/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== HELPERS ====================

def tokenizer_fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_lr(it, total_it):
    """Cosine schedule with linear warmup. Local to sft.py (NOT shared) because
    MAX_LR/MIN_LR/WARMUP_STEPS differ from pretrain — see SFT_TRAIN_PLAN.md §0.1."""
    if it < WARMUP_STEPS:
        return MAX_LR * it / WARMUP_STEPS
    decay_ratio = (it - WARMUP_STEPS) / max(1, (total_it - WARMUP_STEPS))
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def collate(batch):
    """Pad a list of {"input_ids","labels"} to the batch's longest sequence.
    input_ids padded with PAD_ID; labels padded with IGNORE_INDEX (so pad
    positions never contribute to loss). Right-padding + causal attention means
    real tokens never attend to pad, so no attention mask is needed."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    n = len(batch)
    input_ids = torch.full((n, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(ex["labels"], dtype=torch.long)
    return input_ids, labels


def shifted_ce(logits, labels, vocab_size, reduction):
    """Next-token cross-entropy. forward() returns logits[i] predicting token
    i+1, while labels are position-aligned — so we MUST shift: compare
    logits[:, :-1] against labels[:, 1:]. (Pretrain does this shift in its data
    loader instead; SFT stores aligned labels, so it shifts here.)"""
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, vocab_size),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )


@torch.no_grad()
def eval_source(model, val_examples, vocab_size):
    """Token-weighted mean response-token loss over one source's val set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(val_examples), EVAL_BATCH_SIZE):
        input_ids, labels = collate(val_examples[start:start + EVAL_BATCH_SIZE])
        input_ids, labels = input_ids.to(device), labels.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = shifted_ce(logits, labels, vocab_size, reduction="sum")
        total_loss += loss.item()
        total_tokens += (labels[:, 1:] != IGNORE_INDEX).sum().item()
    model.train()
    return total_loss / total_tokens if total_tokens else float("nan")


def eval_all(model, val_by_source, vocab_size):
    per_source = {name: eval_source(model, ex, vocab_size)
                  for name, ex in val_by_source.items()}
    vals = [v for v in per_source.values() if not math.isnan(v)]
    per_source["overall"] = (sum(vals) / len(vals)) if vals else float("nan")
    return per_source


# ==================== DATA LOADING ====================

def load_all_data(syn, tok_id):
    """Returns (train_examples, weights, val_by_source). Per-example weight is
    mix_weight / source_size so each source hits its mix share regardless of
    raw size. Optionally cached to a .pt keyed on tok_id + per-source counts
    (counts in the key so growing a source invalidates a stale cache)."""
    tok_base = os.path.join(syn, "sft_tokenized")
    cache_path = os.path.join(tok_base, "_all_sft_cache.pt")

    # Build a signature from the on-disk train counts so the cache can't go stale
    # when a source is regrown (tok_id alone wouldn't catch that).
    sig_parts = [tok_id]
    for name in sorted(SFT_DATA_MIX):
        p = os.path.join(tok_base, name, "train.jsonl")
        if not os.path.exists(p):
            raise SystemExit(f"[{name}] missing {p} — run tokenize_sft_data.py first")
        sig_parts.append(f"{name}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}")
    signature = "|".join(sig_parts)

    if USE_CACHE and os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        if cache.get("signature") == signature:
            print(f"[data] cache hit → {cache_path}")
            return cache["train"], cache["weights"], cache["val"]
        print("[data] cache stale (data changed) — rebuilding")

    train_examples, weights, val_by_source = [], [], {}
    train_counts = {}
    print("[data] parsing per-source JSONL:")
    for name, mix_w in SFT_DATA_MIX.items():
        d = os.path.join(tok_base, name)
        train = load_jsonl(os.path.join(d, "train.jsonl"))
        val_by_source[name] = load_jsonl(os.path.join(d, "val.jsonl"))
        if not train:
            raise SystemExit(f"[{name}] has 0 train examples — retokenize it or remove it from SFT_DATA_MIX")
        per_ex_w = mix_w / len(train)
        train_counts[name] = len(train)
        train_examples.extend(train)
        weights.extend([per_ex_w] * len(train))
        print(f"  {name:12s} train={len(train):>6,} val={len(val_by_source[name]):>4,} "
              f"weight={mix_w:.2f}")

    # Fail loud if any source would be over-drawn: expected draws/epoch for a
    # source are mix_w * P; per-example repeat = mix_w * P / N must stay under
    # MAX_EPOCH_REPEAT or its examples memorize before the pool finishes.
    P = len(train_examples)
    for name, mix_w in SFT_DATA_MIX.items():
        repeat = mix_w * P / train_counts[name]
        if repeat > MAX_EPOCH_REPEAT:
            raise SystemExit(
                f"[{name}] weight {mix_w} draws each example {repeat:.2f}x/epoch "
                f"(> {MAX_EPOCH_REPEAT}) at N={train_counts[name]:,}, P={P:,} — "
                f"lower its weight or grow the source")

    if USE_CACHE:
        torch.save({"signature": signature, "train": train_examples,
                    "weights": weights, "val": val_by_source}, cache_path)
        print(f"[data] cached → {cache_path}")
    return train_examples, weights, val_by_source


# ==================== CHECKPOINTS ====================

_push_thread = None


def async_push_to_drive(src, dst, wait_prev=False):
    """Copy a (large) checkpoint local -> Drive in a BACKGROUND thread so training
    never blocks on slow Drive FUSE writes. Atomic via tmp+replace. If a previous
    push is still running: skip (default — the next interval carries newer state),
    or with wait_prev=True JOIN it first so this push is never dropped (used for
    durable artifacts: epoch snapshots, sft_best)."""
    global _push_thread
    if _push_thread is not None and _push_thread.is_alive():
        if wait_prev:
            print("  [drive] waiting for previous push (durable artifact — never skipped)")
            _push_thread.join()
        else:
            print("  [drive] previous push still running — skipping this one")
            return
    def _do():
        try:
            tmp = dst + ".tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            print(f"  [drive] pushed {os.path.basename(dst)} "
                  f"({os.path.getsize(dst)/1e9:.1f} GB)")
        except Exception as e:
            print(f"  [drive] push failed (will retry next interval): {e}")
    _push_thread = threading.Thread(target=_do, daemon=True)
    _push_thread.start()


_rclone_thread = None


def rclone_push_async(local_path, remote_dir, wait_prev=False):
    """Background `rclone copyto local -> remote/basename` for VM/RunPod durability
    (mirrors pretrain's _push_to_remote_async). Failures warn, never crash.
    wait_prev=True joins an in-flight push instead of skipping (durable artifacts)."""
    global _rclone_thread
    if not remote_dir:
        return
    if _rclone_thread is not None and _rclone_thread.is_alive():
        if wait_prev:
            print("  [rclone] waiting for previous push (durable artifact — never skipped)")
            _rclone_thread.join()
        else:
            print("  [rclone] previous push still running — skipping this one")
            return
    remote_path = remote_dir + "/" + os.path.basename(local_path)
    def _run():
        try:
            r = subprocess.run(
                ["rclone", "copyto", local_path, remote_path,
                 "--checksum", "--drive-chunk-size=64M"],
                capture_output=True, text=True, timeout=7200)
            if r.returncode == 0:
                print(f"  [rclone] pushed {os.path.basename(local_path)} -> {remote_dir}")
            else:
                print(f"  [rclone] push failed: {r.stderr.strip()[:200]}")
        except Exception as e:
            print(f"  [rclone] push exception: {e}")
    _rclone_thread = threading.Thread(target=_run, daemon=True)
    _rclone_thread.start()


def _clean_state(model):
    """state_dict with torch.compile's _orig_mod. wrapper removed."""
    raw = getattr(model, "_orig_mod", model)
    return raw.state_dict()


def save_checkpoint(path, model, *, model_only, optimizer=None, curr_step=0,
                    total_steps=0, epoch=0, epoch_complete=True, tok_id="",
                    eval_history=None, last_eval=None, baseline=None):
    payload = {
        "schema": "v2",
        "stage": "sft",
        "model": _clean_state(model),
        "epoch": epoch,
        "tokenization_id": tok_id,
        "data_mix": SFT_DATA_MIX,
        "block_size": BLOCK_SIZE,
    }
    if not model_only:
        payload.update({
            "optimizer": optimizer.state_dict(),
            "curr_step": curr_step,
            "total_steps": total_steps,   # pin the LR horizon for exact resume
            "epoch_complete": epoch_complete,  # False = saved mid-epoch
            "eval_history": eval_history or [],
            "last_eval_loss": last_eval,
            "baseline_eval": baseline,
        })
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  saved {'model-only' if model_only else 'full'} → {path}")


# ==================== MAIN ====================

def main():
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    maybe_mount_drive()
    syn = os.environ.get("SYNAPSE_DIR") or default_synapse_dir()
    tokenizer_path = os.path.join(syn, "tokenizer_out", "tokenizer.json")
    manifest_dir = os.path.join(syn, "manifests")
    ckpt_in = os.path.join(syn, "checkpoints",
                           os.environ.get("CHECKPOINT_NAME", "synapse_2b_d2560_l28.pth"))
    out_dir = os.path.join(syn, "sft_checkpoints")          # durable (Drive on Colab)
    os.makedirs(out_dir, exist_ok=True)
    # Frequent FULL checkpoints (~25 GB) go to LOCAL SSD — writing them to Drive
    # FUSE every few hundred steps would stall training for many minutes. Only the
    # small model-only epoch snapshots (~8 GB) are copied to Drive (durable:
    # pick-best + VM-death recovery).
    work_dir = os.environ.get("SFT_WORK_DIR") or (
        "/content/sft_work" if out_dir.startswith("/content/drive") else out_dir)
    os.makedirs(work_dir, exist_ok=True)
    latest_path = os.path.join(work_dir, "sft_latest.pth")  # full, LOCAL (fast)
    drive_latest = os.path.join(out_dir, "sft_latest.pth")  # full, Drive (background push)
    best_path = os.path.join(work_dir, "sft_best.pth")      # model-only, best overall val

    def mirror(path, durable=False):
        """Copy a just-written LOCAL checkpoint off the box, in the background.
        VM/RunPod: rclone to CHECKPOINT_PUSH_REMOTE. Colab: shutil to the
        Drive-FUSE out_dir. No-op when out_dir is already the local work_dir.
        durable=True waits for any in-flight push instead of skipping — H3 fix:
        epoch snapshots and sft_best must NEVER silently miss the mirror."""
        if PUSH_REMOTE:
            rclone_push_async(path, PUSH_REMOTE, wait_prev=durable)
        elif os.path.abspath(work_dir) != os.path.abspath(out_dir):
            async_push_to_drive(path, os.path.join(out_dir, os.path.basename(path)),
                                wait_prev=durable)

    print(f"[setup] SYNAPSE_DIR = {syn}")
    print(f"[setup] device = {device}")
    print(f"[setup] work_dir (local, full ckpts) = {work_dir}")
    print(f"[setup] out_dir  (drive, snapshots)  = {out_dir}")

    # --- 1. Tokenization-ID guard: tokenizer, pretrain manifest, SFT registry ---
    if not os.path.exists(tokenizer_path):
        raise SystemExit(f"tokenizer not found at {tokenizer_path}")
    tok_id = tokenizer_fingerprint(tokenizer_path)
    pre_manifest = os.path.join(manifest_dir, "training_latest.json")
    sft_registry_path = os.path.join(manifest_dir, "sft_data_registry.json")
    for p in (pre_manifest, sft_registry_path):
        if not os.path.exists(p):
            raise SystemExit(f"required manifest missing: {p}")
    pretrain_tok_id = json.load(open(pre_manifest))["tokenization_id"]
    sft_registry = json.load(open(sft_registry_path))
    sft_tok_id = sft_registry["tokenization_id"]
    if not (tok_id == pretrain_tok_id == sft_tok_id == EXPECTED_TOK_ID):
        raise SystemExit(
            f"tokenization_id mismatch — refusing to train:\n"
            f"  tokenizer.json = {tok_id}\n  pretrain        = {pretrain_tok_id}\n"
            f"  sft registry    = {sft_tok_id}\n  expected        = {EXPECTED_TOK_ID}")
    print(f"[setup] tokenization_id OK → {tok_id}")
    reg_block = sft_registry.get("block_size")
    if not isinstance(reg_block, int) or reg_block > BLOCK_SIZE:
        raise SystemExit(
            f"registry block_size {reg_block!r} > model BLOCK_SIZE {BLOCK_SIZE} — "
            f"data was tokenized for a longer context than the model's RoPE tables "
            f"support (would crash mid-training); retokenize with --block-size {BLOCK_SIZE}")

    # --- 2. Resume from an SFT checkpoint, else init from pretrain weights ---
    resuming = False
    saved_epoch = 0
    saved_curr_step = 0
    saved_total_steps = None
    saved_last_eval = None
    saved_epoch_complete = True
    optimizer_state = None
    eval_history, baseline_eval = [], None
    full_ckpt_path = (latest_path if os.path.exists(latest_path)
                      else drive_latest if os.path.exists(drive_latest) else None)
    if full_ckpt_path:
        ckpt = torch.load(full_ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("stage") == "sft" and ckpt.get("tokenization_id") == tok_id:
            print(f"[init] resuming SFT (full) from {full_ckpt_path} (epoch {ckpt.get('epoch')})")
            state_dict = ckpt["model"]
            optimizer_state = ckpt.get("optimizer")
            saved_epoch = int(ckpt.get("epoch", 0))
            saved_curr_step = int(ckpt.get("curr_step", 0))
            saved_total_steps = ckpt.get("total_steps")
            saved_last_eval = ckpt.get("last_eval_loss")
            saved_epoch_complete = bool(ckpt.get("epoch_complete", True))
            eval_history = list(ckpt.get("eval_history", []))
            baseline_eval = ckpt.get("baseline_eval")
            resuming = True
        else:
            raise SystemExit(
                f"{full_ckpt_path} exists but stage/tok mismatch — refusing to overwrite")
    elif glob.glob(os.path.join(out_dir, "sft_epoch*.pth")):
        # No local full checkpoint (new VM / session reset), but durable model-only
        # epoch snapshots survive on Drive. Resume from the newest with a FRESH
        # optimizer — acceptable for a short SFT run.
        snaps = sorted(glob.glob(os.path.join(out_dir, "sft_epoch*.pth")),
                       key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
        snap = snaps[-1]
        snap_epoch = int(re.search(r"epoch(\d+)", snap).group(1))
        print(f"[init] no local ckpt; resuming from Drive snapshot {snap} "
              f"(epoch {snap_epoch}, fresh optimizer)")
        sd = torch.load(snap, map_location="cpu", weights_only=False)
        if sd.get("tokenization_id") not in (None, tok_id):
            raise SystemExit("snapshot tokenization_id mismatch")
        state_dict = sd["model"]
        saved_epoch = snap_epoch          # snapshots are written at epoch end
        resuming = True                   # optimizer_state stays None -> fresh
    else:
        if not os.path.exists(ckpt_in):
            raise SystemExit(f"pretrain checkpoint not found: {ckpt_in}")
        print(f"[init] loading pretrain weights from {ckpt_in} "
              f"(mmap — reads only the ~8 GB model, skips the ~16 GB optimizer)")
        try:
            pre = torch.load(ckpt_in, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError):
            pre = torch.load(ckpt_in, map_location="cpu", weights_only=False)
        if pre.get("tokenization_id") and pre["tokenization_id"] != tok_id:
            raise SystemExit(f"pretrain ckpt tokenization_id {pre['tokenization_id']} != {tok_id}")
        state_dict = pre["model"] if isinstance(pre, dict) and pre.get("schema") == "v2" else pre

    # strip torch.compile prefix; infer vocab from the checkpoint itself
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    print(f"[model] vocab_size (from checkpoint) = {vocab_size}")

    model = SynapseGPT(vocab_size)
    model_state = model.state_dict()
    missing = set(model_state) - set(state_dict)
    mismatched = {k for k in model_state.keys() & state_dict.keys()
                  if model_state[k].shape != state_dict[k].shape}
    extra = set(state_dict) - set(model_state)
    if extra:
        print(f"  [load] ignoring {len(extra)} extra checkpoint keys")
    if missing or mismatched:
        raise SystemExit(f"checkpoint load failed: missing={sorted(missing)[:8]} "
                         f"mismatched={sorted(mismatched)[:8]}")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"[model] SynapseGPT loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # --- 3. Inductor config (mirror pretrain) + compile ---
    import torch._inductor.config as _ic
    _ic.max_fusion_size = 16
    _ic.epilogue_fusion = False
    try:
        _ic.triton.persistent_reductions = False
    except AttributeError:
        pass
    if USE_COMPILE:
        # dynamic=True: SFT batches are variable-length (dynamic padding), unlike
        # pretrain's fixed BLOCK_SIZE — without it every new length recompiles.
        print("[compile] torch.compile(dynamic=True)")
        model = torch.compile(model, dynamic=True)

    # --- 4. Optimizer (fresh; pretrain moments discarded) ---
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=MAX_LR, betas=BETAS, fused=torch.cuda.is_available())
    if resuming and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("[opt] restored optimizer state")

    # --- 5. Data ---
    train_examples, weights, val_by_source = load_all_data(syn, tok_id)
    steps_per_epoch = math.ceil(len(train_examples) / (BATCH_SIZE * GRAD_ACCUM_STEPS))
    total_steps = EPOCHS * steps_per_epoch
    # On resume, keep the ORIGINAL horizon from the checkpoint so the cosine
    # schedule is identical even if the data size (hence steps_per_epoch) changed.
    lr_horizon = saved_total_steps if (resuming and saved_total_steps) else total_steps
    print(f"[plan] {len(train_examples):,} train / "
          f"{sum(len(v) for v in val_by_source.values()):,} val | "
          f"{EPOCHS} epochs x ~{steps_per_epoch} steps = {total_steps} total "
          f"| warmup={WARMUP_STEPS} horizon={lr_horizon}")

    # --- 6. Baseline (pre-SFT) eval — only on a fresh run ---
    if not resuming:
        print("[baseline] per-source val loss on the pre-SFT checkpoint:")
        baseline_eval = eval_all(model, val_by_source, vocab_size)
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in baseline_eval.items()))

    # --- 7. Training loop ---
    # Resume policy: a completed epoch advances to the next; a mid-epoch crash
    # (epoch_complete=False) restarts the in-progress epoch from its boundary —
    # weights + optimizer are preserved, and the LR replays this epoch's range.
    if resuming and not saved_epoch_complete:
        start_epoch = saved_epoch
        curr_step = (saved_epoch - 1) * steps_per_epoch
        print(f"[resume] epoch {saved_epoch} was incomplete — restarting it from step {curr_step}")
    elif resuming:
        start_epoch = saved_epoch + 1
        # model-only Drive resume has no saved curr_step → derive from the boundary
        curr_step = saved_curr_step or (saved_epoch * steps_per_epoch)
    else:
        start_epoch = 1
        curr_step = 0
    last_eval = saved_last_eval
    # Best-val tracking (v2): derived from eval_history so it survives resume
    # without a new checkpoint field. The final model is sft_best.pth, not the
    # endpoint (v1's val bottomed at step 1400/2280 and the bottom was lost).
    best_val = min((e["overall"] for e in eval_history if "overall" in e),
                   default=float("inf"))
    if best_val != float("inf"):
        print(f"[best] resuming with best overall val so far = {best_val:.4f}")

    def maybe_save_best(val, epoch, step):
        nonlocal best_val
        if val["overall"] < best_val:
            best_val = val["overall"]
            save_checkpoint(best_path, model, model_only=True, epoch=epoch, tok_id=tok_id)
            print(f"  [best] overall val {best_val:.4f} @ step {step} → sft_best.pth")
            mirror(best_path, durable=True)

    print(f"\nSTARTING SFT (bf16, shifted masked loss) from epoch {start_epoch}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        gen = torch.Generator().manual_seed(SEED + epoch)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True, generator=gen)
        loader = DataLoader(train_examples, batch_size=BATCH_SIZE, sampler=sampler,
                            collate_fn=collate, num_workers=2,
                            pin_memory=(device.type == "cuda"), drop_last=False)

        micro_in_window = 0
        window_loss, window_tokens = 0.0, 0    # H2: token-weighted accumulation
        log_loss = float("nan")
        t0 = time.time()
        for input_ids, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                # H2 fix: SUM of token losses, not per-microbatch mean. Grads
                # accumulate unscaled; before step we divide by the window's
                # TOTAL learned-token count, giving the true token-mean
                # sum(all_token_losses)/sum(all_tokens) — a short-answer
                # microbatch no longer outweighs a long-CoT one.
                loss = shifted_ce(logits, labels, vocab_size, reduction="sum")
            loss.backward()
            window_loss += loss.item()
            window_tokens += int((labels[:, 1:] != IGNORE_INDEX).sum().item())
            micro_in_window += 1

            if micro_in_window == GRAD_ACCUM_STEPS:
                if window_tokens == 0:
                    raise SystemExit("accumulation window has 0 learned tokens — "
                                     "labels are fully masked; data is broken")
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(window_tokens)
                lr = get_lr(curr_step, lr_horizon)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                log_loss = window_loss / window_tokens
                micro_in_window = 0
                window_loss, window_tokens = 0.0, 0
                curr_step += 1

                if curr_step % 20 == 0:
                    dt = time.time() - t0
                    print(f"  epoch {epoch} step {curr_step}/{total_steps} "
                          f"loss={log_loss:.4f} lr={lr:.2e} "
                          f"grad_norm={grad_norm:.2f} ({dt:.1f}s/20-window)")
                    t0 = time.time()

                if curr_step % EVAL_EVERY == 0:
                    val = eval_all(model, val_by_source, vocab_size)
                    last_eval = val["overall"]
                    eval_history.append({"step": curr_step, "epoch": epoch, **val})
                    print("  [eval] " + "  ".join(f"{k}={v:.3f}" for k, v in val.items()))
                    maybe_save_best(val, epoch, curr_step)

                if curr_step % SAVE_EVERY_STEPS == 0:
                    # mid-epoch crash safety: full checkpoint (LOCAL, fast)
                    save_checkpoint(latest_path, model, model_only=False,
                                    optimizer=optimizer, curr_step=curr_step,
                                    total_steps=lr_horizon, epoch=epoch,
                                    epoch_complete=False, tok_id=tok_id,
                                    eval_history=eval_history, last_eval=last_eval,
                                    baseline=baseline_eval)
                    # periodically mirror it to Drive in the background (survives a
                    # Colab session stop) without blocking training
                    if curr_step % DRIVE_EVERY_STEPS == 0:
                        mirror(latest_path)

        # clean-break flush: apply the leftover partial accumulation window
        # (H2: same token-mean normalization as a full window)
        if micro_in_window > 0:
            if window_tokens == 0:
                raise SystemExit("partial window has 0 learned tokens — data is broken")
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(window_tokens)
            lr = get_lr(curr_step, lr_horizon)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            curr_step += 1

        # end-of-epoch eval + checkpoints
        val = eval_all(model, val_by_source, vocab_size)
        last_eval = val["overall"]
        eval_history.append({"step": curr_step, "epoch": epoch, **val})
        print(f"[epoch {epoch} done] " + "  ".join(f"{k}={v:.3f}" for k, v in val.items()))
        maybe_save_best(val, epoch, curr_step)

        # full latest (with optimizer) — LOCAL only (fast same-session resume)
        save_checkpoint(latest_path, model, model_only=False, optimizer=optimizer,
                        curr_step=curr_step, total_steps=lr_horizon, epoch=epoch,
                        epoch_complete=True, tok_id=tok_id, eval_history=eval_history,
                        last_eval=last_eval, baseline=baseline_eval)
        # mirror the full checkpoint off-box (background) for cross-session resume
        mirror(latest_path, durable=True)
        # model-only snapshot (~8 GB) — local, then mirrored (durable artifact)
        snap_local = os.path.join(work_dir, f"sft_epoch{epoch}.pth")
        save_checkpoint(snap_local, model, model_only=True, epoch=epoch, tok_id=tok_id)
        mirror(snap_local, durable=True)
        write_manifest(manifest_dir, tok_id, epoch, curr_step, total_steps,
                       baseline_eval, eval_history, val)

    # join in-flight background pushes, then do ONE guaranteed synchronous mirror
    # of the final checkpoint (daemon threads would be killed on exit mid-copy)
    for th in (_push_thread, _rclone_thread):
        if th is not None and th.is_alive():
            print("[mirror] waiting for final background push to finish...")
            th.join()
    if PUSH_REMOTE:
        print("[mirror] final rclone push of full checkpoint")
        subprocess.run(["rclone", "copyto", latest_path,
                        PUSH_REMOTE + "/" + os.path.basename(latest_path),
                        "--checksum", "--drive-chunk-size=64M"], check=False)
    elif os.path.abspath(work_dir) != os.path.abspath(out_dir):
        print("[mirror] final sync of full checkpoint -> Drive")
        shutil.copyfile(latest_path, drive_latest)

    print("\nSFT complete.")
    if baseline_eval:
        print(f"  baseline overall = {baseline_eval['overall']:.3f}")
    if last_eval is not None:
        print(f"  final    overall = {last_eval:.3f}")
    if best_val != float("inf"):
        print(f"  best     overall = {best_val:.4f}  → ship sft_best.pth, not the endpoint")


def write_manifest(manifest_dir, tok_id, epoch, curr_step, total_steps,
                   baseline, eval_history, last_val):
    path = os.path.join(manifest_dir, "sft_training_latest.json")
    payload = {
        "stage": "sft",
        "tokenization_id": tok_id,
        "epoch": epoch,
        "curr_step": curr_step,
        "total_steps": total_steps,
        "data_mix": SFT_DATA_MIX,
        "hyperparameters": {
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM_STEPS,
            "max_lr": MAX_LR, "min_lr": MIN_LR, "warmup": WARMUP_STEPS,
            "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY,
        },
        "results": {"baseline_eval": baseline, "latest_eval": last_val,
                    "eval_history": eval_history},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
