#!/usr/bin/env python3
"""SynapseGPT supervised fine-tuning — Alpaca v1.

Loads a v2 pretrain checkpoint, fine-tunes on tokenized ChatML data with
prefix-masked cross-entropy, writes best+latest sft_v1 checkpoints.

Usage:
  SYNAPSE_DIR=/path/to/synapse python sft/sft_train.py

Configurable via environment variables:
  SYNAPSE_DIR                base data dir (default: Drive on Colab, ./synapse else)
  PRETRAIN_CHECKPOINT_NAME   default: synapse_2b_d2560_l28.pth
  SFT_BEST_NAME              default: sft_synapse_best.pth
  SFT_LATEST_NAME            default: sft_synapse_latest.pth
  CHECKPOINT_PUSH_REMOTE     rclone remote dir for SFT checkpoints+manifest

Expects sft_tokenized/<name>/{train,val}.jsonl produced by tokenize_sft_data.py
and refuses to run if the tokenization_id baked into the SFT data doesn't
match the pretrain checkpoint's.

Tech debt: duplicates ~55 lines of helpers from pretrain/train.py
(_push_to_remote_async, file_info, default_synapse_dir, partial_file_hash).
Consolidate into synapse_model/utils.py in a future cleanup PR — out of
scope here because that PR would touch pretrain/train.py again.
"""
import os
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import datetime
import hashlib
import json
import math
import random
import subprocess
import threading
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

from synapse_model import SynapseGPT, BLOCK_SIZE


# ==================== 1. HELPERS (duplicated from pretrain/train.py) ====================
def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def partial_file_hash(path, edge=1 << 20):
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(min(edge, size)))
        if size > 2 * edge:
            f.seek(-edge, 2)
            h.update(f.read(edge))
    h.update(size.to_bytes(8, "little"))
    return h.hexdigest()


def file_info(path, *, full_hash=False):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    size = os.path.getsize(path)
    info = {"path": path, "size_mb": round(size / 1024 / 1024, 2), "size_bytes": size}
    if full_hash:
        info["sha256"] = file_hash(path)
    else:
        info["sha256_partial"] = partial_file_hash(path)
    return info


def _push_to_remote_async(local_path, remote_dir):
    if not remote_dir:
        return
    remote_path = remote_dir.rstrip("/") + "/" + os.path.basename(local_path)
    def _run():
        try:
            r = subprocess.run(
                ["rclone", "copyto", local_path, remote_path,
                 "--checksum", "--drive-chunk-size=64M"],
                capture_output=True, text=True, timeout=3600,
            )
            if r.returncode == 0:
                print(f"  pushed {os.path.basename(local_path)} -> {remote_dir}")
            else:
                print(f"  WARNING: rclone push failed for "
                      f"{os.path.basename(local_path)}: {r.stderr.strip()[:200]}")
        except Exception as e:
            print(f"  WARNING: rclone push exception: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ==================== 2. CONFIGURATION ====================
SYNAPSE = os.environ.get("SYNAPSE_DIR") or default_synapse_dir()
CHECKPOINT_DIR = os.path.join(SYNAPSE, "checkpoints")
MANIFEST_DIR = os.path.join(SYNAPSE, "manifests")
TOKENIZER_PATH = os.path.join(SYNAPSE, "tokenizer_out", "tokenizer.json")
SFT_DATA_DIR = os.path.join(SYNAPSE, "sft_tokenized")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

PRETRAIN_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, os.environ.get("PRETRAIN_CHECKPOINT_NAME", "synapse_2b_d2560_l28.pth"),
)
SFT_BEST_PATH = os.path.join(
    CHECKPOINT_DIR, os.environ.get("SFT_BEST_NAME", "sft_synapse_best.pth"),
)
SFT_LATEST_PATH = os.path.join(
    CHECKPOINT_DIR, os.environ.get("SFT_LATEST_NAME", "sft_synapse_latest.pth"),
)

DATASET_MIX = {
    "alpaca": {"weight": 1.0},
}

# Hyperparams
BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 8           # effective batch = 64 sequences
EPOCHS           = 3
MAX_LR           = 1e-5
MIN_LR           = 1e-6
WARMUP_STEPS     = 100
WEIGHT_DECAY     = 0.1
BETAS            = (0.9, 0.95)
GRAD_CLIP        = 1.0

# Length-bucketed sampler — bucket cap is upper bound (inclusive) in tokens.
BUCKET_EDGES     = [256, 512, 1024, 2048]

# Eval / checkpoint cadence
EVAL_EVERY_STEPS = 100
LATEST_SAVE_EVERY_STEPS = 200

# Set DRY_RUN_STEPS=N to exit cleanly after N optimizer steps without eval/save
# /generation — smoke test that setup, data loading, checkpoint load, forward,
# backward, optimizer all work end-to-end. 0 = normal run.
DRY_RUN_STEPS = int(os.environ.get("DRY_RUN_STEPS", "0"))

# Special token IDs (must match tokenize_sft_data.py)
PAD_ID = 1
EOT_ID = 0

# Hardware
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
else:
    raise SystemExit("SFT requires a CUDA GPU.")


# ==================== 3. LOAD TOKENIZER + VERIFY TOKENIZATION_ID ====================
if not os.path.exists(TOKENIZER_PATH):
    raise SystemExit(f"tokenizer.json missing at {TOKENIZER_PATH}")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()
assert tokenizer.token_to_id("<|pad|>") == PAD_ID, "PAD_ID drifted from tokenizer"
assert tokenizer.token_to_id("<|endoftext|>") == EOT_ID, "EOT_ID drifted from tokenizer"
IM_END_ID = tokenizer.token_to_id("<|im_end|>")
print(f"tokenizer: vocab={VOCAB_SIZE}, pad={PAD_ID}, eot={EOT_ID}, im_end={IM_END_ID}")

# Load tokenization ids from both manifests; they must match each other.
pretrain_manifest = os.path.join(MANIFEST_DIR, "tokenization_latest.json")
sft_manifest_path = os.path.join(MANIFEST_DIR, "sft_tokenization_latest.json")
for p in (pretrain_manifest, sft_manifest_path):
    if not os.path.exists(p):
        raise SystemExit(f"required manifest missing: {p}")
with open(pretrain_manifest) as f:
    pretrain_tok_id = json.load(f)["tokenization_id"]
with open(sft_manifest_path) as f:
    sft_tok_id = json.load(f)["tokenization_id"]
if pretrain_tok_id != sft_tok_id:
    raise SystemExit(
        f"tokenization_id mismatch:\n"
        f"  pretrain: {pretrain_tok_id}\n"
        f"  sft data: {sft_tok_id}\n"
        f"refusing to run — model and SFT data use different tokenizers"
    )
TOKENIZATION_ID = pretrain_tok_id
print(f"tokenization_id: {TOKENIZATION_ID} (verified across pretrain + SFT)")


# ==================== 4. LOAD SFT DATA ====================
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            rows.append({"input_ids": ex["input_ids"], "prefix_len": ex["prefix_len"]})
    return rows


print("\nLoading SFT data...")
train_by_dataset = {}
val_by_dataset = {}
for name, spec in DATASET_MIX.items():
    base = os.path.join(SFT_DATA_DIR, name)
    train_by_dataset[name] = load_jsonl(os.path.join(base, "train.jsonl"))
    val_by_dataset[name] = load_jsonl(os.path.join(base, "val.jsonl"))
    print(f"  {name}: train={len(train_by_dataset[name]):,}, val={len(val_by_dataset[name]):,} "
          f"(weight={spec['weight']})")

# Training pool. With a single dataset this is just the rows; multi-dataset
# weighting comes in Phase 4 when OASST1 lands.
training_pool = []
for name, rows in train_by_dataset.items():
    training_pool.extend(rows)
print(f"  training pool: {len(training_pool):,} examples per epoch")
if len(DATASET_MIX) > 1:
    raise NotImplementedError(
        "weighted multi-dataset mixing not implemented yet — single dataset only for now"
    )


# ==================== 5. LENGTH-BUCKETED BATCH BUILDER ====================
def bucket_index(length, edges):
    for i, e in enumerate(edges):
        if length <= e:
            return i
    return len(edges) - 1  # overflow → put in last bucket (shouldn't happen given truncation)


def build_batches(rows, batch_size, edges, pad_id, rng):
    # Group by bucket, shuffle within bucket, form batches of `batch_size`,
    # pad to bucket-max within batch. Then shuffle the batch order so the
    # model doesn't see short→long sequentially (would bias gradients).
    buckets = [[] for _ in edges]
    for ex in rows:
        buckets[bucket_index(len(ex["input_ids"]), edges)].append(ex)
    batches = []
    for bucket in buckets:
        if not bucket:
            continue
        rng.shuffle(bucket)
        # Drop the last incomplete batch — keeps every step exactly batch_size.
        n_full = (len(bucket) // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            chunk = bucket[i:i + batch_size]
            max_len = max(len(ex["input_ids"]) for ex in chunk)
            ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
            pl = torch.tensor([ex["prefix_len"] for ex in chunk], dtype=torch.long)
            for j, ex in enumerate(chunk):
                row = ex["input_ids"]
                ids[j, :len(row)] = torch.tensor(row, dtype=torch.long)
            batches.append((ids, pl))
    rng.shuffle(batches)
    return batches


# ==================== 6. LOSS WITH PREFIX + PAD MASK ====================
def compute_loss(model, input_ids, prefix_lens):
    # input_ids: [B, T], prefix_lens: [B]
    # Train on labels where the corresponding input is a response token (not prompt, not pad).
    x = input_ids[:, :-1]
    labels = input_ids[:, 1:].clone()
    # Mask pad tokens in the target.
    labels[labels == PAD_ID] = -100
    # Mask prefix tokens per row. label index k corresponds to predicting token k+1
    # of the original sequence; we want to skip when k+1 < prefix_len, i.e. k < prefix_len - 1.
    positions = torch.arange(labels.size(1), device=labels.device)
    prefix_mask = positions.unsqueeze(0) < (prefix_lens - 1).unsqueeze(1)
    labels[prefix_mask] = -100
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
            ignore_index=-100,
        )
    return loss


# ==================== 7. BUILD MODEL FROM PRETRAIN CHECKPOINT ====================
print(f"\nLoading pretrain checkpoint: {PRETRAIN_CHECKPOINT_PATH}")
if not os.path.exists(PRETRAIN_CHECKPOINT_PATH):
    raise SystemExit(f"pretrain checkpoint missing: {PRETRAIN_CHECKPOINT_PATH}")
ckpt = torch.load(PRETRAIN_CHECKPOINT_PATH, map_location="cpu")
if isinstance(ckpt, dict) and ckpt.get("schema") == "v2":
    state = ckpt["model"]
    ckpt_tok_id = ckpt.get("tokenization_id")
    ckpt_step = ckpt.get("curr_step", "?")
    print(f"  v2 checkpoint at step {ckpt_step}, tokenization_id={ckpt_tok_id}")
    if ckpt_tok_id and ckpt_tok_id != TOKENIZATION_ID:
        raise SystemExit(
            f"checkpoint tokenization_id {ckpt_tok_id} != current {TOKENIZATION_ID}"
        )
else:
    state = ckpt
    print("  legacy checkpoint (model-only)")
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

model = SynapseGPT(VOCAB_SIZE, gradient_checkpointing=True).to(device)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f"state_dict mismatch when loading pretrain checkpoint:\n"
        f"  missing ({len(missing)}): {list(missing)[:5]}\n"
        f"  unexpected ({len(unexpected)}): {list(unexpected)[:5]}"
    )
n_params = sum(p.numel() for p in model.parameters())
print(f"  model: {n_params/1e9:.3f}B params (gradient_checkpointing=True)")


# ==================== 8. OPTIMIZER WITH DECAY GROUPS ====================
def make_optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    print(f"  decay: {sum(p.numel() for p in decay)/1e6:.1f}M params, "
          f"no_decay: {sum(p.numel() for p in no_decay)/1e6:.1f}M")
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=MAX_LR, betas=BETAS, fused=True,
    )


optimizer = make_optimizer(model)


# ==================== 9. LR SCHEDULE ====================
# Total optimizer steps over the full SFT run (estimated from epoch 1 batch count;
# refined after we build the first epoch's batches).
def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return MAX_LR * step / max(1, WARMUP_STEPS)
    if step >= total_steps:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# ==================== 10. EVAL ====================
@torch.no_grad()
def run_eval():
    model.eval()
    per_dataset = {}
    overall_loss, overall_count = 0.0, 0
    rng = random.Random(1337)
    for name, rows in val_by_dataset.items():
        batches = build_batches(rows, BATCH_SIZE, BUCKET_EDGES, PAD_ID, rng)
        if not batches:
            per_dataset[name] = float("nan")
            continue
        total, n = 0.0, 0
        for ids, pl in batches:
            ids, pl = ids.to(device, non_blocking=True), pl.to(device, non_blocking=True)
            loss = compute_loss(model, ids, pl).item()
            total += loss
            n += 1
        per_dataset[name] = total / n
        overall_loss += total
        overall_count += n
    model.train()
    return {"overall": overall_loss / max(1, overall_count), **per_dataset}


# ==================== 11. CHECKPOINT SAVES ====================
def save_sft(path, *, include_optimizer, step, eval_history, total_steps):
    payload = {
        "schema": "sft_v1",
        "model": model.state_dict(),
        "tokenization_id": TOKENIZATION_ID,
        "step": step,
        "sft_config": {
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs": EPOCHS,
            "max_lr": MAX_LR, "min_lr": MIN_LR, "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY, "betas": list(BETAS), "grad_clip": GRAD_CLIP,
            "bucket_edges": BUCKET_EDGES,
            "dataset_mix": {k: v["weight"] for k, v in DATASET_MIX.items()},
            "total_steps_estimate": total_steps,
        },
        "eval_history": eval_history,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  saved: {path}  (optimizer={'yes' if include_optimizer else 'no'})")
    _push_to_remote_async(path, os.environ.get("CHECKPOINT_PUSH_REMOTE"))


def write_sft_training_manifest(*, step, total_steps, eval_history, status):
    manifest = {
        "stage": "sft_training",
        "status": status,
        "created": datetime.datetime.now().isoformat(),
        "tokenization_id": TOKENIZATION_ID,
        "step": step,
        "total_steps_estimate": total_steps,
        "best_checkpoint": file_info(SFT_BEST_PATH),
        "latest_checkpoint": file_info(SFT_LATEST_PATH),
        "sft_config": {
            "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs": EPOCHS, "max_lr": MAX_LR, "min_lr": MIN_LR,
            "warmup_steps": WARMUP_STEPS, "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS), "grad_clip": GRAD_CLIP,
            "bucket_edges": BUCKET_EDGES,
            "dataset_mix": {k: v["weight"] for k, v in DATASET_MIX.items()},
        },
        "data": {
            name: {"train": len(train_by_dataset[name]), "val": len(val_by_dataset[name])}
            for name in DATASET_MIX
        },
        "eval_history": eval_history,
    }
    path = os.path.join(MANIFEST_DIR, "sft_training_latest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)
    remote = os.environ.get("CHECKPOINT_PUSH_REMOTE", "").rstrip("/")
    if remote.endswith("/checkpoints"):
        manifest_remote = remote[: -len("/checkpoints")] + "/manifests"
        _push_to_remote_async(path, manifest_remote)


# ==================== 12. TRAINING LOOP ====================
print("\nBuilding epoch-1 batches to estimate total steps...")
rng = random.Random(42)
epoch_batches = build_batches(training_pool, BATCH_SIZE, BUCKET_EDGES, PAD_ID, rng)
batches_per_epoch = len(epoch_batches)
optimizer_steps_per_epoch = batches_per_epoch // GRAD_ACCUM_STEPS
total_steps = optimizer_steps_per_epoch * EPOCHS
print(f"  {batches_per_epoch} batches/epoch × {EPOCHS} epochs = {batches_per_epoch * EPOCHS} forward passes")
print(f"  effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS} sequences")
print(f"  total optimizer steps = {total_steps}")

curr_step = 0
best_eval_loss = float("inf")
eval_history = []
train_start = time.time()

print("\n" + "=" * 60)
print(f"STARTING SFT — {n_params/1e9:.2f}B params, bf16, grad-ckpt")
if DRY_RUN_STEPS:
    print(f"DRY RUN: will exit after {DRY_RUN_STEPS} optimizer steps, no eval/save/gen")
print("=" * 60)

model.train()
for epoch in range(1, EPOCHS + 1):
    if epoch > 1:
        epoch_batches = build_batches(training_pool, BATCH_SIZE, BUCKET_EDGES, PAD_ID, rng)
    pbar = tqdm(epoch_batches, desc=f"Ep {epoch}/{EPOCHS}", dynamic_ncols=True, mininterval=1.0)
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (ids, pl) in enumerate(pbar):
        ids = ids.to(device, non_blocking=True)
        pl = pl.to(device, non_blocking=True)

        lr = get_lr(curr_step, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss = compute_loss(model, ids, pl) / GRAD_ACCUM_STEPS
        loss.backward()
        accum_loss += loss.item()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            curr_step += 1
            # accum_loss is the mean per-micro-batch loss (each .item() was
            # already divided by GRAD_ACCUM_STEPS, summed across the step).
            step_loss = accum_loss
            accum_loss = 0.0
            elapsed = time.time() - train_start
            pbar.set_postfix(
                loss=f"{step_loss:.3f}",
                gnorm=f"{float(grad_norm):.2f}",
                lr=f"{lr:.2e}",
                step=curr_step,
                bktT=ids.size(1),
                hrs=f"{elapsed/3600:.2f}",
            )

            if DRY_RUN_STEPS and curr_step >= DRY_RUN_STEPS:
                vram = torch.cuda.max_memory_allocated() / 1024**3
                print(f"\n{'=' * 60}")
                print(f"DRY RUN COMPLETE — {curr_step} optimizer steps in {elapsed:.1f}s")
                print(f"  last step_loss: {step_loss:.4f}")
                print(f"  last grad_norm: {float(grad_norm):.3f}")
                print(f"  last bucket seq_len: {ids.size(1)}")
                print(f"  peak VRAM: {vram:.2f} GB")
                print(f"  est full-run wall time (per epoch): "
                      f"{(elapsed/curr_step) * optimizer_steps_per_epoch / 60:.1f} min")
                print(f"{'=' * 60}")
                sys.exit(0)

            # Eval
            if curr_step % EVAL_EVERY_STEPS == 0:
                eval_result = run_eval()
                eval_history.append({
                    "step": curr_step,
                    "loss": eval_result["overall"],
                    "by_dataset": {k: v for k, v in eval_result.items() if k != "overall"},
                    "elapsed_hrs": round(elapsed / 3600, 2),
                })
                per_src = " ".join(f"{k}={v:.3f}" for k, v in sorted(eval_result.items())
                                   if k != "overall")
                print(f"\n  [eval @ step {curr_step}] overall={eval_result['overall']:.3f} | {per_src}")
                if eval_result["overall"] < best_eval_loss:
                    best_eval_loss = eval_result["overall"]
                    print(f"  new best eval loss — saving sft_best")
                    save_sft(SFT_BEST_PATH, include_optimizer=False, step=curr_step,
                             eval_history=eval_history, total_steps=total_steps)
                write_sft_training_manifest(
                    step=curr_step, total_steps=total_steps,
                    eval_history=eval_history, status="running",
                )

            # Periodic latest save (crash recovery)
            if curr_step % LATEST_SAVE_EVERY_STEPS == 0:
                save_sft(SFT_LATEST_PATH, include_optimizer=True, step=curr_step,
                         eval_history=eval_history, total_steps=total_steps)


# ==================== 13. FINALIZE ====================
total_time = time.time() - train_start
print(f"\nSFT complete. {total_time/3600:.2f} hours, {curr_step} optimizer steps.")
print(f"Best eval loss: {best_eval_loss:.4f}")

# Final eval + saves
eval_result = run_eval()
eval_history.append({
    "step": curr_step,
    "loss": eval_result["overall"],
    "by_dataset": {k: v for k, v in eval_result.items() if k != "overall"},
    "elapsed_hrs": round(total_time / 3600, 2),
    "final": True,
})
if eval_result["overall"] < best_eval_loss:
    best_eval_loss = eval_result["overall"]
    save_sft(SFT_BEST_PATH, include_optimizer=False, step=curr_step,
             eval_history=eval_history, total_steps=total_steps)
save_sft(SFT_LATEST_PATH, include_optimizer=True, step=curr_step,
         eval_history=eval_history, total_steps=total_steps)
write_sft_training_manifest(
    step=curr_step, total_steps=total_steps,
    eval_history=eval_history, status="completed",
)


# ==================== 14. GENERATION EVAL (sanity check) ====================
@torch.no_grad()
def generate(prompt, max_new_tokens=128):
    model.eval()
    prompt_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([prompt_ids], device=device)
    out_ids = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        next_id = int(logits[0, -1].argmax().item())
        out_ids.append(next_id)
        if next_id == IM_END_ID or next_id == EOT_ID:
            break
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(out_ids)


PROMPTS = [
    "Summarize the plot of Hamlet in two sentences.",
    "Translate to French: Good morning, how are you today?",
    "What is the capital of Australia?",
    "Write a haiku about autumn leaves.",
    "Explain the difference between Python lists and tuples.",
    "If a train leaves at 3pm going 60 mph and another at 4pm going 80 mph, when do they meet?",
    "Give three healthy breakfast ideas.",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "Write a polite email declining a meeting.",
    "What are the primary colors?",
]

print("\n" + "=" * 60)
print("GENERATION EVAL")
print("=" * 60)
for p in PROMPTS:
    chatml = f"<|im_start|>user\n{p}\n<|im_end|>\n<|im_start|>assistant"
    response = generate(chatml, max_new_tokens=128)
    print(f"\n>>> {p}")
    print(f"    {response.strip()}")

print(f"\nDone. Best eval loss = {best_eval_loss:.4f}.")
