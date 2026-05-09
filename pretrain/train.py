#!/usr/bin/env python3
"""SynapseGPT pretraining — runs in Colab or on a bare VM (Lambda, GCP, etc.).

Configurable via environment variables (all optional):
  SYNAPSE_DIR        Base dir holding token_shards_merged/, checkpoints/, manifests/.
                     Default: /content/drive/MyDrive/synapse on Colab, ./synapse elsewhere.
  CHECKPOINT_NAME    Checkpoint filename. Default: synapse_2b_d2560_l28.pth.
  MAX_TOKENS         Token budget. Default: 42_000_000_000.
  EXPECTED_TOK_ID    Required tokenization_id. Default: 7a570a7ba9fc7985.
  SKIP_DRIVE_MOUNT   If "1", don't try to mount Google Drive even on Colab.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import hashlib
import random
import math
import time
import shutil
import datetime
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

torch._logging.set_logs(inductor=logging.WARNING, dynamic=logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


# ==================== 1. ENVIRONMENT SETUP ====================
def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

def maybe_mount_drive():
    if os.environ.get("SKIP_DRIVE_MOUNT") == "1":
        return
    if not in_colab():
        return
    if os.path.isdir("/content/drive/MyDrive"):
        return
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount("/content/drive", force_remount=False)

def default_synapse_dir() -> str:
    if in_colab():
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


# ==================== 2. MANIFEST HELPERS ====================
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def file_info(path):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    return {
        "path": path,
        "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
        "sha256": file_hash(path),
    }

def filter_present_shards(shards, shard_dir, dtype_bytes):
    # Drop manifest entries whose .bin is absent or wrong-size; lets training
    # start while rclone is still mid-upload. Wrong-size catches partial files.
    kept, missing, wrong_size = [], [], []
    for s in shards:
        path = os.path.join(shard_dir, s["shard"])
        if not os.path.exists(path):
            missing.append(s)
            continue
        if os.path.getsize(path) != s["tokens"] * dtype_bytes:
            wrong_size.append(s)
            continue
        kept.append(s)
    return kept, missing, wrong_size


# ==================== 3. CONFIGURATION ====================
maybe_mount_drive()

SYNAPSE = os.environ.get("SYNAPSE_DIR") or default_synapse_dir()
SHARD_DIR = os.path.join(SYNAPSE, "token_shards_merged")
CHECKPOINT_DIR = os.path.join(SYNAPSE, "checkpoints")
MANIFEST_DIR = os.path.join(SYNAPSE, "manifests")
TOKENIZER_PATH = os.path.join(SYNAPSE, "tokenizer_out", "tokenizer.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "synapse_2b_d2560_l28.pth")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

if not os.path.isdir(SHARD_DIR):
    raise FileNotFoundError(
        f"Token shards not found at {SHARD_DIR}. "
        f"Set SYNAPSE_DIR to point at the synapse data root."
    )

# Archive old checkpoint if it exists
if os.path.exists(CHECKPOINT_PATH):
    mod_time = os.path.getmtime(CHECKPOINT_PATH)
    date_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
    archived_name = CHECKPOINT_NAME.replace(".pth", f"_{date_str}.pth")
    archived_path = os.path.join(CHECKPOINT_DIR, archived_name)
    if not os.path.exists(archived_path):
        shutil.copy2(CHECKPOINT_PATH, archived_path)
        print(f"Archived old checkpoint as: {archived_name}")

# -- MODEL ARCHITECTURE (Shape D, ~2.1B params) --
BLOCK_SIZE      = 2048
STRIDE          = BLOCK_SIZE
EMBED_DIM       = 2560
NUM_LAYERS      = 28
NUM_HEADS       = 20            # head_dim = 128
NUM_KV_HEADS    = 4             # GQA, group size = 5
FF_HIDDEN_DIM   = 6912          # ~8/3 * EMBED_DIM, multiple of 128
ROPE_BASE       = 10000.0
RMSNORM_EPS     = 1e-5
GRAD_CHECKPOINT = True

# -- TRAINING HYPERPARAMETERS --
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 64           # effective batch = 256 sequences = 524K tokens/step
EPOCHS           = 1
MAX_LR           = 2e-4
MIN_LR           = 2e-5
WEIGHT_DECAY     = 0.1
WARMUP_STEPS     = 4000
BETAS            = (0.9, 0.95)
GRAD_CLIP        = 1.0

# -- DATA SELECTION --
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 42_000_000_000))

DATA_MIX = {
    "data_wikipedia":       0.40,
    "data_c4":              0.20,
    "data_code":            0.15,
    "data_finemath":        0.10,
    "data_books_gutemberg": 0.05,
    "data_books_faded":     0.03,
    "data_arxiv":           0.02,
    "data_adult":           0.02,
    "data_distilled_facts": 0.03,
}

# -- EVAL --
EVAL_FRACTION_PER_SOURCE = 0.02
EVAL_EVERY_STEPS         = 500
EVAL_BATCHES             = 32
EVAL_SEED                = 1337

# -- MID-EPOCH CHECKPOINT --
SAVE_EVERY_N_SHARDS = 50

# Hardware Settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: no CUDA device found — training on CPU will be unusably slow.")


# ==================== 4. MODEL DEFINITION ====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(in_dtype) * self.weight

def precompute_rope(head_dim, max_seq_len, base, device, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin

def apply_rope(x, cos, sin):
    T = x.size(-2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)

class CausalGQA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.n_rep = num_heads // num_kv_heads
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class SwiGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim,     bias=False)
    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, ff_hidden_dim, rms_eps):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim, eps=rms_eps)
        self.attn  = CausalGQA(embed_dim, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(embed_dim, eps=rms_eps)
        self.ff    = SwiGLU(embed_dim, ff_hidden_dim)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ff(self.norm2(x))
        return x

class SynapseGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.blocks = nn.ModuleList([
            TransformerBlock(EMBED_DIM, NUM_HEADS, NUM_KV_HEADS, FF_HIDDEN_DIM, RMSNORM_EPS)
            for _ in range(NUM_LAYERS)
        ])
        self.final_norm = RMSNorm(EMBED_DIM, eps=RMSNORM_EPS)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight  # tied
        head_dim = EMBED_DIM // NUM_HEADS
        cos, sin = precompute_rope(head_dim, BLOCK_SIZE, ROPE_BASE, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.gradient_checkpointing = GRAD_CHECKPOINT
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        x = self.token_embedding(idx)
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return self.lm_head(self.final_norm(x))


# ==================== 5. SHARDED DATASET ====================
class ShardDataset(Dataset):
    def __init__(self, tokens, block_size, stride):
        self.tokens = tokens
        self.block_size = block_size
        self.stride = stride
        self.length = max(0, (len(tokens) - block_size - 1) // stride)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        s = idx * self.stride
        if s + self.block_size + 1 > len(self.tokens):
            s = len(self.tokens) - self.block_size - 1
        chunk = self.tokens[s : s + self.block_size + 1]
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


# ==================== 6. LOAD META & MODEL ====================
META_PATH = os.path.join(SHARD_DIR, "meta.json")
with open(META_PATH, "r") as f:
    meta = json.load(f)
VOCAB_SIZE_RAW = int(meta["vocab_size"])
VOCAB_SIZE = VOCAB_SIZE_RAW
if VOCAB_SIZE % 64 != 0:
    VOCAB_SIZE = ((VOCAB_SIZE + 63) // 64) * 64
    print(f"Vocab padded to {VOCAB_SIZE} (Tensor Core Optimized)")

SHARD_DTYPE = np.dtype(meta.get("shard_dtype", "uint16"))
assert VOCAB_SIZE_RAW <= np.iinfo(SHARD_DTYPE).max + 1, (
    f"vocab_size {VOCAB_SIZE_RAW} exceeds {SHARD_DTYPE} range -- shards would overflow"
)
print(f"shard_dtype: {SHARD_DTYPE} | vocab raw: {VOCAB_SIZE_RAW} | padded: {VOCAB_SIZE}")

current_tok_id = meta.get("tokenization_id")
print(f"tokenization_id: {current_tok_id}")

EXPECTED_TOK_ID = os.environ.get("EXPECTED_TOK_ID", "7a570a7ba9fc7985")
tokid_txt_path = os.path.join(SHARD_DIR, "tokenization_id.txt")
tokid_txt = open(tokid_txt_path).read().strip()
if not (current_tok_id == tokid_txt == EXPECTED_TOK_ID):
    raise RuntimeError(
        f"tokenization_id mismatch: "
        f"meta.json={current_tok_id!r}, "
        f"tokenization_id.txt={tokid_txt!r}, "
        f"expected={EXPECTED_TOK_ID!r}"
    )

model = SynapseGPT(VOCAB_SIZE).to(device)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {n_params/1e9:.3f}B params ({n_trainable/1e9:.3f}B trainable)")

# --- CHECKPOINT LOADING ---
resume_path = None
if os.path.exists(CHECKPOINT_PATH):
    train_manifest_path = os.path.join(MANIFEST_DIR, "training_latest.json")
    safe_to_load = True
    if os.path.exists(train_manifest_path) and current_tok_id:
        with open(train_manifest_path) as f:
            train_manifest = json.load(f)
        ckpt_tok_id = train_manifest.get("tokenization_id")
        if ckpt_tok_id and ckpt_tok_id != current_tok_id:
            print(f"TOKENIZER MISMATCH — checkpoint: {ckpt_tok_id}, current: {current_tok_id}")
            print(f"  Training from scratch.")
            safe_to_load = False
        else:
            print(f"Tokenizer match confirmed.")
    if safe_to_load:
        print(f"Resuming from: {CHECKPOINT_PATH}")
        resume_path = CHECKPOINT_PATH
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        new_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model_state = model.state_dict()
        safe_state = {k: v for k, v in new_state.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(safe_state, strict=False)
        print(f"State loaded: {len(safe_state)} layers matched.")
else:
    print(f"No checkpoint found — training from scratch.")

# Cap Inductor fusion so RMSNorm forward+backward doesn't get merged into one
# Triton kernel that needs >99 KB of shared memory (Blackwell's opt-in ceiling).
import torch._inductor.config as _ic
_ic.max_fusion_size = 16
_ic.epilogue_fusion = False
try:
    _ic.triton.persistent_reductions = False
except AttributeError:
    pass  # older PyTorch — skip; the two above carry most of the fix

print("Compiling model...")
model = torch.compile(model)

# Optimizer
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params,   'weight_decay': WEIGHT_DECAY},
    {'params': nodecay_params, 'weight_decay': 0.0},
]
optimizer = optim.AdamW(optim_groups, lr=MAX_LR, betas=BETAS, fused=True)
print(f"Optimizer: AdamW(fused) lr={MAX_LR} betas={BETAS} wd={WEIGHT_DECAY}")
print(f"  decay groups: {sum(p.numel() for p in decay_params)/1e6:.1f}M params, "
      f"nodecay: {sum(p.numel() for p in nodecay_params)/1e6:.3f}M params")


# ==================== 7. SELECT SHARDS (TRAIN + EVAL SPLIT) ====================
with open(os.path.join(SHARD_DIR, "shard_manifest.json"), "r") as f:
    shard_manifest = json.load(f)

def get_source_name(shard_entry):
    source = shard_entry.get("source", "")
    parts = source.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("data_"):
            return p
    return "other"

all_shards_raw = shard_manifest["shards"]
all_shards, missing_shards, wrong_size_shards = filter_present_shards(
    all_shards_raw, SHARD_DIR, SHARD_DTYPE.itemsize
)
n_skipped = len(missing_shards) + len(wrong_size_shards)
if n_skipped:
    print(f"\nFiltering manifest against {SHARD_DIR}:")
    print(f"  kept {len(all_shards)}/{len(all_shards_raw)} "
          f"({len(missing_shards)} missing, {len(wrong_size_shards)} wrong-size)")
    skipped_by_source = defaultdict(lambda: {"missing": 0, "wrong_size": 0})
    for s in missing_shards:
        skipped_by_source[get_source_name(s)]["missing"] += 1
    for s in wrong_size_shards:
        skipped_by_source[get_source_name(s)]["wrong_size"] += 1
    for src, counts in sorted(skipped_by_source.items()):
        bits = []
        if counts["missing"]:
            bits.append(f"{counts['missing']} missing")
        if counts["wrong_size"]:
            bits.append(f"{counts['wrong_size']} wrong-size")
        print(f"    {src}: {', '.join(bits)}")
if not all_shards:
    raise FileNotFoundError(
        f"No usable shards in {SHARD_DIR}: all {len(all_shards_raw)} "
        f"manifest entries are missing or wrong-size on disk."
    )

shards_by_source = defaultdict(list)
for s in all_shards:
    shards_by_source[get_source_name(s)].append(s)

print(f"\nAvailable data sources:")
for src, shards in sorted(shards_by_source.items()):
    src_tokens = sum(s["tokens"] for s in shards)
    print(f"  {src}: {len(shards)} shards, {src_tokens:,} tokens")

eval_rng = random.Random(EVAL_SEED)
eval_shards = []
train_pool_by_source = {}
for src, shards in shards_by_source.items():
    pool = shards.copy()
    eval_rng.shuffle(pool)
    n_eval = max(1, int(len(pool) * EVAL_FRACTION_PER_SOURCE)) if len(pool) > 1 else 0
    eval_shards.extend(pool[:n_eval])
    train_pool_by_source[src] = pool[n_eval:]
print(f"\nHeld out {len(eval_shards)} eval shards "
      f"({sum(s['tokens'] for s in eval_shards):,} tokens)")

random.seed(42)
selected_shards = []
if DATA_MIX is None:
    pool = [s for ss in train_pool_by_source.values() for s in ss]
    random.shuffle(pool)
    budget = MAX_TOKENS if MAX_TOKENS else float('inf')
    running = 0
    for s in pool:
        if running >= budget:
            break
        selected_shards.append(s)
        running += s["tokens"]
else:
    budget = MAX_TOKENS if MAX_TOKENS else sum(s["tokens"] for ss in train_pool_by_source.values() for s in ss)
    for source, weight in DATA_MIX.items():
        source_budget = int(budget * weight)
        available = train_pool_by_source.get(source, [])
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

print(f"\nSelected for training:")
selected_by_source = defaultdict(lambda: {"shards": 0, "tokens": 0})
for s in selected_shards:
    src = get_source_name(s)
    selected_by_source[src]["shards"] += 1
    selected_by_source[src]["tokens"] += s["tokens"]
for src, info in sorted(selected_by_source.items()):
    total_available = sum(s["tokens"] for s in shards_by_source.get(src, []))
    pct = info["tokens"] / total_available * 100 if total_available else 0
    print(f"  {src}: {info['shards']} shards, {info['tokens']:,} tokens ({pct:.0f}% of available)")
print(f"\n  TOTAL: {len(selected_shards)} shards, {selected_tokens:,} tokens")


# ==================== 7b. STAGE SHARDS TO LOCAL DISK ====================
# Drive FUSE reads at training time are 5-50 MB/s with random stalls. Copy
# selected + eval shards to local SSD once; train from local for the run.
# Override: STAGE_DIR=<path> or SKIP_STAGE=1.
def _stage_shards_locally(src_dir, selected, evals):
    if os.environ.get("SKIP_STAGE") == "1":
        return src_dir
    stage_dir = os.environ.get("STAGE_DIR") or ("/content/shards" if in_colab() else "")
    if not stage_dir or os.path.abspath(stage_dir) == os.path.abspath(src_dir):
        return src_dir
    os.makedirs(stage_dir, exist_ok=True)
    for fname in ("shard_manifest.json", "meta.json", "tokenization_id.txt"):
        s = os.path.join(src_dir, fname)
        d = os.path.join(stage_dir, fname)
        if os.path.exists(s) and (not os.path.exists(d) or os.path.getsize(s) != os.path.getsize(d)):
            shutil.copy2(s, d)
    to_copy = sorted({s["shard"] for s in selected} | {s["shard"] for s in evals})
    print(f"\nStaging {len(to_copy)} shards: {src_dir} -> {stage_dir}")
    t0 = time.time()
    total = 0
    for i, name in enumerate(to_copy, 1):
        s = os.path.join(src_dir, name)
        d = os.path.join(stage_dir, name)
        s_size = os.path.getsize(s)
        if not (os.path.exists(d) and os.path.getsize(d) == s_size):
            shutil.copy2(s, d)
            if os.path.getsize(d) != s_size:
                raise RuntimeError(f"staged size mismatch for {name}")
        total += s_size
        if i % 10 == 0 or i == len(to_copy):
            elapsed = max(time.time() - t0, 0.01)
            print(f"  {i}/{len(to_copy)} | {total/1024**3:.2f} GB | "
                  f"{(total/1024**2)/elapsed:.0f} MB/s")
    print(f"Staging done in {time.time()-t0:.0f}s.")
    return stage_dir

SHARD_DIR = _stage_shards_locally(SHARD_DIR, selected_shards, eval_shards)


# ==================== 8. COMPUTE TOTAL STEPS ====================
samples_approx = selected_tokens // BLOCK_SIZE
batches_total = samples_approx // BATCH_SIZE
total_steps = (batches_total * EPOCHS) // GRAD_ACCUM_STEPS

print(f"  Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS} sequences "
      f"({BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE:,} tokens/step)")
print(f"  Total optimizer steps: {total_steps:,} | Warmup: {WARMUP_STEPS}")
print(f"  Mid-epoch save every {SAVE_EVERY_N_SHARDS} shards | Eval every {EVAL_EVERY_STEPS} steps")


# ==================== 9. EVAL HELPER ====================
@torch.no_grad()
def run_eval(model, eval_shards, n_batches):
    model.eval()
    losses = []
    rng = random.Random(EVAL_SEED)
    pool = eval_shards.copy()
    rng.shuffle(pool)
    seen = 0
    for shard_info in pool:
        shard_path = os.path.join(SHARD_DIR, shard_info["shard"])
        try:
            tokens = np.fromfile(shard_path, dtype=SHARD_DTYPE).astype(np.int64)
        except FileNotFoundError:
            continue
        ds = ShardDataset(tokens, BLOCK_SIZE, STRIDE)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            losses.append(loss.item())
            seen += 1
            if seen >= n_batches:
                break
        del tokens, ds, loader
        if seen >= n_batches:
            break
    model.train()
    if not losses:
        return float('nan')
    return sum(losses) / len(losses)


# ==================== 10. TRAINING LOOP ====================
def get_lr(it, total_it):
    if it < WARMUP_STEPS:
        return MAX_LR * it / WARMUP_STEPS
    decay_ratio = (it - WARMUP_STEPS) / max(1, (total_it - WARMUP_STEPS))
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

curr_step = 0
final_loss = None
last_grad_norm = None
last_eval_loss = None
eval_history = []
train_start = time.time()

print("\nSTARTING TRAINING (BFloat16, GQA, RoPE, RMSNorm, grad-ckpt)\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_shards = selected_shards.copy()
    random.shuffle(epoch_shards)

    for shard_idx, shard_info in enumerate(epoch_shards):
        shard_path = os.path.join(SHARD_DIR, shard_info["shard"])
        shard_name = shard_info["shard"]
        shard_source = get_source_name(shard_info)

        tokens = np.fromfile(shard_path, dtype=SHARD_DTYPE).astype(np.int64)
        dataset = ShardDataset(tokens, BLOCK_SIZE, STRIDE)
        if len(dataset) == 0:
            continue

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)
        pbar = tqdm(loader,
                    desc=f"Ep {epoch} [{shard_idx+1}/{len(epoch_shards)}] {shard_source}/{shard_name}",
                    dynamic_ncols=True, mininterval=1.0)

        for batch_idx, (xb, yb) in enumerate(pbar):
            lr = get_lr(curr_step, total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                last_grad_norm = float(grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                curr_step += 1
                final_loss = loss.item() * GRAD_ACCUM_STEPS
                elapsed = time.time() - train_start

                pbar.set_postfix(loss=f"{final_loss:.3f}",
                                 gnorm=f"{last_grad_norm:.2f}",
                                 lr=f"{lr:.2e}",
                                 step=curr_step,
                                 hrs=f"{elapsed/3600:.1f}",
                                 eval=("-" if last_eval_loss is None else f"{last_eval_loss:.3f}"))

                if curr_step % EVAL_EVERY_STEPS == 0:
                    last_eval_loss = run_eval(model, eval_shards, EVAL_BATCHES)
                    eval_history.append({"step": curr_step, "loss": last_eval_loss,
                                         "elapsed_hrs": round(elapsed/3600, 2)})
                    print(f"  [eval @ step {curr_step}] loss={last_eval_loss:.4f} "
                          f"(train={final_loss:.4f}, grad_norm={last_grad_norm:.2f})")

        del tokens, dataset, loader

        if (shard_idx + 1) % SAVE_EVERY_N_SHARDS == 0:
            elapsed = time.time() - train_start
            print(f"  Mid-epoch save at shard {shard_idx+1}/{len(epoch_shards)} "
                  f"(step {curr_step}, loss {final_loss:.3f}, {elapsed/3600:.1f}h elapsed)...")
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Saving Epoch {epoch}...")
    torch.save(model.state_dict(), CHECKPOINT_PATH)


# ==================== 11. FINALIZE ====================
total_time = time.time() - train_start
print(f"Training Complete. {total_time/3600:.1f} hours, {curr_step} steps.")
torch.save(model.state_dict(), CHECKPOINT_PATH)


# ==================== 12. SAVE MANIFEST ====================
manifest = {
    "stage": "training",
    "created": datetime.datetime.now().isoformat(),
    "checkpoint": file_info(CHECKPOINT_PATH),
    "tokenization_id": current_tok_id,
    "config": {
        "arch": "synapse_gpt_2b_d2560_l28",
        "block_size": BLOCK_SIZE, "stride": STRIDE,
        "embed_dim": EMBED_DIM, "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV_HEADS,
        "ff_hidden_dim": FF_HIDDEN_DIM,
        "rope_base": ROPE_BASE, "rmsnorm_eps": RMSNORM_EPS,
        "gradient_checkpointing": GRAD_CHECKPOINT,
        "vocab_size_raw": VOCAB_SIZE_RAW, "vocab_size_padded": VOCAB_SIZE,
        "shard_dtype": str(SHARD_DTYPE),
        "n_params": n_params, "n_trainable": n_trainable,
        "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": EPOCHS, "max_lr": MAX_LR, "min_lr": MIN_LR,
        "warmup_steps": WARMUP_STEPS, "weight_decay": WEIGHT_DECAY,
        "betas": list(BETAS), "grad_clip": GRAD_CLIP,
        "precision": "bfloat16", "optimizer": "AdamW (fused)",
        "lr_schedule": "cosine_with_warmup",
    },
    "data_selection": {
        "max_tokens": MAX_TOKENS,
        "data_mix": DATA_MIX,
        "selected_shards": len(selected_shards),
        "selected_tokens": selected_tokens,
        "sources": dict(selected_by_source),
        "eval_shards": len(eval_shards),
        "eval_tokens": sum(s["tokens"] for s in eval_shards),
    },
    "results": {
        "final_loss": final_loss,
        "final_eval_loss": last_eval_loss,
        "eval_history": eval_history,
        "last_grad_norm": last_grad_norm,
        "total_optimizer_steps": curr_step,
        "total_hours": round(total_time / 3600, 2),
        "resumed_from": resume_path,
    },
}
manifest_path = os.path.join(MANIFEST_DIR, "training_latest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest saved: {manifest_path}")
print(f"  tokenization_id: {current_tok_id}")
print(f"  final_loss: {final_loss}")
print(f"  final_eval_loss: {last_eval_loss}")
print(f"  total_steps: {curr_step}")
