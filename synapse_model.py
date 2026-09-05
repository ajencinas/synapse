#!/usr/bin/env python3
"""Shared SynapseGPT model definition + architecture constants.

Single source of truth for the model architecture, imported by both
`pretrain/train.py` and `sft/sft.py`. The architecture here MUST stay
byte-identical across pretrain and SFT — a checkpoint trained with one shape
cannot be loaded into another, and a silent mismatch loads garbage weights.

Contains:
  - Architecture constants (the model config; NOT training hyperparameters,
    which stay per-script: LR, batch size, epochs, weight decay, ...).
  - Environment helpers (Colab detection, Drive mount, default dir).
  - Model: RMSNorm, RoPE, CausalGQA, SwiGLU, TransformerBlock, SynapseGPT.

Extracted verbatim from pretrain/train.py §1, §3 (architecture subset), and §4.
Note: `get_lr` is intentionally NOT here — its cosine constants (MAX_LR, MIN_LR,
WARMUP_STEPS, horizon) differ between pretrain and SFT, so each script keeps its
own copy to avoid reading the wrong module globals.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ==================== ENVIRONMENT HELPERS ====================

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


# ==================== ARCHITECTURE CONSTANTS (Shape D, ~2.1B params) ====================
# These define the model and must match the pretrain checkpoint exactly.
BLOCK_SIZE      = 2048
EMBED_DIM       = 2560
NUM_LAYERS      = 28
NUM_HEADS       = 20            # head_dim = 128
NUM_KV_HEADS    = 4            # GQA, group size = 5
FF_HIDDEN_DIM   = 6912          # ~8/3 * EMBED_DIM, multiple of 128
ROPE_BASE       = 10000.0
RMSNORM_EPS     = 1e-5
GRAD_CHECKPOINT = False         # default off; a trainer may flip model.gradient_checkpointing


# ==================== MODEL DEFINITION ====================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    # Graph-break here: Inductor was fusing RMSNorm fwd+bwd into one Triton
    # kernel that wanted >160 KB of shared memory and couldn't compile.
    # Running RMSNorm eager keeps the rest of the model compiled.
    @torch._dynamo.disable
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
