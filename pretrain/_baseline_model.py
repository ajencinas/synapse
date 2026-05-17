"""Frozen pre-refactor snapshot of the SynapseGPT model classes.

DO NOT MODIFY. Used only by pretrain/_verify_refactor.py to confirm that
the synapse/model.py extraction preserves loss bit-for-bit. Deleted from
the branch once verification passes and before the branch merges to main.

Copied verbatim from pretrain/train.py @ 7d729e1 (lines 183-192, 256-371).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Architecture constants (verbatim from pretrain/train.py)
BLOCK_SIZE      = 2048
STRIDE          = BLOCK_SIZE
EMBED_DIM       = 2560
NUM_LAYERS      = 28
NUM_HEADS       = 20            # head_dim = 128
NUM_KV_HEADS    = 4             # GQA, group size = 5
FF_HIDDEN_DIM   = 6912          # ~8/3 * EMBED_DIM, multiple of 128
ROPE_BASE       = 10000.0
RMSNORM_EPS     = 1e-5
GRAD_CHECKPOINT = False         # 95 GB Blackwell has VRAM headroom; recompute is wasted work


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
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
