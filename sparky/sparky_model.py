"""
SynapseGPT inference model with KV-cache generation.
Architecture matches pretrain/train.py exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import inspect

# -- Config matching pretrain/train.py Shape D, ~2.1B --
BLOCK_SIZE    = 2048
EMBED_DIM     = 2560
NUM_LAYERS    = 28
NUM_HEADS     = 20
NUM_KV_HEADS  = 4
FF_HIDDEN_DIM = 6912
ROPE_BASE     = 10000.0
RMSNORM_EPS   = 1e-5
VOCAB_SIZE    = 64000


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=RMSNORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(in_dtype) * self.weight


def precompute_rope(head_dim, max_seq_len, base, device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def apply_rope(x, cos, sin, pos_offset=0):
    T = x.size(-2)
    cos = cos[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


class CausalGQA(nn.Module):
    """Grouped-query attention with KV-cache support for inference."""
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.n_rep = num_heads // num_kv_heads
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, pos_offset=0):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, pos_offset)
        k = apply_rope(k, cos, sin, pos_offset)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # dim=2 is sequence length
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v)

        if self.n_rep > 1:
            k_expanded = k.repeat_interleave(self.n_rep, dim=1)
            v_expanded = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        use_causal = (kv_cache is None)  # causal only during prefill
        y = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=use_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), new_cache


class SwiGLU(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=FF_HIDDEN_DIM):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(EMBED_DIM)
        self.attn = CausalGQA()
        self.norm2 = RMSNorm(EMBED_DIM)
        self.ff = SwiGLU()

    def forward(self, x, cos, sin, kv_cache=None, pos_offset=0):
        attn_out, new_cache = self.attn(self.norm1(x), cos, sin, kv_cache, pos_offset)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_cache


class SynapseInfer(nn.Module):
    """Inference model — loads checkpoint from pretrain/train.py training runs."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.final_norm = RMSNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.token_embedding.weight = self.lm_head.weight  # tied

        head_dim = EMBED_DIM // NUM_HEADS
        cos, sin = precompute_rope(head_dim, BLOCK_SIZE, ROPE_BASE, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, kv_cache=None, pos_offset=0):
        """idx: (B, T). kv_cache: list of (k, v) tuples per layer or None."""
        x = self.token_embedding(idx)
        cos, sin = self.rope_cos, self.rope_sin
        new_cache = []
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_cache):
            x, kv = block(x, cos, sin, cache, pos_offset)
            new_cache.append(kv)
        logits = self.lm_head(self.final_norm(x))
        return logits, new_cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.9,
                 repetition_penalty=1.0, eot_id=0, stop_tokens=None):
        """Token streaming with KV cache. Yields int tokens one at a time."""
        if stop_tokens is None:
            stop_tokens = {eot_id}

        kv_cache = None
        pos_offset = 0

        # Prefill: process the full prompt
        logits, kv_cache = self(idx, kv_cache, pos_offset)
        pos_offset = (pos_offset + idx.shape[1]) % BLOCK_SIZE

        generated_ids = set(idx[0].tolist())

        # Decode: one token at a time
        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0 and generated_ids:
                tokens_tensor = torch.tensor(list(generated_ids), device=logits_last.device)
                l = logits_last[0, tokens_tensor]
                logits_last[0, tokens_tensor] = torch.where(l < 0, l * repetition_penalty, l / repetition_penalty)

            if top_k > 0:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_last, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits_last[mask] = float('-inf')

            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token in stop_tokens:
                break

            yield next_token

            generated_ids.add(next_token)
            idx_next = torch.tensor([[next_token]], device=idx.device, dtype=idx.dtype)
            logits, kv_cache = self(idx_next, kv_cache, pos_offset)
            pos_offset = (pos_offset + 1) % BLOCK_SIZE

    @classmethod
    def from_checkpoint(cls, ckpt_path, device="cuda", no_compile=False):
        """Load a v2 (pretrain) or sft_v1 (fine-tuned) checkpoint."""
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        schema = ckpt.get("schema") if isinstance(ckpt, dict) else None
        stage = ckpt.get("stage") if isinstance(ckpt, dict) else None
        if schema in ("v2", "sft_v1"):
            state_dict = ckpt["model"]
            is_sft = stage == "sft" or schema == "sft_v1" or "sft_config" in ckpt
            info = {
                "step": ckpt.get("curr_step") or ckpt.get("step"),
                "eval_history": ckpt.get("eval_history"),
                "is_sft": is_sft,
            }
        else:
            state_dict = ckpt
            info = {}

        # Remove _orig_mod. prefix if present (from torch.compile)
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("_orig_mod.", "")] = v

        load_kwargs = {"strict": False}
        supports_assign = "assign" in inspect.signature(nn.Module.load_state_dict).parameters
        if supports_assign:
            load_kwargs["assign"] = True
            with torch.device("meta"):
                model = cls()
        else:
            model = cls()

        # H5: a key mismatch means the checkpoint is not this architecture (or was
        # saved with a different naming) — every weight it fails to load would be
        # left at random init and every downstream number would be garbage. Fail
        # loud. The only tolerated absence is one side of the tied embedding /
        # lm_head pair (some savers store the shared tensor under a single name).
        missing, unexpected = model.load_state_dict(cleaned, **load_kwargs)
        tied = {"token_embedding.weight", "lm_head.weight"}
        missing = [k for k in missing if not (k in tied and (tied - {k}) & set(cleaned))]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint {ckpt_path} does not match SynapseInfer: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]} "
                f"(total {len(missing)} missing / {len(unexpected)} unexpected)")

        model.lm_head.weight = model.token_embedding.weight
        del ckpt, state_dict, cleaned
        gc.collect()
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        # Compile for faster decode (optional — comment out if it causes issues)
        if not no_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("[model] torch.compile succeeded")
            except Exception as e:
                print(f"[model] torch.compile skipped ({e})")
        return model, info
