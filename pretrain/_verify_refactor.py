#!/usr/bin/env python3
"""Verify that synapse_model.model.SynapseGPT is bit-for-bit equivalent to the
pre-refactor pretrain/_baseline_model.SynapseGPT.

Loads the same v2 checkpoint into both, runs a fixed deterministic batch
through each under bf16 autocast, and compares:
  1. state_dict key sets (must be exactly equal)
  2. cross-entropy losses (abs diff <= 1e-4; bf16 matmul order makes
     bit-exactness impossible)
  3. weight tensors of 5 sampled parameters (must be torch.allclose)

Run before merging branch sft-phase-1 to main. Exits 0 on pass, 1 on fail.

Usage (any env with the v2 checkpoint on disk + a CUDA GPU):
  SYNAPSE_DIR=/content/drive/MyDrive/synapse \
      python pretrain/_verify_refactor.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_REPO_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F

from _baseline_model import SynapseGPT as BaselineGPT
from synapse_model.model import SynapseGPT as NewGPT


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


SYNAPSE_DIR = os.environ.get("SYNAPSE_DIR") or default_synapse_dir()
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "synapse_2b_d2560_l28.pth")
CHECKPOINT_PATH = os.path.join(SYNAPSE_DIR, "checkpoints", CHECKPOINT_NAME)

LOSS_TOLERANCE = 1e-4
SEED = 1337
BATCH_SIZE = 2
SEQ_LEN = 128   # short — we only need a forward pass, not real training


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"FAIL: checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)
    print(f"loading checkpoint: {CHECKPOINT_PATH}")
    obj = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(obj, dict) and obj.get("schema") == "v2":
        state = obj["model"]
    else:
        state = obj
    # Strip torch.compile prefix if present.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    return state


def vocab_size_from_state(state):
    emb_w = state["token_embedding.weight"]
    return emb_w.shape[0]


def fixed_batch(vocab_size, device):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    return torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), generator=g).to(device)


def forward_loss(model, x):
    # Same loss form as training: shift-by-one cross-entropy on the full sequence.
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
    return float(loss.item())


def build_loaded(cls, vocab_size, state, device, **kwargs):
    torch.manual_seed(SEED)  # for any random init done in __init__
    model = cls(vocab_size, **kwargs).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, set(missing), set(unexpected)


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available — verification needs a GPU")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"loss tolerance: {LOSS_TOLERANCE}")
    print(f"batch: {BATCH_SIZE}x{SEQ_LEN}, seed={SEED}\n")

    state = load_checkpoint()
    vocab_size = vocab_size_from_state(state)
    print(f"vocab_size (from checkpoint embedding): {vocab_size}\n")

    print("[1/3] building baseline model (pre-refactor classes)")
    baseline, b_missing, b_unexpected = build_loaded(BaselineGPT, vocab_size, state, device)
    print(f"      missing keys: {len(b_missing)}, unexpected keys: {len(b_unexpected)}")

    print("[2/3] building new model (synapse_model.model)")
    new, n_missing, n_unexpected = build_loaded(
        NewGPT, vocab_size, state, device, gradient_checkpointing=False,
    )
    print(f"      missing keys: {len(n_missing)}, unexpected keys: {len(n_unexpected)}")

    failures = []

    # Check 1: state_dict key sets are identical
    b_keys = set(baseline.state_dict().keys())
    n_keys = set(new.state_dict().keys())
    if b_keys != n_keys:
        only_b = b_keys - n_keys
        only_n = n_keys - b_keys
        failures.append(
            f"state_dict key mismatch:\n"
            f"  only in baseline: {sorted(only_b)[:10]}\n"
            f"  only in new:      {sorted(only_n)[:10]}"
        )
    else:
        print(f"\n[check 1/3] state_dict keys identical ({len(b_keys)} keys) — PASS")

    # Check 2: forward losses match within tolerance
    print(f"\n[check 2/3] running forward pass on both models...")
    x = fixed_batch(vocab_size, device)
    baseline.eval()
    new.eval()
    with torch.no_grad():
        l_baseline = forward_loss(baseline, x)
        l_new = forward_loss(new, x)
    diff = abs(l_baseline - l_new)
    print(f"  baseline loss: {l_baseline:.8f}")
    print(f"  new loss:      {l_new:.8f}")
    print(f"  abs diff:      {diff:.2e}  (tolerance {LOSS_TOLERANCE:.0e})")
    if diff > LOSS_TOLERANCE:
        failures.append(f"loss diff {diff:.6e} exceeds tolerance {LOSS_TOLERANCE:.0e}")
    else:
        print(f"  PASS")

    # Check 3: sampled parameter tensors match exactly (loaded from same checkpoint)
    print(f"\n[check 3/3] comparing 5 random parameter tensors")
    rng = torch.Generator().manual_seed(SEED)
    shared = sorted(b_keys & n_keys)
    sample_idx = torch.randperm(len(shared), generator=rng)[:5].tolist()
    b_sd, n_sd = baseline.state_dict(), new.state_dict()
    for i in sample_idx:
        k = shared[i]
        ok = torch.allclose(b_sd[k], n_sd[k])
        marker = "OK" if ok else "MISMATCH"
        print(f"  {marker:8s} {k}  shape={tuple(b_sd[k].shape)}")
        if not ok:
            failures.append(f"parameter tensor mismatch: {k}")

    print()
    if failures:
        print("=" * 60)
        print("VERIFICATION FAILED")
        print("=" * 60)
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("=" * 60)
    print("VERIFICATION PASSED")
    print("=" * 60)
    print("Safe to merge branch (after pretrain completes).")
    sys.exit(0)


if __name__ == "__main__":
    main()
