#!/usr/bin/env python3
"""Diagnose checkpoint files — fast, no GPU needed, just metadata."""
import os
import sys
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "sparky_data")
TOKENIZER_PATH = os.path.join(DATA, "tokenizer.json")

# Collect all .pth files
checkpoints = []
for root, dirs, files in os.walk(os.path.join(REPO, "..")):
    if ".git" in root.split(os.sep):
        continue
    for f in files:
        if f.endswith(".pth") and os.path.getsize(os.path.join(root, f)) > 1e8:
            checkpoints.append(os.path.join(root, f))

# ── 1. Scan all checkpoints ─────────────────────────────────────────
print("=" * 70)
print("CHECKPOINT SCAN")
print("=" * 70)
print(f"{'File':50s} {'Size':>8s}  {'Schema':>12s}  {'Step':>8s}  {'SFT?':>5s}")
print("-" * 70)

sft_found = []
for path in sorted(checkpoints, key=lambda p: os.path.getsize(p)):
    size = os.path.getsize(path) / 1e9
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        schema = ckpt.get("schema", "?")
        step = ckpt.get("step") or ckpt.get("curr_step", "?")
        # Canonical SFT marker (matches sft/sft.py + sparky_model.from_checkpoint):
        # schema='v2' + stage='sft'. Keep old fallbacks for legacy files.
        has_sft = (ckpt.get("stage") == "sft"
                   or schema == "sft_v1" or "sft_config" in ckpt)
        tag = "SFT" if has_sft else schema
        if has_sft:
            sft_found.append(path)
    else:
        schema, step, has_sft, tag = "legacy", "?", False, "legacy"
    fname = os.path.basename(path)
    short = f"{os.path.basename(os.path.dirname(path))}/{fname}"
    print(f"{short:50s} {size:>6.2f}GB  {tag:>12s}  {str(step):>8s}  {'YES' if has_sft else 'no':>5s}")
    if isinstance(ckpt, dict) and has_sft:
        mix = ckpt.get("data_mix") or (ckpt.get("sft_config") or {}).get("dataset_mix")
        print(f"    {'':50s} epoch={ckpt.get('epoch')} data_mix={mix}")

# ── 2. Tokenizer check ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOKENIZER CHECK")
print("=" * 70)
from tokenizers import Tokenizer
if not os.path.exists(TOKENIZER_PATH):
    print(f"  ERROR: tokenizer not found at {TOKENIZER_PATH}")
    sys.exit(1)
tok = Tokenizer.from_file(TOKENIZER_PATH)
vocab = tok.get_vocab_size()
pad_id = tok.token_to_id("<|pad|>")
eot_id = tok.token_to_id("<|endoftext|>")
im_start = tok.token_to_id("<|im_start|>")
im_end = tok.token_to_id("<|im_end|>")
print(f"  vocab: {vocab}")
print(f"  pad_id (<|pad|>):           {pad_id}")
print(f"  eot_id (<|endoftext|>):     {eot_id}")
print(f"  im_start (<|im_start|>):    {im_start}")
print(f"  im_end (<|im_end|>):        {im_end}")

# Quick encode test — SFT template uses reserved role tokens (ids 9/10/11) and
# <|im_end|> separators, matching sft/tokenize_sft_data.py. NOT <|im_start|>.
test = "Hello, how are you?"
ids = tok.encode(test).ids
chatml = f"<|reserved_1|>{test}<|im_end|><|reserved_2|>"
chatml_ids = tok.encode(chatml, add_special_tokens=False).ids
print(f"\n  Raw prompt:    '{test}' -> {len(ids)} tokens")
print(f"  SFT prompt:    '{chatml}' -> {len(chatml_ids)} tokens")
print(f"  SFT prompt ids: {chatml_ids}  (10=user, 3=<|im_end|>, 11=assistant)")

# ── 3. Conclusion ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if sft_found:
    print(f"  ✅ SFT checkpoint FOUND:")
    for p in sft_found:
        sz = os.path.getsize(p) / 1e9
        print(f"     {p} ({sz:.2f}GB)")
    print("\n  sparky is pointing to:")
    drive = "synapse/sft_checkpoints/v3_15source/sft_best.pth"
    print(f"     Drive path: {drive}")
    print(f"     Local:      sparky/sparky_data/sft_latest.pth")
    print()
    for p in sft_found:
        local_name = os.path.basename(p)
        if "sparky_data" in p:
            print(f"  ✅ Already in sparky_data as {local_name}")
        else:
            print(f"  ❌ SFT checkpoint is NOT in sparky_data")
            print(f"     Copy it: cp {p} sparky/sparky_data/")
else:
    print("  ❌ No checkpoint with stage=='sft' found on this machine.")
    print()
    print("  An SFT checkpoint from sft/sft.py has schema='v2' AND stage='sft'")
    print("  (plus data_mix); a pretrain checkpoint has schema='v2' with NO stage.")
    print("  If you only see pretrain files, then sft.py either:")
    print("    1. Hasn't finished an epoch yet (it writes sft_epochN.pth + sft_latest.pth)")
    print("    2. Saved under $SYNAPSE_DIR/sft_checkpoints/ (full ckpt may be local")
    print("       in /content/sft_work until the background Drive mirror completes)")
    print("    3. The Drive copy hasn't synced yet")