#!/usr/bin/env python3
"""
Standalone tokenizer pipeline — runs on local SSD or any fast filesystem.
Mirrors the logic of tokenizer_pipeline.ipynb without Colab/Drive dependencies.

Usage:
  python run_tokenizer.py                    # load existing tokenizer, tokenize new/changed only
  python run_tokenizer.py --train            # train new tokenizer + tokenize all
  python run_tokenizer.py --config path      # use custom config path

Paths (configured via env vars or defaults below):
  DATA_PATH         — source data (data_*/ dirs)
  TOKENIZER_DIR     — where tokenizer.json is written/read
  SHARD_DIR         — uint16 shard output
  MANIFEST_DIR      — run manifest output
  CONFIG_PATH       — tokenizer_config.json
"""

import os
import sys
import json
import random
import time
import hashlib
import argparse
import datetime
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence as PreSequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------- Paths (override via env vars) ----------
DATA_PATH         = os.environ.get("TOKENIZER_DATA_PATH",     "/mnt/ssd/datasets_pretrain")
TOKENIZER_DIR     = os.environ.get("TOKENIZER_OUT_DIR",      "/mnt/ssd/tokenizer_out")
SHARD_DIR         = os.environ.get("TOKENIZER_SHARD_DIR",    "/mnt/ssd/token_shards")
MANIFEST_DIR      = os.environ.get("TOKENIZER_MANIFEST_DIR", "/mnt/ssd/manifests")
CONFIG_PATH       = os.environ.get("TOKENIZER_CONFIG_PATH",  "tokenizer_config.json")

# ---------- Config ----------
with open(CONFIG_PATH) as f:
    config = json.load(f)

VOCAB_SIZE            = config["vocab_size"]
MIN_FREQUENCY         = config["min_frequency"]
BPE_DROPOUT           = config["bpe_dropout"]
TRAIN_SIZE_BYTES      = config["train_subset_bytes"]
ENCODE_BATCH_SIZE     = config["encode_batch_size"]
READ_CHUNK_CHARACTERS = config["read_chunk_characters"]
SHARD_DTYPE           = np.dtype(config["shard_dtype"]).type
NUM_SPECIAL           = config["num_special_tokens"]
NAMED_SPECIAL         = config["named_special_tokens"]
EOT_TOKEN             = config["eot_token"]
PAD_TOKEN             = config["pad_token"]
MAX_EVAL_BYTES        = config["eval"]["max_eval_bytes"]
EVAL_THRESHOLDS       = config["eval"]["thresholds"]

RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(NUM_SPECIAL - len(NAMED_SPECIAL))]
SPECIAL_TOKENS = NAMED_SPECIAL + RESERVED_TOKENS
assert len(SPECIAL_TOKENS) == NUM_SPECIAL, (
    f"Expected {NUM_SPECIAL} special tokens, got {len(SPECIAL_TOKENS)}"
)

TOKENIZER_PATH      = os.path.join(TOKENIZER_DIR, "tokenizer.json")
SHARD_MANIFEST_PATH = os.path.join(SHARD_DIR, "shard_manifest.json")
META_PATH           = os.path.join(SHARD_DIR, "meta.json")
TOK_ID_PATH         = os.path.join(SHARD_DIR, "tokenization_id.txt")
EVAL_REPORT_PATH    = os.path.join(TOKENIZER_DIR, "tokenizer_eval.json")

# ---------- Helpers ----------
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def atomic_dump_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)

def save_meta(path, shard_manifest, vocab_size, eot_id, pad_id, tok_id):
    total_tokens = sum(s["tokens"] for s in shard_manifest.get("shards", []))
    total_shards = len(shard_manifest.get("shards", []))
    meta = {
        "num_tokens": total_tokens,
        "num_shards": total_shards,
        "vocab_size": vocab_size,
        "eot_id": eot_id,
        "pad_id": pad_id,
        "tokenizer": "byte-level-bpe-digits",
        "shard_dtype": np.dtype(SHARD_DTYPE).name,
        "tokenization_id": tok_id,
    }
    atomic_dump_json(path, meta)

def group_by_dir(files):
    d = {}
    for f in files:
        key = os.path.basename(os.path.dirname(f))
        d.setdefault(key, []).append(f)
    return d

def build_stratified_sample(files_by_dir, total_budget, out_path, seed):
    per_dir_budget = total_budget // max(1, len(files_by_dir))
    rng = random.Random(seed)
    total_bytes = 0
    eot_stripped = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out_f:
        for dirname, files in files_by_dir.items():
            sampled = files.copy()
            rng.shuffle(sampled)
            dir_bytes = 0
            for src in sampled:
                if dir_bytes >= per_dir_budget:
                    break
                with open(src, "r", encoding="utf-8", errors="ignore") as in_f:
                    for line in in_f:
                        if EOT_TOKEN in line:
                            eot_stripped += line.count(EOT_TOKEN)
                            line = line.replace(EOT_TOKEN, "\n")
                        out_f.write(line)
                        dir_bytes += len(line.encode("utf-8"))
                        if dir_bytes >= per_dir_budget:
                            break
            print(f"    {dirname}: {dir_bytes / 1024 / 1024:.0f} MB")
            total_bytes += dir_bytes
    print(f"  Sample: {total_bytes / 1024 / 1024 / 1024:.2f} GB in {time.time() - t0:.1f}s "
          f"(stripped {eot_stripped:,} {EOT_TOKEN!r} separators)")
    return total_bytes

def evaluate_tokenizer(tokenizer, sample_path, thresholds, special_ids):
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    encoded = tokenizer.encode(text)
    n_tokens = max(1, len(encoded.ids))
    chars_per_token = n_chars / n_tokens
    bytes_per_token = n_bytes / n_tokens
    id_counts = Counter(encoded.ids)
    non_special = [(tid, c) for tid, c in id_counts.items() if tid not in special_ids]
    non_special.sort(key=lambda x: -x[1])
    top_freq = non_special[0][1] / n_tokens if non_special else 0.0
    top_tokens = [(tokenizer.id_to_token(tid), int(c)) for tid, c in non_special[:20]]
    report = {
        "eval_bytes": n_bytes, "eval_chars": n_chars,
        "total_tokens": n_tokens,
        "chars_per_token": round(chars_per_token, 3),
        "bytes_per_token": round(bytes_per_token, 3),
        "max_single_token_freq": round(top_freq, 4),
        "top_20_tokens": top_tokens, "thresholds": thresholds,
    }
    failures = []
    if chars_per_token < thresholds["chars_per_token_min"]:
        failures.append(f"chars/token={chars_per_token:.3f} < min {thresholds['chars_per_token_min']}")
    if chars_per_token > thresholds["chars_per_token_max"]:
        failures.append(f"chars/token={chars_per_token:.3f} > max {thresholds['chars_per_token_max']}")
    if bytes_per_token < thresholds["bytes_per_token_min"]:
        failures.append(f"bytes/token={bytes_per_token:.3f} < min {thresholds['bytes_per_token_min']}")
    if top_freq > thresholds["max_single_token_freq"]:
        failures.append(f"top token freq={top_freq:.4f} > max {thresholds['max_single_token_freq']}")
    report["passed"] = not failures
    report["failures"] = failures
    return report

# ---------- Worker (process-pool) state ----------
# Populated once per worker process by _worker_init; used by _tokenize_one.
_W_TOKENIZER = None
_W_EOT_ID = None
_W_BATCH = None
_W_CHUNK = None
_W_EOT_TOKEN = None
_W_DTYPE = None

def _worker_init(tokenizer_path, eot_token, batch_size, chunk_chars, shard_dtype_name):
    global _W_TOKENIZER, _W_EOT_ID, _W_BATCH, _W_CHUNK, _W_EOT_TOKEN, _W_DTYPE
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "4"
    _W_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    _W_EOT_ID = _W_TOKENIZER.token_to_id(eot_token)
    _W_EOT_TOKEN = eot_token
    _W_BATCH = batch_size
    _W_CHUNK = chunk_chars
    _W_DTYPE = np.dtype(shard_dtype_name).type

def _tokenize_one(item):
    """Tokenize one source file → write one shard. Returns manifest entry dict."""
    src_path, shard_path, src_size, prior_hash, prior_existing_hash = item

    # Hash check for size-mismatched files (deferred from main pre-flight).
    if prior_existing_hash is not None:
        h = hashlib.sha256()
        with open(src_path, "rb") as fh:
            while chunk := fh.read(8192):
                h.update(chunk)
        src_hash = h.hexdigest()[:16]
        if src_hash == prior_existing_hash:
            return {"_skipped": True, "source": src_path}
    elif prior_hash is not None:
        src_hash = prior_hash
    else:
        h = hashlib.sha256()
        with open(src_path, "rb") as fh:
            while chunk := fh.read(8192):
                h.update(chunk)
        src_hash = h.hexdigest()[:16]

    token_count = doc_count = 0
    docs_buffer = []
    leftover = ""
    with open(shard_path, "wb") as f_bin:
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f_txt:
            while True:
                chunk = f_txt.read(_W_CHUNK)
                if not chunk:
                    break
                buf = leftover + chunk
                pieces = buf.split(_W_EOT_TOKEN)
                leftover = pieces.pop()
                for doc in pieces:
                    doc = doc.strip()
                    if doc:
                        docs_buffer.append(doc)
                        doc_count += 1
                if len(docs_buffer) >= _W_BATCH:
                    encoded = _W_TOKENIZER.encode_batch(docs_buffer)
                    out = []
                    for enc in encoded:
                        out.extend(enc.ids)
                        out.append(_W_EOT_ID)
                    arr = np.array(out, dtype=_W_DTYPE)
                    f_bin.write(arr.tobytes())
                    token_count += len(out)
                    docs_buffer = []
        if leftover.strip():
            docs_buffer.append(leftover.strip())
            doc_count += 1
        if docs_buffer:
            encoded = _W_TOKENIZER.encode_batch(docs_buffer)
            out = []
            for enc in encoded:
                out.extend(enc.ids)
                out.append(_W_EOT_ID)
            arr = np.array(out, dtype=_W_DTYPE)
            f_bin.write(arr.tobytes())
            token_count += len(out)

    shard_mb = os.path.getsize(shard_path) / 1024 / 1024
    return {
        "_skipped": False,
        "shard": os.path.basename(shard_path),
        "source": src_path,
        "source_hash": src_hash,
        "source_size": src_size,
        "tokens": token_count,
        "documents": doc_count,
        "shard_mb": round(shard_mb, 2),
    }

# ---------- Args ----------
parser = argparse.ArgumentParser(description="Run the synapse tokenizer pipeline")
parser.add_argument("--train", action="store_true", help="Train a new tokenizer (default: load existing)")
parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="Parallel worker processes for step 3 tokenization (1 = serial)")
parser.add_argument("--no-merge", action="store_true",
                    help="Skip the post-tokenization shard merge step")
parser.add_argument("--merge-target-bytes", type=int, default=None,
                    help="Override config merge.target_bytes (e.g. 104857600 for 100 MB)")
args = parser.parse_args()

# ---------- Discover source files ----------
os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(SHARD_DIR, exist_ok=True)

all_source_files = []
train_files = []
eval_files = []
data_dirs = sorted([
    d for d in os.listdir(DATA_PATH)
    if d.startswith("data_") and os.path.isdir(os.path.join(DATA_PATH, d))
])
eval_seed = config["eval"]["seed"]
eval_fraction = config["eval"]["held_out_fraction"]
split_rng = random.Random(eval_seed)
for dirname in data_dirs:
    dirpath = os.path.join(DATA_PATH, dirname)
    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".txt"))
    full_paths = [os.path.join(dirpath, f) for f in files]
    all_source_files.extend(full_paths)
    dir_files = full_paths.copy()
    split_rng.shuffle(dir_files)
    n_eval = max(1, int(len(dir_files) * eval_fraction)) if len(dir_files) > 1 else 0
    eval_files.extend(dir_files[:n_eval])
    train_files.extend(dir_files[n_eval:])
    print(f"  {dirname}: {len(files)} files ({len(dir_files[n_eval:])} train, {n_eval} eval)")

total_size = sum(os.path.getsize(f) for f in all_source_files)
print(f"\nTotal: {len(all_source_files)} source files, {total_size / 1024 / 1024 / 1024:.1f} GB")
print(f"  Train pool: {len(train_files)} files | Eval pool: {len(eval_files)} files\n")

# ================= STEP 1: TRAIN OR LOAD TOKENIZER =================
if args.train:
    subset_path = os.path.join(TOKENIZER_DIR, "bpe_subset.txt")
    print(f"[1/3] Building training sample ({TRAIN_SIZE_BYTES / 1024 / 1024 / 1024:.1f} GB) from {len(train_files)} train files...")
    train_by_dir = group_by_dir(train_files)
    print(f"  {len(train_by_dir)} dirs, {(TRAIN_SIZE_BYTES // max(1, len(train_by_dir))) / 1024 / 1024:.0f} MB each")
    build_stratified_sample(train_by_dir, TRAIN_SIZE_BYTES, subset_path, seed=42)

    print(f"  Training Byte-Level BPE (vocab={VOCAB_SIZE}, min_freq={MIN_FREQUENCY}, digit-per-token)...")
    t0 = time.time()
    tokenizer = Tokenizer(BPE(dropout=BPE_DROPOUT))
    tokenizer.pre_tokenizer = PreSequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([subset_path], trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"  Done in {time.time() - t0:.1f}s -> {TOKENIZER_PATH}")
else:
    print(f"[1/3] Loading existing tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

vocab_size_actual = tokenizer.get_vocab_size()
eot_id = tokenizer.token_to_id(EOT_TOKEN)
pad_id = tokenizer.token_to_id(PAD_TOKEN)
assert eot_id is not None, "EOT token missing"
assert pad_id is not None, "PAD token missing"
assert vocab_size_actual <= np.iinfo(SHARD_DTYPE).max + 1, (
    f"Vocab size {vocab_size_actual} exceeds {np.dtype(SHARD_DTYPE).name} ID range "
    f"(max ID {np.iinfo(SHARD_DTYPE).max}). Upgrade shard_dtype to uint32 or lower vocab_size."
)
max_id = max(tokenizer.get_vocab().values())
assert max_id <= np.iinfo(SHARD_DTYPE).max, (
    f"Max token ID {max_id} exceeds {np.dtype(SHARD_DTYPE).name} max "
    f"({np.iinfo(SHARD_DTYPE).max}). Shards would silently overflow."
)
print(f"  Vocab: {vocab_size_actual} | max_id: {max_id} | EOT: {eot_id} | PAD: {pad_id}")

# ================= STEP 2: AUTO-EVAL GATE =================
print(f"\n[2/3] Auto-eval gate: encoding held-out sample...")
eval_report = None
if eval_files:
    eval_sample_path = os.path.join(TOKENIZER_DIR, "bpe_eval_sample.txt")
    eval_by_dir = group_by_dir(eval_files)
    build_stratified_sample(eval_by_dir, MAX_EVAL_BYTES, eval_sample_path, seed=config["eval"]["seed"])
    special_ids = {
        tokenizer.token_to_id(t) for t in SPECIAL_TOKENS
        if tokenizer.token_to_id(t) is not None
    }
    eval_report = evaluate_tokenizer(tokenizer, eval_sample_path, EVAL_THRESHOLDS, special_ids)
    atomic_dump_json(EVAL_REPORT_PATH, eval_report)
    t = EVAL_THRESHOLDS
    print(f"  chars/token:    {eval_report['chars_per_token']:>6}  (ok [{t['chars_per_token_min']}, {t['chars_per_token_max']}])")
    print(f"  bytes/token:    {eval_report['bytes_per_token']:>6}  (min {t['bytes_per_token_min']})")
    print(f"  max token freq: {eval_report['max_single_token_freq']:>6}  (max {t['max_single_token_freq']})")
    if not eval_report["passed"]:
        print("\n  TOP 10 NON-SPECIAL TOKENS (debug):")
        for tok, count in eval_report["top_20_tokens"][:10]:
            print(f"    {count:>10,}  {tok!r}")
        raise RuntimeError(
            "Tokenizer eval FAILED -- aborting before sharding.\n  Issues:\n    - "
            + "\n    - ".join(eval_report["failures"])
            + f"\n  Full report: {EVAL_REPORT_PATH}"
        )
    print(f"  PASSED. Report: {EVAL_REPORT_PATH}")
else:
    print("  SKIP: no eval files available")

# ================= STEP 3: TOKENIZE =================
# Free the main-process tokenizer so workers don't fight it for memory.
import gc
del tokenizer
gc.collect()

current_tok_id = file_hash(TOKENIZER_PATH)[:16]
print(f"\n[3/3] Tokenizing... tokenization_id: {current_tok_id}")

old_tok_id = None
if os.path.exists(TOK_ID_PATH):
    with open(TOK_ID_PATH, "r") as f:
        old_tok_id = f.read().strip()
if old_tok_id and old_tok_id != current_tok_id:
    print(f"  TOKENIZER CHANGED ({old_tok_id} -> {current_tok_id}). Wiping old shards...")
    for f in os.listdir(SHARD_DIR):
        os.remove(os.path.join(SHARD_DIR, f))
    os.makedirs(SHARD_DIR, exist_ok=True)
elif old_tok_id:
    print(f"  Tokenizer unchanged. Tokenizing only new/changed files.")

atomic_write_text(TOK_ID_PATH, current_tok_id)

shard_manifest = {"tokenization_id": current_tok_id, "shards": []}
if os.path.exists(SHARD_MANIFEST_PATH):
    with open(SHARD_MANIFEST_PATH, "r") as f:
        shard_manifest = json.load(f)
existing_shards = {s["source"]: s for s in shard_manifest.get("shards", [])}

# Validate manifest/disk consistency BEFORE assigning new shard names. Any
# inconsistency (duplicates, orphans) means an earlier run produced bad state;
# refuse to proceed rather than silently overwrite or work around it. The user
# fixes the state with repair_shards.py and re-runs.
def _idx_from_shard_name(name: str) -> int:
    try:
        stem = name[len("shard_"):-len(".bin")]
        return int(stem)
    except (ValueError, IndexError, TypeError):
        return -1

manifest_shard_names = [s.get("shard", "") for s in shard_manifest.get("shards", [])]
dup_counts = {}
for n in manifest_shard_names:
    dup_counts[n] = dup_counts.get(n, 0) + 1
duplicates = sorted(n for n, c in dup_counts.items() if c > 1)
if duplicates:
    raise RuntimeError(
        f"Manifest has {len(duplicates)} duplicate shard name(s) — multiple sources "
        f"mapped to the same shard_*.bin file. Examples: {duplicates[:5]}.\n"
        f"  Run:  python tokenize/repair_shards.py --apply\n"
        f"  to keep the entry whose .bin size matches disk and drop the rest."
    )

on_disk_bins = {
    f for f in os.listdir(SHARD_DIR)
    if f.startswith("shard_") and f.endswith(".bin")
       and os.path.isfile(os.path.join(SHARD_DIR, f))
}
manifest_set = set(manifest_shard_names)
orphans = sorted(on_disk_bins - manifest_set)
if orphans:
    raise RuntimeError(
        f"{len(orphans)} orphan .bin file(s) on disk not referenced by the manifest. "
        f"Examples: {orphans[:5]}.\n"
        f"  Run:  python tokenize/repair_shards.py --apply --quarantine-orphans\n"
        f"  to move them out of the way before re-running."
    )

# Sanity check on shard size vs source text size. BPE on any text can produce
# at most ~1 token per source byte (the worst case is no compression at all),
# so shard_bytes can be at most dtype_bytes * source_bytes — that's 2x for
# uint16, 4x for uint32. Anything beyond that means the .bin file's content
# does not correspond to the source it claims (almost always a duplicate-write
# where two sources clobbered the same shard file).
_dtype_bytes = np.dtype(SHARD_DTYPE).itemsize
oversize_violations = []
for s in shard_manifest.get("shards", []):
    bin_path  = os.path.join(SHARD_DIR, s.get("shard", ""))
    src_bytes = s.get("source_size", 0)
    if not os.path.exists(bin_path) or src_bytes <= 0:
        continue
    shard_bytes = os.path.getsize(bin_path)
    if shard_bytes > _dtype_bytes * src_bytes:
        oversize_violations.append({
            "shard": s["shard"], "source": s["source"],
            "src_bytes": src_bytes, "shard_bytes": shard_bytes,
            "ratio": shard_bytes / src_bytes,
        })
if oversize_violations:
    oversize_violations.sort(key=lambda v: -v["ratio"])
    examples = "\n".join(
        f"  {v['shard']}  src={v['src_bytes']:,}B  shard={v['shard_bytes']:,}B  "
        f"({v['ratio']:.1f}x source)  [{v['source']}]"
        for v in oversize_violations[:5]
    )
    raise RuntimeError(
        f"{len(oversize_violations)} shard(s) larger than {_dtype_bytes}x their source "
        f"text size — impossible under any sane BPE compression for {SHARD_DTYPE.__name__}.\n"
        f"Worst offenders:\n{examples}\n"
        f"  Run:  python tokenize/repair_shards.py --apply\n"
        f"  to drop the manifest entries whose .bin size doesn't match disk."
    )

# All-clear: no duplicates, no orphans, no oversize shards. Use max-existing-
# index + 1 so we never collide with an existing shard name even after a
# repair has left gaps in the index sequence. (len(shards) would only be safe
# with strict contiguity.)
shard_idx = max((_idx_from_shard_name(n) for n in manifest_shard_names), default=-1) + 1
new_count = skip_count = total_new_tokens = 0
t0 = time.time()

# ---------- Pre-flight: classify each source as skip vs work ----------
# shard_manifest.json is the source of truth. If the manifest has an entry for
# this source with a matching size, skip it — regardless of whether the .bin is
# physically on disk. This lets cold-tar backups (manifest on Drive, .bins
# archived elsewhere) work without re-tokenizing. The tokenization_id mismatch
# branch above has already wiped the manifest if the tokenizer changed, so any
# entry we see here is guaranteed to be for the current tokenizer.
work_items = []  # list of (src_path, shard_path, src_size, prior_hash, prior_existing_hash)
cold_skip_count = 0  # subset of skip_count where the .bin is not on disk
for src_path in all_source_files:
    src_size = os.path.getsize(src_path)
    prior_existing_hash = None
    if src_path in existing_shards:
        old_entry = existing_shards[src_path]
        shard_path_old = os.path.join(SHARD_DIR, old_entry["shard"])
        if src_size == old_entry.get("source_size", -1):
            skip_count += 1
            if not os.path.exists(shard_path_old):
                cold_skip_count += 1
            continue
        # Size differs — defer hash compare to worker.
        prior_existing_hash = old_entry.get("source_hash")
        shard_name = old_entry["shard"]
    else:
        shard_name = f"shard_{shard_idx:05d}.bin"
        shard_idx += 1
    shard_path = os.path.join(SHARD_DIR, shard_name)
    work_items.append((src_path, shard_path, src_size, None, prior_existing_hash))

print(f"  Pre-flight: {len(work_items)} to tokenize, {skip_count} skipped (manifest match)")
if cold_skip_count:
    print(f"              of {skip_count} skipped, {cold_skip_count} have no .bin on disk (cold)")
print(f"  Workers: {args.workers}")

def _apply_entry(entry):
    """Insert/update a manifest entry, then atomically persist manifest+meta."""
    global total_new_tokens, new_count
    src = entry["source"]
    if src in existing_shards:
        shard_manifest["shards"] = [entry if s["source"] == src else s for s in shard_manifest["shards"]]
    else:
        shard_manifest["shards"].append(entry)
    existing_shards[src] = entry
    total_new_tokens += entry["tokens"]
    new_count += 1
    shard_manifest["tokenization_id"] = current_tok_id
    atomic_dump_json(SHARD_MANIFEST_PATH, shard_manifest)
    save_meta(META_PATH, shard_manifest, vocab_size_actual, eot_id, pad_id, current_tok_id)

if args.workers <= 1 or len(work_items) <= 1:
    # Serial fallback — also inits worker globals in-process to reuse _tokenize_one.
    _worker_init(TOKENIZER_PATH, EOT_TOKEN, ENCODE_BATCH_SIZE, READ_CHUNK_CHARACTERS,
                 np.dtype(SHARD_DTYPE).name)
    for i, item in enumerate(work_items):
        result = _tokenize_one(item)
        src_name = os.path.basename(result["source"])
        if result["_skipped"]:
            skip_count += 1
            print(f"  [{i+1}/{len(work_items)}] {src_name} -> skipped (hash match)")
            continue
        print(f"  [{i+1}/{len(work_items)}] {src_name}: "
              f"{result['tokens']:,} tokens ({result['shard_mb']:.1f} MB, {result['documents']:,} docs)")
        result.pop("_skipped")
        _apply_entry(result)
else:
    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(TOKENIZER_PATH, EOT_TOKEN, ENCODE_BATCH_SIZE, READ_CHUNK_CHARACTERS,
                  np.dtype(SHARD_DTYPE).name),
    )
    try:
        futures = {pool.submit(_tokenize_one, item): item for item in work_items}
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            src_name = os.path.basename(result["source"])
            if result["_skipped"]:
                skip_count += 1
                print(f"  [{done}/{len(work_items)}] {src_name} -> skipped (hash match)")
                continue
            print(f"  [{done}/{len(work_items)}] {src_name}: "
                  f"{result['tokens']:,} tokens ({result['shard_mb']:.1f} MB, {result['documents']:,} docs)")
            result.pop("_skipped")
            _apply_entry(result)
    finally:
        pool.shutdown(wait=True)

# Re-sort manifest entries to source-file order for determinism across runs.
order = {p: i for i, p in enumerate(all_source_files)}
shard_manifest["shards"].sort(key=lambda s: order.get(s["source"], len(order)))
atomic_dump_json(SHARD_MANIFEST_PATH, shard_manifest)
save_meta(META_PATH, shard_manifest, vocab_size_actual, eot_id, pad_id, current_tok_id)

elapsed = time.time() - t0
total_tokens = sum(s["tokens"] for s in shard_manifest["shards"])
total_shards = len(shard_manifest["shards"])
print(f"\nDone in {elapsed:.1f}s!")
print(f"  New/updated: {new_count} files ({total_new_tokens:,} tokens)")
print(f"  Skipped:     {skip_count} files (unchanged)")
print(f"  Total:       {total_shards} shards, {total_tokens:,} tokens")

# ================= MANIFEST =================
os.makedirs(MANIFEST_DIR, exist_ok=True)
eval_report_embed = None
if os.path.exists(EVAL_REPORT_PATH):
    with open(EVAL_REPORT_PATH) as f:
        eval_report_embed = json.load(f)

manifest = {
    "stage": "tokenization",
    "created": datetime.datetime.now().isoformat(),
    "tokenization_id": current_tok_id,
    "tokenizer": {
        "path": TOKENIZER_PATH,
        "vocab_size": vocab_size_actual,
        "type": "byte-level-bpe-digits",
        "num_special_tokens": NUM_SPECIAL,
        "named_specials": NAMED_SPECIAL,
        "eot_id": eot_id,
        "pad_id": pad_id,
    },
    "config": config,
    "config_source": os.path.abspath(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else None,
    "eval": eval_report_embed,
    "shards": {
        "dir": SHARD_DIR,
        "num_shards": total_shards,
        "total_tokens": total_tokens,
        "dtype": np.dtype(SHARD_DTYPE).name,
    },
    "sources": {
        "num_data_dirs": len(data_dirs),
        "num_source_files": len(all_source_files),
        "num_train_files": len(train_files),
        "num_eval_files": len(eval_files),
        "data_dirs": data_dirs,
    },
}
manifest_path = os.path.join(MANIFEST_DIR, "tokenization_latest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved: {manifest_path}")
print(f"  tokenization_id: {current_tok_id}")

# ================= STEP 4: MERGE SHARDS (optional) =================
merge_cfg = config.get("merge", {})
merge_enabled = merge_cfg.get("enabled", False) and not args.no_merge
if merge_enabled:
    # Fail loud before merge if any manifest entry has no .bin on disk —
    # merge_shards.py would sys.exit on the first missing input, but doing the
    # check up front gives a clearer message and a full count.
    missing_bins = [
        s for s in shard_manifest.get("shards", [])
        if not os.path.exists(os.path.join(SHARD_DIR, s.get("shard", "")))
    ]
    if missing_bins:
        examples = ", ".join(s["shard"] for s in missing_bins[:5])
        raise RuntimeError(
            f"{len(missing_bins)} shard(s) in the manifest have no .bin on disk "
            f"(examples: {examples}). Restore the tar archives into {SHARD_DIR} "
            f"or re-run with --no-merge."
        )
    target_bytes = args.merge_target_bytes or merge_cfg.get("target_bytes", 100 * 1024 * 1024)
    merged_dir = SHARD_DIR.rstrip("/") + "_merged"
    print(f"\n[4/4] Merging shards -> {merged_dir} (target {target_bytes / 1024 / 1024:.0f} MB)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from merge_shards import merge_shards as _merge_shards
    _merge_shards(SHARD_DIR, merged_dir, target_bytes)
elif args.no_merge:
    print("\n[4/4] Merge skipped (--no-merge).")
else:
    print("\n[4/4] Merge skipped (config merge.enabled=false).")
print(f"  {total_shards} shards, {total_tokens:,} tokens")
