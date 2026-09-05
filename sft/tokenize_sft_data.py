#!/usr/bin/env python3
"""Tokenize raw SFT chat JSONL → token IDs + per-token loss labels.

Usage:
  SYNAPSE_DIR=/path/to/synapse python sft/tokenize_sft_data.py \
      --datasets all --block-size 2048 --val-fraction 0.02

Reads:  $SYNAPSE_DIR/datasets_sft/<name>/<name>_raw.jsonl   {"system","messages"}
        $SYNAPSE_DIR/tokenizer_out/tokenizer.json
        $SYNAPSE_DIR/manifests/tokenization_latest.json     (tokenization_id check)

Writes: $SYNAPSE_DIR/sft_tokenized/<name>/train.jsonl   {"input_ids":[...],"labels":[...]}
        $SYNAPSE_DIR/sft_tokenized/<name>/val.jsonl
        $SYNAPSE_DIR/sft_tokenized/<name>/meta.json
        $SYNAPSE_DIR/sft_tokenized/tokenization_id.txt
        $SYNAPSE_DIR/manifests/sft_tokenization_latest.json   (upsert)

Refuses to run if the tokenizer's tokenization_id doesn't match pretrain's
(prevents accidentally SFT-ing with a tokenizer the model never saw).

Chat template (atomic reserved role tokens — see sft/SFT_DATA_PLAN.md):
  <|system|> sys <|im_end|> <|user|> q <|im_end|> <|assistant|> a <|im_end|> ... <|endoftext|>
Loss mask: only assistant response tokens + their <|im_end|> and the final
<|endoftext|> are learned (labels != -100). Everything else is -100.

Train/val split (global, deterministic — SFT v2 Phase 1.1): an example goes to
val iff sha256(normalized first user message) % 1000 < val_fraction*1000. The
split is a pure function of the question, so the same question lands on the
same side in EVERY source and every retokenization. See SPLIT_RULE below.

Incremental: a source is skipped if its outputs already exist AND its stored
tokenization_id/block_size match the current run. Use --force to retokenize a
named source. The manifest is upserted, never overwritten.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

from tokenizers import Tokenizer

# Canonical tool-call serializer — single source of truth (shared with the
# generator + chat template) so tool-call JSON tokenizes byte-identically.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools_runtime import dump_tool_call

IGNORE = -100

# Reserved-pool role markers (IDs 9/10/11) + the dedicated tool-result role (id 8).
# No tokenizer retrain — all already exist. Verified atomic + ID-matched below.
ROLE_TOKEN = {
    "system": "<|reserved_0|>",
    "user": "<|reserved_1|>",
    "assistant": "<|reserved_2|>",
    "tool": "<|tool_result|>",        # tool-result turns (injected; masked)
}
EXPECTED_ROLE_ID = {"system": 9, "user": 10, "assistant": 11, "tool": 8}
# Inline marker emitted INSIDE an assistant turn when it has a tool_call dict —
# NOT a role, resolved separately.
TOOL_CALL = "<|tool_call|>"
EXPECTED_TOOL_CALL_ID = 7
IM_END = "<|im_end|>"
EOT = "<|endoftext|>"


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


# Global deterministic train/val split (SFT v2 Phase 1.1 — see
# sft/SFT_V2_EXECUTION_PLAN.md §2 1.1). The split is a pure function of the
# example's first user message, which fixes v1's structural val leakage:
#   - cross-source twins (tool_use / reasoning_distill share ~22k problem ids)
#     now land on the SAME side of the split in both sources;
#   - exact internal duplicates (metamath has 3,518) can no longer straddle it;
#   - retokenizing after a source grows (tulu3 18k -> 50k) no longer reshuffles
#     which examples are in val — the val set only grows, never churns.
# Bump the version suffix if the rule ever changes; is_fresh() treats any
# meta.json without this exact string as stale and retokenizes.
SPLIT_RULE = "sha256(normalized_first_user_msg)[:8] % 1000 < val_fraction*1000 (v1)"
MIN_RESP_OVERRIDE = {"format_following": 1, "nli": 1}   # deliberate short answers (v3): "PONG", "Yes."


def split_bucket(ex):
    """Permille bucket in [0, 1000) from the normalized first user message."""
    for m in ex.get("messages", []):
        if m.get("role") == "user":
            q = " ".join(str(m.get("content", "")).split()).casefold()
            if not q:
                break
            h = hashlib.sha256(q.encode("utf-8")).digest()
            return int.from_bytes(h[:8], "big") % 1000
    raise SystemExit(
        "example has no non-empty user message — can't compute split bucket; "
        "raw data violates the chat invariants (re-run download checks)")


def tokenizer_id(tokenizer_path):
    h = hashlib.sha256()
    with open(tokenizer_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def resolve_special_ids(tok):
    """Return (role_id, im_end_id, eot_id, tool_call_id) — fail loud if any token is
    missing or doesn't match its expected id."""
    role_id = {}
    for role, name in ROLE_TOKEN.items():
        tid = tok.token_to_id(name)
        if tid is None:
            raise SystemExit(f"role token {name!r} not in tokenizer — can't map {role}")
        if tid != EXPECTED_ROLE_ID[role]:
            raise SystemExit(
                f"role token {name!r} has id {tid}, expected {EXPECTED_ROLE_ID[role]} "
                f"— reserved-pool layout changed; refusing to write")
        role_id[role] = tid
    im_end_id = tok.token_to_id(IM_END)
    eot_id = tok.token_to_id(EOT)
    if im_end_id is None or eot_id is None:
        raise SystemExit("tokenizer missing <|im_end|> or <|endoftext|>")
    tool_call_id = tok.token_to_id(TOOL_CALL)
    if tool_call_id is None:
        raise SystemExit(f"tokenizer missing {TOOL_CALL}")
    if tool_call_id != EXPECTED_TOOL_CALL_ID:
        raise SystemExit(
            f"{TOOL_CALL} has id {tool_call_id}, expected {EXPECTED_TOOL_CALL_ID} "
            f"— refusing to write")
    return role_id, im_end_id, eot_id, tool_call_id


def encode_example(tok, ex, role_id, im_end_id, eot_id, tool_call_id):
    """Returns (input_ids, labels, response_token_count).

    Records without a `tool` role or `tool_call` field tokenize EXACTLY as before
    (the new branches never fire), so existing sources are unaffected."""
    ids, labels = [], []
    resp_tokens = 0

    def add(tok_ids, learn):
        ids.extend(tok_ids)
        labels.extend(tok_ids if learn else [IGNORE] * len(tok_ids))

    def enc(text):
        return tok.encode(text, add_special_tokens=False).ids

    if ex.get("system"):
        add([role_id["system"]], False)
        add(enc(ex["system"]), False)
        add([im_end_id], False)

    for m in ex["messages"]:
        role = m["role"]
        learn = role == "assistant"
        add([role_id[role]], False)            # role marker always masked
        content_ids = enc(m.get("content") or "")
        add(content_ids, learn)
        if learn:
            resp_tokens += len(content_ids)
        # inline tool call inside an assistant turn (learned), before the closing
        # <|im_end|> — keeps reasoning + call in ONE turn (one im_end).
        if role == "assistant" and m.get("tool_call"):
            add([tool_call_id], True)
            tc_ids = enc(dump_tool_call(m["tool_call"]))
            add(tc_ids, True)
            resp_tokens += len(tc_ids)
        add([im_end_id], learn)                # learn to STOP on assistant turns

    add([eot_id], True)                        # learn end-of-sequence
    return ids, labels, resp_tokens


def pcts(xs):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(0.95 * n))],
        "p99": s[min(n - 1, int(0.99 * n))],
        "max": s[-1],
        "mean": round(statistics.mean(s), 1),
    }


def raw_fingerprint(raw_path):
    """sha256 of the raw jsonl — freshness must track CONTENT, not just config
    (v3 lesson: tool_negative was rebuilt with +2k rows; config-only freshness
    would silently skip it and train without them)."""
    h = hashlib.sha256()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_fresh(out_dir, tok_id, block_size, val_fraction, raw_path):
    """True if outputs exist and were built with this tokenizer + block_size +
    split rule. Sources tokenized before the global hash split (no split_rule
    in meta.json) are stale by construction and get retokenized."""
    meta_path = os.path.join(out_dir, "meta.json")
    needed = [meta_path, os.path.join(out_dir, "train.jsonl"),
              os.path.join(out_dir, "val.jsonl")]
    if not all(os.path.exists(p) for p in needed):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (meta.get("tokenization_id") == tok_id
            and meta.get("block_size") == block_size
            and meta.get("split_rule") == SPLIT_RULE
            and meta.get("val_fraction") == val_fraction
            and os.path.exists(raw_path)
            and meta.get("raw_sha256") == raw_fingerprint(raw_path))


def process(name, raw_dir, out_dir, tok, ids_triplet, tok_id,
            block_size, val_fraction):
    role_id, im_end_id, eot_id, tool_call_id = ids_triplet
    raw_path = os.path.join(raw_dir, f"{name}_raw.jsonl")
    if not os.path.exists(raw_path):
        raise SystemExit(f"[{name}] missing raw file: {raw_path} — run download_sft_data.py first")

    os.makedirs(out_dir, exist_ok=True)
    with open(raw_path) as f:
        raw_rows = [json.loads(line) for line in f]

    val_threshold = int(round(val_fraction * 1000))
    train, val = [], []
    # format_following's one-word answers ("PONG", "yes", "6") ARE the training
    # signal — exact-format compliance. The <3-token junk filter must not eat them.
    min_resp = MIN_RESP_OVERRIDE.get(name, 3)
    drops = {"short_response": 0, "too_long": 0}
    total_lens, response_lens = [], []
    for ex in raw_rows:
        input_ids, labels, resp_tokens = encode_example(
            tok, ex, role_id, im_end_id, eot_id, tool_call_id)
        if resp_tokens < min_resp:
            drops["short_response"] += 1
            continue
        if len(input_ids) > block_size:
            drops["too_long"] += 1            # never truncate mid-response
            continue
        rec = {"input_ids": input_ids, "labels": labels}
        (val if split_bucket(ex) < val_threshold else train).append(rec)
        total_lens.append(len(input_ids))
        response_lens.append(resp_tokens)

    n_kept = len(train) + len(val)
    if n_kept and not val:
        raise SystemExit(
            f"[{name}] global hash split produced 0 val examples out of "
            f"{n_kept:,} kept (val_fraction={val_fraction}) — source too small "
            f"or val_fraction too low; refusing to write an empty val set")
    if n_kept and not train:
        raise SystemExit(f"[{name}] global hash split produced 0 TRAIN examples "
                         f"— val_fraction={val_fraction} looks wrong")

    def write_jsonl(path, rows):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)

    meta = {
        "dataset": name,
        "tokenization_id": tok_id,
        "block_size": block_size,
        "val_fraction": val_fraction,
        "split_rule": SPLIT_RULE,
        "raw_count": len(raw_rows),
        "raw_sha256": raw_fingerprint(raw_path),
        "kept": n_kept,
        "train_count": len(train),
        "val_count": len(val),
        "drops": drops,
        "total_len_stats": pcts(total_lens),
        "response_len_stats": pcts(response_lens),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{name}] kept {n_kept:,} (train={len(train):,}, val={len(val):,}) "
          f"drops={drops} total_p95={meta['total_len_stats'].get('p95')} "
          f"response_p95={meta['response_len_stats'].get('p95')}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated names, or 'all'")
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--force", action="store_true",
                    help="retokenize named sources even if outputs are fresh")
    ap.add_argument("--synapse-dir",
                    default=os.environ.get("SYNAPSE_DIR") or default_synapse_dir())
    args = ap.parse_args()

    syn = args.synapse_dir
    raw_base = os.path.join(syn, "datasets_sft")
    out_base = os.path.join(syn, "sft_tokenized")
    manifest_dir = os.path.join(syn, "manifests")
    tokenizer_path = os.path.join(syn, "tokenizer_out", "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        raise SystemExit(f"tokenizer not found at {tokenizer_path}")

    tok = Tokenizer.from_file(tokenizer_path)
    tok_id = tokenizer_id(tokenizer_path)
    ids_triplet = resolve_special_ids(tok)

    pretrain_manifest = os.path.join(manifest_dir, "tokenization_latest.json")
    if not os.path.exists(pretrain_manifest):
        raise SystemExit(
            f"pretrain tokenization manifest missing: {pretrain_manifest} "
            f"— can't verify tokenization_id match")
    with open(pretrain_manifest) as f:
        pretrain_tok_id = json.load(f)["tokenization_id"]
    if tok_id != pretrain_tok_id:
        raise SystemExit(
            f"tokenization_id mismatch:\n"
            f"  tokenizer.json fingerprint: {tok_id}\n"
            f"  pretrain manifest:          {pretrain_tok_id}\n"
            f"refusing to write — the model was trained on a different tokenizer")

    if args.datasets == "all":
        names = sorted(n for n in os.listdir(raw_base)
                       if os.path.isdir(os.path.join(raw_base, n)))
    else:
        names = args.datasets.split(",")

    os.makedirs(out_base, exist_ok=True)
    processed = {}
    for name in names:
        out_dir = os.path.join(out_base, name)
        raw_path = os.path.join(raw_base, name, f"{name}_raw.jsonl")
        if not args.force and is_fresh(out_dir, tok_id, args.block_size,
                                       args.val_fraction, raw_path):
            print(f"[{name}] skip — already tokenized (tokenization_id + block_size match)")
            continue
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            print(f"[{name}] retokenizing (stale or --force)")
        processed[name] = process(
            name=name, raw_dir=os.path.join(raw_base, name), out_dir=out_dir,
            tok=tok, ids_triplet=ids_triplet, tok_id=tok_id,
            block_size=args.block_size, val_fraction=args.val_fraction)

    with open(os.path.join(out_base, "tokenization_id.txt"), "w") as f:
        f.write(tok_id + "\n")

    # --- Manifest upsert: keep untouched datasets, refresh requested ones. ---
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "sft_tokenization_latest.json")
    manifest = {"stage": "sft_tokenization", "tokenization_id": tok_id,
                "block_size": args.block_size, "datasets": {}}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                prev = json.load(f)
            if isinstance(prev.get("datasets"), dict):
                manifest["datasets"] = prev["datasets"]
        except (OSError, json.JSONDecodeError):
            pass
    # refresh entries for every requested name from its on-disk meta (covers
    # both freshly processed and up-to-date skips)
    for name in names:
        meta_path = os.path.join(out_base, name, "meta.json")
        if name in processed:
            manifest["datasets"][name] = processed[name]
        elif os.path.exists(meta_path):
            with open(meta_path) as f:
                manifest["datasets"][name] = json.load(f)
    manifest["total_train"] = sum(d.get("train_count", 0)
                                  for d in manifest["datasets"].values())
    manifest["total_val"] = sum(d.get("val_count", 0)
                                for d in manifest["datasets"].values())

    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    print(f"manifest: {manifest_path} "
          f"(total_train={manifest['total_train']:,}, total_val={manifest['total_val']:,})")


if __name__ == "__main__":
    main()
