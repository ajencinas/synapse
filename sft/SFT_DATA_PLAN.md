# SFT Data Management Plan

Data-collection plan for SFT-ing the SynapseGPT base checkpoint. Scope here is
**data only** (download → decontaminate → tokenize → consolidate). The training
loop (`sft.py`) is a later phase.

**Status: implemented.** `download_sft_data.py`, `tokenize_sft_data.py`,
`consolidate_sft_data.py`, `sft_data_prep.ipynb`, and `push_to_drive.py` all
follow this plan. Sections below double as the design reference for that code.

## 0. The base model this data must fit

| Property | Value |
|---|---|
| Params / shape | ~2B — `EMBED_DIM=2560`, 28 layers, FFN 6912, GQA 20q/4kv, RoPE 10000, RMSNorm, SwiGLU |
| Context | `BLOCK_SIZE = 2048` — **hard ceiling**, no positions beyond it |
| Vocab | 64k BBPE, 256 reserved specials, **digit-per-token** pre-tokenizer |
| Embeddings | **tied** (`token_embedding.weight = lm_head.weight`) |
| Tokenizer fingerprint | `tokenization_id = 7a570a7ba9fc7985` (enforced by `tokenize_sft_data.py`) |
| Pretrain corpus | **arxiv 21% + finemath 21% + math variants ~5% ≈ 47% STEM**, code 15%, fineweb 20%, wiki 13%, books 3% |

**Implication:** the base is a STEM/reasoning specialist. Per the SFT guide §4
("quality and *fit to the pretrained distribution* beat raw volume"), math and
reasoning SFT data lands best; chatty/creative data is the weakest fit. The mix
in §6 reflects this plus a deliberate *cut code / tilt reasoning+prose*.

Two consequences of model internals for data prep:
- **Digit-per-token** → a 5-digit number is 5 tokens; long CoT inflates fast.
  Filter math/CoT on **tokenized** length, not chars/words, and budget math more
  conservatively.
- **Tied embeddings + reserved role tokens** → role-token rows already exist in
  the checkpoint (at init values, never updated in pretrain); SFT trains them.
  **No vocab resize.**

---

## 1. Role Tokens (from reserved pool — zero tokenizer impact)

Assign 3 unused reserved specials to ChatML role markers. No tokenizer retrain,
`tokenization_id` stays `7a570a7ba9fc7985`.

| ID | Current Name | New Role | Purpose |
|---|---|---|---|
| 9 | `<\|reserved_0\|>` | `<\|system\|>` | System message |
| 10 | `<\|reserved_1\|>` | `<\|user\|>` | User turn |
| 11 | `<\|reserved_2\|>` | `<\|assistant\|>` | Assistant turn |

Plus existing tokens: `<|im_end|>` (ID 3) ends each turn, `<|endoftext|>` (ID 0)
ends the sequence.

**Decision — atomic role tokens, not `<|im_start|>` + text.** The old
`tokenize_sft_data.py` used `<|im_start|>user\n…` where "user"/"assistant" are
BPE-split text. We replace this with single atomic role tokens (9/10/11) so the
masking boundary is unambiguous and BPE can never merge a role word into adjacent
content. Apply the *identical* template at train and inference — a mismatch here
is the most common silent SFT bug.

---

## 2. Decontamination

**8-gram overlap filter** against a guard set, applied in `download_sft_data.py`.
On by default; disable with `--no-decontaminate`. The guard set is built
automatically from **GSM8K + HumanEval** (test splits) — no path argument needed.
It is built once per run and only when at least one requested source still needs
downloading (idempotent reruns skip it).

Normalization: lowercase, strip punctuation, collapse whitespace. Build the eval
n-gram index once; drop any training example sharing an 8-gram with it. For eval
items < 8 tokens, fall back to exact normalized-substring match.

```python
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()
def ngrams(s, n=8):
    t = norm(s).split()
    return {" ".join(t[i:i+n]) for i in range(len(t) - n + 1)}
```

**Guard set: GSM8K + HumanEval** (filtering only — *not* the evaluation set).
Keep the real eval holdout separate. More can be added later.

---

## 3. Data Sources

Per-source adapter maps each source into the uniform format
`{system, messages: [{role, content}, ...]}` (alternating user/assistant, ending
on assistant) before templating. The `max_examples` cap bounds *collection*
(disk/download); training proportions come from the §6 mix, not these caps.

| Source | HF path | Cap | Adapter notes |
|---|---|---|---|
| Tulu-3 SFT mixture | `allenai/tulu-3-sft-mixture` | 18k | streamed; reads its `messages` list; system pulled out. |
| OASST1 | `OpenAssistant/oasst1` | 8k | message tree → English-only, best-`rank` reply per turn; multi-turn. **Supply-limited (~2.3k actual).** |
| Dolly-15k | `databricks/databricks-dolly-15k` | 15k | `instruction`(+`context`)/`response`. Near-exhausted (~15k). |
| MetaMath | `meta-math/MetaMathQA` | 22k | streamed; `query`/`response`. Reasoning anchor. **Filter on tokenized length.** |
| OpenCodeInstruct | `nvidia/OpenCodeInstruct` | 6k | streamed; tries `input/instruction/question` → `output/response/solution/completion`. |
| SAMSum | `knkarthick/samsum` | 5k | Parquet mirror; `dialogue`→`summary` wrapped in a "Summarize…" instruction. |
| AlpacaGPT-4 | `vicgalle/alpaca-gpt4` | 4k | GPT-4 answers only — **never** original Alpaca (guide §5). |

**Target ~70k collected examples** (reasoning-tilted scale-up). Realistic kept
after dedup/decontam ≈ tulu3 18k + metamath 22k + dolly ~15k + opencode 6k +
samsum 5k + alpaca_gpt4 4k + oasst1 ~2.3k ≈ **~72k**. With weighted sampling the
collected pool *is* the training pool; the §6 mix sets per-batch proportions.
Schema caveat: `opencode` column names vary by revision — if everything drops as
`adapter_rejected`, inspect the columns and adjust `adapter_opencode`.

**Getting more from one source.** Caps are CLI-overridable and downloads are
idempotent, so raise a cap + force the re-download, then retokenize just that
source:
```bash
python sft/download_sft_data.py --datasets metamath --limit metamath=20000 --force
python sft/tokenize_sft_data.py --datasets metamath --force   # raw changed → --force
python sft/consolidate_sft_data.py
```
`--force` is needed on tokenize because skip-if-fresh keys on tokenizer/block-size
identity, not raw content (no content hash yet — see the open item below).

---

## 4. Tokenization (`tokenize_sft_data.py`)

**Per-token `labels` with `-100` masking** (replaces the old `prefix_len`).

```
[<|user|>(M)  content(M)  <|im_end|>(M)  <|assistant|>(M)  response(L)  <|im_end|>(L)  <|endoftext|>(L)]
  M = -100 (ignore in loss), L = token ID (learned)
```

- Mask: system/user content + all role markers + the *assistant role marker*.
- Learn: each assistant response **and its `<|im_end|>`** (so the model learns to
  stop). `<|endoftext|>` after the final turn is also learned.
- **Multi-turn:** every assistant turn is unmasked (N response spans), all
  user/system/markers masked.
- **Filter on total tokenized length; drop examples > `block_size`.** Never
  truncate mid-response (teaches early stopping). Budget math/CoT conservatively
  because of digit-split.
- Keep the existing `tokenization_id` guard (refuse if ≠ pretrain fingerprint).
- **Round-trip verify:** decode a few samples, confirm the loss mask aligns with
  assistant turns — catches almost every template bug.

**Exact template:** an atomic role token *immediately precedes* its content (no
extra newline/separator), content is encoded with `add_special_tokens=False`, and
each turn closes with `<|im_end|>`. Concretely:
`<|system|>sys<|im_end|><|user|>q<|im_end|><|assistant|>a<|im_end|>…<|endoftext|>`.
This identical template **must** be reproduced at inference.

Output: per-example `{input_ids, labels}` (parallel arrays) + `meta.json` (which
stores `tokenization_id` + `block_size` for the staleness guard).

### Incremental tokenization (adding a source does NOT retokenize everything)

Implemented so a new source is cheap to add:

1. **Per-source skip-if-exists.** Like `download_sft_data.py`, skip a source when
   its `sft_tokenized/<name>/{train,val}.jsonl` + `meta.json` already exist. So
   `--datasets all` tokenizes only *new/changed* sources. Add `--force` to
   retokenize a named source (e.g. after changing the template/block_size).
2. **Manifest upsert, not overwrite.** Load the existing
   `sft_tokenization_latest.json`, update only the entries for sources processed
   this run, and rewrite — never drop untouched sources' records.
3. **Staleness guard.** Re-tokenize (ignore the skip) if a source's stored
   `tokenization_id`/`block_size` in its `meta.json` differs from the current
   run's — so a tokenizer or block-size change can't silently leave mismatched
   shards behind. Fail loud on mismatch rather than mixing.

Resulting flow when adding a source:
```bash
python sft/download_sft_data.py --datasets newsource   # downloads only new (already idempotent)
python sft/tokenize_sft_data.py --datasets newsource   # tokenizes only new; manifest upsert
python sft/consolidate_sft_data.py                     # rescans dirs → registry
```
The registry (§5) is the authoritative full picture; it rescans `sft_tokenized/`
directories directly, so it's always consistent regardless of the manifest.

---

## 5. Data Registry (`consolidate_sft_data.py`)

Rescans all `sft_tokenized/<name>/` dirs (source pulled from each
`datasets_sft/<name>/meta_raw.json`), writes `manifests/sft_data_registry.json`:

```json
{
  "tokenization_id": "7a570a7ba9fc7985",
  "block_size": 2048,
  "datasets": {
    "tulu3": {
      "source": "allenai/tulu-3-sft-mixture",
      "kept": 12800, "train_count": 12544, "val_count": 256,
      "val_path": "sft_tokenized/tulu3/val.jsonl",
      "drops": {"short_response": 4, "too_long": 31},
      "stats": {
        "total_len": {"p50": 210, "p95": 690, "max": 2041, "mean": 268.0},
        "response_len": {"p50": 120, "p95": 312, "max": 980, "mean": 142.0}
      }
    }
  },
  "total_train": 39200, "total_val": 800
}
```

The per-source `val.jsonl` is kept in its own dir, so the **directory name is the
source tag** (recorded as `val_path`) — training tracks per-source val loss from
these without merging. Fail-loud if sources disagree on `tokenization_id`/
`block_size`. Analogue of pretrain's `shard_manifest.json`.

---

## 6. Data Mix Selection (reasoning/prose tilt, ~70k scale)

Tilted toward the STEM base and the *cut-code* preference. Reasoning ~0.28,
code ~0.06, prose/general ~0.66.

```python
SFT_DATA_MIX = {
    "tulu3":      0.30,   # general backbone
    "metamath":   0.28,   # reasoning anchor — UP (best fit to base)
    "dolly":      0.16,   # clean simple
    "alpaca_gpt4":0.08,   # prose (GPT-4 answers only)
    "samsum":     0.07,   # summarization
    "opencode":   0.06,   # code — DOWN (cut-code pref)
    "oasst1":     0.05,   # multi-turn prose (supply-limited ~2.3k)
}
```

Weighted sampling per batch from the consolidated registry. Per-dataset val loss
tracked independently. The training manifest (later phase) records the exact mix,
source-checkpoint identity, and per-source eval history.

**Oversampling note:** `oasst1` has only ~2.3k unique examples, so even at weight
0.05 it is drawn more than once per epoch-equivalent over a ~72k pool — watch its
per-source val loss for early overfitting and lower its weight if it diverges.

---

## Build order

1. [x] Role tokens 9/10/11 → `system`/`user`/`assistant` (resolved + ID-verified in `tokenize_sft_data.py`).
2. [x] `download_sft_data.py` — per-source adapters → `{system, messages[]}`, §3 sources, 8-gram decontam (GSM8K + HumanEval), dedup.
3. [x] `tokenize_sft_data.py` — per-token `labels`, multi-turn, drop-on-overflow (tokenized length), `tokenization_id` guard, **incremental** (skip-if-fresh + `--force`, manifest upsert, staleness guard — see §4).
4. [x] `consolidate_sft_data.py` — builds `sft_data_registry.json`, val tagged by source dir.
5. [x] `sft_data_prep.ipynb` — new `--datasets` list, tokenize `--datasets all`, consolidate cell, verify cell ported to the `labels` mask. (`push_to_drive.py` also pushes the registry.)
6. [ ] (Later) `sft.py` — load checkpoint weights-only (`strict=False`, inspect
   missing/unexpected keys) + fresh optimizer, masked-label loss, LR ~1e-5–2e-5,
   1–3 epochs, per-source response-token val loss.
