# 3B Model — Tokenizer Pipeline Config Plan

## Current vs. 3B Recommended Changes

### `tokenizer_config.json`

| Parameter | Current | 3B Recommendation | Why |
|---|---|---|---|
| `vocab_size` | 64000 | 32000–128000 | 3B models commonly use 32K (Phi-3) to 128K (Llama-3.2). 64K is fine too — design choice. |
| `shard_dtype` | `"uint16"` | `"uint32"` if vocab > 65535 | **Must change** if vocab exceeds 65K, otherwise token IDs overflow silently. Doubles shard disk usage. Already validated at runtime (`run_tokenizer.py:347–355`). |
| `min_frequency` | 2 | 2 | Fine as-is. |
| `bpe_dropout` | `null` | `0.1` | Adds regularization; standard for modern subword training. |
| `train_subset_bytes` | 4294967296 (4 GB) | 4–10 GB | 4 GB of text is enough for tokenizer training regardless of final model size. |
| `encode_batch_size` | 500 | 500 | Fine as-is. |
| `read_chunk_characters` | 8388608 | 8388608 | Fine as-is. |
| `num_special_tokens` | 256 | 256 | Fine as-is. |
| `named_special_tokens` | 9 tokens (im_start, fim_*, tool_*) | Same | Depends on chat template / tool-calling requirements. |
| `eval.thresholds` | chars/token 3–6, bytes/token ≥ 2.5, max freq ≤ 0.10 | Recalibrate for new vocab size | Larger vocab = higher compression (more chars/token). Auto-eval gate catches bad configs. |

### Data Volume

| Metric | 3B Target |
|---|---|
| Training tokens | ~2T (Chinchilla-optimal) |
| Current pipeline | Tokenizes whatever is in `data_*/` dirs |
| Action | Verify source data can yield enough tokens |

### No Code Changes Required

`run_tokenizer.py` is fully parameterized from the config. The dtype-vs-vocab assertion at lines 347–355 prevents silent overflow.

### Steps to Switch

1. Edit `tokenizer_config.json` with new values
2. Re-train tokenizer: `python run_tokenizer.py`
3. Verify eval report passes auto-gate
4. Feed shards to model training
