# Tokenization Pipeline

Two main entry points — use one or the other depending on where you're running:

### `tokenizer_pipeline.ipynb`
Colab notebook. Reads raw text from Google Drive, trains a BPE tokenizer,
tokenizes shards, merges them, uploads to Drive. Best for Drive-native flows.

### `run_tokenizer.py`
Standalone script for VMs / local machines. Mirrors the notebook logic but
reads/writes local filesystem instead of Drive. Run with `--help` for options.

**Note:** The notebook and standalone script share logic but are NOT kept
in sync automatically. If you change one, check if the other needs the same fix.

---

## Supporting scripts

### `merge_shards.py`
Incremental, order-preserving merger for tokenizer `.bin` shards. Merges small
per-source shards into larger merged shards grouped by data source.

### `validate_shards.py`
Four pre-merge sanity checks: correct dtype, valid token IDs, non-empty shards,
manifest consistency. Run before merging.

### `upload_to_drive.py`
Uploads local tokenizer outputs (shards, manifests, tokenizer files) to Google
Drive via rclone. Used after running `run_tokenizer.py` on a VM.

---

## Config

### `tokenizer_config.json`
Tokenizer hyperparameters (vocab size, min frequency, etc.). Loaded verbatim
into the run manifest for full reproducibility. Edit before running the pipeline.
