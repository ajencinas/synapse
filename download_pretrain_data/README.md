# download_pretrain_data/ — public pretraining corpora

Standalone Jupyter notebooks that download and lightly clean the public corpora
used for pretraining. Each writes plain-text output into the Drive layout
(`gdrive:synapse/datasets_pretrain/data_<name>/`) that the tokenizer pipeline
(`tokenize/`) then shards.

| Notebook | Source |
|---|---|
| `download_wikipedia.ipynb` | Wikipedia |
| `download_c4.ipynb` | C4 (Common Crawl) |
| `download_fineweb.ipynb` | FineWeb |
| `download_finemath.ipynb` | FineMath |
| `download_code.ipynb` | Code corpora |
| `download_redpajama.ipynb` | RedPajama (arXiv slice) |
| `download_reasoning.ipynb` | Reasoning traces |

Run them independently, in any order. Books (Gutenberg/FadedPage) and the
synthetic corpora come from private pipelines that are not part of this repo;
the pretraining mix in `pretrain/train.py` documents every source it expects.
