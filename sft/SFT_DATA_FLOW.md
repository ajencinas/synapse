# SynapseGPT — SFT data flow

All SFT inputs, the notebook/script that produces each, where the data lives on
Drive, and how it flows to training. The Mermaid diagram below renders inline on
GitHub/VS Code; a rendered image is also committed
(`SFT_DATA_FLOW.png` / `.svg`, regenerate with
`dot -Tpng sft/SFT_DATA_FLOW.dot -o sft/SFT_DATA_FLOW.png`).

```mermaid
flowchart TB
  subgraph OTS["OTS download · sft_data_prep.ipynb → download_sft_data.py"]
    direction LR
    tulu3 & oasst1 & dolly & metamath & opencode & samsum & alpaca_gpt4
    nr["no_robots (NEW)"]
  end

  subgraph CUSTOM["Custom generation · DeepSeek V4 Flash teacher"]
    direction TB
    rd["reasoning_distill<br/>generate_reasoning_distill.py<br/>26,250 verified"]
    tu["tool_use<br/>generate_tool_use.py<br/>python + search/facts"]
    cr["creative<br/>generate_creative.py<br/>teacher + LLM-judge (NEW)"]
    facts[("Wikidata fact DB<br/>generate_tool_problems.py")]
    rt{{"tools_runtime.py<br/>sandbox + Brave + tokenizer"}}
  end

  dl["download_sft_data.py<br/>decontaminate + normalize"]
  raw[("datasets_sft/&lt;name&gt;/&lt;name&gt;_raw.jsonl<br/>{system, messages} · Google Drive")]
  tok["tokenize_sft_data.py<br/>ChatML · tool tokens 7/8 · loss mask · ≤2048"]
  shards[("sft_tokenized/&lt;name&gt;/{train,val}.jsonl")]
  con["consolidate_sft_data.py"]
  reg[("manifests/sft_data_registry.json")]
  mix["sft.py · SFT_DATA_MIX<br/>weighted per-batch sampling"]
  train["sft_pre_train.ipynb<br/>train SynapseGPT (~2B)"]

  tulu3 & oasst1 & dolly & metamath & opencode & samsum & alpaca_gpt4 & nr --> dl --> raw
  facts -. facts pool .-> tu
  rt -. tools .-> tu
  rd --> raw
  tu --> raw
  cr --> raw
  raw --> tok --> shards --> con --> reg --> mix --> train
  rt -. "same tools at inference<br/>(sparky_chatbot loop, TODO)" .-> train

  classDef done fill:#d7ecd9,stroke:#5a8f5e;
  classDef new fill:#cfe2ff,stroke:#1a73e8;
  classDef prog fill:#ffe0b3,stroke:#b5530a;
  class tulu3,oasst1,dolly,metamath,opencode,samsum,alpaca_gpt4,rd done;
  class nr,cr new;
  class tu prog;
```

**Status legend:** 🟩 on Drive · 🟦 new/ready · 🟧 in progress

## Source table

| Source | Type | Notebook → script | Raw file (`datasets_sft/<name>/`) | Status |
|---|---|---|---|---|
| tulu3, oasst1, dolly, metamath, opencode, samsum, alpaca_gpt4 | OTS | `sft_data_prep.ipynb` → `download_sft_data.py` | `<name>_raw.jsonl` | 🟩 on Drive |
| **no_robots** | OTS | ″ | `no_robots_raw.jsonl` | 🟦 not downloaded yet |
| reasoning_distill | custom | `generate_reasoning_distill_colab.ipynb` | `reasoning_distill_raw.jsonl` | 🟩 26,250 |
| tool_use | custom | `generate_tool_use_colab.ipynb` | `tool_use_raw.jsonl` | 🟧 regenerating (clean) |
| **creative** | custom | `generate_creative_colab.ipynb` | `creative_raw.jsonl` | 🟦 tested, not run |

**Helpers (not sources):** `generate_tool_problems.py` builds the Wikidata fact DB feeding `tool_use --mode search`; `tools_runtime.py` is the shared sandbox/Brave/tokenizer runtime used by `generate_tool_use` and (TODO) the inference tool-loop in `sparky_chatbot.py`.

## Open item
`SFT_DATA_MIX` in `sft.py` lists only the 7 original OTS sources. `reasoning_distill`,
`tool_use`, `creative`, `no_robots` must be **added after each is tokenized** (the
trainer hard-fails on a mix source with no `train.jsonl`).
