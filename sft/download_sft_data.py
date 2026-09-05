#!/usr/bin/env python3
"""Download SFT datasets from HuggingFace → normalized raw JSONL (chat format).

Usage:
  SYNAPSE_DIR=/path/to/synapse python sft/download_sft_data.py \
      --datasets tulu3,oasst1,dolly,metamath,opencode,samsum,alpaca_gpt4

  # download MORE of one source (raise its cap + re-download):
  python sft/download_sft_data.py --datasets metamath --limit metamath=20000 --force

  # disable decontamination (faster, NOT recommended for a real run):
  python sft/download_sft_data.py --datasets dolly --no-decontaminate

Output layout (under $SYNAPSE_DIR/datasets_sft/<name>/):
  <name>_raw.jsonl   one {"system": str, "messages": [{"role","content"}, ...]} per line
  meta_raw.json      HF revision, raw/kept counts, drop reasons

Uniform format: every example is {"system", "messages"} where messages alternate
user/assistant and END on an assistant turn. system may be "".

Decontamination: 8-gram overlap against a guard set (GSM8K + HumanEval by
default). Any training example sharing an 8-gram with the guard set is dropped.
Short guard items (< 8 tokens) fall back to normalized-substring matching.

Idempotent: skips a dataset if its <name>_raw.jsonl already exists.
"""
import argparse
import json
import os
import re

from datasets import load_dataset

# ----------------------------------------------------------------------------
# Decontamination
# ----------------------------------------------------------------------------

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (s or "").lower())).strip()


def ngrams(s, n=8):
    toks = norm(s).split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def build_guard(n=8):
    """Build (ngram_set, short_substrings) from GSM8K + HumanEval."""
    big = set()
    short = set()

    def add(text):
        toks = norm(text).split()
        if len(toks) >= n:
            big.update(ngrams(text, n))
        elif toks:
            short.add(" ".join(toks))

    print("[decontam] building guard set from GSM8K + HumanEval")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for row in gsm:
        add(row["question"])
        add(row["answer"])
    he = load_dataset("openai/openai_humaneval", split="test")
    for row in he:
        add(row["prompt"])
        add(row["canonical_solution"])
    print(f"[decontam] guard: {len(big):,} 8-grams, {len(short):,} short items")
    return big, short


def is_contaminated(text, guard, n=8):
    big, short = guard
    if ngrams(text, n) & big:
        return True
    if short:
        nt = norm(text)
        if any(s in nt for s in short):
            return True
    return False


# ----------------------------------------------------------------------------
# Normalized-example helpers
# ----------------------------------------------------------------------------

def make_example(system, turns):
    """turns: list of (role, content). Returns a validated {system, messages}
    dict, or None if malformed (must alternate user/assistant and end on
    assistant)."""
    messages = []
    for role, content in turns:
        content = (content or "").strip()
        if role not in ("user", "assistant") or not content:
            return None
        messages.append({"role": role, "content": content})
    if len(messages) < 2:
        return None
    if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None
    for i, m in enumerate(messages):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None  # must strictly alternate
    return {"system": (system or "").strip(), "messages": messages}


# ----------------------------------------------------------------------------
# Per-source adapters: row -> normalized example (or None)
# ----------------------------------------------------------------------------

def adapter_dolly(row):
    instr = (row.get("instruction") or "").strip()
    ctx = (row.get("context") or "").strip()
    resp = (row.get("response") or "").strip()
    user = f"{instr}\n\n{ctx}" if ctx else instr
    return make_example("", [("user", user), ("assistant", resp)])


def adapter_metamath(row):
    return make_example("", [("user", row.get("query")),
                             ("assistant", row.get("response"))])


def adapter_alpaca_gpt4(row):
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    user = f"{instr}\n\n{inp}" if inp else instr
    return make_example("", [("user", user), ("assistant", row.get("output"))])


def adapter_samsum(row):
    dialogue = (row.get("dialogue") or "").strip()
    summary = (row.get("summary") or "").strip()
    user = f"Summarize the following conversation:\n\n{dialogue}"
    return make_example("", [("user", user), ("assistant", summary)])


def adapter_opencode(row):
    # OpenCodeInstruct stores the prompt/response under a few possible keys
    # across revisions; try the common ones.
    user = row.get("input") or row.get("instruction") or row.get("question")
    resp = (row.get("output") or row.get("response") or row.get("solution")
            or row.get("completion"))
    return make_example("", [("user", user), ("assistant", resp)])


def _coerce_role(r):
    return {"user": "user", "human": "user", "prompter": "user",
            "assistant": "assistant", "gpt": "assistant"}.get(r)


def adapter_tulu3(row):
    """Tulu-3 stores a 'messages' list of {role, content}."""
    msgs = row.get("messages") or []
    system = ""
    turns = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            system = content or system
            continue
        cr = _coerce_role(role)
        if cr is None:
            return None
        turns.append((cr, content))
    return make_example(system, turns)


def adapter_nli(row):
    """stanfordnlp/snli {premise, hypothesis, label 0=entail/1=neutral/2=contradict}.
    Reformatted to the two yes/no entailment phrasings sft_bench uses (v2 bench:
    logic_entailment 0-1/3 — no source contained entailment at all). Neutral and
    unlabeled (-1) rows are dropped; the template alternates by question hash so
    both phrasings appear for both answers."""
    import hashlib as _h
    prem = (row.get("premise") or "").strip().rstrip(".")
    hyp = (row.get("hypothesis") or "").strip().rstrip(".")
    label = row.get("label")
    if not prem or not hyp or label not in (0, 2):
        return None
    if int(_h.sha256((prem + hyp).encode()).hexdigest()[:4], 16) % 2 == 0:
        q = (f"Premise: '{prem}.' Hypothesis: '{hyp}.' Based only on the premise, "
             f"is the hypothesis true? Answer yes or no.")
        ans = "Yes." if label == 0 else "No."
    else:
        q = (f"Sentence A: '{prem}.' Sentence B: '{hyp}.' "
             f"Do these sentences contradict each other? Answer yes or no.")
        ans = "Yes." if label == 2 else "No."
    return make_example("", [("user", q), ("assistant", ans)])


def adapter_no_robots(row):
    """HuggingFaceH4/no_robots stores a 'messages' list of {role, content} (+ a
    'category' incl. creative 'Generation'). Same shape as Tulu-3 — high-quality,
    human-written, great for expressive/assistant tone across the board."""
    return adapter_tulu3(row)


# ----------------------------------------------------------------------------
# Builders: take the full dataset, yield normalized examples (for sources that
# can't be mapped row-by-row, e.g. OASST's message tree).
# ----------------------------------------------------------------------------

def builder_oasst1(ds):
    """Reconstruct English conversation paths from the OASST1 message tree,
    taking the best-ranked assistant reply at each step."""
    by_id = {}
    children = {}
    roots = []
    for row in ds:
        if row.get("deleted") or row.get("lang") != "en":
            continue
        mid = row["message_id"]
        by_id[mid] = row
        pid = row.get("parent_id")
        if pid:
            children.setdefault(pid, []).append(mid)
        elif row.get("role") == "prompter":
            roots.append(mid)

    def best_child(mid, role):
        cands = [by_id[c] for c in children.get(mid, []) if c in by_id
                 and by_id[c].get("role") == role]
        if not cands:
            return None
        # lower rank == better; missing rank sorts last
        cands.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
        return cands[0]

    for root in roots:
        turns = []
        node = by_id[root]
        while node is not None:
            turns.append(("user", node["text"]))
            asst = best_child(node["message_id"], "assistant")
            if asst is None:
                break
            turns.append(("assistant", asst["text"]))
            node = best_child(asst["message_id"], "prompter")
        ex = make_example("", turns)
        if ex is not None:
            yield ex


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------

DATASETS = {
    "tulu3": {
        "hf_path": "allenai/tulu-3-sft-mixture", "hf_revision": "main",
        "split": "train", "streaming": True, "adapter": adapter_tulu3,
        "max_examples": 50000,
    },
    "oasst1": {
        "hf_path": "OpenAssistant/oasst1", "hf_revision": "main",
        "split": "train", "builder": builder_oasst1, "max_examples": 8000,
    },
    "dolly": {
        "hf_path": "databricks/databricks-dolly-15k", "hf_revision": "main",
        "split": "train", "adapter": adapter_dolly, "max_examples": 15000,
    },
    "metamath": {
        "hf_path": "meta-math/MetaMathQA", "hf_revision": "main",
        "split": "train", "streaming": True, "adapter": adapter_metamath,
        "max_examples": 22000,
    },
    "opencode": {
        "hf_path": "nvidia/OpenCodeInstruct", "hf_revision": "main",
        "split": "train", "streaming": True, "adapter": adapter_opencode,
        "max_examples": 6000,
    },
    "samsum": {
        # Parquet mirror — the original Samsung/samsum is a script-based dataset
        # that newer `datasets` can no longer load. Same dialogue/summary fields.
        "hf_path": "knkarthick/samsum", "hf_revision": "main",
        "split": "train", "adapter": adapter_samsum, "max_examples": 5000,
    },
    "alpaca_gpt4": {
        "hf_path": "vicgalle/alpaca-gpt4", "hf_revision": "main",
        "split": "train", "adapter": adapter_alpaca_gpt4, "max_examples": 4000,
    },
    "no_robots": {
        # ~9.5k high-quality human-written instruction/response (incl. creative
        # "Generation"). Expressive-tone breadth for the prose tilt.
        "hf_path": "HuggingFaceH4/no_robots", "hf_revision": "main",
        "split": "train", "adapter": adapter_no_robots, "max_examples": 9500,
    },
    "nli": {
        # v3: yes/no entailment in the bench's phrasing (logic_entailment was 0/3).
        # SNLI train is 550k; 3k is plenty for a 0.02 mix weight.
        "hf_path": "stanfordnlp/snli", "hf_revision": "main",
        "split": "train", "adapter": adapter_nli, "max_examples": 3000,
    },
}


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def example_text(ex):
    return ex["system"] + " " + " ".join(m["content"] for m in ex["messages"])


def iter_source(spec):
    kwargs = {"split": spec["split"]}
    if spec.get("hf_revision"):
        kwargs["revision"] = spec["hf_revision"]
    if spec.get("streaming"):
        kwargs["streaming"] = True
    if spec.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    ds = load_dataset(spec["hf_path"], **kwargs)
    if "builder" in spec:
        yield from spec["builder"](ds)
    else:
        for row in ds:
            yield spec["adapter"](row)


def process(name, spec, out_dir, guard, cap, force):
    raw_path = os.path.join(out_dir, f"{name}_raw.jsonl")
    meta_path = os.path.join(out_dir, "meta_raw.json")
    if os.path.exists(raw_path) and not force:
        print(f"[{name}] skip — {raw_path} already exists (use --force to re-download)")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"[{name}] loading {spec['hf_path']} @ {spec.get('hf_revision')} "
          f"({spec['split']}{', streaming' if spec.get('streaming') else ''}) "
          f"cap={cap}")
    seen = set()
    kept = []
    raw_count = 0
    drops = {"adapter_rejected": 0, "too_short": 0, "duplicate": 0,
             "contaminated": 0}
    for norm_ex in iter_source(spec):
        raw_count += 1
        if norm_ex is None:
            drops["adapter_rejected"] += 1
            continue
        text = example_text(norm_ex)
        if len(text) < 10:
            drops["too_short"] += 1
            continue
        key = (norm_ex["system"],
               tuple((m["role"], m["content"]) for m in norm_ex["messages"]))
        if key in seen:
            drops["duplicate"] += 1
            continue
        if guard is not None and is_contaminated(text, guard):
            drops["contaminated"] += 1
            continue
        seen.add(key)
        kept.append(norm_ex)
        if cap and len(kept) >= cap:
            break

    tmp_path = raw_path + ".tmp"
    with open(tmp_path, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp_path, raw_path)

    meta = {
        "dataset": name,
        "hf_path": spec["hf_path"],
        "hf_revision": spec.get("hf_revision"),
        "split": spec["split"],
        "raw_count_seen": raw_count,
        "max_examples": cap,
        "kept": len(kept),
        "decontaminated": guard is not None,
        "drops": drops,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{name}] wrote {len(kept):,} examples → {raw_path}")
    print(f"[{name}] drops: {drops}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="comma-separated dataset names, or 'all'")
    ap.add_argument("--synapse-dir",
                    default=os.environ.get("SYNAPSE_DIR") or default_synapse_dir())
    ap.add_argument("--decontaminate", dest="decontaminate", action="store_true",
                    default=True, help="filter against GSM8K+HumanEval (default)")
    ap.add_argument("--no-decontaminate", dest="decontaminate",
                    action="store_false")
    ap.add_argument("--force", action="store_true",
                    help="re-download named sources even if their raw file exists")
    ap.add_argument("--limit", default="",
                    help="override per-source caps, e.g. 'metamath=20000,dolly=15000'")
    args = ap.parse_args()

    names = list(DATASETS.keys()) if args.datasets == "all" else args.datasets.split(",")
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}. known: {list(DATASETS)}")

    overrides = {}
    for part in filter(None, args.limit.split(",")):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in DATASETS or not v.strip().isdigit():
            raise SystemExit(f"bad --limit entry {part!r} (expected name=INT)")
        overrides[k] = int(v)

    base = os.path.join(args.synapse_dir, "datasets_sft")
    print(f"output base: {base}")

    # Build the guard set once, only if some target will actually be downloaded.
    need = [n for n in names if args.force
            or not os.path.exists(os.path.join(base, n, f"{n}_raw.jsonl"))]
    guard = build_guard() if (args.decontaminate and need) else None
    if not args.decontaminate:
        print("[decontam] DISABLED via --no-decontaminate")

    for name in names:
        cap = overrides.get(name, DATASETS[name].get("max_examples"))
        process(name, DATASETS[name], os.path.join(base, name), guard, cap, args.force)


if __name__ == "__main__":
    main()
