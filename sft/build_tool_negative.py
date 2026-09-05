#!/usr/bin/env python3
"""Build the `tool_negative` SFT source — SFT v2 Phase 0.3 (zero API cost).

Relabels existing verified records with the canonical tool system prompt so the
dataset finally contains BOTH behaviors under the same prompt:

    tool prompt + hard math          -> call python        (tool_use, 35k)
    tool prompt + trivial math/prose -> answer directly    (this source, ~6k)

Without this, 100% of tool-prompt examples call a tool, and the model learns
"tool prompt in context => emit <|tool_call|>" instead of tool judgment
(SFT_V2_EXECUTION_PLAN.md §1b).

Donors (answers are already correct — nothing is generated here):
  - reasoning_distill: teacher CoT kept only when the final answer matched the
    gold answer at generation time (generate_reasoning_distill.py). Admitted
    only if (a) the question does NOT appear in tool_use (never show the same
    question with and without a call) and (b) trivial arithmetic — every
    integer in question+solution has <= MAX_DIGITS digits and the solution has
    no decimal-point arithmetic — so answering by hand is exactly what the
    prompt's "non-trivial calculation" clause asks for.
  - dolly + no_robots: human-written prose. Admitted only if no turn asks for
    computation (keyword + digit-density filters) and donor system is empty.

Selection within each admitted pool is deterministic: sorted by
sha256(donor id). Output records keep the donor's messages byte-identical and
only swap in system = CANONICAL_TOOL_SYSTEM. Because the train/val split
(tokenize_sft_data.split_bucket) hashes the first user message, every negative
lands on the same side of the split as its donor — relabeling cannot create
train/val leakage.

Usage:
  SYNAPSE_DIR=~/synapse_data python sft/build_tool_negative.py
Writes: $SYNAPSE_DIR/datasets_sft/tool_negative/{tool_negative_raw.jsonl,meta_raw.json}
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools_runtime import CANONICAL_TOOL_SYSTEM

N_MATH = 4000
N_PROSE_PER_DONOR = 1700          # dolly + no_robots (raised 1000->1700: rd tiers
                                  # supply only ~2.6k math; keeps total ~6k so the
                                  # mix weight 0.05 clears the 1.65x epoch ceiling)
N_SYNTH = 2000                    # v3: synthesized trivial arithmetic. v2 shipped
                                  # 2,629 math negatives vs 35k tool_use and the
                                  # boundary lost (eval: 43/43 trivial-math false
                                  # calls). Roughly double the math side.
SYNTH_SEED = 11
MAX_DIGITS = 2                    # "trivial" = every integer fits in 2 digits

COMPUTE_PAT = re.compile(
    r"\b(calculat\w*|comput\w*|how many|how much|sum of|multipl\w*|divid\w*|"
    r"subtract\w*|add up|solve|equation|percent\w*|average|total of|"
    r"square root|remainder|fraction)\b|[=+^]|\d+\s*[*/×÷-]\s*\d+",
    re.IGNORECASE)


def norm_q(ex):
    for m in ex["messages"]:
        if m["role"] == "user":
            return " ".join(m["content"].split()).casefold()
    raise SystemExit(f"record {ex.get('id')!r} has no user message")


def load_jsonl(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing donor file: {path} — pull it from Drive first")
    with open(path) as f:
        return [json.loads(line) for line in f]


def math_tier(ex):
    """0 = trivial (ints <= 2 digits), 1 = easy (ints <= 3 digits), else None.
    Both tiers: no decimal arithmetic, answer line present (prompt requires it)."""
    text = ex["messages"][0]["content"] + "\n" + ex["messages"][-1]["content"]
    if "the answer is" not in ex["messages"][-1]["content"].lower():
        return None
    if re.search(r"\d+\.\d+", text):
        return None
    longest = max((len(n) for n in re.findall(r"\d+", text)), default=0)
    if longest <= MAX_DIGITS:
        return 0
    if longest <= MAX_DIGITS + 1:
        return 1
    return None


def is_toolfree_prose(ex):
    if ex.get("system"):
        return False              # donor's own persona/system would be lost
    user_text = " ".join(m["content"] for m in ex["messages"] if m["role"] == "user")
    if COMPUTE_PAT.search(user_text):
        return False
    if len(re.findall(r"\d+", user_text)) >= 3:     # digit-dense — play safe
        return False
    # one-word answers ("No", "Casablanca") tokenize to <3 response tokens and
    # would be dropped by tokenize_sft_data anyway — exclude here so counts are real
    return all(len(m["content"].split()) >= 3
               for m in ex["messages"] if m["role"] == "assistant")


def donor_id(ex, donor):
    """Stable id: donor's own id, else hash of the normalized first question
    (dolly/no_robots carry no id field)."""
    return ex.get("id") or f"{donor}_{hashlib.sha256(norm_q(ex).encode()).hexdigest()[:16]}"


def det_sort(rows, donor):
    return sorted(rows, key=lambda r: hashlib.sha256(donor_id(r, donor).encode()).hexdigest())


def relabel(ex, donor):
    return {"id": f"neg_{donor_id(ex, donor)}", "donor": donor,
            "system": CANONICAL_TOOL_SYSTEM, "messages": ex["messages"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR"))
    args = ap.parse_args()
    if not args.synapse_dir:
        raise SystemExit("set SYNAPSE_DIR (refusing to guess — ./synapse is the venv)")
    base = os.path.join(args.synapse_dir, "datasets_sft")

    tool_use = load_jsonl(os.path.join(base, "tool_use", "tool_use_raw.jsonl"))
    rd = load_jsonl(os.path.join(base, "reasoning_distill", "reasoning_distill_raw.jsonl"))
    dolly = load_jsonl(os.path.join(base, "dolly", "dolly_raw.jsonl"))
    no_robots = load_jsonl(os.path.join(base, "no_robots", "no_robots_raw.jsonl"))

    tool_use_qs = {norm_q(ex) for ex in tool_use}

    # --- math negatives: tier 0 (trivial) first, then tier 1 (easy); NEVER
    # admit genuinely heavy computation — that would teach "skip the tool even
    # when it's needed". If both tiers together fall short of N_MATH, we ship
    # fewer math negatives rather than dilute the boundary.
    rd_unique = [ex for ex in rd if norm_q(ex) not in tool_use_qs]
    tier0 = det_sort([ex for ex in rd_unique if math_tier(ex) == 0], "reasoning_distill")
    tier1 = det_sort([ex for ex in rd_unique if math_tier(ex) == 1], "reasoning_distill")
    print(f"[math] rd={len(rd):,} unique-vs-tool_use={len(rd_unique):,} "
          f"tier0={len(tier0):,} tier1={len(tier1):,}")
    math_pool = (tier0 + tier1)[:N_MATH]
    if len(math_pool) < N_MATH:
        print(f"[math] tiers exhausted at {len(math_pool):,} < {N_MATH:,} — "
              f"shipping fewer math negatives (by design)")
    math = [relabel(ex, "reasoning_distill") for ex in math_pool]

    # --- prose negatives (dedupe on question globally: dolly repeats a few
    # questions, and nothing may collide with a math negative either) ---
    used_qs = {norm_q(ex) for ex in math_pool}
    prose = []
    for donor_name, rows in (("dolly", dolly), ("no_robots", no_robots)):
        ok = []
        for ex in det_sort([ex for ex in rows if is_toolfree_prose(ex)], donor_name):
            q = norm_q(ex)
            if q in used_qs:
                continue
            used_qs.add(q)
            ok.append(ex)
        if len(ok) < N_PROSE_PER_DONOR:
            raise SystemExit(f"[{donor_name}] only {len(ok):,} tool-free prose "
                             f"donors < {N_PROSE_PER_DONOR} — filters too strict")
        prose += [relabel(ex, donor_name) for ex in ok[:N_PROSE_PER_DONOR]]

    # --- v3: synthesized trivial arithmetic (answer directly under the tool
    # prompt). Deterministic; verified by construction; only 1-2 digit operands
    # and sub-3-digit results, so "non-trivial calculation" never applies.
    import random
    rng = random.Random(SYNTH_SEED)
    synth = []
    while len(synth) < N_SYNTH:
        op = rng.choice("+-*")
        if op == "*":
            a, b = rng.randint(2, 12), rng.randint(2, 9)
        else:
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            if op == "-" and b > a:
                a, b = b, a
        val = {"+": a + b, "-": a - b, "*": a * b}[op]
        opw = {"+": rng.choice(["plus", "+", "added to"]),
               "-": rng.choice(["minus", "-"]),
               "*": rng.choice(["times", "*", "multiplied by"])}[op]
        q = rng.choice([f"What is {a} {opw} {b}?", f"Compute {a} {opw} {b}.",
                        f"{a} {opw} {b} = ?"])
        nq = " ".join(q.split()).casefold()
        if nq in used_qs or nq in tool_use_qs:
            continue
        used_qs.add(nq)
        sym = {"plus": "+", "added to": "+", "minus": "-", "times": "×",
               "multiplied by": "×", "+": "+", "-": "-", "*": "×"}[opw]
        synth.append({"id": f"neg_synth_{hashlib.sha256(nq.encode()).hexdigest()[:16]}",
                      "donor": "synthetic",
                      "system": CANONICAL_TOOL_SYSTEM,
                      "messages": [{"role": "user", "content": q},
                                   {"role": "assistant",
                                    "content": f"{a} {sym} {b} = {val}.\n\nThe answer is: {val}"}]})
    print(f"[synth] {len(synth):,} trivial-arithmetic negatives")

    out_rows = math + prose + synth

    # --- fail-loud invariants ---
    seen_ids = set()
    for r in out_rows:
        if r["id"] in seen_ids:
            raise SystemExit(f"duplicate id {r['id']}")
        seen_ids.add(r["id"])
        if norm_q(r) in tool_use_qs:
            raise SystemExit(f"{r['id']} overlaps a tool_use question")
        for m in r["messages"]:
            if "tool_call" in m or m["role"] == "tool":
                raise SystemExit(f"{r['id']} contains a tool call/result — not a negative")
            if m["role"] not in ("user", "assistant") or not m["content"].strip():
                raise SystemExit(f"{r['id']} bad turn: role={m['role']!r}")
        if r["messages"][0]["role"] != "user" or r["messages"][-1]["role"] != "assistant":
            raise SystemExit(f"{r['id']} doesn't start user / end assistant")

    out_dir = os.path.join(base, "tool_negative")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "tool_negative_raw.jsonl")
    tmp = raw_path + ".tmp"
    with open(tmp, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, raw_path)

    meta = {
        "dataset": "tool_negative",
        "hf_path": "derived: reasoning_distill + dolly + no_robots (relabeled, no generation)",
        "built_by": "sft/build_tool_negative.py",
        "system_prompt": "CANONICAL_TOOL_SYSTEM (tools_runtime.py)",
        "kept": len(out_rows),
        "by_donor": {"reasoning_distill": len(math),
                     "dolly": N_PROSE_PER_DONOR, "no_robots": N_PROSE_PER_DONOR,
                     "synthetic": len(synth)},
        "filters": {
            "math": f"question not in tool_use; tier0 ints <= {MAX_DIGITS} "
                    f"digits then tier1 <= {MAX_DIGITS + 1}; no decimals; "
                    f"has answer line; sha256(id) order",
            "prose": "empty donor system; no compute keywords; < 3 numbers "
                     "in user turns; sha256(id) order",
        },
        "donor_pool_stats": {
            "rd_total": len(rd), "rd_not_in_tool_use": len(rd_unique),
            "rd_tier0": len(tier0), "rd_tier1": len(tier1),
            "dolly_toolfree": sum(is_toolfree_prose(ex) for ex in dolly),
            "no_robots_toolfree": sum(is_toolfree_prose(ex) for ex in no_robots),
        },
    }
    with open(os.path.join(out_dir, "meta_raw.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta["donor_pool_stats"], indent=2))
    print(f"wrote {len(out_rows):,} records "
          f"(math={len(math):,}, prose={len(prose):,}) -> {raw_path}")


if __name__ == "__main__":
    main()
