#!/usr/bin/env python3
"""Build the `format_following` SFT source — SFT v3 Phase C (zero API cost).

sft_bench v2: instruction_following 1/4, text_manipulation 0/3 — the model cannot
"reply with only PONG", "count 1 to 5 separated by spaces", or "reverse 'stop'".
No training source contains exact-format instructions. This builder GENERATES them
programmatically, so every answer is correct by construction AND re-verified by an
independent checker per task type (generation bug -> loud crash, never bad data).

Records are plain prose SFT: {"id", "system": "", "messages": [user, assistant]}.
Empty system = the trained no-tools distribution. Deterministic (seeded), deduped
on the normalized question, sha256-sorted like every derived source.

Usage:
  SYNAPSE_DIR=~/synapse_data python sft/build_format_following.py
Writes: $SYNAPSE_DIR/datasets_sft/format_following/{format_following_raw.jsonl,meta_raw.json}
"""
import argparse
import hashlib
import json
import os
import random

N_TOTAL = 5000
PER_TASK_CAP = 0.20     # no task type may exceed this share (dedupe drains small
                        # pools; without the cap t_yesno filled 84% of the file)
SEED = 7

_BASE_WORDS = ("stop spark cloud river stone light music dream horse tiger apple grape lemon "
         "mango peach bread water glass chair table plant piano robot cabin field maple "
         "ocean planet silver copper garden window candle basket rocket circle square "
         "orange purple yellow winter summer spring autumn monday friday sunday north "
         "south spoon knife plate brick paper pencil letter number market castle bridge "
         "tunnel forest desert island valley meadow shadow thunder breeze pebble "
         "anchor bottle camera dragon engine feather guitar hammer icicle jacket kettle "
         "ladder magnet needle otter parrot quartz rabbit saddle turtle umbrella violet "
         "wagon xylophone yogurt zephyr acorn badger canyon dolphin ember falcon glacier "
         "harbor ivory jungle kayak lantern mirror nectar orchid puzzle quiver ribbon "
         "sunset timber urchin velvet walnut yonder zeal almond beacon cactus dagger "
         "eagle fabric gadget hazel iguana jigsaw karma lagoon mantle nickel onion "
         "pepper quill raven socket toffee unicorn vessel willow zipper amber blaze "
         "coral drift ferry grove haven inlet jewel knoll lodge marsh nook oasis "
         "pond quay reef slope trail vault wharf bluff creek dune fjord").split()
WORDS = _BASE_WORDS + [w + "s" for w in _BASE_WORDS if not w.endswith("s")]
CATEGORIES = {
    "colors": "red blue green yellow purple orange pink brown black white".split(),
    "animals": "dog cat horse cow sheep goat lion tiger bear wolf".split(),
    "fruits": "apple banana orange grape mango peach pear plum cherry kiwi".split(),
    "planets": "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune".split(),
    "weekdays": "Monday Tuesday Wednesday Thursday Friday".split(),
    "months": "January February March April May June July August September October".split(),
    "instruments": "piano guitar violin drums flute trumpet cello harp oboe banjo".split(),
    "vegetables": "carrot potato onion tomato cucumber spinach broccoli pepper corn pea".split(),
}
_SUBJ = ["The quick brown fox", "A gentle breeze", "Seven tired students", "The old lighthouse",
         "A silver river", "The night train", "One curious child", "The village baker",
         "A distant thunderstorm", "The patient gardener", "A paper lantern", "The clever otter"]
_VERB = ["jumps over", "drifted past", "watched", "guided", "circled", "followed",
         "painted", "carried", "crossed", "discovered"]
_OBJ = ["the lazy dog", "the quiet valley", "the crowded market", "a narrow bridge",
        "the stormy harbor", "an empty field", "the winding path", "a wooden cabin"]
SENTENCES = [f"{a} {v} {o}" for a in _SUBJ for v in _VERB for o in _OBJ][::7][:120]


# Each task: build(rng) -> (question, answer); check(question_args, answer) -> bool.
# The checker recomputes the truth independently of how build produced it.

def t_echo(rng):
    w = rng.choice(WORDS).upper() if rng.random() < 0.5 else rng.choice(WORDS)
    q = rng.choice([f"Reply with only the word '{w}' and nothing else.",
                    f"Respond with exactly the word '{w}' — nothing more."])
    return q, w, lambda a: a == w

def t_count(rng):
    a0 = rng.randint(1, 60); b = a0 + rng.randint(2, 8)
    sep_name, sep = rng.choice([("spaces", " "), ("commas", ", ")])
    q = f"Count from {a0} to {b}, separated by {sep_name}."
    ans = sep.join(str(i) for i in range(a0, b + 1))
    return q, ans, lambda a: [x.strip() for x in a.replace(",", " ").split()] == [str(i) for i in range(a0, b + 1)]

def t_list_exact(rng):
    cat = rng.choice(list(CATEGORIES)); pool = CATEGORIES[cat]
    n = rng.randint(2, min(5, len(pool)))
    items = rng.sample(pool, n)
    q = f"List exactly {n} {cat}, separated by commas."
    ans = ", ".join(items)
    return q, ans, lambda a: len([x for x in a.split(",") if x.strip()]) == n

def t_case(rng):
    w = rng.choice(WORDS); up = rng.random() < 0.5
    q = f"Convert the word '{w}' to {'uppercase' if up else 'lowercase'}."
    ans = w.upper() if up else w.lower()
    return q, ans, lambda a: a == ans

def t_reverse(rng):
    w = rng.choice(WORDS)
    q = f"Reverse the word '{w}'."
    ans = w[::-1]
    return q, ans, lambda a: a == w[::-1]

def t_letter_count(rng):
    w = rng.choice(WORDS)
    q = f"How many letters are in the word '{w}'?"
    ans = str(len(w))
    return q, ans, lambda a: a == str(len(w))

def t_first_last(rng):
    w = rng.choice(WORDS); first = rng.random() < 0.5
    q = f"What is the {'first' if first else 'last'} letter of the word '{w}'?"
    ans = w[0] if first else w[-1]
    return q, ans, lambda a: a == ans

def t_nth_word(rng):
    s = rng.choice(SENTENCES); words = s.split()
    i = rng.randint(1, len(words))
    ordn = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}[i]
    q = f"What is the {ordn} word of this sentence: \"{s}\"?"
    ans = words[i - 1]
    return q, ans, lambda a: a == words[i - 1]

def t_word_count(rng):
    s = rng.choice(SENTENCES)
    q = f"How many words are in this sentence: \"{s}\"?"
    ans = str(len(s.split()))
    return q, ans, lambda a: a == str(len(s.split()))

def t_yesno(rng):
    x, y = rng.randint(1, 99), rng.randint(1, 99)
    while x == y:
        y = rng.randint(1, 99)
    gt = rng.random() < 0.5
    q = f"Answer with only 'yes' or 'no': Is {x} {'greater' if gt else 'less'} than {y}?"
    ans = "yes" if ((x > y) == gt) else "no"
    return q, ans, lambda a: a == ans

def t_one_per_line(rng):
    cat = rng.choice(list(CATEGORIES)); pool = CATEGORIES[cat]
    n = rng.randint(2, 4)
    items = rng.sample(pool, n)
    q = f"List exactly {n} {cat}, one per line."
    ans = "\n".join(items)
    return q, ans, lambda a: len([x for x in a.split("\n") if x.strip()]) == n

TASKS = [t_echo, t_count, t_list_exact, t_case, t_reverse, t_letter_count,
         t_first_last, t_nth_word, t_word_count, t_yesno, t_one_per_line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR"))
    ap.add_argument("-n", type=int, default=N_TOTAL)
    args = ap.parse_args()
    if not args.synapse_dir:
        raise SystemExit("set SYNAPSE_DIR (refusing to guess — ./synapse is the venv)")

    rng = random.Random(SEED)
    rows, seen = [], set()
    per_task = {}
    attempts = 0
    while len(rows) < args.n:
        attempts += 1
        if attempts > args.n * 50:
            raise SystemExit(f"exhausted variation at {len(rows)} records — enlarge WORDS/CATEGORIES")
        task = rng.choice(TASKS)
        if per_task.get(task.__name__, 0) >= int(PER_TASK_CAP * args.n):
            continue
        q, ans, check = task(rng)
        if not check(ans):                       # independent re-verification
            raise SystemExit(f"builder bug: {task.__name__} produced a wrong answer for {q!r}")
        key = " ".join(q.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        per_task[task.__name__] = per_task.get(task.__name__, 0) + 1
        rows.append({"id": f"fmt_{task.__name__[2:]}_{hashlib.sha256(key.encode()).hexdigest()[:12]}",
                     "system": "", "messages": [{"role": "user", "content": q},
                                                {"role": "assistant", "content": ans}]})

    rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    # fail-loud invariants (mirror build_tool_negative)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids")
    for r in rows:
        if r["messages"][0]["role"] != "user" or r["messages"][-1]["role"] != "assistant":
            raise SystemExit(f"{r['id']} bad turn structure")
        if not r["messages"][-1]["content"].strip():
            raise SystemExit(f"{r['id']} empty answer")

    out_dir = os.path.join(args.synapse_dir, "datasets_sft", "format_following")
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, "format_following_raw.jsonl")
    tmp = raw + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, raw)
    meta = {"dataset": "format_following", "built_by": "sft/build_format_following.py",
            "hf_path": "synthesized: programmatic generation + independent per-task checkers",
            "kept": len(rows), "seed": SEED, "by_task": per_task}
    with open(os.path.join(out_dir, "meta_raw.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["by_task"], indent=None))
    print(f"wrote {len(rows):,} records -> {raw}")


if __name__ == "__main__":
    main()
