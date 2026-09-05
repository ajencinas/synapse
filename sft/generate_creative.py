#!/usr/bin/env python3
"""Custom creative-writing SFT generator (teacher + LLM-judge).

Storytelling has NO verifiable gold, so we replace answer-verification with an
LLM-JUDGE: the teacher (DeepSeek V4 Flash) writes a SHORT, COMPLETE piece from a
programmatic prompt; a judge call scores it 1-10; we keep only high-scoring,
budget-fitting pieces. Output is the `creative` SFT source ({system, messages}).

Design choices for a 2B with a 2048 ceiling and weak narrative pretraining:
  - prompts ask for SHORT complete pieces (~250 words) so they fit 2048.
  - high judge threshold (default 8) — a 2B imitates exactly, so only clean,
    polished exemplars should make it in.
  - system="" so the expressive style generalizes to ordinary answers (the user
    instruction carries the creative ask), matching the other sources.

Caveat: the teacher judging itself is somewhat lenient; the high threshold + length
discipline compensate. Validate output with tests/test_sft_source_validation.py.

Usage:
  SYNAPSE_DIR=/path python sft/generate_creative.py --limit 200   # calibrate
  SYNAPSE_DIR=/path python sft/generate_creative.py               # full
Env: DEEPSEEK_API_KEY or OPENROUTER_API_KEY.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_reasoning_distill as grd
import tools_runtime as tr

FORMS = ["short story", "piece of flash fiction", "fable", "quiet scene",
         "dramatic monologue", "vignette", "short letter", "fairy tale"]
GENRES = ["science fiction", "fantasy", "mystery", "literary fiction",
          "historical fiction", "magical realism", "adventure", "slice of life",
          "gentle horror", "comedy", "fairy tale", "noir"]
THEMES = [
    "a lighthouse keeper's last night on duty", "a city where it never stops raining",
    "two strangers sharing an umbrella", "a clockmaker who can pause time",
    "the last library on Earth", "a letter that arrives fifty years late",
    "a child who collects lost sounds", "the morning after the machines woke up",
    "a garden that only grows at midnight", "a ferryman who has forgotten the far shore",
    "a recipe handed down through a haunted kitchen", "the day the moon went missing",
    "a cartographer mapping a country that no longer exists", "an old dog waiting at a station",
    "a violinist playing to an empty theater", "a town that votes on the weather",
    "the keeper of a museum of small regrets", "two rival bakers and one last winter",
    "a robot learning to grieve", "a door that opens onto yesterday",
    "a fisherman who catches a single word each day", "the last phone booth in the world",
    "a girl who trades her shadow for a song", "a soldier coming home to a changed orchard",
    "a bookshop cat who guards a secret",
]
STYLES = ["", "", "", "with a quiet twist at the end", "in warm, nostalgic prose",
          "told mostly through dialogue", "in the second person", "with a hopeful ending",
          "in spare, understated language"]

STORY_SYSTEM = ""   # stored in the record; kept empty so style generalizes
WRITE_INSTRUCTION = (
    "You are a gifted storyteller. Write a vivid, emotionally resonant, COMPLETE "
    "piece with a clear beginning, middle, and end. Keep it under ~250 words. "
    "Output only the piece — no title, no preamble, no commentary.")

JUDGE_SYSTEM = (
    "You are a discerning literary editor. Rate the piece 1-10 on vividness, "
    "coherence, emotional engagement, completeness, and how well it fulfills the "
    "request. 8+ means polished, complete, publishable-quality. Reply with EXACTLY "
    "one line: 'Score: N'.")


def build_prompts(n, seed=42):
    rng = random.Random(seed)
    combos = []
    for form in FORMS:
        for genre in GENRES:
            for theme in THEMES:
                combos.append((form, genre, theme))
    rng.shuffle(combos)
    out, seen = [], set()
    for form, genre, theme in combos:
        style = rng.choice(STYLES)
        prompt = f"Write a {form} of {genre} about {theme}"
        if style:
            prompt += f", {style}"
        prompt += "."
        pid = "creative_" + hashlib.sha1(prompt.lower().encode()).hexdigest()[:12]
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"id": pid, "prompt": prompt, "form": form, "genre": genre})
        if n and len(out) >= n:
            break
    return out


def parse_score(text):
    """Pull an integer 1-10 from a judge reply ('Score: 8'). None if unparseable."""
    if not text:
        return None
    m = re.search(r"score\s*[:=]?\s*(\d+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b(10|[1-9])\b", text)
    if not m:
        return None
    v = int(m.group(1))
    return v if 1 <= v <= 10 else None


def _call(client, model, system, user, temperature, max_tokens, retries=4):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            u = getattr(resp, "usage", None)
            it = (getattr(u, "prompt_tokens", 0) or 0) if u else 0
            ot = (getattr(u, "completion_tokens", 0) or 0) if u else 0
            return (resp.choices[0].message.content or "").strip(), it, ot
        except Exception:
            if attempt == retries - 1:
                return None, 0, 0
            time.sleep(2 ** attempt)
    return None, 0, 0


def make_record(client, model, problem, args):
    """Write -> judge -> keep. Returns (record_or_None, transient, in_tok, out_tok,
    status). status: ok|low_score|too_long|short|judge_fail|api."""
    in_tok = out_tok = 0
    story, it, ot = _call(client, model, WRITE_INSTRUCTION, problem["prompt"],
                          temperature=0.9, max_tokens=args.max_output_tok)
    in_tok += it
    out_tok += ot
    if story is None:
        return None, (it == 0 and ot == 0), in_tok, out_tok, "api"
    # length: prompt + story + markers must fit the SFT budget
    total = tr.synapse_token_len(problem["prompt"]) + tr.synapse_token_len(story) + 5
    if tr.synapse_token_len(story) < 20:
        return None, False, in_tok, out_tok, "short"
    if total > args.trace_max_tok:
        return None, False, in_tok, out_tok, "too_long"
    # judge — DeepSeek V4 Flash is a reasoning model: it spends tokens on hidden
    # reasoning before emitting "Score: N", so max_tokens must be generous or the
    # final content comes back EMPTY (the reasoning isn't the answer). 16 was far
    # too low (every judge call failed). Give it room.
    judge_user = f"Request: {problem['prompt']}\n\nPiece:\n{story}"
    verdict, it, ot = _call(client, model, JUDGE_SYSTEM, judge_user,
                            temperature=0.0, max_tokens=args.judge_max_tok)
    in_tok += it
    out_tok += ot
    score = parse_score(verdict)
    if score is None:
        return None, False, in_tok, out_tok, "judge_fail"
    if score < args.min_score:
        return None, False, in_tok, out_tok, "low_score"
    rec = {"id": problem["id"], "system": STORY_SYSTEM, "source": "creative",
           "score": score, "form": problem["form"], "genre": problem["genre"],
           "messages": [{"role": "user", "content": problem["prompt"]},
                        {"role": "assistant", "content": story}]}
    return rec, False, in_tok, out_tok, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-score", type=int, default=8)
    ap.add_argument("--max-output-tok", type=int, default=2048,
                    help="teacher budget: reasoning + story (reasoning not stored)")
    ap.add_argument("--judge-max-tok", type=int, default=1024,
                    help="judge budget: reasoning + 'Score: N' (reasoning model needs room)")
    ap.add_argument("--trace-max-tok", type=int, default=1984)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--provider", default="auto", choices=["auto", "openrouter", "deepseek"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--synapse-dir", default=os.environ.get("SYNAPSE_DIR") or grd.default_synapse_dir())
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    except ImportError:
        pass

    provider = args.provider
    if provider == "auto":
        provider = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else (
            "openrouter" if os.environ.get("OPENROUTER_API_KEY") else None)
        if provider is None:
            raise SystemExit("no API key — set DEEPSEEK_API_KEY or OPENROUTER_API_KEY")
    cfg = grd.PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"{cfg['key_env']} not set for provider={provider}")
    model = args.model or cfg["model"]
    print(f"[provider] {provider} model={model} min_score={args.min_score}")

    out_dir = os.path.join(args.synapse_dir, "datasets_sft", "creative")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "creative_raw.jsonl")
    prog_path = os.path.join(out_dir, "progress.log")
    meta_path = os.path.join(out_dir, "meta_raw.json")

    done = set()
    for p in (prog_path, raw_path):
        if os.path.exists(p):
            with open(p) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            done.add(json.loads(ln).get("id") if p == raw_path else ln)
                        except json.JSONDecodeError:
                            pass
    done.discard(None)
    if done:
        print(f"[resume] {len(done):,} already processed")

    pool = build_prompts(args.limit if args.limit else 0)
    full = len(build_prompts(0))
    pool = [p for p in pool if p["id"] not in done]
    print(f"[run] {len(pool):,} prompts (of {full:,} unique combos) workers={args.workers}")

    client = grd.make_client(api_key, cfg["base_url"])
    lock = threading.Lock()
    stop = threading.Event()
    stats = {"processed": 0, "kept": 0, "in_tok": 0, "out_tok": 0, "reasons": {}}
    raw_f = open(raw_path, "a")
    prog_f = open(prog_path, "a")
    t0 = time.time()

    def handle(problem):
        if stop.is_set():
            return
        rec, transient, it, ot, status = make_record(client, model, problem, args)
        with lock:
            stats["in_tok"] += it
            stats["out_tok"] += ot
            if args.budget_usd:
                cur = stats["in_tok"] * grd.PRICE_IN_PER_TOK + stats["out_tok"] * grd.PRICE_OUT_PER_TOK
                if cur >= args.budget_usd and not stop.is_set():
                    stop.set()
                    print(f"  [budget] ${cur:.2f} ≥ ${args.budget_usd:.0f} — stopping (resumable)")
            if transient:
                return
            stats["reasons"][status] = stats["reasons"].get(status, 0) + 1
            if rec:
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw_f.flush()
                stats["kept"] += 1
            prog_f.write(problem["id"] + "\n")
            prog_f.flush()
            done.add(problem["id"])
            stats["processed"] += 1
            if stats["processed"] % 100 == 0:
                _progress(stats, t0)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(handle, p) for p in pool]):
            pass
    raw_f.close()
    prog_f.close()
    _progress(stats, t0)

    cost = stats["in_tok"] * grd.PRICE_IN_PER_TOK + stats["out_tok"] * grd.PRICE_OUT_PER_TOK
    yield_pct = 100 * stats["kept"] / max(1, stats["processed"])
    json.dump({"source": "creative", "model": model, "min_score": args.min_score,
               "processed": stats["processed"], "kept": stats["kept"],
               "yield_pct": round(yield_pct, 1), "cost_usd": round(cost, 2),
               "in_tok": stats["in_tok"], "out_tok": stats["out_tok"],
               "reject_reasons": stats["reasons"]}, open(meta_path, "w"), indent=2)
    print(f"\n[done] kept {stats['kept']:,}/{stats['processed']:,} ({yield_pct:.1f}%) cost=${cost:.2f}")
    print(f"[reasons] {dict(sorted(stats['reasons'].items(), key=lambda kv: -kv[1]))}")


def _progress(stats, t0):
    dt = time.time() - t0
    cost = stats["in_tok"] * grd.PRICE_IN_PER_TOK + stats["out_tok"] * grd.PRICE_OUT_PER_TOK
    top = dict(sorted(stats.get("reasons", {}).items(), key=lambda kv: -kv[1])[:4])
    print(f"  processed={stats['processed']:,} kept={stats['kept']:,} "
          f"({100*stats['kept']/max(1,stats['processed']):.0f}%) "
          f"{stats['processed']/max(1e-9,dt):.1f}/s cost=${cost:.2f} drops={top}")


if __name__ == "__main__":
    main()
