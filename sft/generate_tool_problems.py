#!/usr/bin/env python3
"""Custom tool-REQUIRED problem DB for tool-use SFT.

Builds verifiable (question, gold) pairs whose answer is KNOWN independently of any
tool, so the agentic generator (generate_tool_use.py) can rejection-sample exactly.
This is better than mining HF math/QA sets because (a) every item genuinely needs a
tool -> high tool-positive yield, (b) the gold is canonical -> exact verification,
(c) `search`/`python` get their OWN pool (no overlap with the reasoning sources).

Kind `facts` (for the `search` tool): short canonical facts pulled from **Wikidata**
(unambiguous entities, stable short answers) or a local `--seed` JSONL. The student
must look the fact up and COPY the value out of the result — extraction, not
summarization, which is what a ~2B can actually do. The fact is the gold; at
generation time the teacher really searches and we keep the trace only if its answer
matches the gold.

Pick families that are stable + canonical + obscure-enough that a 2B benefits from
looking them up (ISO codes, atomic numbers, symbols) plus some well-known ones
(capitals) for contrast.

Output: JSONL {id, question, gold, family, kind} -> feed to generate_tool_use.py
(--problems). No network in the seed path; Wikidata path needs network + a UA header.

Usage:
  python sft/generate_tool_problems.py --kind facts --source wikidata \
      --families element_atomic_number,country_iso2,country_capital -n 4000 --out facts.jsonl
  python sft/generate_tool_problems.py --kind facts --source seed --seed my_facts.jsonl --out facts.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "synapse-sft-toolproblems/1.0 (https://github.com/ajencinas/synapse)"

# Each family: a SPARQL that SELECTs ?entity (en label) + ?answer (literal or en
# label), a question template with {e}, and an answer kind for normalization.
# A family = a SPARQL returning ?entity (en label) + ?answer (literal or en label),
# a question template with {e}, and an answer kind. Chosen for diverse domains with
# UNAMBIGUOUS entities + CANONICAL short answers (so answer-verification works).
def _literal(qid_type, prop):   # ?answer is a literal value (codes, numbers, symbols)
    return (f"SELECT ?entity ?answer WHERE {{ ?x wdt:P31 wd:{qid_type} ; wdt:{prop} ?answer ;"
            f" rdfs:label ?entity . FILTER(LANG(?entity)='en') }}")


def _entity(qid_type, prop):    # ?answer is an entity -> take its en label
    return (f"SELECT ?entity ?answer WHERE {{ ?x wdt:P31 wd:{qid_type} ; rdfs:label ?entity ;"
            f" wdt:{prop} ?a . FILTER(LANG(?entity)='en')"
            f" ?a rdfs:label ?answer . FILTER(LANG(?answer)='en') }}")


FAMILIES = {
    # --- chemistry ---
    "element_atomic_number": {"kind": "int", "question": "What is the atomic number of the element {e}?",
                              "sparql": _literal("Q11344", "P1086")},
    "element_symbol": {"kind": "str", "question": "What is the chemical symbol of the element {e}?",
                       "sparql": _literal("Q11344", "P246")},
    # --- geography ---
    "country_capital": {"kind": "str", "question": "What is the capital city of {e}?",
                        "sparql": _entity("Q6256", "P36")},
    "country_continent": {"kind": "str", "question": "On which continent is {e} located?",
                          "sparql": _entity("Q6256", "P30")},
    "us_state_capital": {"kind": "str", "question": "What is the capital of the U.S. state of {e}?",
                         "sparql": _entity("Q35657", "P36")},
    "mountain_country": {"kind": "str", "question": "In which country is the mountain {e}?",
                         "sparql": _entity("Q8502", "P17")},
    # --- codes / economy ---
    "country_iso2": {"kind": "str", "question": "What is the ISO 3166-1 alpha-2 country code for {e}?",
                     "sparql": _literal("Q6256", "P297")},
    "country_iso3": {"kind": "str", "question": "What is the ISO 3166-1 alpha-3 country code for {e}?",
                     "sparql": _literal("Q6256", "P298")},
    "country_calling_code": {"kind": "str", "question": "What is the international calling code for {e}?",
                             "sparql": _literal("Q6256", "P474")},
    "country_currency": {"kind": "str", "question": "What is the official currency of {e}?",
                         "sparql": _entity("Q6256", "P38")},
    # --- culture ---
    "book_author": {"kind": "str", "question": "Who is the author of the book \"{e}\"?",
                    "sparql": _entity("Q7725634", "P50")},
    "film_director": {"kind": "str", "question": "Who directed the film \"{e}\"?",
                      "sparql": _entity("Q11424", "P57")},
    "company_founder": {"kind": "str", "question": "Who founded the company {e}?",
                        "sparql": _entity("Q4830453", "P112")},
    # --- v3 additions (SFT v3 Phase B: broaden beyond the 4 dominant families) ---
    "element_discoverer": {"kind": "str", "question": "Who discovered the chemical element {e}?",
                           "sparql": _entity("Q11344", "P61")},
    "chemical_formula": {"kind": "str", "question": "What is the chemical formula of {e}?",
                         "sparql": _literal("Q11173", "P274")},
    "country_official_language": {"kind": "str", "question": "What is the official language of {e}?",
                                  "sparql": _entity("Q6256", "P37")},
    "river_mouth": {"kind": "str", "question": "Into which body of water does the river {e} flow?",
                    "sparql": _entity("Q4022", "P403")},
    "airport_iata": {"kind": "str", "question": "What is the IATA code of {e}?",
                     "sparql": _literal("Q1248784", "P238")},
    "language_iso639_1": {"kind": "str", "question": "What is the ISO 639-1 code of the {e} language?",
                          "sparql": _literal("Q34770", "P218")},
    "opera_composer": {"kind": "str", "question": "Who composed the opera \"{e}\"?",
                       "sparql": _entity("Q1344", "P86")},
    "building_architect": {"kind": "str", "question": "Who was the architect of {e}?",
                           "sparql": _entity("Q41176", "P84")},
}


# --- normalization ---------------------------------------------------------
def normalize_gold(kind, value):
    """Canonicalize the answer string so verification is robust. Returns '' to drop."""
    v = (value or "").strip()
    if not v:
        return ""
    if kind == "int":
        try:
            return str(int(float(v)))
        except ValueError:
            return ""
    return v


# --- Wikidata (network; the fetch is a module fn so tests can monkeypatch it) ---
def _sparql_get(query, *, endpoint=WIKIDATA_ENDPOINT, timeout=60):
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_sparql(json_text, ent_var="entity", ans_var="answer"):
    """Return [(entity_label, answer_value), ...] from a SPARQL JSON result."""
    data = json.loads(json_text)
    out = []
    for b in data.get("results", {}).get("bindings", []):
        e = b.get(ent_var, {}).get("value")
        a = b.get(ans_var, {}).get("value")
        if e and a is not None:
            out.append((e, a))
    return out


def _with_retry(query, fetch, retries=8, wait=60):
    """Retry on 429/5xx/transport errors. Wikidata throttles to ~1 request/MINUTE
    during WDQS outages, so we IGNORE its (often ~1000s) Retry-After and instead wait
    ~`wait` seconds — long enough to cross into a fresh minute window — escalating
    slightly each try, for `retries` attempts, then give up so the caller can re-run."""
    import urllib.error
    last = None
    for attempt in range(retries):
        try:
            return fetch(query)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise                                    # 4xx (bad query/auth): won't fix itself
            last = e
        except urllib.error.URLError as e:
            last = e
        if attempt == retries - 1:
            raise last
        w = wait + 10 * attempt                          # 60, 70, 80, ... crosses the 1/min window
        print(f"[wikidata] transient ({getattr(last,'code','net')}) — retrying in {w}s "
              f"(attempt {attempt + 1}/{retries})")
        time.sleep(w)


def fetch_family(name, limit, *, fetch=None, retries=8, wait=60):
    """Fetch up to `limit` (entity, raw_answer) pairs for a family via SPARQL."""
    fetch = fetch or _sparql_get
    spec = FAMILIES[name]
    query = spec["sparql"] + f" LIMIT {int(limit)}"
    return parse_sparql(_with_retry(query, fetch, retries, wait))


# --- problem assembly (pure; fully testable) -------------------------------
def _qid(question):
    return hashlib.sha1(question.strip().lower().encode()).hexdigest()[:12]


def make_problem(family, entity, raw_answer):
    """Build one {id, question, gold, family, kind} record, or None if unusable."""
    spec = FAMILIES[family]
    gold = normalize_gold(spec["kind"], raw_answer)
    if not gold:
        return None
    question = spec["question"].format(e=entity.strip())
    return {"id": f"facts_{family}_{_qid(question)}", "question": question,
            "gold": gold, "family": family, "kind": "facts"}


def build_from_pairs(family, pairs):
    rows = []
    for entity, ans in pairs:
        rec = make_problem(family, entity, ans)
        if rec:
            rows.append(rec)
    return rows


def load_seed(path):
    """Seed JSONL: {question, gold, [family]} per line (offline / custom facts)."""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            q, g = d.get("question", "").strip(), str(d.get("gold", "")).strip()
            if not (q and g):
                continue
            rows.append({"id": f"facts_seed_{_qid(q)}", "question": q, "gold": g,
                         "family": d.get("family", "seed"), "kind": "facts"})
    return rows


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="facts", choices=["facts"])
    ap.add_argument("--source", default="wikidata", choices=["wikidata", "seed"])
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma-separated family names (wikidata source)")
    ap.add_argument("--seed", help="seed JSONL path (seed source)")
    ap.add_argument("-n", "--limit", type=int, default=4000, help="total problems to emit")
    ap.add_argument("--per-family-fetch", type=int, default=3000,
                    help="max rows to pull per family from Wikidata before sampling")
    ap.add_argument("--per-family-cap", type=float, default=0.10,
                    help="max share of the EMITTED set any one family may hold "
                         "(v1 lesson: 93%% concentration in 4 families); <=0 disables")
    ap.add_argument("--retries", type=int, default=8, help="retries per family on 429/5xx")
    ap.add_argument("--retry-wait", type=int, default=60,
                    help="base wait (s) between retries; Wikidata throttles ~1 req/min")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-rng", type=int, default=42)
    args = ap.parse_args()

    rows = []
    if args.source == "seed":
        if not args.seed:
            raise SystemExit("--source seed requires --seed <path>")
        rows = load_seed(args.seed)
    else:
        for fam in args.families.split(","):
            fam = fam.strip()
            if fam not in FAMILIES:
                raise SystemExit(f"unknown family {fam!r}; known: {', '.join(FAMILIES)}")
            try:
                pairs = fetch_family(fam, args.per_family_fetch,
                                     retries=args.retries, wait=args.retry_wait)
            except Exception as e:
                print(f"[warn] {fam}: fetch failed ({e}); skipping")
                continue
            fam_rows = build_from_pairs(fam, pairs)
            print(f"[wikidata] {fam}: {len(fam_rows):,} usable")
            rows.extend(fam_rows)

    rows = dedup(rows)
    rng = random.Random(args.seed_rng)
    rng.shuffle(rows)
    if args.limit and args.per_family_cap and args.per_family_cap > 0:
        cap = max(1, int(args.per_family_cap * args.limit))
        taken, out = {}, []
        for r in rows:                       # rows already shuffled -> a fair sample per family
            if taken.get(r["family"], 0) >= cap:
                continue
            taken[r["family"]] = taken.get(r["family"], 0) + 1
            out.append(r)
        dropped = len(rows) - len(out)
        if dropped:
            print(f"[cap] per-family cap {cap} dropped {dropped:,} rows from dominant families")
        rows = out
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_fam = {}
    for r in rows:
        by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
    print(f"[done] wrote {len(rows):,} problems -> {args.out}  by_family={by_fam}")


if __name__ == "__main__":
    main()
