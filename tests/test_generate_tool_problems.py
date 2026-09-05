"""Tests for sft/generate_tool_problems.py (custom fact-lookup DB builder).

Pure logic + a mocked SPARQL fetch (no network). A live Wikidata pull runs only if
RUN_LIVE_WIKIDATA=1.

Run:  python -m unittest tests.test_generate_tool_problems -v
"""
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sft"))

import generate_tool_problems as gtp  # noqa: E402


def _sparql_json(rows, ent="entity", ans="answer"):
    return json.dumps({"results": {"bindings": [
        {ent: {"value": e}, ans: {"value": a}} for e, a in rows]}})


class TestParseSparql(unittest.TestCase):
    def test_parse(self):
        txt = _sparql_json([("Tungsten", "74"), ("Iron", "26")])
        self.assertEqual(gtp.parse_sparql(txt), [("Tungsten", "74"), ("Iron", "26")])

    def test_skips_missing(self):
        txt = json.dumps({"results": {"bindings": [
            {"entity": {"value": "X"}},                 # no answer
            {"answer": {"value": "5"}},                 # no entity
            {"entity": {"value": "Y"}, "answer": {"value": "9"}},
        ]}})
        self.assertEqual(gtp.parse_sparql(txt), [("Y", "9")])


class TestNormalize(unittest.TestCase):
    def test_int(self):
        self.assertEqual(gtp.normalize_gold("int", "74"), "74")
        self.assertEqual(gtp.normalize_gold("int", "74.0"), "74")
        self.assertEqual(gtp.normalize_gold("int", "  26 "), "26")
        self.assertEqual(gtp.normalize_gold("int", "not-a-number"), "")

    def test_str(self):
        self.assertEqual(gtp.normalize_gold("str", "  Tokyo "), "Tokyo")
        self.assertEqual(gtp.normalize_gold("str", ""), "")


class TestMakeProblem(unittest.TestCase):
    def test_templating_and_fields(self):
        r = gtp.make_problem("element_atomic_number", "Tungsten", "74")
        self.assertEqual(r["question"], "What is the atomic number of the element Tungsten?")
        self.assertEqual(r["gold"], "74")
        self.assertEqual(r["family"], "element_atomic_number")
        self.assertEqual(r["kind"], "facts")
        self.assertTrue(r["id"].startswith("facts_element_atomic_number_"))

    def test_drop_on_bad_answer(self):
        self.assertIsNone(gtp.make_problem("element_atomic_number", "Foo", "xyz"))

    def test_id_is_deterministic_and_question_keyed(self):
        a = gtp.make_problem("country_capital", "Japan", "Tokyo")
        b = gtp.make_problem("country_capital", "Japan", "Kyoto")   # same Q, diff gold
        self.assertEqual(a["id"], b["id"])                          # id keyed on question
        c = gtp.make_problem("country_capital", "France", "Paris")
        self.assertNotEqual(a["id"], c["id"])


class TestFetchAndBuild(unittest.TestCase):
    def test_fetch_family_uses_injected_fetcher(self):
        captured = {}

        def fake_fetch(query, **kw):
            captured["q"] = query
            return _sparql_json([("Iron", "26"), ("Gold", "79")])

        pairs = gtp.fetch_family("element_atomic_number", 50, fetch=fake_fetch)
        self.assertEqual(pairs, [("Iron", "26"), ("Gold", "79")])
        self.assertIn("LIMIT 50", captured["q"])                    # limit appended

    def test_429_then_success_retries(self):
        import urllib.error
        calls = {"n": 0}
        slept = []

        def flaky_fetch(query, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                # huge Retry-After must be IGNORED in favour of the fast backoff
                raise urllib.error.HTTPError("u", 429, "rate", {"Retry-After": "1000"}, None)
            return _sparql_json([("Iron", "26")])

        orig_sleep = gtp.time.sleep
        gtp.time.sleep = lambda s: slept.append(s)
        try:
            pairs = gtp.fetch_family("element_atomic_number", 5, fetch=flaky_fetch, wait=60)
        finally:
            gtp.time.sleep = orig_sleep
        self.assertEqual(pairs, [("Iron", "26")])
        self.assertEqual(calls["n"], 2)
        self.assertEqual(slept, [60])             # ~1min wait, Retry-After:1000 ignored

    def test_non_429_http_error_raises(self):
        import urllib.error

        def bad(query, **kw):
            raise urllib.error.HTTPError("u", 400, "bad", {}, None)

        with self.assertRaises(urllib.error.HTTPError):
            gtp.fetch_family("element_atomic_number", 5, fetch=bad, retries=2)

    def test_build_from_pairs_filters_bad(self):
        rows = gtp.build_from_pairs("element_atomic_number",
                                    [("Iron", "26"), ("Bad", "??"), ("Gold", "79")])
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["gold"] for r in rows}, {"26", "79"})


class TestSeedAndDedup(unittest.TestCase):
    def test_load_seed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"question": "Capital of Japan?", "gold": "Tokyo"}) + "\n")
            f.write(json.dumps({"question": "", "gold": "x"}) + "\n")        # dropped
            f.write(json.dumps({"question": "Atomic number of iron?", "gold": 26}) + "\n")
            path = f.name
        try:
            rows = gtp.load_seed(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["gold"], "Tokyo")
        self.assertEqual(rows[1]["gold"], "26")                     # int coerced to str

    def test_dedup(self):
        a = gtp.make_problem("country_capital", "Japan", "Tokyo")
        dup = dict(a)
        rows = gtp.dedup([a, dup, gtp.make_problem("country_capital", "France", "Paris")])
        self.assertEqual(len(rows), 2)


@unittest.skipUnless(os.environ.get("RUN_LIVE_WIKIDATA") == "1", "live Wikidata disabled")
class TestLiveWikidata(unittest.TestCase):
    def test_atomic_numbers(self):
        pairs = gtp.fetch_family("element_atomic_number", 10)
        self.assertGreater(len(pairs), 0)
        rows = gtp.build_from_pairs("element_atomic_number", pairs)
        self.assertTrue(all(r["gold"].isdigit() for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
