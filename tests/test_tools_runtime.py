"""Thorough tests for sft/tools_runtime.py (the shared tool runtime).

Run from repo root:  python -m unittest tests.test_tools_runtime  (or -v)

Covers: canonical serializer, python sandbox (success/error/timeout/network-block/
limits), rate limiter (timing + aggregate concurrency bound), tokenizer counting +
truncation, the 7/8 token regression guard, and run_search retry/parse logic via a
mocked HTTP layer. A live Brave call runs only if RUN_LIVE_SEARCH=1.
"""
import hashlib
import os
import sys
import threading
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sft"))

CANONICAL_FP = "7a570a7ba9fc7985"   # the tokenization_id the pipeline enforces


def _fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def _find_canonical_tokenizer():
    """Locate a tokenizer.json whose fingerprint matches the canonical one, so the
    7/8 id assertions are meaningful (the stale trainingdata copy must be ignored)."""
    candidates = []
    if os.environ.get("SYNAPSE_TOKENIZER"):
        candidates.append(os.environ["SYNAPSE_TOKENIZER"])
    syn = os.environ.get("SYNAPSE_DIR")
    if syn:
        candidates.append(os.path.join(syn, "tokenizer_out", "tokenizer.json"))
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(home, "Downloads", "tokenizer_out", "tokenizer.json"),
        os.path.join(home, "Python 2025", "trainingdata", "tokenizer_out", "tokenizer.json"),
    ]
    for p in candidates:
        try:
            if os.path.exists(p) and _fingerprint(p) == CANONICAL_FP:
                return p
        except OSError:
            continue
    return None


# Point the runtime at the canonical tokenizer BEFORE first use, if we can find it.
_CANON = _find_canonical_tokenizer()
if _CANON:
    os.environ["SYNAPSE_TOKENIZER"] = _CANON

import tools_runtime as tr  # noqa: E402

HAVE_TOK = _CANON is not None
skip_no_tok = unittest.skipUnless(HAVE_TOK, "canonical tokenizer not found locally")


# ---------------------------------------------------------------------------
class TestSerializer(unittest.TestCase):
    def test_key_order_stable_and_compact(self):
        a = tr.dump_tool_call({"tool": "python", "code": "print(1)"})
        b = tr.dump_tool_call({"code": "print(1)", "tool": "python"})
        self.assertEqual(a, b)                       # input order irrelevant
        self.assertEqual(a, '{"code":"print(1)","tool":"python"}')  # sorted, no spaces
        self.assertNotIn(", ", a)
        self.assertNotIn(": ", a)

    def test_unicode_literal(self):
        s = tr.dump_tool_call({"tool": "search", "query": "café ∑ π"})
        self.assertIn("café", s)
        self.assertNotIn("\\u", s)


# ---------------------------------------------------------------------------
class TestRunPython(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(tr.run_python("print(2847*391)"), "1113177")

    def test_sympy_available(self):
        out = tr.run_python("import sympy; print(sympy.simplify('(x**2-1)/(x-1)'))")
        self.assertEqual(out, "x + 1")

    def test_multiline_and_loop(self):
        out = tr.run_python("t=0\nfor i in range(5): t+=i\nprint(t)")
        self.assertEqual(out, "10")

    def test_error_surfaced_not_raised(self):
        out = tr.run_python("1/0")
        self.assertIn("[error]", out)
        self.assertIn("ZeroDivisionError", out)

    def test_no_output_marker(self):
        self.assertEqual(tr.run_python("x = 1 + 1"), "[no output]")   # assignment, no echo

    def test_repl_echoes_last_expression(self):
        self.assertEqual(tr.run_python("56/7"), "8.0")                # bare expr -> echoed
        self.assertEqual(tr.run_python("(1+2, 3*4)"), "(3, 12)")
        self.assertEqual(tr.run_python("x=5\nx*2"), "10")             # stmts then bare expr

    def test_explicit_print_not_double_echoed(self):
        self.assertEqual(tr.run_python("print(56/7)"), "8.0")         # print -> None, no dup

    def test_timeout(self):
        t0 = time.monotonic()
        out = tr.run_python("while True: pass", timeout=2)
        dt = time.monotonic() - t0
        self.assertIn("timed out", out)
        self.assertLess(dt, 6, "timeout did not kill the process promptly")

    def test_network_blocked(self):
        out = tr.run_python(
            "import urllib.request as u; u.urlopen('http://example.com', timeout=3)")
        self.assertIn("[error]", out)
        # socket is monkeypatched to raise OSError before any real connection
        self.assertTrue("network disabled" in out or "OSError" in out or "URLError" in out)

    def test_secrets_not_in_child_env(self):
        os.environ["SECRET_CANARY_XYZ"] = "leak-me"
        try:
            out = tr.run_python("import os; print(os.environ.get('SECRET_CANARY_XYZ'))")
            self.assertEqual(out, "None")
        finally:
            del os.environ["SECRET_CANARY_XYZ"]

    def test_stdout_captured_in_order(self):
        out = tr.run_python("print('a'); print('b')")
        self.assertEqual(out, "a\nb")


# ---------------------------------------------------------------------------
class TestRateLimiter(unittest.TestCase):
    def test_serial_spacing(self):
        rl = tr.RateLimiter(rate=20)            # 50ms apart
        t0 = time.monotonic()
        for _ in range(5):
            rl.acquire()
        dt = time.monotonic() - t0
        self.assertGreaterEqual(dt, 4 * 0.05 - 0.01)   # >= 4 gaps

    def test_aggregate_bound_under_threads(self):
        rate = 50
        rl = tr.RateLimiter(rate=rate)
        n = 20
        stamps = []
        lock = threading.Lock()

        def worker():
            rl.acquire()
            with lock:
                stamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(n)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        span = max(stamps) - t0
        # n requests at `rate`/s must take >= (n-1)/rate regardless of thread count
        self.assertGreaterEqual(span, (n - 1) / rate - 0.02)

    def test_rejects_bad_rate(self):
        with self.assertRaises(ValueError):
            tr.RateLimiter(0)


# ---------------------------------------------------------------------------
@skip_no_tok
class TestTokenizer(unittest.TestCase):
    def test_len_positive(self):
        self.assertGreater(tr.synapse_token_len("hello world"), 0)
        self.assertEqual(tr.synapse_token_len(""), 0)

    def test_digit_per_token(self):
        # digit-per-token: a 5-digit number costs ~5 tokens, far more than a short word
        self.assertGreaterEqual(tr.synapse_token_len("12345"), 5)

    def test_truncate_keeps_budget(self):
        text = " ".join(str(i) for i in range(500))    # long, digit-heavy
        out = tr.truncate_tokens(text, 32)
        self.assertIn("…[truncated]", out)
        # the kept prefix (minus the marker) is <= budget
        prefix = out.replace(" …[truncated]", "")
        self.assertLessEqual(tr.synapse_token_len(prefix), 32)

    def test_truncate_noop_when_short(self):
        self.assertEqual(tr.truncate_tokens("short", 1000), "short")

    def test_verify_tool_tokens_passes(self):
        self.assertTrue(tr.verify_tool_tokens())


# ---------------------------------------------------------------------------
class _FakeTok:
    """Minimal tokenizer stand-in for verify_tool_tokens negative tests."""
    def __init__(self, ids, atomic=True):
        self._ids = ids
        self._atomic = atomic

    def token_to_id(self, name):
        return self._ids.get(name)

    def encode(self, name, add_special_tokens=False):
        class E:
            pass
        e = E()
        e.ids = [self._ids.get(name, 0)] if self._atomic else [0, 0]
        return e


class TestVerifyTokensNegative(unittest.TestCase):
    def test_wrong_id_raises(self):
        bad = _FakeTok({"<|tool_call|>": 99, "<|tool_result|>": 8})
        with self.assertRaises(SystemExit):
            tr.verify_tool_tokens(bad)

    def test_non_atomic_raises(self):
        bad = _FakeTok({"<|tool_call|>": 7, "<|tool_result|>": 8}, atomic=False)
        with self.assertRaises(SystemExit):
            tr.verify_tool_tokens(bad)


# ---------------------------------------------------------------------------
class TestRunSearch(unittest.TestCase):
    """run_search logic with a mocked HTTP layer (no network, no quota)."""

    def setUp(self):
        self._orig_get = tr._http_get
        self._orig_backoff = tr._backoff
        tr._backoff = lambda attempt: None          # no real sleeping in tests
        self.fast = tr.RateLimiter(rate=10000)

    def tearDown(self):
        tr._http_get = self._orig_get
        tr._backoff = self._orig_backoff

    def _set_responses(self, responses):
        """responses: list of (status, body) or an Exception to raise, served in order;
        last one repeats."""
        seq = list(responses)
        calls = {"n": 0}

        def fake(url, headers, params, timeout):
            i = min(calls["n"], len(seq) - 1)
            calls["n"] += 1
            item = seq[i]
            if isinstance(item, Exception):
                raise item
            return item
        tr._http_get = fake
        return calls

    def _body(self, results):
        import json
        return json.dumps({"web": {"results": results}})

    def test_success_parse(self):
        body = self._body([
            {"title": "Related rates", "description": "derivative method"},
            {"title": "Calc", "description": "chain rule"},
        ])
        self._set_responses([(200, body)])
        out = tr.run_search("x", api_key="k", limiter=self.fast)
        self.assertEqual(out, "Related rates: derivative method | Calc: chain rule")

    def test_empty_results(self):
        self._set_responses([(200, self._body([]))])
        out = tr.run_search("x", api_key="k", limiter=self.fast)
        self.assertEqual(out, tr.NO_RESULTS)

    def test_429_then_success(self):
        body = self._body([{"title": "T", "description": "D"}])
        calls = self._set_responses([(429, ""), (200, body)])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=4)
        self.assertEqual(out, "T: D")
        self.assertEqual(calls["n"], 2)             # retried once

    def test_all_429_exhausts_to_unavailable(self):
        calls = self._set_responses([(429, "")])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=3)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)
        self.assertEqual(calls["n"], 3)

    def test_5xx_retries(self):
        calls = self._set_responses([(503, "")])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=2)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)
        self.assertEqual(calls["n"], 2)

    def test_4xx_gives_up_immediately(self):
        calls = self._set_responses([(401, "bad key")])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=4)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)
        self.assertEqual(calls["n"], 1)             # did NOT burn retries

    def test_transport_error_retries(self):
        import urllib.error
        calls = self._set_responses([urllib.error.URLError("dns")])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=2)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)
        self.assertEqual(calls["n"], 2)

    def test_no_api_key(self):
        # hermetic: no key passed AND none in the environment -> unavailable.
        # (an empty api_key arg intentionally falls back to env in production.)
        import unittest.mock
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAVE_API_KEY", None)
            out = tr.run_search("x", api_key=None, limiter=self.fast)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)

    def test_malformed_json_retries_then_unavailable(self):
        calls = self._set_responses([(200, "not json")])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=2)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)
        self.assertEqual(calls["n"], 2)

    def test_error_envelope_with_http_200_is_not_trusted(self):
        # Brave can return an error envelope even with HTTP 200 — must NOT be passed
        # through as a result.
        import json as _json
        body = _json.dumps({"type": "ErrorResponse", "error": {"code": "RATE_LIMITED"}})
        calls = self._set_responses([(200, body)])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=3)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)
        self.assertEqual(calls["n"], 3)            # treated as transient -> retried

    def test_top_level_error_key_with_200(self):
        import json as _json
        calls = self._set_responses([(200, _json.dumps({"error": "quota exceeded"}))])
        out = tr.run_search("x", api_key="k", limiter=self.fast, retries=2)
        self.assertEqual(out, tr.SEARCH_UNAVAILABLE)
        self.assertEqual(calls["n"], 2)


# ---------------------------------------------------------------------------
@unittest.skipUnless(os.environ.get("RUN_LIVE_SEARCH") == "1" and os.environ.get("BRAVE_API_KEY"),
                     "live search disabled (set RUN_LIVE_SEARCH=1 + BRAVE_API_KEY)")
class TestRunSearchLive(unittest.TestCase):
    def test_live_brave(self):
        out = tr.run_search("related rates calculus method")
        self.assertNotIn(tr.SEARCH_UNAVAILABLE, out)
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
