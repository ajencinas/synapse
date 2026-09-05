"""Tests for sft/generate_tool_use.py — the agentic tool-use generator.

Driven by a FakeClient with scripted teacher responses (no API / no spend). Real
run_python is used (fast, deterministic); search is monkeypatched. Covers the happy
path, transcode structure, every rejection reason, retry, API-transient, and usage
accounting.

Run:  python -m unittest tests.test_generate_tool_use -v
"""
import hashlib
import json
import os
import sys
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sft"))

CANONICAL_FP = "7a570a7ba9fc7985"


def _fp(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def _find_tok():
    cands = []
    if os.environ.get("SYNAPSE_TOKENIZER"):
        cands.append(os.environ["SYNAPSE_TOKENIZER"])
    if os.environ.get("SYNAPSE_DIR"):
        cands.append(os.path.join(os.environ["SYNAPSE_DIR"], "tokenizer_out", "tokenizer.json"))
    home = os.path.expanduser("~")
    cands.append(os.path.join(home, "Downloads", "tokenizer_out", "tokenizer.json"))
    for p in cands:
        try:
            if os.path.exists(p) and _fp(p) == CANONICAL_FP:
                return p
        except OSError:
            pass
    return None


_CANON = _find_tok()
if _CANON:
    os.environ["SYNAPSE_TOKENIZER"] = _CANON

import generate_tool_use as gtu  # noqa: E402
import tools_runtime as tr       # noqa: E402

skip_no_tok = unittest.skipUnless(_CANON, "canonical tokenizer not found locally")
SN = types.SimpleNamespace


# --- scripted-response helpers --------------------------------------------
def tool_resp(name, args, content="", pt=10, ct=5):
    arguments = args if isinstance(args, str) else json.dumps(args)
    call = SN(id="call_1", type="function", function=SN(name=name, arguments=arguments))
    msg = SN(content=content, tool_calls=[call])
    return SN(choices=[SN(message=msg)], usage=SN(prompt_tokens=pt, completion_tokens=ct))


def final_resp(content, pt=10, ct=5):
    msg = SN(content=content, tool_calls=None)
    return SN(choices=[SN(message=msg)], usage=SN(prompt_tokens=pt, completion_tokens=ct))


class FakeClient:
    def __init__(self, responses):
        self._r = list(responses)
        self._i = 0
        self.calls = []
        self.chat = SN(completions=SN(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        item = self._r[min(self._i, len(self._r) - 1)]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return item


def agent(responses, *, gold="1113177", mode="python", max_calls=3, trace_max_tok=1984):
    client = FakeClient(responses)
    r = gtu.run_agentic(client, "m", "What is 2847*391?", gold, mode,
                        max_calls=max_calls, trace_max_tok=trace_max_tok)
    return client, r


# --------------------------------------------------------------------------
@skip_no_tok
class TestRunAgentic(unittest.TestCase):
    def test_python_happy_path(self):
        _, r = agent([
            tool_resp("python", {"code": "print(2847*391)"}),
            final_resp("So it is 1113177.\nThe answer is: 1113177"),
        ])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["used_tool"], 1)
        msgs = r["trace"]
        # user, assistant(+tool_call), tool, assistant
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(msgs[1]["tool_call"], {"tool": "python", "code": "print(2847*391)"})
        self.assertEqual(msgs[2]["content"], "1113177")        # real run_python result
        self.assertNotIn("tool_call", msgs[3])                 # final turn has no call

    def test_serializer_roundtrip_on_trace(self):
        _, r = agent([
            tool_resp("python", {"code": "print(2847*391)"}),
            final_resp("The answer is: 1113177"),
        ])
        tc = r["trace"][1]["tool_call"]
        self.assertEqual(tr.dump_tool_call(tc), '{"code":"print(2847*391)","tool":"python"}')

    def test_wrong_answer_rejected(self):
        _, r = agent([
            tool_resp("python", {"code": "print(1)"}),
            final_resp("The answer is: 999"),
        ])
        self.assertEqual(r["status"], "wrong")

    def test_no_tool_used(self):
        # immediate correct final, no tool -> status ok but used_tool 0
        _, r = agent([final_resp("The answer is: 1113177")])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["used_tool"], 0)

    def test_bad_json_args(self):
        _, r = agent([tool_resp("python", "{not valid json")])
        self.assertEqual(r["status"], "badjson")

    def test_bad_tool_name(self):
        _, r = agent([tool_resp("calculator", {"x": 1})])
        self.assertEqual(r["status"], "badtool")

    def test_maxcalls(self):
        # teacher never gives a final answer -> exhausts calls
        _, r = agent([tool_resp("python", {"code": "print(1)"})], max_calls=2)
        self.assertEqual(r["status"], "maxcalls")
        self.assertEqual(r["used_tool"], 2)

    def test_overflow(self):
        _, r = agent([final_resp("The answer is: 1113177")], trace_max_tok=5)
        self.assertEqual(r["status"], "overflow")

    def test_long_final(self):
        big = "x " * (gtu.FINAL_ANSWER_TOK + 50)
        _, r = agent([final_resp(big + "The answer is: 1113177")])
        self.assertEqual(r["status"], "long")

    def test_api_failure(self):
        _, r = agent([RuntimeError("api down")])
        self.assertEqual(r["status"], "api")
        self.assertEqual(r["in_tok"], 0)

    def test_usage_accounting(self):
        _, r = agent([
            tool_resp("python", {"code": "print(2847*391)"}, pt=100, ct=20),
            final_resp("The answer is: 1113177", pt=50, ct=10),
        ])
        self.assertEqual(r["in_tok"], 150)
        self.assertEqual(r["out_tok"], 30)

    def test_search_path(self):
        orig = tr.run_search
        tr.run_search = lambda q, **k: "Related rates: use implicit differentiation"
        try:
            _, r = agent([
                tool_resp("search", {"query": "related rates method"}),
                final_resp("The answer is: 1113177"),
            ], mode="search")
        finally:
            tr.run_search = orig
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["trace"][1]["tool_call"], {"tool": "search", "query": "related rates method"})
        self.assertIn("implicit differentiation", r["trace"][2]["content"])

    def test_search_unavailable_rejected(self):
        orig = tr.run_search
        tr.run_search = lambda q, **k: tr.SEARCH_UNAVAILABLE
        try:
            _, r = agent([
                tool_resp("search", {"query": "x"}),
                final_resp("The answer is: 1113177"),
            ], mode="search")
        finally:
            tr.run_search = orig
        self.assertEqual(r["status"], "searchfail")

    def test_force_first_tool_steers_not_forces(self):
        # DeepSeek thinking mode 400s on a forced tool_choice, so "force" is now a
        # MUST-call steering line + rejection sampling; tool_choice stays "auto".
        orig = tr.run_search
        tr.run_search = lambda q, **k: "Tokyo is the capital of Japan."
        try:
            client = FakeClient([
                tool_resp("search", {"query": "capital of Japan"}),
                final_resp("The answer is: Tokyo"),
            ])
            gtu.run_agentic(client, "m", "Capital of Japan?", "Tokyo", "search",
                            max_calls=2, trace_max_tok=1984, force_first_tool="search")
        finally:
            tr.run_search = orig
        self.assertEqual(client.calls[0]["tool_choice"], "auto")
        self.assertEqual(client.calls[1]["tool_choice"], "auto")
        sys_msg = client.calls[0]["messages"][0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIn("You MUST call the `search` tool", sys_msg["content"])
        # the steering line is transient — never in the STORED system prompt
        self.assertNotIn("MUST call", gtu.CANONICAL_TOOL_SYSTEM)

    def test_no_force_is_auto(self):
        client = FakeClient([final_resp("The answer is: 1113177")])
        gtu.run_agentic(client, "m", "q", "1113177", "python",
                        max_calls=3, trace_max_tok=1984)
        self.assertEqual(client.calls[0]["tool_choice"], "auto")

    def test_only_answered_call_in_transcript(self):
        # the OpenAI working transcript must contain exactly the call we answered
        client, _ = agent([
            tool_resp("python", {"code": "print(2847*391)"}),
            final_resp("The answer is: 1113177"),
        ])
        second_call_msgs = client.calls[1]["messages"]
        roles = [m["role"] for m in second_call_msgs]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        assistant = second_call_msgs[2]
        self.assertEqual(len(assistant["tool_calls"]), 1)
        self.assertEqual(second_call_msgs[3]["tool_call_id"], "call_1")


# --------------------------------------------------------------------------
@skip_no_tok
class TestSolveProblem(unittest.TestCase):
    def _args(self, **over):
        d = dict(retries=1, max_calls=None, trace_max_tok=1984)
        d.update(over)
        return SN(**d)

    def test_kept_record_shape(self):
        client = FakeClient([
            tool_resp("python", {"code": "print(2847*391)"}),
            final_resp("The answer is: 1113177"),
        ])
        rec, transient, it, ot, _st, _dbg = gtu.solve_problem(
            client, "m", {"id": "python_abc", "question": "2847*391?", "gold": "1113177"},
            "python", self._args(), limiter=None, brave_key=None)
        self.assertIsNotNone(rec)
        self.assertFalse(transient)
        self.assertEqual(rec["id"], "python_abc")
        self.assertEqual(rec["mode"], "python")
        self.assertEqual(rec["system"], gtu.CANONICAL_TOOL_SYSTEM)
        self.assertEqual(rec["messages"][0]["role"], "user")

    def test_no_tool_not_kept(self):
        client = FakeClient([final_resp("The answer is: 1113177")])
        rec, transient, _, _, _st, _dbg = gtu.solve_problem(
            client, "m", {"id": "python_x", "question": "q", "gold": "1113177"},
            "python", self._args(), limiter=None, brave_key=None)
        self.assertIsNone(rec)              # verified but no tool used -> dropped
        self.assertFalse(transient)

    def test_retry_recovers(self):
        # attempt 0 wrong, attempt 1 correct -> kept
        client = FakeClient([
            tool_resp("python", {"code": "print(1)"}), final_resp("The answer is: 0"),  # wrong
            tool_resp("python", {"code": "print(2847*391)"}), final_resp("The answer is: 1113177"),
        ])
        rec, _, _, _, _st, _dbg = gtu.solve_problem(
            client, "m", {"id": "python_y", "question": "q", "gold": "1113177"},
            "python", self._args(retries=1), limiter=None, brave_key=None)
        self.assertIsNotNone(rec)

    def test_transient_left_undone(self):
        client = FakeClient([RuntimeError("down")])
        rec, transient, it, ot, _st, _dbg = gtu.solve_problem(
            client, "m", {"id": "python_z", "question": "q", "gold": "1"},
            "python", self._args(retries=0), limiter=None, brave_key=None)
        self.assertIsNone(rec)
        self.assertTrue(transient)          # zero tokens spent -> retry next run


@skip_no_tok
class TestLoadProblems(unittest.TestCase):
    def test_loads_and_skips_bad(self):
        import tempfile, json as _json
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(_json.dumps({"id": "facts_1", "question": "Capital of Japan?", "gold": "Tokyo"}) + "\n")
            f.write("not json\n")                                   # skipped
            f.write(_json.dumps({"question": "", "gold": "x"}) + "\n")   # skipped
            f.write(_json.dumps({"question": "Atomic number of iron?", "gold": 26}) + "\n")
            path = f.name
        try:
            pool = gtu.load_problems(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(pool), 2)
        self.assertEqual(pool[0]["id"], "facts_1")
        self.assertEqual(pool[1]["gold"], "26")                     # coerced to str
        self.assertTrue(pool[1]["id"])                              # synthesized id


if __name__ == "__main__":
    unittest.main(verbosity=2)
