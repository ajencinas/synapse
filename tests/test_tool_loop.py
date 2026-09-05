#!/usr/bin/env python3
"""sft/tool_loop.py — the shared inference tool loop, driven by a scripted
"model" over the REAL Synapse tokenizer (ids 7/8 + role markers must line up
with the trained wire format). No GPU.

Run:  SYNAPSE_DIR=~/synapse_data python -m unittest tests.test_tool_loop
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sft"))
sys.path.insert(0, os.path.join(REPO_ROOT, "sparky"))

import tool_loop as tl                                   # noqa: E402
import tools_runtime as tr                               # noqa: E402
from sparky_chat_template import (                        # noqa: E402
    build_sft_prompt, ROLE_ASSISTANT, ROLE_TOOL, IM_END)


def _tokenizer():
    from tokenizers import Tokenizer
    cands = []
    if os.environ.get("SYNAPSE_TOKENIZER"):
        cands.append(os.environ["SYNAPSE_TOKENIZER"])
    if os.environ.get("SYNAPSE_DIR"):
        cands.append(os.path.join(os.environ["SYNAPSE_DIR"], "tokenizer_out", "tokenizer.json"))
    cands.append(os.path.expanduser("~/synapse_data/tokenizer_out/tokenizer.json"))
    for p in cands:
        if os.path.exists(p):
            return Tokenizer.from_file(p)
    raise unittest.SkipTest("Synapse tokenizer not found (set SYNAPSE_DIR)")


class Scripted:
    """A fake model: each call to generate_ids() plays the next scripted turn.
    A turn is (prose, call_dict_or_None). Records every prompt it was given."""

    def __init__(self, tok, turns):
        self.tok, self.turns, self.prompts = tok, list(turns), []

    def __call__(self, prompt_ids):
        self.prompts.append(list(prompt_ids))
        if not self.turns:
            raise AssertionError("model asked for more turns than scripted")
        prose, call = self.turns.pop(0)
        ids = self.tok.encode(prose, add_special_tokens=False).ids if prose else []
        if call is not None:
            raw = call if isinstance(call, str) else tr.dump_tool_call(call)
            ids = ids + [tl.TOOL_CALL_ID] + self.tok.encode(raw, add_special_tokens=False).ids
        yield from ids           # stop token is NOT yielded (contract)


class TestToolLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()
        cls.user = [{"role": "user", "content": "What is 6 times 7?"}]

    def run_loop(self, turns, system=tr.CANONICAL_TOOL_SYSTEM, **kw):
        model = Scripted(self.tok, turns)
        events = list(tl.run_tool_loop(model, self.tok, self.user, system, **kw))
        return model, events, events[-1]

    def test_python_call_round_trip(self):
        call = {"tool": "python", "code": "6*7\n"}
        model, events, final = self.run_loop([("", call), ("The answer is: 42", None)])
        kinds = [e["type"] for e in events]
        collapsed = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]
        self.assertEqual(collapsed, ["tool_call", "tool_result", "token", "final"])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "The answer is: 42")
        self.assertEqual(events[0]["call"], call)
        self.assertEqual(events[1]["text"], "42")            # sandbox really ran it
        self.assertEqual(final["status"], "answered")
        self.assertEqual(final["rounds"], 1)
        self.assertEqual(final["messages"], [
            {"role": "assistant", "content": "", "tool_call": call},
            {"role": "tool", "content": "42"},
            {"role": "assistant", "content": "The answer is: 42"},
        ])
        # The second prompt must be exactly the trained wire format of the trace so
        # far: ...<|reserved_2|><|tool_call|>{json}<|im_end|><|tool_result|>42<|im_end|><|reserved_2|>
        expected = build_sft_prompt(self.user + final["messages"][:2], system=tr.CANONICAL_TOOL_SYSTEM)
        self.assertEqual(model.prompts[1], self.tok.encode(expected, add_special_tokens=False).ids)
        self.assertTrue(expected.endswith(ROLE_TOOL + "42" + IM_END + ROLE_ASSISTANT))

    def test_no_tool_streams_prose(self):
        model, events, final = self.run_loop([("It is 42.", None)], system="")
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "It is 42.")
        self.assertEqual(final["status"], "answered")
        self.assertEqual(final["messages"], [{"role": "assistant", "content": "It is 42."}])
        self.assertEqual(len(model.prompts), 1)
        # empty system => no system segment at all
        self.assertNotIn("<|reserved_0|>", self.tok.decode(model.prompts[0], skip_special_tokens=False))

    def test_malformed_json_is_fed_back_as_error(self):
        model, events, final = self.run_loop([("", '{"tool": "python", "code": '),   # truncated JSON
                                              ("The answer is: 42", None)])
        self.assertIsNone(events[0]["tool"])
        self.assertTrue(events[1]["text"].startswith("[error] malformed tool call"))
        self.assertEqual(final["status"], "answered")
        # malformed call is not re-rendered into the prompt (can't be serialized)
        self.assertNotIn("tool_call", final["messages"][0])
        self.assertEqual(final["messages"][1]["role"], "tool")

    def test_python_error_is_fed_back(self):
        model, events, final = self.run_loop([("", {"tool": "python", "code": "1/0\n"}),
                                              ("Division by zero — undefined.", None)])
        self.assertIn("[error]", events[1]["text"])
        self.assertIn("ZeroDivisionError", events[1]["text"])
        self.assertEqual(final["status"], "answered")

    def test_unknown_tool(self):
        model, events, final = self.run_loop([("", {"tool": "calculator", "expr": "6*7"}),
                                              ("The answer is: 42", None)])
        self.assertIn("unknown tool 'calculator'", events[1]["text"])

    def test_repeated_identical_call_stops(self):
        call = {"tool": "python", "code": "6*7\n"}
        model, events, final = self.run_loop([("", call), ("", call), ("never", None)])
        self.assertEqual(final["status"], "repeat")
        self.assertEqual(sum(e["type"] == "tool_result" for e in events), 1)  # ran once
        self.assertEqual(len(model.turns), 1)                                   # 'never' unused

    def test_max_rounds(self):
        calls = [("", {"tool": "python", "code": f"{i}+1\n"}) for i in range(10)]
        model, events, final = self.run_loop(calls, max_rounds=3)
        self.assertEqual(final["status"], "max_rounds")
        self.assertEqual(sum(e["type"] == "tool_result" for e in events), 3)
        self.assertEqual(final["rounds"], 3)

    def test_prompt_budget(self):
        model, events, final = self.run_loop([("x", None)], max_prompt_tokens=5)
        self.assertEqual(final["status"], "budget")
        self.assertEqual(final["messages"], [])

    def test_final_text_helper(self):
        model = Scripted(self.tok, [("", {"tool": "python", "code": "6*7\n"}), ("The answer is: 42", None)])
        text, final = tl.final_text(tl.run_tool_loop(model, self.tok, self.user, tr.CANONICAL_TOOL_SYSTEM))
        self.assertEqual(text, "The answer is: 42")
        self.assertEqual(final["status"], "answered")


if __name__ == "__main__":
    unittest.main()
