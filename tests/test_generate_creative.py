"""Tests for sft/generate_creative.py (teacher + LLM-judge creative generator).

Prompt-bank + score-parsing run offline. make_record (write->judge->keep) is driven
by a FakeClient (no API). Needs the canonical tokenizer for length checks.

Run:  python -m unittest tests.test_generate_creative -v
"""
import hashlib
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sft"))

CANONICAL_FP = "7a570a7ba9fc7985"


def _fp(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8192):
            h.update(c)
    return h.hexdigest()[:16]


def _find_tok():
    cands = []
    if os.environ.get("SYNAPSE_TOKENIZER"):
        cands.append(os.environ["SYNAPSE_TOKENIZER"])
    cands.append(os.path.join(os.path.expanduser("~"), "Downloads", "tokenizer_out", "tokenizer.json"))
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

import generate_creative as gc  # noqa: E402

SN = types.SimpleNamespace
skip_no_tok = unittest.skipUnless(_CANON, "canonical tokenizer not found")


def resp(content, pt=50, ct=120):
    return SN(choices=[SN(message=SN(content=content))],
              usage=SN(prompt_tokens=pt, completion_tokens=ct))


class FakeClient:
    """Returns queued responses in order (story call, then judge call)."""
    def __init__(self, responses):
        self._r, self._i = list(responses), 0
        self.chat = SN(completions=SN(create=self._create))

    def _create(self, **kw):
        r = self._r[min(self._i, len(self._r) - 1)]
        self._i += 1
        return r


class TestBuildPrompts(unittest.TestCase):
    def test_count_unique_wellformed(self):
        ps = gc.build_prompts(50)
        self.assertEqual(len(ps), 50)
        self.assertEqual(len({p["id"] for p in ps}), 50)        # unique
        for p in ps:
            self.assertTrue(p["prompt"].startswith("Write a"))
            self.assertTrue(p["id"].startswith("creative_"))
            self.assertIn(p["form"], gc.FORMS)

    def test_deterministic(self):
        self.assertEqual([p["id"] for p in gc.build_prompts(20)],
                         [p["id"] for p in gc.build_prompts(20)])


class TestParseScore(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(gc.parse_score("Score: 8"), 8)
        self.assertEqual(gc.parse_score("score = 10"), 10)
        self.assertEqual(gc.parse_score("I'd give it a 7."), 7)
        self.assertEqual(gc.parse_score("Score: 9/10"), 9)

    def test_invalid(self):
        self.assertIsNone(gc.parse_score(""))
        self.assertIsNone(gc.parse_score("no number here"))
        self.assertIsNone(gc.parse_score("Score: 15"))          # out of range


@skip_no_tok
class TestMakeRecord(unittest.TestCase):
    def _args(self, **over):
        d = dict(max_output_tok=1400, judge_max_tok=1024, trace_max_tok=1984, min_score=8)
        d.update(over)
        return SN(**d)

    def _problem(self):
        return {"id": "creative_abc", "prompt": "Write a fable about a patient river.",
                "form": "fable", "genre": "fantasy"}

    def test_kept_high_score(self):
        story = "The river waited, and the mountain learned to listen. " * 6
        client = FakeClient([resp(story), resp("Score: 9")])
        rec, transient, it, ot, status = gc.make_record(client, "m", self._problem(), self._args())
        self.assertEqual(status, "ok")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["system"], "")
        self.assertEqual(rec["score"], 9)
        self.assertEqual([m["role"] for m in rec["messages"]], ["user", "assistant"])
        self.assertEqual(rec["messages"][1]["content"], story.strip())

    def test_low_score_rejected(self):
        story = "A mediocre tale that goes nowhere in particular. " * 6
        client = FakeClient([resp(story), resp("Score: 5")])
        rec, _, _, _, status = gc.make_record(client, "m", self._problem(), self._args())
        self.assertIsNone(rec)
        self.assertEqual(status, "low_score")

    def test_too_short_rejected(self):
        client = FakeClient([resp("Too short."), resp("Score: 9")])
        rec, _, _, _, status = gc.make_record(client, "m", self._problem(), self._args())
        self.assertIsNone(rec)
        self.assertEqual(status, "short")

    def test_too_long_rejected(self):
        long_story = "word " * 5000
        client = FakeClient([resp(long_story), resp("Score: 9")])
        rec, _, _, _, status = gc.make_record(client, "m", self._problem(), self._args())
        self.assertIsNone(rec)
        self.assertEqual(status, "too_long")

    def test_judge_unparseable(self):
        story = "The river waited, and the mountain learned to listen. " * 6
        client = FakeClient([resp(story), resp("hmm, hard to say")])
        rec, _, _, _, status = gc.make_record(client, "m", self._problem(), self._args())
        self.assertIsNone(rec)
        self.assertEqual(status, "judge_fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
