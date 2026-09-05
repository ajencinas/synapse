"""Validate every NEW SFT source so it can't break the model/pipeline.

For a source's records this checks: valid {system,messages} schema, clean
tokenization via the REAL encode_example (no exceptions), correct loss masking
(role markers + user/system masked; assistant learned), and that a healthy fraction
survives the 2048 block-size filter (guards the silent "all creative text too long
-> ~0 kept" failure).

- Schema/masking invariants run offline (always).
- Live HF source checks run only with RUN_SOURCE_VALIDATION=1 (need network + tokenizer).

Run all:   RUN_SOURCE_VALIDATION=1 python -m unittest tests.test_sft_source_validation -v
"""
import hashlib
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sft"))

CANONICAL_FP = "7a570a7ba9fc7985"
BLOCK = 2048


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
    if os.environ.get("SYNAPSE_DIR"):
        cands.append(os.path.join(os.environ["SYNAPSE_DIR"], "tokenizer_out", "tokenizer.json"))
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

import download_sft_data as dl          # noqa: E402
import tokenize_sft_data as tk          # noqa: E402

LIVE = os.environ.get("RUN_SOURCE_VALIDATION") == "1" and _CANON
IGNORE = -100


def _validate_examples(examples, label="src"):
    """Run examples through the real tokenizer/encoder. Returns a stats dict and
    asserts the hard invariants (raises AssertionError on any breakage)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(_CANON)
    role_id, im_end_id, eot_id, tool_call_id = tk.resolve_special_ids(tok)
    lens, kept, short, too_long, bad_mask = [], 0, 0, 0, 0
    for ex in examples:
        # schema: must be the uniform shape
        assert isinstance(ex.get("system", ""), str)
        assert ex["messages"] and ex["messages"][0]["role"] == "user"
        assert ex["messages"][-1]["role"] == "assistant"
        ids, labels, resp = tk.encode_example(tok, ex, role_id, im_end_id, eot_id, tool_call_id)
        assert len(ids) == len(labels), "ids/labels length mismatch"
        # masking: there must be BOTH masked (prompt) and learned (response) tokens
        if not (any(l == IGNORE for l in labels) and any(l != IGNORE for l in labels)):
            bad_mask += 1
        # role markers themselves must be masked
        for rid in role_id.values():
            for i, t in enumerate(ids):
                if t == rid:
                    assert labels[i] == IGNORE, f"role marker {rid} not masked"
                    break
        lens.append(len(ids))
        if resp < 3:
            short += 1
        elif len(ids) > BLOCK:
            too_long += 1
        else:
            kept += 1
    lens.sort()
    n = len(lens)
    return {"n": n, "kept": kept, "short": short, "too_long": too_long,
            "bad_mask": bad_mask, "kept_frac": round(kept / max(1, n), 2),
            "p50": lens[n // 2], "p95": lens[min(n - 1, int(0.95 * n))], "max": lens[-1]}


# --------------------------------------------------------------------------
class TestSchemaInvariantsOffline(unittest.TestCase):
    """No network: hand-crafted records exercise the masking contract + the
    drop behaviour the pipeline relies on (so a new source can't silently break)."""

    @unittest.skipUnless(_CANON, "tokenizer not found")
    def test_good_example_masks_correctly(self):
        ex = {"system": "S", "messages": [
            {"role": "user", "content": "Tell me a story about the sea."},
            {"role": "assistant", "content": "The tide remembered every ship it had ever held."}]}
        stats = _validate_examples([ex])
        self.assertEqual(stats["bad_mask"], 0)
        self.assertEqual(stats["kept"], 1)

    def test_make_example_rejects_malformed(self):
        # adapter/make_example must reject non-alternating / wrong-ending records,
        # so they never reach the tokenizer.
        self.assertIsNone(dl.make_example("", [("assistant", "hi")]))           # starts on assistant
        self.assertIsNone(dl.make_example("", [("user", "a"), ("user", "b")]))  # no alternation
        self.assertIsNone(dl.make_example("", [("user", "a"), ("assistant", "")]))  # empty content
        self.assertIsNotNone(dl.make_example("", [("user", "a"), ("assistant", "b")]))


# --------------------------------------------------------------------------
@unittest.skipUnless(LIVE, "set RUN_SOURCE_VALIDATION=1 (+ tokenizer) for live HF source checks")
class TestNoRobotsSource(unittest.TestCase):
    def test_no_robots_is_safe(self):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/no_robots", split="train")
        sample = [dl.adapter_no_robots(r) for r in ds.select(range(300))]
        valid = [e for e in sample if e is not None]
        adapter_frac = len(valid) / len(sample)
        stats = _validate_examples(valid, "no_robots")
        print(f"\n[no_robots] adapter_ok={adapter_frac:.2f} {stats}")
        self.assertGreater(adapter_frac, 0.85, "adapter rejects too many rows")
        self.assertEqual(stats["bad_mask"], 0, "masking broken on some records")
        self.assertGreater(stats["kept_frac"], 0.3,
                           "too many dropped by 2048 filter — source would mostly vanish")


if __name__ == "__main__":
    unittest.main(verbosity=2)
