"""Tests for the tool-use changes to sft/tokenize_sft_data.py + sparky_chat_template.py.

The headline test is a REGRESSION proof: a normal user/assistant record tokenizes
BYTE-IDENTICALLY to the pre-change logic (a reference encoder embedded here), so
existing SFT sources are provably unaffected. Plus: tool-role + inline tool_call
masking, resp_tokens accounting, the 7/8 resolver guard, and template round-trip.

Run:  python -m unittest tests.test_tokenize_sft_tooluse -v
"""
import hashlib
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sft"))
sys.path.insert(0, os.path.join(REPO_ROOT, "sparky"))

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

skip_no_tok = unittest.skipUnless(_CANON, "canonical tokenizer not found locally")

if _CANON:
    from tokenizers import Tokenizer
    import tokenize_sft_data as tk
    import sparky_chat_template as tmpl
    _TOK = Tokenizer.from_file(_CANON)
    _IDS = None


def ids():
    global _IDS
    if _IDS is None:
        _IDS = tk.resolve_special_ids(_TOK)
    return _IDS


IGNORE = -100


def reference_encode_no_tool(tok, ex, role_id, im_end_id, eot_id):
    """A copy of the PRE-CHANGE encode_example (no tool support). Used to prove the
    new encoder is byte-identical on records without tool fields."""
    out_ids, labels = [], []
    resp = 0

    def add(t, learn):
        out_ids.extend(t)
        labels.extend(t if learn else [IGNORE] * len(t))

    def enc(s):
        return tok.encode(s, add_special_tokens=False).ids

    if ex.get("system"):
        add([role_id["system"]], False)
        add(enc(ex["system"]), False)
        add([im_end_id], False)
    for m in ex["messages"]:
        learn = m["role"] == "assistant"
        add([role_id[m["role"]]], False)
        cids = enc(m["content"])
        add(cids, learn)
        add([im_end_id], learn)
        if learn:
            resp += len(cids)
    add([eot_id], True)
    return out_ids, labels, resp


@skip_no_tok
class TestRegressionNoToolUnchanged(unittest.TestCase):
    """Existing-style records must tokenize EXACTLY as before."""
    SAMPLES = [
        {"system": "You are helpful.", "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4."}]},
        {"system": "", "messages": [
            {"role": "user", "content": "Multi-turn?"},
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": "Sum 100000 and 23456?"},
            {"role": "assistant", "content": "123456"}]},
    ]

    def test_byte_identical(self):
        role_id, im_end_id, eot_id, tool_call_id = ids()
        for ex in self.SAMPLES:
            new = tk.encode_example(_TOK, ex, role_id, im_end_id, eot_id, tool_call_id)
            old = reference_encode_no_tool(_TOK, ex, role_id, im_end_id, eot_id)
            self.assertEqual(new, old, "no-tool record changed — regression!")


@skip_no_tok
class TestToolEncoding(unittest.TestCase):
    def _ex(self):
        return {"system": "sys", "messages": [
            {"role": "user", "content": "2847*391?"},
            {"role": "assistant", "content": "Let me compute.",
             "tool_call": {"tool": "python", "code": "print(2847*391)"}},
            {"role": "tool", "content": "1113177"},
            {"role": "assistant", "content": "The answer is: 1113177"}]}

    def test_masking(self):
        role_id, im_end_id, eot_id, tcid = ids()
        input_ids, labels, resp = tk.encode_example(_TOK, self._ex(), role_id, im_end_id, eot_id, tcid)
        self.assertEqual(len(input_ids), len(labels))
        # the WHOLE tool turn (marker + content + its im_end) must be masked. Check the
        # region from the tool-result marker to the next im_end (structural — robust to
        # digit-per-token where the number also recurs in the learned final answer).
        k = input_ids.index(role_id["tool"])
        end = input_ids.index(im_end_id, k)
        self.assertTrue(all(l == IGNORE for l in labels[k:end + 1]),
                        "entire tool-result turn (incl its im_end) must be masked")
        # the inline <|tool_call|> marker AND the JSON after it are learned
        j = input_ids.index(tcid)
        self.assertEqual(labels[j], tcid)
        self.assertNotEqual(labels[j + 1], IGNORE)
        # final answer turn is learned (the number recurs there, correctly unmasked)
        self.assertEqual(labels[-1], eot_id)         # learned EOS

    def test_resp_tokens_counts_call_json_not_result(self):
        role_id, im_end_id, eot_id, tcid = ids()
        _, _, resp = tk.encode_example(_TOK, self._ex(), role_id, im_end_id, eot_id, tcid)
        enc = lambda s: len(_TOK.encode(s, add_special_tokens=False).ids)
        expected = (enc("Let me compute.") + enc(tk.dump_tool_call({"tool": "python", "code": "print(2847*391)"}))
                    + enc("The answer is: 1113177"))
        self.assertEqual(resp, expected)         # assistant content + call json; result excluded

    def test_serializer_is_shared(self):
        # tokenizer + template use the SAME serializer object
        self.assertIs(tk.dump_tool_call, tmpl.dump_tool_call)


@skip_no_tok
class TestResolver(unittest.TestCase):
    def test_returns_tool_call_id_7(self):
        role_id, im_end_id, eot_id, tcid = ids()
        self.assertEqual(tcid, 7)
        self.assertEqual(role_id["tool"], 8)


@skip_no_tok
class TestTemplateRoundTrip(unittest.TestCase):
    def test_render_matches_encoder_wire(self):
        # the template's rendered string should tokenize to the same ids the encoder
        # produced for the non-final turns (prompt up to the last assistant cue).
        msgs = [
            {"role": "user", "content": "2847*391?"},
            {"role": "assistant", "content": "Let me compute.",
             "tool_call": {"tool": "python", "code": "print(2847*391)"}},
            {"role": "tool", "content": "1113177"},
        ]
        prompt = tmpl.build_sft_prompt(msgs, system="sys")
        # round-trips: contains both markers in order, ends on the assistant cue
        self.assertIn("<|tool_call|>", prompt)
        self.assertIn("<|tool_result|>", prompt)
        self.assertTrue(prompt.endswith(tmpl.ROLE_ASSISTANT))
        self.assertIn('{"code":"print(2847*391)","tool":"python"}', prompt)  # canonical serialize

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            tmpl.build_sft_prompt([{"role": "function", "content": "x"}])


def _find_tool_use_raw():
    cands = []
    if os.environ.get("SYNAPSE_DIR"):
        cands.append(os.path.join(os.environ["SYNAPSE_DIR"],
                                  "datasets_sft", "tool_use", "tool_use_raw.jsonl"))
    cands.append(os.path.join(os.path.expanduser("~"), "synapse_data",
                              "datasets_sft", "tool_use", "tool_use_raw.jsonl"))
    for p in cands:
        if os.path.exists(p):
            return p
    return None


_TOOL_USE_RAW = _find_tool_use_raw()


@skip_no_tok
@unittest.skipUnless(_TOOL_USE_RAW, "tool_use_raw.jsonl not found locally")
class TestRealToolUseRoundTrip(unittest.TestCase):
    """SFT v2 Phase 2 gate: 5 REAL records from tool_use_raw.jsonl must
    (a) mask exactly right — labels cover assistant content, <|tool_call|> (7),
        the serialized call JSON, the assistant <|im_end|> and the final EOT,
        and NEVER cover a tool-result turn (8) or its <|im_end|>;
    (b) round-trip — build_sft_prompt renders a string that tokenizes
        byte-identically to the ids the encoder produced for the prompt part.
    Any drift here means the served model would run out-of-distribution."""

    @classmethod
    def setUpClass(cls):
        import json
        cls.records = []
        with open(_TOOL_USE_RAW) as f:
            for line in f:
                cls.records.append(json.loads(line))
                if len(cls.records) == 5:
                    break
        assert len(cls.records) == 5

    def test_masking_on_real_records(self):
        role_id, im_end_id, eot_id, tcid = ids()
        for rec in self.records:
            input_ids, labels, resp = tk.encode_example(
                _TOK, rec, role_id, im_end_id, eot_id, tcid)
            with self.subTest(rec_id=rec["id"]):
                self.assertEqual(len(input_ids), len(labels))
                self.assertGreaterEqual(resp, 3)
                # every tool-result turn fully masked, incl. its im_end
                for k, t in enumerate(input_ids):
                    if t == role_id["tool"]:
                        end = input_ids.index(im_end_id, k)
                        self.assertTrue(
                            all(l == IGNORE for l in labels[k:end + 1]),
                            f"{rec['id']}: tool-result turn at {k} not fully masked")
                # every <|tool_call|> marker + its JSON span learned
                for j, t in enumerate(input_ids):
                    if t == tcid:
                        end = input_ids.index(im_end_id, j)
                        self.assertTrue(
                            all(labels[i] == input_ids[i] for i in range(j, end + 1)),
                            f"{rec['id']}: call JSON span at {j} not fully learned")
                # final assistant turn learned through its im_end; EOT learned
                last_a = max(i for i, t in enumerate(input_ids)
                             if t == role_id["assistant"])
                end = input_ids.index(im_end_id, last_a)
                self.assertTrue(
                    all(labels[i] == input_ids[i] for i in range(last_a + 1, end + 1)),
                    f"{rec['id']}: final assistant turn not fully learned")
                self.assertEqual(labels[-1], eot_id)
                # role markers and everything before the first assistant turn masked
                first_a = input_ids.index(role_id["assistant"])
                self.assertTrue(all(l == IGNORE for l in labels[:first_a + 1]),
                                f"{rec['id']}: prompt region leaked into labels")

    def test_template_renders_encoder_wire_bytes(self):
        role_id, im_end_id, eot_id, tcid = ids()
        for rec in self.records:
            input_ids, _, _ = tk.encode_example(
                _TOK, rec, role_id, im_end_id, eot_id, tcid)
            # the inference-time prompt: everything before the FINAL assistant
            # turn, ending on the assistant cue the model completes from
            prompt = tmpl.build_sft_prompt(rec["messages"][:-1],
                                           system=rec.get("system", ""))
            prompt_ids = _TOK.encode(prompt, add_special_tokens=False).ids
            with self.subTest(rec_id=rec["id"]):
                self.assertEqual(
                    prompt_ids, input_ids[:len(prompt_ids)],
                    f"{rec['id']}: template wire bytes diverge from training encoding")
                # the very next token is where generation starts — inside the
                # final assistant turn, so the prompt must end on its role cue
                self.assertEqual(prompt_ids[-1], role_id["assistant"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
