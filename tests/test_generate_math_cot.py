"""Regression tests for the math CoT generator.

Catches the bugs we fixed in May 2026:
  - algebra x/a=b returning wrong x
  - order-of-ops evaluating + as / (bad MD/AS split in _evaluate_flat)
  - raw Python tuple repr ('+',) leaking into prose
  - "Step N: Step N:" duplication in fractions/decimals
  - "Step N: N." duplication in order_of_ops

Run from repo root: `python -m unittest tests.test_generate_math_cot`
"""

import os
import re
import sys
import tempfile
import unittest
from fractions import Fraction

# Make the generator importable.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(REPO_ROOT, "download_pretrain_others", "src")
sys.path.insert(0, GEN_DIR)

import generate_math_operations_cot as g  # noqa: E402


BANNED_SUBSTRINGS = ["('+'", "('-'", "('×'", "('*'", "('/'"]
STEP_DUP_PATTERNS = [
    re.compile(r"Step \d+: Step \d+:"),
    re.compile(r"Step \d+: \d+\."),
]


def _declared_answer(text):
    m = re.search(r"^Answer:\s*(.+?)\s*$", text, re.MULTILINE)
    if not m:
        return None
    s = m.group(1).strip()
    m2 = re.match(r"\\boxed\{(.+)\}\s*$", s)
    s = m2.group(1).strip() if m2 else s
    return re.sub(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", s).strip()


def _problem_line(text):
    m = re.search(r"^Problem:\s*(.+?)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _strip_surface(p):
    p = re.sub(r"^Compute\s+", "", p)
    p = re.sub(r"^>>> solve\(", "", p)
    p = re.sub(r"^>>>\s*", "", p)
    p = p.replace("\\(", "").replace("\\)", "")
    p = re.sub(r"^Solve\s+", "", p)
    p = re.sub(r"\s*for x\.?\s*$", "", p)
    if p.count("(") < p.count(")"):
        p = p.rstrip(")")
    return p.rstrip(".").strip()


class CotGeneratorRegression(unittest.TestCase):
    """Each category generates 200 examples; all must pass the same gates."""

    PER_CAT = 200
    GENERATORS = [
        ("addition", lambda n: g.gen_addition_cot(n)),
        ("subtraction", lambda n: g.gen_subtraction_cot(n)),
        ("multiplication", lambda n: g.gen_multiplication_cot(n)),
        ("division", lambda n: g.gen_division_cot(n)),
        ("order_of_ops", lambda n: g.gen_order_of_ops_cot(n)),
        ("fractions", lambda n: g.gen_fractions_cot(n)),
        ("decimals", lambda n: g.gen_decimals_cot(n)),
        ("algebra_one_step", lambda n: g.gen_algebra_one_step_cot(n)),
        ("algebra_two_step", lambda n: g.gen_algebra_two_step_cot(n)),
        ("algebra_multi_step", lambda n: g.gen_algebra_multi_step_cot(n)),
        ("exponents", lambda n: g.gen_exponents_cot(n)),
    ]

    def test_no_banned_substrings_or_step_duplication(self):
        for name, gen in self.GENERATORS:
            with self.subTest(category=name):
                for text in gen(self.PER_CAT):
                    for banned in BANNED_SUBSTRINGS:
                        self.assertNotIn(banned, text, f"{name}: banned substring {banned!r}")
                    for pat in STEP_DUP_PATTERNS:
                        self.assertIsNone(pat.search(text),
                                          f"{name}: pattern {pat.pattern!r} matched")

    def test_arithmetic_answers_are_correct(self):
        """For categories whose printed problem is a single 'a op b' form, the
        printed Answer must match the ground-truth computation."""

        def check_simple(name, docs, opmap):
            for d in docs:
                p = _strip_surface(_problem_line(d) or "")
                for k, v in opmap.items():
                    p = p.replace(k, v)
                p = p.replace("−", "-")
                m = re.match(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)$", p)
                if not m:
                    continue
                a, c, b = int(m.group(1)), m.group(2), int(m.group(3))
                if c == "+":
                    expect = str(a + b)
                elif c == "-":
                    expect = str(a - b)
                elif c == "*":
                    expect = str(a * b)
                else:
                    q, r = divmod(a, b)
                    expect = str(q) if r == 0 else f"{q} remainder {r}"
                self.assertEqual(_declared_answer(d), expect, f"{name}: wrong answer in:\n{d}")

        for name, gen, opmap in [
            ("addition", g.gen_addition_cot, {}),
            ("subtraction", g.gen_subtraction_cot, {}),
            ("multiplication", g.gen_multiplication_cot,
             {"\\times": "*", "×": "*"}),
            ("division", g.gen_division_cot,
             {"\\div": "/", "÷": "/"}),
        ]:
            with self.subTest(category=name):
                check_simple(name, gen(self.PER_CAT), opmap)

    def test_order_of_ops_answers_match_python_eval(self):
        for d in g.gen_order_of_ops_cot(self.PER_CAT):
            p = _problem_line(d) or ""
            e = (_strip_surface(p)
                 .replace("×", "*").replace("÷", "/").replace("−", "-")
                 .replace(" = ?", "").replace("=?", "").strip())
            try:
                true_v = eval(e)
            except Exception:
                continue  # parser limitation, not a generator failure
            self.assertAlmostEqual(float(true_v), float(_declared_answer(d)),
                                   places=3, msg=f"order_of_ops wrong:\n{d}")

    def test_algebra_substitution_balances(self):
        for name, gen in [
            ("algebra_one_step", g.gen_algebra_one_step_cot),
            ("algebra_two_step", g.gen_algebra_two_step_cot),
            ("algebra_multi_step", g.gen_algebra_multi_step_cot),
        ]:
            with self.subTest(category=name):
                for d in gen(self.PER_CAT):
                    da = _declared_answer(d)
                    try:
                        x_val = int(da)
                    except Exception:
                        continue
                    eq = _strip_surface(_problem_line(d) or "").rstrip(".")
                    if "=" not in eq:
                        continue
                    lhs, rhs = eq.split("=", 1)

                    def ev(side):
                        s = side.replace("×", "*")
                        s = re.sub(r"(\d)x", r"\1*x", s)
                        s = re.sub(r"(\d)\(", r"\1*(", s)
                        s = s.replace("x", f"({x_val})")
                        try:
                            return eval(s)
                        except Exception:
                            return None
                    lv, rv = ev(lhs), ev(rhs)
                    if lv is None or rv is None:
                        continue
                    self.assertAlmostEqual(lv, rv, places=3,
                                           msg=f"{name}: x={x_val} doesn't balance\n{d}")

    def test_main_validation_gate_runs_and_writes_output(self):
        """End-to-end: generator -> validation gate -> file write."""
        original_categories = g.DEFAULT_CATEGORIES
        original_output_dir = g.OUTPUT_DIR
        original_output_file = g.OUTPUT_FILE
        try:
            g.DEFAULT_CATEGORIES = [(name, gen, 50) for name, gen, _ in original_categories]
            g.OUTPUT_DIR = tempfile.mkdtemp(prefix="cot_test_main_")
            g.OUTPUT_FILE = os.path.join(g.OUTPUT_DIR, "math_operations_cot.txt")
            g.main()
            self.assertTrue(os.path.exists(g.OUTPUT_FILE))
            with open(g.OUTPUT_FILE, encoding="utf-8") as f:
                content = f.read()
            self.assertGreater(content.count("<|endoftext|>"), 500)
        finally:
            g.DEFAULT_CATEGORIES = original_categories
            g.OUTPUT_DIR = original_output_dir
            g.OUTPUT_FILE = original_output_file


if __name__ == "__main__":
    unittest.main()
