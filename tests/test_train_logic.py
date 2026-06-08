"""Regression tests for two pretrain/train.py fixes:

  1. Clean-break gradient flush at shard boundaries — the trailing partial
     accumulation window must be rescaled so its gradient is the true mean over
     the micro-batches it actually contains (not under-weighted by 1/GRAD_ACCUM).
  2. seen_shards as a MULTISET — multi-epoch (max_epochs>1) passes must be
     subtracted by count on resume so repeated sources converge instead of
     being retrained every resume.

train.py is a top-level script (importing it would launch training and needs a
GPU + data), so these tests mirror the exact arithmetic/logic of the relevant
blocks rather than importing them. Keep them in sync with section 8 and the
training loop in pretrain/train.py.
"""

import unittest
from collections import defaultdict


GRAD_ACCUM_STEPS = 64


# --- mirrors of the train.py logic under test -------------------------------

def window_grad_buggy(micro_grads):
    """Old behaviour: every micro-batch loss divided by GRAD_ACCUM_STEPS and
    summed. Correct for a full window, WRONG (under-weighted) for a partial one.
    """
    return sum(g / GRAD_ACCUM_STEPS for g in micro_grads)


def window_grad_fixed(micro_grads):
    """Clean-break flush: accumulate the same way, then rescale the partial
    window by GRAD_ACCUM_STEPS / micro_in_window before the step."""
    acc = sum(g / GRAD_ACCUM_STEPS for g in micro_grads)
    w = len(micro_grads)
    scale = GRAD_ACCUM_STEPS / w
    return acc * scale


def true_mean(micro_grads):
    """The gradient we actually want: mean over the micro-batches present."""
    return sum(micro_grads) / len(micro_grads)


def subtract_seen(plan_shards, seen_passes):
    """Mirror of section 8: drop already-trained passes by COUNT, not membership."""
    counts = defaultdict(int)
    for name in seen_passes:
        counts[name] += 1
    remaining = []
    for s in plan_shards:
        if counts[s] > 0:
            counts[s] -= 1
        else:
            remaining.append(s)
    return remaining


def trim_to_need(shard_tokens, needed_tokens):
    """Mirror of the fresh-selection trim: take shards until the budget is met."""
    out, acc = [], 0
    for t in shard_tokens:
        out.append(t)
        acc += t
        if acc >= needed_tokens:
            break
    return out


# --- tests ------------------------------------------------------------------

class TestPartialWindowRescale(unittest.TestCase):
    def test_full_window_unchanged(self):
        grads = [float(i) for i in range(GRAD_ACCUM_STEPS)]
        # Full window: buggy == fixed == true mean.
        self.assertAlmostEqual(window_grad_buggy(grads), true_mean(grads), places=12)
        self.assertAlmostEqual(window_grad_fixed(grads), true_mean(grads), places=12)

    def test_partial_window_buggy_is_underweighted(self):
        grads = [1.0] * 10  # 10 of 64
        # Buggy path divides by 64 -> 10/64, far below the true mean of 1.0.
        self.assertAlmostEqual(window_grad_buggy(grads), 10 / 64, places=12)
        self.assertLess(window_grad_buggy(grads), true_mean(grads))

    def test_partial_window_fixed_matches_true_mean(self):
        for w in (1, 3, 7, 17, 63):
            grads = [0.5 * i + 1.0 for i in range(w)]
            self.assertAlmostEqual(
                window_grad_fixed(grads), true_mean(grads), places=12,
                msg=f"rescale wrong for window size {w}",
            )

    def test_fixed_vectorwise(self):
        # Same property must hold component-wise (gradients are tensors).
        w = 5
        micro = [[1.0, 2.0, 3.0], [0.0, 1.0, 1.0], [2.0, 2.0, 2.0],
                 [1.0, 0.0, 4.0], [3.0, 3.0, 0.0]]
        for dim in range(3):
            col = [m[dim] for m in micro[:w]]
            self.assertAlmostEqual(window_grad_fixed(col), true_mean(col), places=12)


class TestSeenShardsMultiset(unittest.TestCase):
    def test_single_epoch_subtraction(self):
        plan = ["a", "b", "c"]
        seen = ["a", "c"]  # trained before
        self.assertEqual(subtract_seen(plan, seen), ["b"])

    def test_multi_epoch_partial(self):
        # wikipedia shard with 4 planned passes, 2 already done -> 2 remain.
        plan = ["w", "w", "w", "w"]
        seen = ["w", "w"]
        self.assertEqual(subtract_seen(plan, seen), ["w", "w"])

    def test_multi_epoch_converges_with_multiset(self):
        # Multiset (append per pass) -> after one resume trains all 4, seen=4,
        # a second resume of the same plan leaves nothing. Converges.
        plan = ["w", "w", "w", "w"]
        seen = []
        seen += [s for s in subtract_seen(plan, seen)]  # train remaining -> append each
        self.assertEqual(len(seen), 4)
        self.assertEqual(subtract_seen(plan, seen), [])

    def test_set_behaviour_would_not_converge(self):
        # Demonstrates the bug being fixed: if seen were a SET, a 4-pass shard
        # records once, so 3 passes get retrained on every resume forever.
        plan = ["w", "w", "w", "w"]
        seen_as_set = sorted(set(plan))  # the old save format
        leftover = subtract_seen(plan, seen_as_set)
        self.assertEqual(leftover, ["w", "w", "w"])  # never converges

    def test_old_set_checkpoint_backward_compatible(self):
        # An existing checkpoint saved unique-only (set) still subtracts one pass
        # per shard correctly for the common max_epochs==1 case.
        plan = ["a", "b", "c", "d"]
        old_seen = ["a", "b"]  # unique entries from a set-format checkpoint
        self.assertEqual(subtract_seen(plan, old_seen), ["c", "d"])


class TestFreshTrim(unittest.TestCase):
    def test_trim_reaches_need(self):
        shards = [50] * 100  # 50M-token shards
        picked = trim_to_need(shards, 260)  # need ~260M
        self.assertEqual(len(picked), 6)      # 6*50=300 >= 260
        self.assertGreaterEqual(sum(picked), 260)

    def test_trim_takes_all_when_short(self):
        shards = [50, 50, 50]
        picked = trim_to_need(shards, 1000)
        self.assertEqual(picked, shards)


if __name__ == "__main__":
    unittest.main()
