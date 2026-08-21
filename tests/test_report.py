"""Tests for report aggregation.

The comparability check is the point: `--limit N` takes the *first* N frozen indices, so runs at
different N cover different problems. Comparing them produces a plausible-looking ordering that is
purely an artifact of which subset each model saw.
"""

from __future__ import annotations

import unittest

from opd.report import build_headline, find_item_count_mismatches


def row(model: str, precision: str, tag: str, benchmark: str, items: int, accuracy: float) -> dict:
    return {
        "model": model,
        "precision": precision,
        "tag": tag,
        "benchmark": benchmark,
        "items": items,
        "accuracy": accuracy,
        "truncation_rate": 0.3,
        "teacher_memory_gib": None,
    }


class ItemCountMismatchTest(unittest.TestCase):
    def test_matching_item_counts_produce_no_warning(self) -> None:
        rows = [
            row("Qwen/Qwen3-1.7B", "bf16", "baseline", "omnimath", 300, 0.253),
            row("Qwen/Qwen3-14B", "int4", "teacher", "omnimath", 300, 0.31),
        ]
        self.assertEqual(find_item_count_mismatches(rows), [])

    def test_differing_item_counts_are_flagged(self) -> None:
        """The exact situation hit in practice: student at 300, teachers at 100."""
        rows = [
            row("Qwen/Qwen3-1.7B", "bf16", "baseline", "omnimath", 300, 0.253),
            row("Qwen/Qwen3-4B", "bf16", "teacher", "omnimath", 100, 0.20),
            row("Qwen/Qwen3-14B", "bf16", "teacher", "omnimath", 100, 0.23),
        ]
        mismatches = find_item_count_mismatches(rows)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["benchmark"], "omnimath")
        self.assertEqual(sorted(mismatches[0]["item_counts"]), [100, 300])

    def test_each_benchmark_is_checked_independently(self) -> None:
        rows = [
            row("a", "bf16", "t", "math500", 500, 0.8),
            row("b", "bf16", "t", "math500", 500, 0.8),
            row("a", "bf16", "t", "omnimath", 300, 0.2),
            row("b", "bf16", "t", "omnimath", 100, 0.2),
        ]
        mismatches = find_item_count_mismatches(rows)
        self.assertEqual([m["benchmark"] for m in mismatches], ["omnimath"])

    def test_offending_conditions_are_named(self) -> None:
        rows = [
            row("Qwen/Qwen3-1.7B", "bf16", "baseline", "omnimath", 300, 0.253),
            row("Qwen/Qwen3-4B", "bf16", "teacher", "omnimath", 100, 0.20),
        ]
        counts = find_item_count_mismatches(rows)[0]["item_counts"]
        self.assertIn("Qwen/Qwen3-4B:bf16:teacher", counts[100])
        self.assertIn("Qwen/Qwen3-1.7B:bf16:baseline", counts[300])

    def test_no_evaluations_is_not_a_mismatch(self) -> None:
        self.assertEqual(find_item_count_mismatches([]), [])


class HeadlineTest(unittest.TestCase):
    def test_one_row_per_condition_and_benchmark(self) -> None:
        rows = [
            row("Qwen/Qwen3-1.7B", "bf16", "baseline", "omnimath", 300, 0.253),
            row("Qwen/Qwen3-1.7B", "bf16", "baseline", "math500", 500, 0.70),
            row("Qwen/Qwen3-14B", "int4", "teacher", "omnimath", 300, 0.31),
        ]
        headline = build_headline(rows)
        self.assertEqual(len(headline), 3)
        self.assertEqual(
            sorted((r["condition"], r["benchmark"]) for r in headline),
            [("baseline", "math500"), ("baseline", "omnimath"), ("teacher", "omnimath")],
        )


if __name__ == "__main__":
    unittest.main()
