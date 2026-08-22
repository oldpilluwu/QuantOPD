"""Tests for the plotting helpers.

Only the pure logic is tested -- rendering needs matplotlib, which lives in the optional `viz`
group. The colour guard is the important one: cycling a categorical palette silently gives two
models the same identity, which is a wrong chart rather than an ugly one.
"""

from __future__ import annotations

import unittest

from opd.plots import SERIES, TEACHER_MEMORY_GIB, series_colour, short_name


class SeriesColourTest(unittest.TestCase):
    def test_hues_are_assigned_in_fixed_order(self) -> None:
        self.assertEqual([series_colour(i) for i in range(3)], list(SERIES[:3]))

    def test_every_slot_is_distinct(self) -> None:
        self.assertEqual(len(set(SERIES)), len(SERIES))

    def test_exceeding_the_palette_raises_rather_than_cycling(self) -> None:
        """A 9th series must not silently reuse slot 1 -- that happened and put two models in blue."""
        with self.assertRaises(ValueError) as context:
            series_colour(len(SERIES))
        self.assertIn("exceeds", str(context.exception))

    def test_the_five_current_conditions_fit_without_cycling(self) -> None:
        colours = [series_colour(i) for i in range(5)]
        self.assertEqual(len(set(colours)), 5)


class ShortNameTest(unittest.TestCase):
    def test_precision_is_spelled_out_by_quantizer(self) -> None:
        self.assertEqual(short_name("Qwen/Qwen3-14B", "int4"), "14B NF4")
        self.assertEqual(short_name("Qwen/Qwen3-14B", "bf16"), "14B BF16")
        self.assertEqual(short_name("Qwen/Qwen3-4B", "bf16"), "4B BF16")

    def test_prequantized_checkpoints_do_not_repeat_the_quantizer(self) -> None:
        """The repo id already carries -AWQ / -GPTQ-Int4; the label should not say it twice."""
        self.assertEqual(short_name("Qwen/Qwen3-14B-AWQ", "awq"), "14B AWQ")
        self.assertEqual(short_name("JunHowie/Qwen3-14B-GPTQ-Int4", "gptq"), "14B GPTQ")

    def test_labels_are_unique_across_the_configured_conditions(self) -> None:
        labels = [short_name(model, precision) for model, precision in TEACHER_MEMORY_GIB]
        self.assertEqual(len(set(labels)), len(labels))


class TeacherMemoryTest(unittest.TestCase):
    def test_the_equal_memory_pair_really_is_close(self) -> None:
        """The study's central comparison only makes sense if these budgets nearly match."""
        four_b = TEACHER_MEMORY_GIB[("Qwen/Qwen3-4B", "bf16")]
        fourteen_b_int4 = TEACHER_MEMORY_GIB[("Qwen/Qwen3-14B", "int4")]
        self.assertLess(abs(fourteen_b_int4 - four_b) / four_b, 0.25)

    def test_the_three_four_bit_quantizers_share_a_budget(self) -> None:
        sizes = [
            TEACHER_MEMORY_GIB[("Qwen/Qwen3-14B", "int4")],
            TEACHER_MEMORY_GIB[("Qwen/Qwen3-14B-AWQ", "awq")],
            TEACHER_MEMORY_GIB[("JunHowie/Qwen3-14B-GPTQ-Int4", "gptq")],
        ]
        self.assertLess(max(sizes) - min(sizes), 0.5)

    def test_bf16_is_far_larger_than_any_four_bit_variant(self) -> None:
        self.assertGreater(
            TEACHER_MEMORY_GIB[("Qwen/Qwen3-14B", "bf16")],
            3 * TEACHER_MEMORY_GIB[("Qwen/Qwen3-14B", "int4")] - 1,
        )


if __name__ == "__main__":
    unittest.main()
