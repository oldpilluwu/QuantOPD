"""Grading tests.

These run Math-Verify for real rather than patching it, because the behaviour worth pinning is
exactly the library's: that a bare LaTeX gold like ``x+1`` parses once wrapped in ``\\boxed{}``,
and that mathematically equivalent answers in different forms compare equal.
"""

from __future__ import annotations

import unittest
from unittest import mock

from math_verify.errors import TimeoutException

from opd.grading import extract_gold, grade, gsm8k_gold, summarize, wilson_interval


class Gsm8kGoldTest(unittest.TestCase):
    def test_extracts_the_answer_after_the_marker(self) -> None:
        self.assertEqual(gsm8k_gold("Janet sells 9 eggs.\nShe makes 9*2 = 18.\n#### 18"), "18")

    def test_strips_thousands_separators_and_currency(self) -> None:
        self.assertEqual(gsm8k_gold("Working...\n#### $1,234"), "1234")

    def test_rejects_an_answer_with_no_marker(self) -> None:
        with self.assertRaises(ValueError):
            gsm8k_gold("There is no marker here")

    def test_extract_gold_dispatches_on_benchmark(self) -> None:
        self.assertEqual(extract_gold("gsm8k", "text\n#### 7"), "7")
        self.assertEqual(extract_gold("math500", r"  \frac{1}{2} "), r"\frac{1}{2}")


class GradeTest(unittest.TestCase):
    def test_exact_match_is_correct(self) -> None:
        result = grade("10", r"The answer is \boxed{10}.")
        self.assertTrue(result.correct)
        self.assertTrue(result.gold_parseable)
        self.assertTrue(result.prediction_parseable)

    def test_wrong_answer_is_incorrect(self) -> None:
        self.assertFalse(grade("10", r"The answer is \boxed{11}.").correct)

    def test_equivalent_forms_compare_equal(self) -> None:
        self.assertTrue(grade(r"\frac{1}{2}", r"The answer is \boxed{0.5}").correct)

    def test_symbolically_equivalent_expressions_compare_equal(self) -> None:
        """Bare 'x+1' is unparseable without the \\boxed{} wrapper; this pins the wrapper."""
        result = grade("x+1", r"answer: \boxed{1+x}")
        self.assertTrue(result.gold_parseable)
        self.assertTrue(result.correct)

    def test_text_gold_parses_once_wrapped(self) -> None:
        result = grade(r"\text{even}", r"the answer is $\boxed{\text{even}}$")
        self.assertTrue(result.gold_parseable)
        self.assertTrue(result.correct)

    def test_unparseable_prediction_counts_as_wrong_not_missing(self) -> None:
        result = grade("10", "I am not going to answer that.")
        self.assertFalse(result.correct)
        self.assertFalse(result.prediction_parseable)
        self.assertTrue(result.gold_parseable)


class GradingFailureTest(unittest.TestCase):
    """A grading failure must be counted -- never raised, never scored as a wrong answer.

    Math-Verify's TimeoutException subclasses BaseException, so `except Exception` does not catch
    it and one pathological olympiad expression aborts an entire evaluation run.
    """

    def test_timeout_exception_is_not_an_exception_subclass(self) -> None:
        """The property that makes this easy to get wrong."""
        self.assertFalse(issubclass(TimeoutException, Exception))
        self.assertTrue(issubclass(TimeoutException, BaseException))

    def test_comparison_timeout_is_caught_and_recorded(self) -> None:
        def boom(*args, **kwargs):
            raise TimeoutException("Timeout during comparison")

        with mock.patch("opd.grading.verify", side_effect=boom):
            result = grade("10", r"\boxed{10}")
        self.assertFalse(result.correct)
        self.assertTrue(result.prediction_parseable)
        self.assertIn("Timeout", result.error)

    def test_parse_timeout_is_caught_and_recorded(self) -> None:
        def boom(*args, **kwargs):
            raise TimeoutException("Timeout during parsing")

        with mock.patch("opd.grading.parse", side_effect=boom):
            result = grade("10", r"\boxed{10}")
        self.assertFalse(result.correct)
        self.assertIn("Timeout", result.error)

    def test_sympy_style_exception_is_caught_and_recorded(self) -> None:
        with mock.patch("opd.grading.verify", side_effect=ValueError("sympy exploded")):
            result = grade("10", r"\boxed{10}")
        self.assertFalse(result.correct)
        self.assertIn("ValueError", result.error)

    def test_timeouts_are_counted_separately_from_other_errors(self) -> None:
        def boom(*args, **kwargs):
            raise TimeoutException("Timeout during comparison")

        with mock.patch("opd.grading.verify", side_effect=boom):
            timed_out = grade("10", r"\boxed{10}")
        with mock.patch("opd.grading.verify", side_effect=ValueError("nope")):
            errored = grade("10", r"\boxed{10}")
        summary = summarize([timed_out, errored, grade("10", r"\boxed{10}")])
        self.assertEqual(summary["verification_errors"], 2)
        self.assertEqual(summary["verification_timeouts"], 1)


class WilsonIntervalTest(unittest.TestCase):
    """Headline accuracies at n=100 carry roughly +/-8 points; the interval makes that visible."""

    def test_interval_brackets_the_estimate(self) -> None:
        interval = wilson_interval(27, 100)
        self.assertLess(interval["lower"], 0.27)
        self.assertGreater(interval["upper"], 0.27)

    def test_interval_narrows_with_more_samples(self) -> None:
        narrow = wilson_interval(270, 1000)
        wide = wilson_interval(27, 100)
        self.assertLess(narrow["upper"] - narrow["lower"], wide["upper"] - wide["lower"])

    def test_interval_stays_inside_zero_one(self) -> None:
        for correct, total in ((0, 10), (10, 10), (1, 3)):
            interval = wilson_interval(correct, total)
            self.assertGreaterEqual(interval["lower"], 0.0)
            self.assertLessEqual(interval["upper"], 1.0)

    def test_empty_sample_has_no_interval(self) -> None:
        self.assertIsNone(wilson_interval(0, 0))

    def test_observed_omnimath_conditions_are_not_separated(self) -> None:
        """4B at 0.20 and 14B-NF4 at 0.27 over n=100 overlap, so the ordering is not real yet."""
        four_b = wilson_interval(20, 100)
        fourteen_b = wilson_interval(27, 100)
        self.assertLess(fourteen_b["lower"], four_b["upper"])


class SummarizeTest(unittest.TestCase):
    def test_accuracy_and_parse_rate_are_reported_separately(self) -> None:
        grades = [
            grade("10", r"\boxed{10}"),  # correct
            grade("10", r"\boxed{11}"),  # wrong but parseable
            grade("10", "no answer here"),  # unparseable
        ]
        summary = summarize(grades)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["correct"], 1)
        self.assertAlmostEqual(summary["accuracy"], 1 / 3)
        # One of the two parseable predictions was right.
        self.assertAlmostEqual(summary["accuracy_on_parseable"], 1 / 2)
        self.assertAlmostEqual(summary["prediction_parse_failure_rate"], 1 / 3)
        self.assertIsNotNone(summary["accuracy_ci95"])

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
