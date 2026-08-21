"""Tests for the distribution metrics.

These pin the things that would otherwise be silently wrong rather than loudly broken: the
prompt/completion logit alignment, the self-comparison identities, and shift invariance.
"""

from __future__ import annotations

import math
import unittest

import torch

from opd.metrics import (
    _bootstrap_ci,
    aggregate,
    completion_logits,
    merge_position_profile,
    score_trajectory,
    summarize_divergence,
    trajectory_means,
)

TOP_KS = (1, 5)
VOCAB = 11


class FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    """Returns logits whose value encodes the absolute position, so alignment is checkable."""

    def __init__(self, total_length: int, vocab: int = VOCAB, honour_logits_to_keep: bool = True) -> None:
        self.total_length = total_length
        self.vocab = vocab
        self.honour_logits_to_keep = honour_logits_to_keep
        # Position i gets argmax at token (i % vocab), so a row's identity is recoverable.
        full = torch.zeros(1, total_length, vocab)
        for position in range(total_length):
            full[0, position, position % vocab] = 10.0
        self.full = full

    def __call__(self, input_ids: torch.Tensor, use_cache: bool = False, logits_to_keep: int | None = None):
        if logits_to_keep is not None:
            if not self.honour_logits_to_keep:
                return FakeOutput(self.full)
            return FakeOutput(self.full[:, -logits_to_keep:, :])
        return FakeOutput(self.full)


class FakeModelWithoutLogitsToKeep(FakeModel):
    def __call__(self, input_ids: torch.Tensor, use_cache: bool = False, logits_to_keep: int | None = None):
        if logits_to_keep is not None:
            raise TypeError("unexpected keyword argument 'logits_to_keep'")
        return FakeOutput(self.full)


class CompletionLogitsAlignmentTest(unittest.TestCase):
    prompt_length = 7
    completion_length = 4

    @property
    def total(self) -> int:
        return self.prompt_length + self.completion_length

    def _expected_argmax(self) -> list[int]:
        # Completion token j is predicted by absolute position P-1+j.
        return [(self.prompt_length - 1 + j) % VOCAB for j in range(self.completion_length)]

    def test_alignment_with_logits_to_keep(self) -> None:
        model = FakeModel(self.total)
        input_ids = torch.zeros(1, self.total, dtype=torch.long)
        logits = completion_logits(model, input_ids, self.prompt_length, self.completion_length)
        self.assertEqual(logits.shape, (self.completion_length, VOCAB))
        self.assertEqual(logits.argmax(dim=-1).tolist(), self._expected_argmax())

    def test_alignment_when_logits_to_keep_is_unsupported(self) -> None:
        model = FakeModelWithoutLogitsToKeep(self.total)
        input_ids = torch.zeros(1, self.total, dtype=torch.long)
        logits = completion_logits(model, input_ids, self.prompt_length, self.completion_length)
        self.assertEqual(logits.argmax(dim=-1).tolist(), self._expected_argmax())

    def test_alignment_when_logits_to_keep_is_ignored(self) -> None:
        model = FakeModel(self.total, honour_logits_to_keep=False)
        input_ids = torch.zeros(1, self.total, dtype=torch.long)
        logits = completion_logits(model, input_ids, self.prompt_length, self.completion_length)
        self.assertEqual(logits.argmax(dim=-1).tolist(), self._expected_argmax())

    def test_all_three_paths_agree(self) -> None:
        input_ids = torch.zeros(1, self.total, dtype=torch.long)
        results = [
            completion_logits(model, input_ids, self.prompt_length, self.completion_length)
            for model in (
                FakeModel(self.total),
                FakeModelWithoutLogitsToKeep(self.total),
                FakeModel(self.total, honour_logits_to_keep=False),
            )
        ]
        self.assertTrue(torch.equal(results[0], results[1]))
        self.assertTrue(torch.equal(results[1], results[2]))

    def test_rejects_batched_input(self) -> None:
        model = FakeModel(self.total)
        with self.assertRaises(ValueError):
            completion_logits(model, torch.zeros(2, self.total, dtype=torch.long), self.prompt_length, 4)

    def test_rejects_length_mismatch(self) -> None:
        model = FakeModel(self.total)
        with self.assertRaises(ValueError):
            completion_logits(model, torch.zeros(1, self.total, dtype=torch.long), self.prompt_length, 99)


class SelfComparisonTest(unittest.TestCase):
    """A teacher that IS the student must produce exactly zero divergence."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.logits = torch.randn(6, VOCAB)
        self.tokens = self.logits.argmax(dim=-1).tolist()

    def _means(self, student: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
        accumulator = score_trajectory(student, teacher, self.tokens, TOP_KS, chunk_size=4, profile_bins=3)
        return trajectory_means(accumulator, TOP_KS)

    def test_zero_divergence_against_itself(self) -> None:
        means = self._means(self.logits, self.logits)
        self.assertAlmostEqual(means["reverse_kl"], 0.0, places=10)
        self.assertAlmostEqual(means["forward_kl"], 0.0, places=10)
        self.assertAlmostEqual(means["entropy_shift"], 0.0, places=10)
        self.assertAlmostEqual(means["top_1_approx_reverse_kl"], 0.0, places=10)

    def test_full_agreement_against_itself(self) -> None:
        means = self._means(self.logits, self.logits)
        self.assertAlmostEqual(means["top_1_agreement"], 1.0, places=10)
        self.assertAlmostEqual(means["top_5_agreement"], 1.0, places=10)

    def test_entropies_match_against_itself(self) -> None:
        means = self._means(self.logits, self.logits)
        self.assertAlmostEqual(means["student_entropy"], means["teacher_entropy"], places=10)

    def test_no_divergence_position_when_tokens_are_argmax(self) -> None:
        accumulator = score_trajectory(self.logits, self.logits, self.tokens, TOP_KS, 4, 3)
        self.assertIsNone(accumulator.first_divergence)


class InvarianceTest(unittest.TestCase):
    def test_kl_is_invariant_to_a_constant_logit_shift(self) -> None:
        torch.manual_seed(1)
        student = torch.randn(5, VOCAB)
        teacher = torch.randn(5, VOCAB)
        tokens = [0, 1, 2, 3, 4]

        base = trajectory_means(score_trajectory(student, teacher, tokens, TOP_KS, 8, 2), TOP_KS)
        shifted = trajectory_means(
            score_trajectory(student + 3.5, teacher - 1.25, tokens, TOP_KS, 8, 2), TOP_KS
        )
        for key in ("reverse_kl", "forward_kl", "student_entropy", "teacher_entropy"):
            self.assertAlmostEqual(base[key], shifted[key], places=6, msg=key)

    def test_chunk_size_does_not_change_results(self) -> None:
        torch.manual_seed(2)
        student = torch.randn(9, VOCAB)
        teacher = torch.randn(9, VOCAB)
        tokens = list(range(9))
        a = trajectory_means(score_trajectory(student, teacher, tokens, TOP_KS, 1, 3), TOP_KS)
        b = trajectory_means(score_trajectory(student, teacher, tokens, TOP_KS, 9, 3), TOP_KS)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=9, msg=key)

    def test_divergent_teacher_gives_positive_kl(self) -> None:
        student = torch.zeros(3, VOCAB)
        teacher = torch.zeros(3, VOCAB)
        teacher[:, 0] = 8.0
        means = trajectory_means(score_trajectory(student, teacher, [1, 1, 1], TOP_KS, 4, 2), TOP_KS)
        self.assertGreater(means["reverse_kl"], 0.0)
        self.assertGreater(means["forward_kl"], 0.0)
        self.assertTrue(math.isfinite(means["reverse_kl"]))

    def test_vocab_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            score_trajectory(torch.zeros(2, 5), torch.zeros(2, 7), [0, 1], TOP_KS, 4, 2)
        self.assertIn("not comparable", str(context.exception))


class TeacherMassTest(unittest.TestCase):
    def test_mass_on_full_support_is_one(self) -> None:
        torch.manual_seed(3)
        student = torch.randn(4, VOCAB)
        teacher = torch.randn(4, VOCAB)
        accumulator = score_trajectory(student, teacher, [0, 1, 2, 3], (VOCAB,), 4, 2)
        means = trajectory_means(accumulator, (VOCAB,))
        self.assertAlmostEqual(means[f"teacher_mass_on_student_top_{VOCAB}"], 1.0, places=6)

    def test_mass_is_monotonic_in_k(self) -> None:
        torch.manual_seed(4)
        student = torch.randn(4, VOCAB)
        teacher = torch.randn(4, VOCAB)
        means = trajectory_means(score_trajectory(student, teacher, [0, 1, 2, 3], (1, 5, 10), 4, 2), (1, 5, 10))
        self.assertLessEqual(means["teacher_mass_on_student_top_1"], means["teacher_mass_on_student_top_5"])
        self.assertLessEqual(means["teacher_mass_on_student_top_5"], means["teacher_mass_on_student_top_10"])


class FirstDivergenceTest(unittest.TestCase):
    def test_reports_first_position_where_teacher_disagrees(self) -> None:
        teacher = torch.zeros(4, VOCAB)
        for position in range(4):
            teacher[position, position] = 5.0
        # Tokens match the teacher argmax until position 2.
        accumulator = score_trajectory(torch.zeros(4, VOCAB), teacher, [0, 1, 9, 3], TOP_KS, 2, 2)
        self.assertEqual(accumulator.first_divergence, 2)

    def test_summary_handles_never_diverging_trajectories(self) -> None:
        teacher = torch.zeros(2, VOCAB)
        teacher[:, 0] = 5.0
        accumulator = score_trajectory(torch.zeros(2, VOCAB), teacher, [0, 0], TOP_KS, 2, 2)
        summary = summarize_divergence([accumulator])
        self.assertEqual(summary["trajectories_with_divergence"], 0)
        self.assertIsNone(summary["mean_first_divergence_position"])


class AggregationTest(unittest.TestCase):
    def test_prompt_and_token_weighting_differ_as_expected(self) -> None:
        per_trajectory = [{"reverse_kl": 1.0}, {"reverse_kl": 3.0}]
        token_counts = [1, 99]
        result = aggregate(per_trajectory, token_counts, bootstrap_samples=0, bootstrap_seed=1)
        self.assertAlmostEqual(result["reverse_kl"]["prompt_weighted"]["mean"], 2.0)
        # Token weighting is dominated by the long trajectory.
        self.assertAlmostEqual(result["reverse_kl"]["token_weighted"]["mean"], (1.0 + 3.0 * 99) / 100)

    def test_bootstrap_is_deterministic_under_a_fixed_seed(self) -> None:
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        first = _bootstrap_ci(values, None, samples=200, seed=2026)
        second = _bootstrap_ci(values, None, samples=200, seed=2026)
        self.assertEqual(first, second)
        self.assertLess(first["lower"], first["upper"])

    def test_bootstrap_interval_brackets_the_sample_mean(self) -> None:
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        interval = _bootstrap_ci(values, None, samples=500, seed=2026)
        mean = sum(values) / len(values)
        self.assertLessEqual(interval["lower"], mean)
        self.assertGreaterEqual(interval["upper"], mean)

    def test_bootstrap_returns_none_for_a_single_trajectory(self) -> None:
        self.assertIsNone(_bootstrap_ci([0.5], None, 100, 1))

    def test_empty_aggregation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate([], [], 10, 1)


class PositionProfileTest(unittest.TestCase):
    def test_profile_covers_every_token_exactly_once(self) -> None:
        torch.manual_seed(5)
        tokens = [i % VOCAB for i in range(20)]
        accumulator = score_trajectory(torch.randn(20, VOCAB), torch.randn(20, VOCAB), tokens, TOP_KS, 7, 4)
        profile = merge_position_profile([accumulator], bins=4)
        self.assertEqual(sum(entry["tokens"] for entry in profile), 20)

    def test_profile_mean_matches_overall_mean(self) -> None:
        torch.manual_seed(6)
        tokens = [i % VOCAB for i in range(12)]
        accumulator = score_trajectory(torch.randn(12, VOCAB), torch.randn(12, VOCAB), tokens, TOP_KS, 5, 3)
        profile = merge_position_profile([accumulator], bins=3)
        weighted = sum(entry["mean_reverse_kl"] * entry["tokens"] for entry in profile) / 12
        self.assertAlmostEqual(weighted, trajectory_means(accumulator, TOP_KS)["reverse_kl"], places=9)


if __name__ == "__main__":
    unittest.main()
