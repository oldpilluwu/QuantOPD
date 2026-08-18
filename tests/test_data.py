from unittest import TestCase

from opd_phase0.data import make_split_indices, select_prompt_field


class DataTests(TestCase):
    def test_split_indices_are_deterministic_and_disjoint(self) -> None:
        first = make_split_indices(100, calibration_size=10, training_size=20, smoke_size=4, seed=7)
        second = make_split_indices(100, calibration_size=10, training_size=20, smoke_size=4, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(set(first.calibration).isdisjoint(first.training))
        self.assertEqual(first.smoke, first.training[:4])

    def test_prompt_field_uses_first_available_candidate(self) -> None:
        field = select_prompt_field(["solution", "question", "problem"], ("problem", "question"))
        self.assertEqual(field, "problem")
