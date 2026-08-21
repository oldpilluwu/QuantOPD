from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from opd.config import DEFAULT_CONFIG, VALID_PRECISIONS, load_config


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(Path(DEFAULT_CONFIG))

    def test_models_and_precisions_are_valid(self) -> None:
        self.assertTrue(self.config.models.student)
        self.assertTrue(set(self.config.models.precisions) <= VALID_PRECISIONS)
        self.assertIn("Qwen/Qwen3-4B", self.config.models.teachers)
        self.assertIn("Qwen/Qwen3-14B", self.config.models.teachers)

    def test_pilot_runs_non_thinking(self) -> None:
        self.assertFalse(self.config.models.enable_thinking)

    def test_data_subsets_fit_the_source(self) -> None:
        self.assertGreater(self.config.data.calibration_size, 0)
        self.assertLessEqual(self.config.data.smoke_size, self.config.data.training_size)

    def test_benchmarks_are_addressable_by_name(self) -> None:
        self.assertEqual(self.config.evaluation_dataset("math500").question_field, "problem")
        self.assertEqual(self.config.evaluation_dataset("omnimath").question_field, "problem")
        with self.assertRaises(ValueError) as context:
            self.config.evaluation_dataset("nonexistent")
        self.assertIn("omnimath", str(context.exception))

    def test_only_oversized_benchmarks_are_subsampled(self) -> None:
        """MATH-500 is already the right size; Omni-MATH's 4428 rows are not affordable."""
        self.assertIsNone(self.config.evaluation_dataset("math500").subsample_size)
        self.assertEqual(self.config.evaluation_dataset("omnimath").subsample_size, 300)

    def test_benchmarks_declare_grouping_columns(self) -> None:
        """Difficulty slicing is what shows whether a headline score is ceilinged."""
        self.assertIn("level", self.config.evaluation_dataset("math500").group_fields)
        self.assertIn("difficulty", self.config.evaluation_dataset("omnimath").group_fields)

    def test_eval_context_fits_prompt_plus_completion(self) -> None:
        self.assertGreater(self.config.eval.vllm_max_model_length, self.config.eval.max_new_tokens)

    def test_objective_is_fully_on_policy_reverse_kl(self) -> None:
        self.assertEqual(self.config.opd.lmbda, 1.0)
        self.assertEqual(self.config.opd.beta, 1.0)

    def test_opd_lengths_are_internally_consistent(self) -> None:
        opd = self.config.opd
        self.assertLessEqual(opd.max_prompt_length + opd.max_completion_length, opd.max_length)

    def test_scoring_and_trajectory_subsets_are_disjoint_from_training(self) -> None:
        """Diagnostics must not be measured on prompts OPD trained on."""
        self.assertNotEqual(self.config.trajectories.source_subset, self.config.opd.source_subset)


class ConfigValidationTest(unittest.TestCase):
    """The loader must reject configurations that would silently invalidate the results."""

    def setUp(self) -> None:
        with Path(DEFAULT_CONFIG).open("rb") as handle:
            self.raw = tomllib.load(handle)

    def _write(self, path: Path, raw: dict) -> None:
        def dump(value: object) -> str:
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, str):
                return f'"{value}"'
            if isinstance(value, list):
                return "[" + ", ".join(dump(item) for item in value) + "]"
            return str(value)

        lines: list[str] = []
        for section, body in raw.items():
            if section == "evaluation":
                for item in body["datasets"]:
                    lines.append("[[evaluation.datasets]]")
                    lines += [f"{k} = {dump(v)}" for k, v in item.items()]
                continue
            lines.append(f"[{section}]")
            lines += [f"{k} = {dump(v)}" for k, v in body.items()]
        path.write_text("\n".join(lines), encoding="utf-8")

    def test_rejects_rollout_settings_that_disagree_between_stages(self) -> None:
        import tempfile

        raw = self.raw
        raw["trajectories"]["temperature"] = 0.7  # [opd] still says 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            self._write(path, raw)
            with self.assertRaises(ValueError) as context:
                load_config(path)
        self.assertIn("rollout settings disagree", str(context.exception))

    def test_rejects_lengths_that_do_not_fit(self) -> None:
        import tempfile

        raw = self.raw
        raw["opd"]["max_completion_length"] = raw["opd"]["max_length"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            self._write(path, raw)
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
