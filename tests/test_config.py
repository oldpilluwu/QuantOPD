from pathlib import Path
from unittest import TestCase

from opd_phase0.config import load_config


class ConfigTests(TestCase):
    def test_phase0_config_loads(self) -> None:
        config = load_config(Path("configs/phase0.toml"))
        self.assertEqual(config.models.student, "Qwen/Qwen3-1.7B")
        self.assertEqual(config.models.precisions, ("bf16", "int8", "int4"))
        self.assertEqual(config.training.max_steps, 2)
        self.assertLess(config.training.max_completion_length, config.training.max_length)
