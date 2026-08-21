"""Tests for precision handling.

The guard here matters because the failure it prevents is silent: loading a BF16 checkpoint while
reporting AWQ produces a plausible-looking number that says "4-bit costs nothing".
"""

from __future__ import annotations

import unittest

import torch
from transformers import BitsAndBytesConfig

from opd.config import PREQUANTIZED_PRECISIONS, VALID_PRECISIONS
from opd.models import make_quantization_config, verify_prequantized


class Config:
    def __init__(self, quantization_config=None) -> None:
        if quantization_config is not None:
            self.quantization_config = quantization_config


class Model:
    def __init__(self, quantization_config=None) -> None:
        self.config = Config(quantization_config)


class QuantizationConfigTest(unittest.TestCase):
    def test_bf16_needs_no_quantization_config(self) -> None:
        self.assertIsNone(make_quantization_config("bf16"))

    def test_prequantized_needs_no_quantization_config(self) -> None:
        """AWQ weights ship quantized; passing a config would try to re-quantize them."""
        for precision in PREQUANTIZED_PRECISIONS:
            self.assertIsNone(make_quantization_config(precision), msg=precision)

    def test_bitsandbytes_precisions_produce_a_config(self) -> None:
        self.assertIsInstance(make_quantization_config("int8"), BitsAndBytesConfig)
        int4 = make_quantization_config("int4")
        self.assertIsInstance(int4, BitsAndBytesConfig)
        self.assertEqual(int4.bnb_4bit_quant_type, "nf4")
        self.assertEqual(int4.bnb_4bit_compute_dtype, torch.bfloat16)
        self.assertTrue(int4.bnb_4bit_use_double_quant)

    def test_unknown_precision_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_quantization_config("fp8")

    def test_prequantized_precisions_are_valid(self) -> None:
        for precision in ("awq", "gptq"):
            self.assertIn(precision, VALID_PRECISIONS, msg=precision)
            self.assertIn(precision, PREQUANTIZED_PRECISIONS, msg=precision)


class VerifyPrequantizedTest(unittest.TestCase):
    def test_accepts_a_matching_checkpoint(self) -> None:
        verify_prequantized(Model({"quant_method": "awq", "bits": 4}), "awq")

    def test_accepts_an_object_style_quantization_config(self) -> None:
        class Quant:
            quant_method = "awq"

        verify_prequantized(Model(Quant()), "awq")

    def test_is_case_insensitive(self) -> None:
        verify_prequantized(Model({"quant_method": "AWQ"}), "awq")

    def test_rejects_an_unquantized_checkpoint(self) -> None:
        """--model Qwen/Qwen3-14B --precision awq would otherwise measure BF16 and report AWQ."""
        with self.assertRaises(RuntimeError) as context:
            verify_prequantized(Model(), "awq")
        self.assertIn("quant_method=None", str(context.exception))

    def test_rejects_a_different_quantizer(self) -> None:
        """An AWQ request must not be satisfied by a GPTQ checkpoint, or vice versa."""
        with self.assertRaises(RuntimeError) as context:
            verify_prequantized(Model({"quant_method": "gptq"}), "awq")
        self.assertIn("gptq", str(context.exception))
        with self.assertRaises(RuntimeError):
            verify_prequantized(Model({"quant_method": "awq"}), "gptq")

    def test_accepts_a_matching_gptq_checkpoint(self) -> None:
        verify_prequantized(Model({"quant_method": "gptq", "bits": 4}), "gptq")


if __name__ == "__main__":
    unittest.main()
