"""Prompt rendering, especially the non-thinking shim.

TRL's collator calls ``apply_chat_template`` with no ``chat_template_kwargs``, so the only way the
trainer renders non-thinking prompts is if the tokenizer itself has the flag bound. These tests use
a stub template that mimics Qwen3's behaviour so they run without downloading a model.
"""

from __future__ import annotations

import unittest

from opd.models import force_non_thinking
from opd.prompts import build_messages, is_non_thinking, render_prompt, render_prompt_ids


class StubTokenizer:
    """Mimics Qwen3: enable_thinking defaults to True and appends an empty think block when False."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        self.calls.append({"enable_thinking": enable_thinking, "add_generation_prompt": add_generation_prompt})
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            body += "<|im_start|>assistant\n"
            if not enable_thinking:
                body += "<think>\n\n</think>\n\n"
        return body

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) % 256 for character in text]}


class NonThinkingShimTest(unittest.TestCase):
    def test_thinking_is_the_default_without_the_shim(self) -> None:
        tokenizer = StubTokenizer()
        self.assertFalse(is_non_thinking(render_prompt(tokenizer, "2+2?")))

    def test_shim_makes_non_thinking_the_default(self) -> None:
        tokenizer = force_non_thinking(StubTokenizer())
        self.assertTrue(is_non_thinking(render_prompt(tokenizer, "2+2?")))

    def test_shim_applies_to_a_bare_trl_style_call(self) -> None:
        """The exact call shape TRL's DistillationDataCollator uses."""
        tokenizer = force_non_thinking(StubTokenizer())
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
        )
        self.assertTrue(is_non_thinking(rendered))
        self.assertFalse(tokenizer.calls[-1]["enable_thinking"])

    def test_shim_does_not_override_an_explicit_argument(self) -> None:
        """setdefault, not force: a caller that explicitly asks for thinking still gets it."""
        tokenizer = force_non_thinking(StubTokenizer())
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        self.assertFalse(is_non_thinking(rendered))

    def test_shim_is_idempotent(self) -> None:
        tokenizer = force_non_thinking(force_non_thinking(StubTokenizer()))
        self.assertTrue(is_non_thinking(render_prompt(tokenizer, "2+2?")))


class ThinkBlockDetectionTest(unittest.TestCase):
    def test_detects_the_real_qwen3_rendering(self) -> None:
        self.assertTrue(is_non_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_rejects_a_generation_prompt_with_no_think_block(self) -> None:
        self.assertFalse(is_non_thinking("<|im_start|>assistant\n"))

    def test_rejects_a_think_block_with_content(self) -> None:
        self.assertFalse(is_non_thinking("<think>let me work this out</think>"))


class MessageConstructionTest(unittest.TestCase):
    def test_question_is_paired_with_the_answer_format_instruction(self) -> None:
        messages = build_messages("  What is 2+2?  ")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].startswith("What is 2+2?"))
        self.assertIn(r"\boxed{}", messages[0]["content"])

    def test_prompt_ids_match_the_rendered_text(self) -> None:
        tokenizer = force_non_thinking(StubTokenizer())
        text = render_prompt(tokenizer, "2+2?")
        self.assertEqual(render_prompt_ids(tokenizer, "2+2?"), tokenizer(text)["input_ids"])


if __name__ == "__main__":
    unittest.main()
