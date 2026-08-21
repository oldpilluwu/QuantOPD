"""Tests for the Transformers generation path.

`generate_hf` sorts prompts by length for batching efficiency, trims at the first EOS, and has to
restore the caller's ordering afterwards. None of that is exercised by the metrics tests, and all
of it would corrupt results silently: a completion attached to the wrong prompt still grades.
"""

from __future__ import annotations

import unittest

import torch

from opd.generate import generate_hf

PAD, EOS, FILLER = 0, 2, 5


class StubTokenizer:
    pad_token_id = PAD
    eos_token_id = EOS

    def decode(self, ids, skip_special_tokens=True):  # noqa: FBT002
        return " ".join(str(i) for i in ids if not (skip_special_tokens and i == EOS))


class StubModel(torch.nn.Module):
    """Emits ``lengths[row]`` tokens then EOS, letting straggler behaviour be constructed."""

    def __init__(self, lengths_by_prompt: dict[int, int]) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.lengths_by_prompt = lengths_by_prompt
        self.seen_prompts: list[list[int]] = []

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        batch, width = input_ids.shape
        lengths = []
        for row in range(batch):
            real = [int(t) for t, m in zip(input_ids[row].tolist(), attention_mask[row].tolist(), strict=True) if m]
            self.seen_prompts.append(real)
            lengths.append(min(self.lengths_by_prompt[len(real)], max_new_tokens))
        steps = max(lengths)
        out = torch.full((batch, width + steps), PAD, dtype=torch.long)
        out[:, :width] = input_ids
        for row, n in enumerate(lengths):
            out[row, width : width + n] = FILLER
            if n < max_new_tokens:
                out[row, width + n - 1] = EOS
        return out


class GenerateHfTest(unittest.TestCase):
    def setUp(self) -> None:
        # Prompt lengths 10..14; deliberately not in the order generate_hf will batch them.
        self.prompts = [[1] * n for n in (13, 10, 14, 11, 12)]
        self.lengths = {10: 4, 11: 30, 12: 6, 13: 8, 14: 5}

    def _run(self, batch_size: int, max_new_tokens: int = 64):
        model = StubModel(self.lengths)
        return model, generate_hf(
            model, StubTokenizer(), self.prompts, max_new_tokens, batch_size, progress=False
        )[0]

    def test_completions_are_returned_in_caller_order(self) -> None:
        """Batching sorts by length internally; the caller must not see that reordering."""
        _, completions = self._run(batch_size=2)
        self.assertEqual([c.index for c in completions], [0, 1, 2, 3, 4])

    def test_each_completion_matches_its_own_prompt(self) -> None:
        _, completions = self._run(batch_size=2)
        for position, completion in enumerate(completions):
            expected = self.lengths[len(self.prompts[position])]
            self.assertEqual(len(completion.token_ids), expected, msg=f"prompt {position}")

    def test_tokens_after_the_first_eos_are_dropped(self) -> None:
        _, completions = self._run(batch_size=5)
        # Every sequence stops early here, so none may carry right-padding.
        for completion in completions:
            self.assertNotIn(PAD, completion.token_ids)
            self.assertEqual(completion.token_ids.count(EOS), 1)
            self.assertEqual(completion.token_ids[-1], EOS)

    def test_hitting_the_cap_is_reported_as_truncated(self) -> None:
        _, completions = self._run(batch_size=5, max_new_tokens=6)
        by_prompt = {len(self.prompts[c.index]): c for c in completions}
        # Prompt of length 11 wants 30 tokens but the cap is 6.
        self.assertTrue(by_prompt[11].truncated)
        self.assertEqual(by_prompt[11].finish_reason, "length")
        # Prompt of length 10 wants only 4.
        self.assertFalse(by_prompt[10].truncated)
        self.assertEqual(by_prompt[10].finish_reason, "stop")

    def test_prompts_are_left_padded_so_generation_starts_at_one_offset(self) -> None:
        model, _ = self._run(batch_size=5)
        # Reconstructed unpadded prompts must all be intact, i.e. padding never ate real tokens.
        self.assertEqual(sorted(len(p) for p in model.seen_prompts), [10, 11, 12, 13, 14])

    def test_batch_size_does_not_change_the_result(self) -> None:
        _, one = self._run(batch_size=1)
        _, many = self._run(batch_size=5)
        self.assertEqual([c.token_ids for c in one], [c.token_ids for c in many])
        self.assertEqual([c.finish_reason for c in one], [c.finish_reason for c in many])


if __name__ == "__main__":
    unittest.main()
