"""Single definition of how a question becomes model input.

Every stage -- benchmark evaluation, trajectory generation, scoring, and OPD training -- must
render prompts identically, or the measurements are not comparable. Training goes through TRL's
collator rather than this module, which is why the non-thinking switch is bound to the tokenizer
in :func:`opd.models.force_non_thinking` instead of being passed at each call site.
"""

from __future__ import annotations

import re

from transformers import PreTrainedTokenizer

# Kept deliberately plain: no few-shot examples and no "think step by step" nudge, so the measured
# difference between conditions is the model and the teacher, not prompt engineering.
MATH_INSTRUCTION = "Put your final answer within \\boxed{}."

# Qwen3 with enable_thinking=False appends a pre-closed, empty think block to the generation
# prompt: "<|im_start|>assistant\n<think>\n\n</think>\n\n".
EMPTY_THINK_BLOCK = re.compile(r"<think>\s*</think>")


def build_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{question.strip()}\n\n{MATH_INSTRUCTION}"}]


def render_prompt(tokenizer: PreTrainedTokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_prompt_ids(tokenizer: PreTrainedTokenizer, question: str) -> list[int]:
    """Token ids for a rendered prompt.

    vLLM builds its own tokenizer internally and would not see the non-thinking shim, so callers
    pass these ids to vLLM rather than raw text.
    """
    return tokenizer(render_prompt(tokenizer, question), add_special_tokens=False)["input_ids"]


def is_non_thinking(rendered_prompt: str) -> bool:
    """Whether a rendered prompt carries Qwen3's pre-closed, empty think block."""
    return EMPTY_THINK_BLOCK.search(rendered_prompt) is not None
