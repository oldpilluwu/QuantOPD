from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

from .config import VALID_PRECISIONS


def require_single_cuda_gpu() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 0 model diagnostics and training")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA GPU is visible")
    return torch.cuda.current_device()


def make_quantization_config(precision: str) -> BitsAndBytesConfig | None:
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"Unsupported teacher precision: {precision}")
    if precision == "bf16":
        return None
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def force_non_thinking(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Bind ``enable_thinking=False`` into the tokenizer's chat template.

    Qwen3's template defaults to thinking mode, and TRL's ``DistillationDataCollator`` calls
    ``apply_chat_template`` with no ``chat_template_kwargs`` (trl 1.6.0,
    ``trl/experimental/distillation/distillation_trainer.py``), so the flag cannot be routed
    through the dataset. Binding it to the tokenizer is the only way every consumer -- evaluation,
    trajectory generation, scoring, and the trainer -- renders prompts identically.
    """
    original = tokenizer.apply_chat_template

    def apply_chat_template(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = apply_chat_template  # type: ignore[method-assign]
    return tokenizer


def load_tokenizer(model_id: str, revision: str, enable_thinking: bool = True) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if not enable_thinking:
        force_non_thinking(tokenizer)
    return tokenizer


def load_teacher(
    model_id: str,
    revision: str,
    precision: str,
    attention_implementation: str,
) -> PreTrainedModel:
    device_index = require_single_cuda_gpu()
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": torch.bfloat16,
        "device_map": {"": device_index},
        "low_cpu_mem_usage": True,
        "attn_implementation": attention_implementation,
    }
    quantization_config = make_quantization_config(precision)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    model.requires_grad_(False)
    return model


def load_bf16_student(model_id: str, revision: str, attention_implementation: str) -> PreTrainedModel:
    device_index = require_single_cuda_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        attn_implementation=attention_implementation,
    )
    model.config.use_cache = False
    model.train()
    return model


def load_bf16_inference_model(model_id: str, revision: str, attention_implementation: str) -> PreTrainedModel:
    """Load a frozen BF16 model for scoring. Same weights as the student, but eval-mode."""
    device_index = require_single_cuda_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        attn_implementation=attention_implementation,
    )
    model.config.use_cache = True
    model.eval()
    model.requires_grad_(False)
    return model


def model_footprint_bytes(model: PreTrainedModel) -> int:
    try:
        return int(model.get_memory_footprint(return_buffers=True))
    except TypeError:
        return int(model.get_memory_footprint())
