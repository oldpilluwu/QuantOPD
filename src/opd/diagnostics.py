from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .common import atomic_write_json, slug
from .config import DEFAULT_CONFIG, load_config
from .hub import resolve_model_revision
from .models import load_bf16_student, load_teacher, load_tokenizer, model_footprint_bytes
from .prompts import is_non_thinking

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 0 model, tokenizer, gradient, and latency checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--precision", choices=("bf16", "int8", "int4"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _render(tokenizer: Any, prompt: str, explicit_thinking: bool | None) -> tuple[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if explicit_thinking is not None:
        kwargs["enable_thinking"] = explicit_thinking
    text = tokenizer.apply_chat_template(messages, **kwargs)
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    return text, input_ids


def _timed_teacher_forward(model: Any, input_ids: torch.Tensor, warmup_steps: int, timed_steps: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(timed_steps):
            model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / timed_steps


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.teacher not in config.models.teachers:
        raise ValueError(f"Teacher {args.teacher!r} is not listed in {args.config}")

    torch.manual_seed(config.opd.seed)
    torch.cuda.manual_seed_all(config.opd.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    student_revision = resolve_model_revision(config.models.student, config.models.revision)
    teacher_revision = resolve_model_revision(args.teacher, config.models.revision)
    enable_thinking = config.models.enable_thinking
    student_tokenizer = load_tokenizer(config.models.student, student_revision, enable_thinking)
    teacher_tokenizer = load_tokenizer(args.teacher, teacher_revision, enable_thinking)

    # The shimmed tokenizers must render the *configured* mode when called the way TRL calls them,
    # i.e. with no explicit enable_thinking argument.
    student_text, student_ids = _render(student_tokenizer, config.diagnostics.prompt, explicit_thinking=None)
    teacher_text, teacher_ids = _render(teacher_tokenizer, config.diagnostics.prompt, explicit_thinking=None)

    allocated_before_models = torch.cuda.memory_allocated()
    teacher = load_teacher(
        args.teacher,
        teacher_revision,
        args.precision,
        config.models.attention_implementation,
    )
    allocated_after_teacher = torch.cuda.memory_allocated()
    student = load_bf16_student(
        config.models.student,
        student_revision,
        config.models.attention_implementation,
    )
    allocated_after_models = torch.cuda.memory_allocated()
    peak_during_model_load = torch.cuda.max_memory_allocated()

    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device
    if student_device != teacher_device:
        raise RuntimeError(f"Student is on {student_device}, but teacher is on {teacher_device}")
    input_ids = student_ids.to(student_device)

    torch.cuda.reset_peak_memory_stats()
    latency_seconds = _timed_teacher_forward(
        teacher,
        input_ids,
        config.diagnostics.warmup_steps,
        config.diagnostics.timed_steps,
    )

    with torch.inference_mode():
        teacher_logits_first = teacher(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()
        teacher_logits_second = teacher(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()
        teacher_log_probs = F.log_softmax(teacher_logits_first, dim=-1)
        teacher_probability_sum = teacher_log_probs.exp().sum(dim=-1)

    student.zero_grad(set_to_none=True)
    student_logits = student(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_probs = student_log_probs.exp()
    reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
    reverse_kl.backward()

    first_student_gradient = next(
        (parameter.grad for parameter in student.parameters() if parameter.grad is not None), None
    )
    student_gradient_is_valid = (
        first_student_gradient is not None
        and bool(torch.isfinite(first_student_gradient).all().item())
        and bool((first_student_gradient != 0).any().item())
    )
    teacher_has_gradient = any(parameter.grad is not None for parameter in teacher.parameters())
    student_floating_dtypes = sorted(
        {str(parameter.dtype) for parameter in student.parameters() if parameter.is_floating_point()}
    )
    deterministic_max_abs_diff = (teacher_logits_first - teacher_logits_second).abs().max().item()

    checks = {
        "configured_thinking_mode_applied": is_non_thinking(student_text) is not enable_thinking,
        "teacher_student_template_text_matches": student_text == teacher_text,
        "teacher_student_token_ids_match": torch.equal(student_ids, teacher_ids),
        "teacher_student_vocab_matches": student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab(),
        "logits_shapes_match": tuple(student_logits.shape) == tuple(teacher_logits_first.shape),
        "teacher_probability_normalized": bool(
            torch.allclose(teacher_probability_sum, torch.ones_like(teacher_probability_sum), atol=1e-5)
        ),
        "reverse_kl_is_finite": math.isfinite(reverse_kl.item()),
        "reverse_kl_is_nonnegative": reverse_kl.item() >= -1e-5,
        "student_gradient_is_valid": student_gradient_is_valid,
        "teacher_is_frozen": not any(parameter.requires_grad for parameter in teacher.parameters()),
        "teacher_has_no_gradient": not teacher_has_gradient,
        "student_is_bf16": student_floating_dtypes == ["torch.bfloat16"],
        "teacher_forward_is_deterministic": deterministic_max_abs_diff == 0.0,
    }
    report = {
        "schema_version": 1,
        "student": {"model": config.models.student, "revision": student_revision},
        "teacher": {
            "model": args.teacher,
            "revision": teacher_revision,
            "precision": args.precision,
            "footprint_bytes": model_footprint_bytes(teacher),
        },
        "objective_check": {
            "type": "full-vocabulary reverse KL on the final prompt token",
            "reverse_kl": reverse_kl.item(),
        },
        "metrics": {
            "input_tokens": input_ids.shape[1],
            "teacher_latency_seconds": latency_seconds,
            "teacher_tokens_per_second": input_ids.shape[1] / latency_seconds,
            "teacher_deterministic_max_abs_diff": deterministic_max_abs_diff,
            "cuda_allocated_before_models_gib": allocated_before_models / GIB,
            "cuda_allocated_after_teacher_gib": allocated_after_teacher / GIB,
            "cuda_allocated_after_models_gib": allocated_after_models / GIB,
            "cuda_peak_during_model_load_gib": peak_during_model_load / GIB,
            "cuda_peak_during_forward_backward_gib": torch.cuda.max_memory_allocated() / GIB,
            "student_floating_dtypes": student_floating_dtypes,
        },
        "checks": checks,
        "environment": {
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "trl": importlib.metadata.version("trl"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        },
    }

    output = args.output or config.diagnostics.output_dir / f"{slug(args.teacher)}-{args.precision}.json"
    atomic_write_json(output, report)
    print(json.dumps({"output": str(output), "checks": checks}, indent=2))
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise SystemExit(f"Phase 0 diagnostics failed: {', '.join(failed_checks)}")


if __name__ == "__main__":
    main()
