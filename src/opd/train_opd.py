"""`opd-train` -- on-policy distillation with TRL's `DistillationTrainer`.

TRL owns the rollout loop and the divergence; this module only supplies fixed data, the condition,
and the reporting. That boundary is deliberate: a project-specific KL implementation would make
results incomparable with the rest of the ecosystem for no scientific gain.

`--max-steps 2` reproduces the original Phase 0 smoke test. The invariant checks that used to live
in `train_smoke.py` now run on every training run, however long, because they are cheap and the
failure they catch (a teacher that is quietly receiving gradients) is invisible in the loss curve.
"""

from __future__ import annotations

import argparse
import json
import math

from datasets import load_dataset
from transformers import set_seed
from trl.experimental.distillation import DistillationConfig, DistillationTrainer

from .common import atomic_write_json
from .config import DEFAULT_CONFIG, VALID_PRECISIONS, load_config
from .hub import resolve_model_revision
from .models import load_teacher, load_tokenizer, model_footprint_bytes
from .runtime import GIB, condition_slug, load_manifest, peak_vram_gib, reset_vram_tracking, runtime_environment

SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run on-policy distillation into the BF16 student.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--precision", choices=sorted(VALID_PRECISIONS), required=True)
    parser.add_argument("--max-steps", type=int, help="Override the configured step budget.")
    parser.add_argument("--subset", help="Override the training subset, e.g. 'smoke' for a fast check.")
    parser.add_argument("--optimizer", help="Override the optimizer, for example adamw_8bit to cut memory.")
    parser.add_argument("--no-vllm", action="store_true", help="Disable colocated vLLM generation (OOM fallback).")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        help="Override the colocated vLLM engine's share of total GPU memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config.opd
    if args.teacher not in config.models.teachers:
        raise ValueError(f"Teacher {args.teacher!r} is not listed in {args.config}")

    manifest = load_manifest(args.manifest)
    subset = args.subset or settings.source_subset
    if subset not in manifest["files"]:
        raise ValueError(f"Unknown subset {subset!r}; manifest has {sorted(manifest['files'])}")
    train_dataset = load_dataset("json", data_files=str(manifest["files"][subset]), split="train")

    max_steps = args.max_steps or settings.max_steps
    set_seed(settings.seed)
    reset_vram_tracking()

    student_revision = resolve_model_revision(config.models.student, config.models.revision)
    teacher_revision = resolve_model_revision(args.teacher, config.models.revision)
    # The shim binds enable_thinking into the tokenizer because TRL's collator calls
    # apply_chat_template with no chat_template_kwargs and would otherwise train in thinking mode.
    tokenizer = load_tokenizer(config.models.student, student_revision, config.models.enable_thinking)
    teacher = load_teacher(args.teacher, teacher_revision, args.precision, config.models.attention_implementation)
    teacher_footprint = model_footprint_bytes(teacher)

    output_dir = settings.output_dir / condition_slug(args.teacher, args.precision)
    use_vllm = settings.use_vllm and not args.no_vllm
    generation_batch_size = settings.per_device_train_batch_size * settings.gradient_accumulation_steps

    training_args = DistillationConfig(
        output_dir=str(output_dir),
        model_init_kwargs={
            "revision": student_revision,
            "dtype": "bfloat16",
            "attn_implementation": config.models.attention_implementation,
            "low_cpu_mem_usage": True,
        },
        max_steps=max_steps,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        generation_batch_size=generation_batch_size,
        learning_rate=settings.learning_rate,
        optim=args.optimizer or settings.optimizer,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=settings.max_length,
        max_prompt_length=settings.max_prompt_length,
        max_completion_length=settings.max_completion_length,
        lmbda=settings.lmbda,
        beta=settings.beta,
        loss_top_k=settings.loss_top_k,
        loss_add_tail=settings.loss_add_tail,
        reverse_kl_top_1_mode=settings.reverse_kl_top_1_mode,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=(
            args.vllm_gpu_memory_utilization
            if args.vllm_gpu_memory_utilization is not None
            else settings.vllm_gpu_memory_utilization
        ),
        # A checkpoint is required to evaluate the trained student afterwards.
        save_strategy="steps",
        save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit,
        logging_steps=settings.logging_steps,
        logging_first_step=True,
        report_to="none",
        seed=settings.seed,
        data_seed=settings.seed,
        dataloader_num_workers=0,
    )
    trainer = DistillationTrainer(
        model=config.models.student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    teacher_has_gradient = any(parameter.grad is not None for parameter in teacher.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    checks = {
        "training_loss_is_finite": math.isfinite(result.training_loss),
        "teacher_is_frozen": not any(parameter.requires_grad for parameter in teacher.parameters()),
        "teacher_has_no_gradient": not teacher_has_gradient,
        "student_is_full_weight_trainable": trainable == total,
        "fully_on_policy": trainer.lmbda == settings.lmbda == 1.0,
        "reverse_kl": trainer.beta == settings.beta == 1.0,
        "top_1_loss_support": trainer.loss_top_k == settings.loss_top_k,
        "checkpoint_written": (final_dir / "config.json").exists(),
    }

    history = [entry for entry in trainer.state.log_history if "loss" in entry]
    report = {
        "schema_version": SCHEMA_VERSION,
        "student": {"model": config.models.student, "revision": student_revision, "precision": "bf16"},
        "teacher": {
            "model": args.teacher,
            "revision": teacher_revision,
            "precision": args.precision,
            "footprint_bytes": teacher_footprint,
            "footprint_gib": teacher_footprint / GIB,
        },
        "data": {"manifest": str(manifest["files"][subset]), "subset": subset},
        "objective": {
            "lmbda": settings.lmbda,
            "beta": settings.beta,
            "loss_top_k": settings.loss_top_k,
            "loss_add_tail": settings.loss_add_tail,
            "reverse_kl_top_1_mode": settings.reverse_kl_top_1_mode,
            "temperature": settings.temperature,
        },
        "budget": {
            "max_steps": max_steps,
            "generation_batch_size": generation_batch_size,
            "prompts_consumed": max_steps * generation_batch_size,
            "max_completion_length": settings.max_completion_length,
            "learning_rate": settings.learning_rate,
            "optimizer": args.optimizer or settings.optimizer,
            "use_vllm": use_vllm,
        },
        "result": {
            "global_step": result.global_step,
            "training_loss": result.training_loss,
            "runtime_seconds": result.metrics.get("train_runtime"),
            "samples_per_second": result.metrics.get("train_samples_per_second"),
            "cuda_peak_allocated_gib": peak_vram_gib(),
            "first_logged_loss": history[0]["loss"] if history else None,
            "last_logged_loss": history[-1]["loss"] if history else None,
        },
        "loss_history": history,
        "checkpoint": str(final_dir),
        "checks": checks,
        "environment": runtime_environment(),
    }
    report_path = output_dir / "training-report.json"
    atomic_write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "checkpoint": str(final_dir), "checks": checks}, indent=2))

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"OPD training invariants failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
