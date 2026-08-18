from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import set_seed
from trl.experimental.distillation import DistillationConfig, DistillationTrainer

from .common import atomic_write_json, resolve_project_path, slug
from .config import load_config
from .hub import resolve_model_revision
from .models import load_teacher, load_tokenizer, model_footprint_bytes

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short, full-weight, fully on-policy TRL training check.")
    parser.add_argument("--config", default="configs/phase0.toml")
    parser.add_argument("--manifest", default="data/phase0/manifest.json")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--precision", choices=("bf16", "int8", "int4"), required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--optimizer", help="Override the optimizer, for example adamw_8bit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.teacher not in config.models.teachers:
        raise ValueError(f"Teacher {args.teacher!r} is not listed in {args.config}")

    manifest_path = resolve_project_path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    smoke_path = Path(manifest["files"]["smoke"])
    train_dataset = load_dataset("json", data_files=str(smoke_path), split="train")

    set_seed(config.training.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    student_revision = resolve_model_revision(config.models.student, config.models.revision)
    teacher_revision = resolve_model_revision(args.teacher, config.models.revision)
    tokenizer = load_tokenizer(config.models.student, student_revision)
    teacher = load_teacher(
        args.teacher,
        teacher_revision,
        args.precision,
        config.models.attention_implementation,
    )

    output_dir = config.training.output_dir / f"{slug(args.teacher)}-{args.precision}"
    training_args = DistillationConfig(
        output_dir=str(output_dir),
        model_init_kwargs={
            "revision": student_revision,
            "dtype": "bfloat16",
            "attn_implementation": config.models.attention_implementation,
            "low_cpu_mem_usage": True,
        },
        max_steps=args.max_steps or config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        generation_batch_size=(
            config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
        ),
        learning_rate=config.training.learning_rate,
        optim=args.optimizer or config.training.optimizer,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config.training.max_length,
        max_completion_length=config.training.max_completion_length,
        lmbda=1.0,
        beta=1.0,
        loss_top_k=1,
        loss_add_tail=True,
        reverse_kl_top_1_mode="sampled",
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        use_vllm=False,
        save_strategy="no",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        seed=config.training.seed,
        data_seed=config.training.seed,
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

    teacher_has_gradient = any(parameter.grad is not None for parameter in teacher.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    checks = {
        "training_loss_is_finite": math.isfinite(result.training_loss),
        "teacher_is_frozen": not any(parameter.requires_grad for parameter in teacher.parameters()),
        "teacher_has_no_gradient": not teacher_has_gradient,
        "student_is_full_weight_trainable": trainable_parameters == total_parameters,
        "fully_on_policy": trainer.lmbda == 1.0,
        "reverse_kl": trainer.beta == 1.0,
        "top_1_loss_support": trainer.loss_top_k == 1,
    }
    report = {
        "schema_version": 1,
        "student": {"model": config.models.student, "revision": student_revision, "precision": "bf16"},
        "teacher": {
            "model": args.teacher,
            "revision": teacher_revision,
            "precision": args.precision,
            "footprint_bytes": model_footprint_bytes(teacher),
        },
        "dataset_manifest": str(manifest_path),
        "objective": {
            "lmbda": 1.0,
            "beta": 1.0,
            "loss_top_k": 1,
            "loss_add_tail": True,
        },
        "result": {
            "global_step": result.global_step,
            "training_loss": result.training_loss,
            "runtime_seconds": result.metrics.get("train_runtime"),
            "samples_per_second": result.metrics.get("train_samples_per_second"),
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        },
        "checks": checks,
    }
    report_path = output_dir / "phase0-smoke-report.json"
    atomic_write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "checks": checks}, indent=2))
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise SystemExit(f"Phase 0 training smoke test failed: {', '.join(failed_checks)}")


if __name__ == "__main__":
    main()
