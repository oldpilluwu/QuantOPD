"""`opd-score` -- teacher/student distribution metrics over student-generated trajectories.

Both models are resident simultaneously and each trajectory is scored one at a time (batch 1), so
there is no padding and therefore no masking to get wrong. Logits are reduced to scalars inside
:mod:`opd.metrics` and discarded; nothing large is ever written to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .common import atomic_write_json, resolve_project_path
from .config import DEFAULT_CONFIG, VALID_PRECISIONS, load_config
from .hub import resolve_model_revision
from .metrics import (
    aggregate,
    completion_logits,
    merge_position_profile,
    score_trajectory,
    summarize_divergence,
    trajectory_means,
)
from .models import load_bf16_inference_model, load_teacher, load_tokenizer, model_footprint_bytes
from .runtime import GIB, condition_slug, peak_vram_gib, reset_vram_tracking, runtime_environment

SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score student trajectories against one teacher condition.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--precision", choices=sorted(VALID_PRECISIONS), required=True)
    parser.add_argument("--trajectories", default="artifacts/trajectories/baseline/manifest.json")
    parser.add_argument("--student-checkpoint", type=Path, help="Score a trained student's distributions.")
    parser.add_argument("--limit", type=int, help="Score only the first N trajectories (smoke testing).")
    parser.add_argument("--tag", help="Label for this scoring run.")
    return parser.parse_args()


def read_trajectories(manifest_path: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows: list[dict[str, Any]] = []
    with Path(manifest["files"]["trajectories"]).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No trajectories found via {manifest_path}")
    return manifest, rows


def check_shared_vocabulary(student: Any, teacher: Any, student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """A KL between distributions over different supports is meaningless. Fail loudly, early."""
    student_vocab = student.get_output_embeddings().weight.shape[0]
    teacher_vocab = teacher.get_output_embeddings().weight.shape[0]
    if student_vocab != teacher_vocab:
        raise RuntimeError(f"Student output vocab {student_vocab} != teacher {teacher_vocab}")
    probe = "The quick brown fox \\boxed{42}"
    if student_tokenizer(probe)["input_ids"] != teacher_tokenizer(probe)["input_ids"]:
        raise RuntimeError("Student and teacher tokenizers disagree; trajectories are not transferable")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config.scoring
    if args.teacher not in config.models.teachers:
        raise ValueError(f"Teacher {args.teacher!r} is not listed in {args.config}")

    manifest_path = resolve_project_path(args.trajectories)
    trajectory_manifest, rows = read_trajectories(manifest_path, args.limit)

    student_id = config.models.student
    student_revision = resolve_model_revision(student_id, config.models.revision)
    teacher_revision = resolve_model_revision(args.teacher, config.models.revision)
    student_source = str(args.student_checkpoint) if args.student_checkpoint else student_id

    student_tokenizer = load_tokenizer(student_id, student_revision, config.models.enable_thinking)
    teacher_tokenizer = load_tokenizer(args.teacher, teacher_revision, config.models.enable_thinking)

    reset_vram_tracking()
    teacher = load_teacher(args.teacher, teacher_revision, args.precision, config.models.attention_implementation)
    teacher_footprint = model_footprint_bytes(teacher)
    allocated_after_teacher = torch.cuda.memory_allocated() / GIB if torch.cuda.is_available() else None
    student = load_bf16_inference_model(student_source, student_revision, config.models.attention_implementation)
    student_footprint = model_footprint_bytes(student)

    check_shared_vocabulary(student, teacher, student_tokenizer, teacher_tokenizer)

    device = next(student.parameters()).device
    if next(teacher.parameters()).device != device:
        raise RuntimeError("Student and teacher must be on the same device")

    per_trajectory: list[dict[str, float]] = []
    token_counts: list[int] = []
    accumulators = []
    started = time.perf_counter()

    with torch.inference_mode():
        for row in rows:
            prompt_ids = row["prompt_token_ids"]
            completion_ids = row["completion_token_ids"]
            input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)

            student_logits = completion_logits(student, input_ids, len(prompt_ids), len(completion_ids))
            teacher_logits = completion_logits(teacher, input_ids, len(prompt_ids), len(completion_ids))
            accumulator = score_trajectory(
                student_logits,
                teacher_logits,
                completion_ids,
                settings.top_ks,
                settings.position_chunk_size,
                settings.position_profile_bins,
            )
            del student_logits, teacher_logits

            per_trajectory.append(trajectory_means(accumulator, settings.top_ks))
            token_counts.append(accumulator.tokens)
            accumulators.append(accumulator)

    elapsed = time.perf_counter() - started

    aggregates = aggregate(
        per_trajectory,
        token_counts,
        bootstrap_samples=settings.bootstrap_samples,
        bootstrap_seed=settings.bootstrap_seed,
    )
    reverse_kl = aggregates["reverse_kl"]["token_weighted"]["mean"]
    if not math.isfinite(reverse_kl) or reverse_kl <= 0:
        raise RuntimeError(f"Reverse KL is {reverse_kl}; expected finite and positive. The scoring setup is wrong.")

    tag = args.tag or trajectory_manifest.get("tag", "baseline")
    output_dir = settings.output_dir / tag / condition_slug(args.teacher, args.precision)
    report = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "teacher": {
            "model": args.teacher,
            "revision": teacher_revision,
            "precision": args.precision,
            "footprint_bytes": teacher_footprint,
            "footprint_gib": teacher_footprint / GIB,
        },
        "student": {
            "model": student_id,
            "revision": student_revision,
            "weights_source": student_source,
            "precision": "bf16",
            "footprint_gib": student_footprint / GIB,
        },
        "trajectories": {
            "manifest": str(manifest_path),
            "count": len(rows),
            "tokens": sum(token_counts),
            "sampling": trajectory_manifest.get("sampling"),
        },
        "aggregates": aggregates,
        "position_profile": merge_position_profile(accumulators, settings.position_profile_bins),
        "first_divergence": summarize_divergence(accumulators),
        "efficiency": {
            "scoring_seconds": elapsed,
            "tokens_per_second": sum(token_counts) / elapsed if elapsed > 0 else None,
            "cuda_allocated_after_teacher_load_gib": allocated_after_teacher,
            "cuda_peak_during_paired_scoring_gib": peak_vram_gib(),
        },
        "bootstrap": {"samples": settings.bootstrap_samples, "seed": settings.bootstrap_seed},
        "environment": runtime_environment(),
    }
    report_path = output_dir / "report.json"
    atomic_write_json(report_path, report)

    headline = {
        "report": str(report_path),
        "reverse_kl_token_weighted": reverse_kl,
        "reverse_kl_prompt_weighted": aggregates["reverse_kl"]["prompt_weighted"]["mean"],
        "forward_kl_token_weighted": aggregates["forward_kl"]["token_weighted"]["mean"],
        "top_1_agreement": aggregates["top_1_agreement"]["token_weighted"]["mean"],
        "entropy_shift": aggregates["entropy_shift"]["token_weighted"]["mean"],
        "top_1_approx_reverse_kl": aggregates["top_1_approx_reverse_kl"]["token_weighted"]["mean"],
    }
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
