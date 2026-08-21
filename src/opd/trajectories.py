"""`opd-trajectories` -- student rollouts for the teacher-scoring diagnostics.

Prompts come from the *calibration* subset, which `opd.data.make_split_indices` builds disjoint
from the OPD training subset. The diagnostics therefore never report on prompts the student was
trained on.

Sampling settings mirror `[opd]` exactly (the config loader enforces this), so these trajectories
stand in for the states OPD will actually visit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .common import atomic_write_json, write_jsonl
from .config import DEFAULT_CONFIG, load_config
from .generate import build_vllm_engine, generate_vllm
from .hub import resolve_model_revision
from .models import load_tokenizer
from .prompts import is_non_thinking, render_prompt, render_prompt_ids
from .runtime import load_manifest, load_prompt_subset, runtime_environment

SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate untouched-student trajectories for teacher scoring.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--checkpoint", type=Path, help="Score a trained student instead of the base weights.")
    parser.add_argument("--limit", type=int, help="Generate only the first N trajectories (smoke testing).")
    parser.add_argument("--tag", help="Label for this set, e.g. 'baseline' or 'opd-qwen3-14b-int4'.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing trajectory set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config.trajectories
    manifest = load_manifest(args.manifest)

    tag = args.tag or ("checkpoint" if args.checkpoint else "baseline")
    output_dir = settings.output_dir / tag
    trajectory_path = output_dir / "trajectories.jsonl"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} already exists; pass --force to regenerate it intentionally")

    count = args.limit or settings.prompt_count
    rows = load_prompt_subset(manifest, settings.source_subset, limit=count)
    if len(rows) < count:
        raise ValueError(f"Subset {settings.source_subset!r} has only {len(rows)} prompts; {count} requested")

    student_id = config.models.student
    revision = resolve_model_revision(student_id, config.models.revision)
    weights_source = str(args.checkpoint) if args.checkpoint else student_id
    tokenizer = load_tokenizer(student_id, revision, config.models.enable_thinking)

    prompt_ids = [render_prompt_ids(tokenizer, row["prompt"]) for row in rows]
    if any(is_non_thinking(render_prompt(tokenizer, row["prompt"])) is config.models.enable_thinking for row in rows):
        raise RuntimeError("Rendered prompts do not match the configured thinking mode; refusing to generate.")

    engine = build_vllm_engine(
        model_id=weights_source,
        revision=None if args.checkpoint else revision,
        gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
        max_model_length=settings.vllm_max_model_length,
        seed=settings.seed,
    )
    # Per-request seeds so a trajectory depends on its prompt, not on batch composition.
    seeds = [settings.seed + row["prompt_index"] for row in rows]

    started = time.perf_counter()
    completions = generate_vllm(
        engine,
        prompt_ids,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        seeds=seeds,
    )
    elapsed = time.perf_counter() - started

    records = []
    for row, ids, completion in zip(rows, prompt_ids, completions, strict=True):
        if not completion.token_ids:
            continue  # nothing to score
        records.append(
            {
                "prompt_index": row["prompt_index"],
                "source_index": row["source_index"],
                "seed": settings.seed + row["prompt_index"],
                "prompt_token_ids": ids,
                "completion_token_ids": completion.token_ids,
                "completion_text": completion.text,
                "finish_reason": completion.finish_reason,
            }
        )
    if not records:
        raise RuntimeError("Every completion was empty; nothing to score")

    write_jsonl(trajectory_path, records)
    completion_tokens = sum(len(record["completion_token_ids"]) for record in records)
    truncated = sum(1 for record in records if record["finish_reason"] == "length")
    trajectory_manifest = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "student": {
            "model": student_id,
            "revision": revision,
            "weights_source": weights_source,
            "precision": "bf16",
            "enable_thinking": config.models.enable_thinking,
        },
        "source": {
            "subset": settings.source_subset,
            "requested": count,
            "kept": len(records),
            "dropped_empty": len(rows) - len(records),
        },
        "sampling": {
            "seed": settings.seed,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "max_new_tokens": settings.max_new_tokens,
        },
        "generation": {
            "completion_tokens": completion_tokens,
            "mean_completion_tokens": completion_tokens / len(records),
            "truncation_rate": truncated / len(records),
            "seconds": elapsed,
            "completion_tokens_per_second": completion_tokens / elapsed if elapsed > 0 else None,
        },
        "files": {"trajectories": str(trajectory_path.resolve())},
        "environment": runtime_environment(),
    }
    atomic_write_json(manifest_path, trajectory_manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "trajectories": len(records),
                "mean_completion_tokens": trajectory_manifest["generation"]["mean_completion_tokens"],
                "truncation_rate": trajectory_manifest["generation"]["truncation_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
