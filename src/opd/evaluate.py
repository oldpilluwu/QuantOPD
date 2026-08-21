"""`opd-eval` -- benchmark accuracy for a student, a teacher, or a trained checkpoint.

Greedy decoding throughout: pass@1 with temperature 0 is the lowest-variance way to compare
conditions, and this pilot compares conditions rather than chasing a headline number.

Backend follows the model, not the benchmark: BF16 models go through vLLM for speed, quantized
teachers go through Transformers so the benchmarked teacher is byte-for-byte the teacher that
supervises during OPD.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .common import atomic_write_json, slug
from .config import DEFAULT_CONFIG, PREQUANTIZED_PRECISIONS, VALID_PRECISIONS, load_config
from .generate import Completion, build_vllm_engine, generate_hf, generate_vllm
from .grading import extract_gold, grade, summarize
from .hub import resolve_model_revision
from .models import load_bf16_inference_model, load_teacher, load_tokenizer, model_footprint_bytes
from .prompts import is_non_thinking, render_prompt, render_prompt_ids
from .runtime import (
    GIB,
    condition_slug,
    load_benchmark,
    load_manifest,
    peak_vram_gib,
    reset_vram_tracking,
    runtime_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure benchmark accuracy for one model condition.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--model", help="Hub id. Defaults to the configured student.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local trained-student directory to evaluate instead of the Hub weights.",
    )
    parser.add_argument("--precision", choices=sorted(VALID_PRECISIONS), default="bf16")
    # Choices are validated against the config rather than hardcoded, so adding a benchmark is a
    # config change alone. config.evaluation_dataset() raises listing the configured names.
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N items (smoke testing).")
    parser.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Transformers-path generation batch size. bitsandbytes dequantizes weights on every "
            "forward pass, so throughput scales strongly with batch size until the KV cache fills "
            "the card. Overrides [eval] batch_size."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        help="Override [models] attention_implementation, e.g. flash_attention_2.",
    )
    parser.add_argument("--tag", help="Label for this run, e.g. 'baseline' or 'opd-qwen3-14b-int4'.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def choose_backend(requested: str, precision: str) -> str:
    if requested != "auto":
        return requested
    # Quantized weights default to the Transformers path so evaluation and OPD share one teacher
    # implementation. A pre-quantized checkpoint *can* be served by vLLM (see the guard below), but
    # only on request, because it changes the kernels relative to the OPD teacher.
    return "vllm" if precision == "bf16" else "hf"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest = load_manifest(args.manifest)
    benchmark = config.evaluation_dataset(args.benchmark)
    settings = config.eval

    attn_implementation = args.attn_implementation or config.models.attention_implementation
    batch_size = args.batch_size or settings.batch_size
    model_id = args.model or config.models.student
    revision = resolve_model_revision(model_id, config.models.revision)
    # A local checkpoint carries the student's architecture but not a Hub revision; keep the base
    # revision in the report so the lineage of the trained weights stays visible.
    weights_source = str(args.checkpoint) if args.checkpoint else model_id
    backend = choose_backend(args.backend, args.precision)

    tokenizer = load_tokenizer(model_id, revision, config.models.enable_thinking)
    items = load_benchmark(manifest, benchmark, args.limit)
    prompts = [render_prompt(tokenizer, item["question"]) for item in items]
    prompt_ids = [render_prompt_ids(tokenizer, item["question"]) for item in items]

    thinking_as_configured = all(is_non_thinking(prompt) is not config.models.enable_thinking for prompt in prompts)
    if not thinking_as_configured:
        raise RuntimeError("Rendered prompts do not match the configured thinking mode; refusing to evaluate.")

    # bitsandbytes quantization is a load-time flag that build_vllm_engine does not pass, so vLLM
    # would serve BF16 weights under an INT4/INT8 label. A pre-quantized checkpoint is different:
    # it carries quantization_config in its own config.json and vLLM detects the format from there,
    # so serving it is correct -- just with different kernels than the Transformers teacher.
    if backend == "vllm" and args.precision not in PREQUANTIZED_PRECISIONS and args.precision != "bf16":
        raise SystemExit(
            f"--backend vllm cannot serve {args.precision}: build_vllm_engine does not pass a "
            "bitsandbytes quantization config, so the run would report quantized precision while "
            "measuring BF16. Use the default 'auto' backend, which routes it through Transformers."
        )
    if backend == "vllm" and args.precision in PREQUANTIZED_PRECISIONS:
        print(
            f"[eval] WARNING: serving {args.precision} through vLLM. vLLM detects the format from "
            "the checkpoint, so the weights are correct, but its kernels differ from the "
            "Transformers teacher used during OPD. Do not compare this number directly against a "
            "Transformers-path teacher; the backend is recorded in the report.",
            file=sys.stderr,
            flush=True,
        )

    reset_vram_tracking()
    footprint_bytes: int | None = None
    if backend == "vllm":
        engine = build_vllm_engine(
            model_id=weights_source,
            revision=None if args.checkpoint else revision,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            max_model_length=settings.vllm_max_model_length,
            seed=settings.seed,
            # A pre-quantized checkpoint states its own compute dtype; AWQ is fp16.
            dtype="auto" if args.precision in PREQUANTIZED_PRECISIONS else "bfloat16",
        )
        started = time.perf_counter()
        completions: list[Completion] = generate_vllm(
            engine,
            prompt_ids,
            max_new_tokens=settings.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        elapsed = time.perf_counter() - started
    else:
        if args.precision == "bf16":
            model = load_bf16_inference_model(weights_source, revision, attn_implementation)
        else:
            model = load_teacher(
                weights_source,
                revision,
                args.precision,
                attn_implementation,
            )
        footprint_bytes = model_footprint_bytes(model)
        # Echo what actually got loaded before a long run starts, so a slow run is diagnosable
        # while it is still running rather than only from the report afterwards.
        header = (
            f"[eval] {model_id} {args.precision} | backend=hf | batch={batch_size}"
            f" | attn={attn_implementation}"
        )
        detail = (
            f"[eval] weights={footprint_bytes / GIB:.2f} GiB"
            f" | device={next(model.parameters()).device}"
            f" | {len(items)} prompts | max_new_tokens={settings.max_new_tokens}"
            f" | benchmark={args.benchmark}"
        )
        print(header, file=sys.stderr, flush=True)
        print(detail, file=sys.stderr, flush=True)
        completions, elapsed = generate_hf(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=settings.max_new_tokens,
            batch_size=batch_size,
            greedy=True,
        )

    grades = []
    records = []
    for item, completion in zip(items, completions, strict=True):
        gold = extract_gold(args.benchmark, item["answer"])
        result = grade(gold, completion.text)
        grades.append(result)
        records.append(
            {
                "item_index": item["item_index"],
                "source_index": item["source_index"],
                "gold": gold,
                "completion_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                **item["groups"],
                **result.as_dict(),
            }
        )

    completion_tokens = sum(len(completion.token_ids) for completion in completions)
    truncated = sum(1 for completion in completions if completion.truncated)
    accuracy = summarize(grades)

    # Math-Verify falls back to extracting a bare number when there is no \boxed{}, so a completion
    # truncated mid-reasoning still "parses" -- it is graded against whatever figure it stopped
    # near. Truncation therefore hides inside the parse-failure rate. Split it out explicitly so
    # the budget's cost is visible rather than inferred.
    finished = [g for g, c in zip(grades, completions, strict=True) if not c.truncated]
    accuracy["accuracy_on_finished"] = (
        sum(1 for g in finished if g.correct) / len(finished) if finished else None
    )
    accuracy["correct_while_truncated"] = sum(
        1 for g, c in zip(grades, completions, strict=True) if c.truncated and g.correct
    )

    # Slice accuracy by whatever grouping columns the benchmark declares (MATH-500 difficulty
    # level, subject). Free -- the completions already exist -- and it shows immediately whether a
    # headline score is hiding a ceiling on the easy items.
    breakdown: dict[str, dict[str, Any]] = {}
    for field_name in benchmark.group_fields:
        buckets: dict[str, list[tuple[int, int]]] = {}
        for item, g, c in zip(items, grades, completions, strict=True):
            buckets.setdefault(str(item["groups"][field_name]), []).append(
                (1 if g.correct else 0, 1 if c.truncated else 0)
            )
        breakdown[field_name] = {
            key: {
                "count": len(entries),
                "accuracy": sum(correct for correct, _ in entries) / len(entries),
                "truncation_rate": sum(cut for _, cut in entries) / len(entries),
            }
            for key, entries in sorted(buckets.items())
        }

    tag = args.tag or ("checkpoint" if args.checkpoint else "base")
    output_dir = settings.output_dir / f"{condition_slug(model_id, args.precision)}-{slug(tag)}" / args.benchmark
    report = {
        "schema_version": 1,
        "tag": tag,
        "benchmark": {
            "name": args.benchmark,
            "dataset": benchmark.dataset,
            "revision": manifest["evaluation"][args.benchmark]["revision"],
            "items": len(items),
            "limit": args.limit,
        },
        "model": {
            "model": model_id,
            "revision": revision,
            "weights_source": weights_source,
            "precision": args.precision,
            "backend": backend,
            "enable_thinking": config.models.enable_thinking,
            "footprint_bytes": footprint_bytes,
            "footprint_gib": None if footprint_bytes is None else footprint_bytes / (1024**3),
        },
        "decoding": {
            "greedy": True,
            "max_new_tokens": settings.max_new_tokens,
            "batch_size": batch_size if backend == "hf" else None,
            "attn_implementation": attn_implementation if backend == "hf" else None,
        },
        "accuracy": accuracy,
        "accuracy_by_group": breakdown,
        "generation": {
            "completion_tokens": completion_tokens,
            "mean_completion_tokens": completion_tokens / len(completions),
            # A high truncation rate means the budget, not the model, is being measured.
            "truncation_rate": truncated / len(completions),
            "seconds": elapsed,
            "completion_tokens_per_second": completion_tokens / elapsed if elapsed > 0 else None,
        },
        "peak_vram_gib": peak_vram_gib(),
        "environment": runtime_environment(),
    }

    report_path = args.output or (output_dir / "report.json")
    atomic_write_json(report_path, report)
    atomic_write_json(output_dir / "per-item.json", records)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "accuracy": accuracy["accuracy"],
                "accuracy_on_finished": accuracy["accuracy_on_finished"],
                "truncation_rate": report["generation"]["truncation_rate"],
                "mean_completion_tokens": report["generation"]["mean_completion_tokens"],
                "parse_failure_rate": accuracy["prediction_parse_failure_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
