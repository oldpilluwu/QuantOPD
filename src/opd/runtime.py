"""Shared plumbing for the measurement CLIs: manifests, environment capture, VRAM accounting."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from .common import resolve_project_path
from .config import EvaluationDatasetConfig, ExperimentConfig

GIB = 1024**3

TRACKED_PACKAGES = ("torch", "transformers", "trl", "bitsandbytes", "datasets", "math-verify", "vllm")


def runtime_environment() -> dict[str, Any]:
    """Everything needed to tell whether two reports are comparable."""
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    device: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device |= {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": f"{properties.major}.{properties.minor}",
            "cuda_version": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": device,
    }


def peak_vram_gib() -> float | None:
    return torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else None


def reset_vram_tracking() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = resolve_project_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run `opd-prepare-data` first.")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt_subset(manifest: dict[str, Any], subset: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Read one of the fixed OpenR1-Math subsets written by `opd-prepare-data`."""
    files = manifest["files"]
    if subset not in files:
        raise ValueError(f"Unknown subset {subset!r}; manifest has {sorted(files)}")
    rows: list[dict[str, Any]] = []
    with Path(files[subset]).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            row["prompt_index"] = index
            rows.append(row)
    return rows


def load_benchmark(
    manifest: dict[str, Any],
    benchmark: EvaluationDatasetConfig,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load benchmark items at the frozen revision and frozen indices.

    Indices come from the manifest rather than being re-drawn, so every model is scored on exactly
    the same items.
    """
    recorded = manifest["evaluation"][benchmark.name]
    load_kwargs: dict[str, Any] = {
        "path": benchmark.dataset,
        "split": benchmark.split,
        "revision": recorded["revision"],
    }
    if benchmark.config:
        load_kwargs["name"] = benchmark.config
    dataset = load_dataset(**load_kwargs)

    indices = recorded["indices"]
    if limit is not None:
        indices = indices[:limit]
    available = set(dataset.column_names)
    missing = [name for name in benchmark.group_fields if name not in available]
    if missing:
        raise ValueError(f"{benchmark.name} has no column(s) {missing}; available: {sorted(available)}")

    return [
        {
            "item_index": position,
            "source_index": source_index,
            "question": dataset[source_index][benchmark.question_field],
            "answer": dataset[source_index][benchmark.answer_field],
            "groups": {name: dataset[source_index][name] for name in benchmark.group_fields},
        }
        for position, source_index in enumerate(indices)
    ]


def condition_slug(model_id: str, precision: str) -> str:
    from .common import slug

    return f"{slug(model_id)}-{precision}"


def model_descriptor(config: ExperimentConfig, model_id: str, precision: str, revision: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "revision": revision,
        "precision": precision,
        "attention_implementation": config.models.attention_implementation,
        "enable_thinking": config.models.enable_thinking,
    }
