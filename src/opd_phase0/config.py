from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .common import resolve_project_path

VALID_PRECISIONS = frozenset({"bf16", "int8", "int4"})


@dataclass(frozen=True)
class ModelsConfig:
    student: str
    teachers: tuple[str, ...]
    precisions: tuple[str, ...]
    revision: str
    attention_implementation: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    split: str
    revision: str
    prompt_fields: tuple[str, ...]
    seed: int
    calibration_size: int
    training_size: int
    smoke_size: int
    output_dir: Path


@dataclass(frozen=True)
class EvaluationDatasetConfig:
    name: str
    dataset: str
    config: str | None
    split: str
    revision: str


@dataclass(frozen=True)
class DiagnosticsConfig:
    prompt: str
    warmup_steps: int
    timed_steps: int
    output_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    max_steps: int
    max_length: int
    max_completion_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    optimizer: str
    output_dir: Path


@dataclass(frozen=True)
class Phase0Config:
    models: ModelsConfig
    data: DataConfig
    evaluation: tuple[EvaluationDatasetConfig, ...]
    diagnostics: DiagnosticsConfig
    training: TrainingConfig


def load_config(path: str | Path = "configs/phase0.toml") -> Phase0Config:
    config_path = resolve_project_path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    models_raw = raw["models"]
    data_raw = raw["data"]
    diagnostics_raw = raw["diagnostics"]
    training_raw = raw["training"]

    models = ModelsConfig(
        student=models_raw["student"],
        teachers=tuple(models_raw["teachers"]),
        precisions=tuple(models_raw["precisions"]),
        revision=models_raw["revision"],
        attention_implementation=models_raw["attention_implementation"],
    )
    unknown_precisions = set(models.precisions) - VALID_PRECISIONS
    if unknown_precisions:
        raise ValueError(f"Unknown teacher precision(s): {sorted(unknown_precisions)}")

    data = DataConfig(
        dataset=data_raw["dataset"],
        split=data_raw["split"],
        revision=data_raw["revision"],
        prompt_fields=tuple(data_raw["prompt_fields"]),
        seed=int(data_raw["seed"]),
        calibration_size=int(data_raw["calibration_size"]),
        training_size=int(data_raw["training_size"]),
        smoke_size=int(data_raw["smoke_size"]),
        output_dir=resolve_project_path(data_raw["output_dir"]),
    )
    if data.smoke_size > data.training_size:
        raise ValueError("smoke_size cannot exceed training_size")
    if min(data.calibration_size, data.training_size, data.smoke_size) < 1:
        raise ValueError("All Phase 0 dataset sizes must be positive")

    evaluation = tuple(
        EvaluationDatasetConfig(
            name=item["name"],
            dataset=item["dataset"],
            config=item.get("config"),
            split=item["split"],
            revision=item["revision"],
        )
        for item in raw.get("evaluation", {}).get("datasets", [])
    )
    diagnostics = DiagnosticsConfig(
        prompt=diagnostics_raw["prompt"],
        warmup_steps=int(diagnostics_raw["warmup_steps"]),
        timed_steps=int(diagnostics_raw["timed_steps"]),
        output_dir=resolve_project_path(diagnostics_raw["output_dir"]),
    )
    training = TrainingConfig(
        seed=int(training_raw["seed"]),
        max_steps=int(training_raw["max_steps"]),
        max_length=int(training_raw["max_length"]),
        max_completion_length=int(training_raw["max_completion_length"]),
        per_device_train_batch_size=int(training_raw["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_raw["gradient_accumulation_steps"]),
        learning_rate=float(training_raw["learning_rate"]),
        optimizer=training_raw["optimizer"],
        output_dir=resolve_project_path(training_raw["output_dir"]),
    )
    if training.max_completion_length >= training.max_length:
        raise ValueError("max_completion_length must be smaller than max_length")

    return Phase0Config(
        models=models,
        data=data,
        evaluation=evaluation,
        diagnostics=diagnostics,
        training=training,
    )
