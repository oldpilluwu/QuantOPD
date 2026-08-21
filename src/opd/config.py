from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .common import resolve_project_path

VALID_PRECISIONS = frozenset({"bf16", "int8", "int4"})
DEFAULT_CONFIG = "configs/experiment.toml"


@dataclass(frozen=True)
class ModelsConfig:
    student: str
    teachers: tuple[str, ...]
    precisions: tuple[str, ...]
    revision: str
    attention_implementation: str
    enable_thinking: bool


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
    question_field: str
    answer_field: str
    subsample_size: int | None


@dataclass(frozen=True)
class DiagnosticsConfig:
    prompt: str
    warmup_steps: int
    timed_steps: int
    output_dir: Path


@dataclass(frozen=True)
class EvalConfig:
    seed: int
    max_new_tokens: int
    batch_size: int
    vllm_gpu_memory_utilization: float
    vllm_max_model_length: int
    output_dir: Path


@dataclass(frozen=True)
class TrajectoriesConfig:
    seed: int
    prompt_count: int
    source_subset: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    vllm_gpu_memory_utilization: float
    vllm_max_model_length: int
    output_dir: Path


@dataclass(frozen=True)
class ScoringConfig:
    top_ks: tuple[int, ...]
    position_chunk_size: int
    position_profile_bins: int
    bootstrap_samples: int
    bootstrap_seed: int
    output_dir: Path


@dataclass(frozen=True)
class OpdConfig:
    seed: int
    source_subset: str
    max_steps: int
    max_length: int
    max_completion_length: int
    max_prompt_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    optimizer: str
    lmbda: float
    beta: float
    loss_top_k: int
    loss_add_tail: bool
    reverse_kl_top_1_mode: str
    temperature: float
    top_p: float
    top_k: int
    use_vllm: bool
    vllm_gpu_memory_utilization: float
    save_steps: int
    save_total_limit: int
    logging_steps: int
    output_dir: Path


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    models: ModelsConfig
    data: DataConfig
    evaluation: tuple[EvaluationDatasetConfig, ...]
    diagnostics: DiagnosticsConfig
    eval: EvalConfig
    trajectories: TrajectoriesConfig
    scoring: ScoringConfig
    opd: OpdConfig
    report: ReportConfig

    def evaluation_dataset(self, name: str) -> EvaluationDatasetConfig:
        for item in self.evaluation:
            if item.name == name:
                return item
        known = sorted(item.name for item in self.evaluation)
        raise ValueError(f"Unknown benchmark {name!r}; configured benchmarks are {known}")


def load_config(path: str | Path = DEFAULT_CONFIG) -> ExperimentConfig:
    config_path = resolve_project_path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    models_raw = raw["models"]
    data_raw = raw["data"]
    diagnostics_raw = raw["diagnostics"]
    eval_raw = raw["eval"]
    trajectories_raw = raw["trajectories"]
    scoring_raw = raw["scoring"]
    opd_raw = raw["opd"]

    models = ModelsConfig(
        student=models_raw["student"],
        teachers=tuple(models_raw["teachers"]),
        precisions=tuple(models_raw["precisions"]),
        revision=models_raw["revision"],
        attention_implementation=models_raw["attention_implementation"],
        enable_thinking=bool(models_raw["enable_thinking"]),
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
        raise ValueError("All dataset sizes must be positive")

    evaluation = tuple(
        EvaluationDatasetConfig(
            name=item["name"],
            dataset=item["dataset"],
            config=item.get("config"),
            split=item["split"],
            revision=item["revision"],
            question_field=item["question_field"],
            answer_field=item["answer_field"],
            subsample_size=(None if item.get("subsample_size") is None else int(item["subsample_size"])),
        )
        for item in raw.get("evaluation", {}).get("datasets", [])
    )

    diagnostics = DiagnosticsConfig(
        prompt=diagnostics_raw["prompt"],
        warmup_steps=int(diagnostics_raw["warmup_steps"]),
        timed_steps=int(diagnostics_raw["timed_steps"]),
        output_dir=resolve_project_path(diagnostics_raw["output_dir"]),
    )
    evaluation_settings = EvalConfig(
        seed=int(eval_raw["seed"]),
        max_new_tokens=int(eval_raw["max_new_tokens"]),
        batch_size=int(eval_raw["batch_size"]),
        vllm_gpu_memory_utilization=float(eval_raw["vllm_gpu_memory_utilization"]),
        vllm_max_model_length=int(eval_raw["vllm_max_model_length"]),
        output_dir=resolve_project_path(eval_raw["output_dir"]),
    )
    trajectories = TrajectoriesConfig(
        seed=int(trajectories_raw["seed"]),
        prompt_count=int(trajectories_raw["prompt_count"]),
        source_subset=trajectories_raw["source_subset"],
        temperature=float(trajectories_raw["temperature"]),
        top_p=float(trajectories_raw["top_p"]),
        top_k=int(trajectories_raw["top_k"]),
        max_new_tokens=int(trajectories_raw["max_new_tokens"]),
        vllm_gpu_memory_utilization=float(trajectories_raw["vllm_gpu_memory_utilization"]),
        vllm_max_model_length=int(trajectories_raw["vllm_max_model_length"]),
        output_dir=resolve_project_path(trajectories_raw["output_dir"]),
    )
    scoring = ScoringConfig(
        top_ks=tuple(int(value) for value in scoring_raw["top_ks"]),
        position_chunk_size=int(scoring_raw["position_chunk_size"]),
        position_profile_bins=int(scoring_raw["position_profile_bins"]),
        bootstrap_samples=int(scoring_raw["bootstrap_samples"]),
        bootstrap_seed=int(scoring_raw["bootstrap_seed"]),
        output_dir=resolve_project_path(scoring_raw["output_dir"]),
    )
    opd = OpdConfig(
        seed=int(opd_raw["seed"]),
        source_subset=opd_raw["source_subset"],
        max_steps=int(opd_raw["max_steps"]),
        max_length=int(opd_raw["max_length"]),
        max_completion_length=int(opd_raw["max_completion_length"]),
        max_prompt_length=int(opd_raw["max_prompt_length"]),
        per_device_train_batch_size=int(opd_raw["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(opd_raw["gradient_accumulation_steps"]),
        learning_rate=float(opd_raw["learning_rate"]),
        optimizer=opd_raw["optimizer"],
        lmbda=float(opd_raw["lmbda"]),
        beta=float(opd_raw["beta"]),
        loss_top_k=int(opd_raw["loss_top_k"]),
        loss_add_tail=bool(opd_raw["loss_add_tail"]),
        reverse_kl_top_1_mode=opd_raw["reverse_kl_top_1_mode"],
        temperature=float(opd_raw["temperature"]),
        top_p=float(opd_raw["top_p"]),
        top_k=int(opd_raw["top_k"]),
        use_vllm=bool(opd_raw["use_vllm"]),
        vllm_gpu_memory_utilization=float(opd_raw["vllm_gpu_memory_utilization"]),
        save_steps=int(opd_raw["save_steps"]),
        save_total_limit=int(opd_raw["save_total_limit"]),
        logging_steps=int(opd_raw["logging_steps"]),
        output_dir=resolve_project_path(opd_raw["output_dir"]),
    )
    if opd.max_prompt_length + opd.max_completion_length > opd.max_length:
        raise ValueError("max_prompt_length + max_completion_length must not exceed max_length")

    # The scoring diagnostics only mean something if the trajectories were sampled the way OPD
    # samples. Catch a config drift here rather than in the interpretation of the results.
    rollout_mismatch = {
        "temperature": (trajectories.temperature, opd.temperature),
        "top_p": (trajectories.top_p, opd.top_p),
        "top_k": (trajectories.top_k, opd.top_k),
        "max_new_tokens": (trajectories.max_new_tokens, opd.max_completion_length),
    }
    differing = {name: pair for name, pair in rollout_mismatch.items() if pair[0] != pair[1]}
    if differing:
        raise ValueError(f"[trajectories] and [opd] rollout settings disagree: {differing}")

    report = ReportConfig(output_dir=resolve_project_path(raw["report"]["output_dir"]))

    return ExperimentConfig(
        models=models,
        data=data,
        evaluation=evaluation,
        diagnostics=diagnostics,
        eval=evaluation_settings,
        trajectories=trajectories,
        scoring=scoring,
        opd=opd,
        report=report,
    )
