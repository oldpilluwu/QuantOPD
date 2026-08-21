from __future__ import annotations

import argparse
import hashlib
import json
import random
from typing import Any

from datasets import Dataset, load_dataset

from .common import atomic_write_json, write_jsonl
from .config import DEFAULT_CONFIG, load_config
from .data import make_split_indices, select_prompt_field
from .hub import resolve_dataset_revision


def _rows_for_indices(dataset: Dataset, indices: tuple[int, ...], prompt_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index in indices:
        prompt = dataset[source_index][prompt_field]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty or non-text prompt at source row {source_index}")
        rows.append(
            {
                "source_index": source_index,
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
    return rows


def _prompt_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["source_index"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["prompt"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create immutable Phase 0 prompt subsets and index manifests.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="Replace an existing generated manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = config.data.output_dir
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} already exists; pass --force to regenerate it intentionally")

    source_revision = resolve_dataset_revision(config.data.dataset, config.data.revision)
    source = load_dataset(
        config.data.dataset,
        split=config.data.split,
        revision=source_revision,
    )
    prompt_field = select_prompt_field(source.column_names, config.data.prompt_fields)
    splits = make_split_indices(
        population_size=len(source),
        calibration_size=config.data.calibration_size,
        training_size=config.data.training_size,
        smoke_size=config.data.smoke_size,
        seed=config.data.seed,
    )

    calibration_rows = _rows_for_indices(source, splits.calibration, prompt_field)
    training_rows = _rows_for_indices(source, splits.training, prompt_field)
    smoke_rows = training_rows[: config.data.smoke_size]
    files = {
        "calibration": output_dir / "calibration.jsonl",
        "training": output_dir / "training.jsonl",
        "smoke": output_dir / "smoke.jsonl",
    }
    write_jsonl(files["calibration"], calibration_rows)
    write_jsonl(files["training"], training_rows)
    write_jsonl(files["smoke"], smoke_rows)

    evaluation: dict[str, Any] = {}
    for evaluation_config in config.evaluation:
        revision = resolve_dataset_revision(evaluation_config.dataset, evaluation_config.revision)
        load_kwargs: dict[str, Any] = {
            "path": evaluation_config.dataset,
            "split": evaluation_config.split,
            "revision": revision,
        }
        if evaluation_config.config:
            load_kwargs["name"] = evaluation_config.config
        evaluation_dataset = load_dataset(**load_kwargs)
        indices = list(range(len(evaluation_dataset)))
        # A benchmark larger than the budget is subsampled once, here, and the indices are frozen
        # in the manifest. Every model must be scored on identical items for the comparison to
        # mean anything, so this must never be re-drawn at evaluation time.
        if evaluation_config.subsample_size is not None and evaluation_config.subsample_size < len(indices):
            sampler = random.Random(f"{config.data.seed}:{evaluation_config.name}")
            indices = sorted(sampler.sample(indices, evaluation_config.subsample_size))
        evaluation[evaluation_config.name] = {
            "dataset": evaluation_config.dataset,
            "config": evaluation_config.config,
            "split": evaluation_config.split,
            "revision": revision,
            "fingerprint": evaluation_dataset._fingerprint,
            "question_field": evaluation_config.question_field,
            "answer_field": evaluation_config.answer_field,
            "row_count": len(evaluation_dataset),
            "indices": indices,
        }

    manifest = {
        "schema_version": 1,
        "seed": config.data.seed,
        "source": {
            "dataset": config.data.dataset,
            "split": config.data.split,
            "revision": source_revision,
            "fingerprint": source._fingerprint,
            "prompt_field": prompt_field,
            "row_count": len(source),
        },
        "subsets": {
            "calibration": {
                "indices": list(splits.calibration),
                "prompt_sha256": _prompt_hash(calibration_rows),
            },
            "training": {
                "indices": list(splits.training),
                "prompt_sha256": _prompt_hash(training_rows),
            },
            "smoke": {
                "indices": list(splits.smoke),
                "prompt_sha256": _prompt_hash(smoke_rows),
            },
        },
        "files": {name: str(path.resolve()) for name, path in files.items()},
        "evaluation": evaluation,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "source_revision": source_revision}, indent=2))


if __name__ == "__main__":
    main()
