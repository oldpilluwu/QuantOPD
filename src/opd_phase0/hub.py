from __future__ import annotations

from huggingface_hub import HfApi


def resolve_model_revision(repo_id: str, revision: str) -> str:
    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for model {repo_id}@{revision}")
    return info.sha


def resolve_dataset_revision(repo_id: str, revision: str) -> str:
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for dataset {repo_id}@{revision}")
    return info.sha
