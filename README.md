# Quantized Teachers for On-Policy Distillation

This repository contains the Phase 0 pipeline for the experiments described in `PLAN.md`. It deliberately reuses
Hugging Face Transformers, bitsandbytes, Datasets, Accelerate, and TRL's `DistillationTrainer`; the project code only
handles reproducibility, model loading, diagnostics, and experiment launch.

## Fixed Phase 0 decisions

- Student: `Qwen/Qwen3-1.7B`, full-weight BF16 training.
- Teachers: `Qwen/Qwen3-4B` and `Qwen/Qwen3-14B` in BF16, bitsandbytes INT8, or bitsandbytes NF4.
- Qwen3 thinking mode: enabled. The diagnostics verify that it is the tokenizer's default.
- Data: prompt-only examples from `open-r1/OpenR1-Math-220k`.
- OPD: fully on-policy (`lmbda=1`) with reverse KL (`beta=1`).
- Smoke-training loss: sampled top-1 support plus a tail bucket (`loss_top_k=1`, `loss_add_tail=true`).
- Diagnostic loss: exact full-vocabulary reverse KL on one token.
- Runtime: one Linux NVIDIA GPU with BF16 support. No vLLM or distributed setup is used in Phase 0.

Dependency versions are recorded in `uv.lock`. TRL is pinned because `DistillationTrainer` is experimental.

## Server setup

Use Python 3.11 or 3.12 on the Linux server. Install `uv`, clone the repository, then run:

```bash
bash scripts/bootstrap.sh
```

If the server has several physical GPUs, expose one for this phase, for example `CUDA_VISIBLE_DEVICES=0`. The scripts
will place both models on the first visible GPU.

If Hugging Face authentication is required, set `HF_TOKEN` before downloading models or datasets.

## Prepare fixed data

```bash
uv run opd-prepare-data
```

This resolves every dataset branch to an immutable Hub commit, creates disjoint calibration and training indices,
writes prompt-only JSONL files, and records hashes and evaluation indices in `data/phase0/manifest.json`. It refuses to
replace that manifest unless `--force` is passed.

## Run one diagnostic condition

```bash
uv run opd-diagnose --teacher Qwen/Qwen3-4B --precision int4
```

The diagnostic verifies tokenizer and chat-template identity, thinking mode, teacher freezing, BF16 student weights,
finite reverse KL, student gradients, deterministic teacher logits, probability normalization, model footprint, CUDA
memory, and teacher latency. Reports are written under `artifacts/phase0/diagnostics/`.

## Run the diagnostic matrix

```bash
bash scripts/run_phase0_matrix.sh
```

Conditions run sequentially so only one student/teacher pair occupies the GPU. The script prepares data first. If the
manifest already exists, run the six diagnostic commands individually rather than regenerating it.

## Run a two-step OPD smoke test

```bash
uv run opd-train-smoke --teacher Qwen/Qwen3-4B --precision int8
```

To run all three 4B teacher smoke conditions after the diagnostic matrix:

```bash
RUN_TRAIN_SMOKE=1 bash scripts/run_phase0_matrix.sh
```

The smoke test uses TRL's native generation path and updates every student parameter. It does not save a multi-GB
student checkpoint. It writes a compact report under `outputs/phase0/`.

An A6000 may not fit full-weight student training together with the 14B BF16 teacher. Start with the 4B smoke matrix.
The 14B loading/forward diagnostics are still expected to fit. An A100 80GB has more room, but actual peak memory is
always recorded rather than assumed.

## Local checks

The model commands require the Linux GPU server. Pure project tests can run without downloading models:

```bash
uv run pytest
uv run ruff check .
```

All scientific settings live in `configs/phase0.toml`. Generated data, model caches, diagnostics, and training outputs
are intentionally excluded from Git.
