#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# vLLM is the rollout backend for benchmarks, trajectories, and colocated OPD generation.
# INSTALL_VLLM=0 gives a CPU-testable environment without it.
if [[ "${INSTALL_VLLM:-1}" == "1" ]]; then
  uv sync --group dev --extra vllm
else
  uv sync --group dev
fi

uv run opd-check-env

