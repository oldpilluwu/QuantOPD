#!/usr/bin/env bash
set -euo pipefail

PHASE0_CONFIG="${1:-configs/phase0.toml}"

if [[ ! -f data/phase0/manifest.json ]]; then
  uv run opd-prepare-data --config "${PHASE0_CONFIG}"
else
  echo "Reusing data/phase0/manifest.json"
fi

for TEACHER_MODEL in Qwen/Qwen3-4B Qwen/Qwen3-14B; do
  for TEACHER_PRECISION in bf16 int8 int4; do
    uv run opd-diagnose \
      --config "${PHASE0_CONFIG}" \
      --teacher "${TEACHER_MODEL}" \
      --precision "${TEACHER_PRECISION}"
  done
done

if [[ "${RUN_TRAIN_SMOKE:-0}" == "1" ]]; then
  TRAIN_TEACHER="${TRAIN_TEACHER:-Qwen/Qwen3-4B}"
  for TEACHER_PRECISION in ${TRAIN_PRECISIONS:-bf16 int8 int4}; do
    uv run opd-train-smoke \
      --config "${PHASE0_CONFIG}" \
      --teacher "${TRAIN_TEACHER}" \
      --precision "${TEACHER_PRECISION}"
  done
fi
