#!/usr/bin/env bash
# End-to-end pilot: baseline benchmarks -> trajectories + teacher scoring -> OPD -> re-evaluation.
#
# The default conditions are PLAN.md's critical near-equal-memory comparison:
#   Qwen3-4B  BF16  (~7.5 GiB teacher)
#   Qwen3-14B INT4  (~9.1 GiB teacher)
#
# Each stage skips work that is already on disk, so an interrupted run can be restarted with the
# same command. Override the grid with, for example:
#   CONDITIONS="Qwen/Qwen3-14B:int4" bash scripts/run_pilot.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/experiment.toml}"
CONDITIONS="${CONDITIONS:-Qwen/Qwen3-4B:bf16 Qwen/Qwen3-14B:int4}"
BENCHMARKS="${BENCHMARKS:-math500 gsm8k}"
LIMIT_ARGS="${LIMIT_ARGS:-}"

run() {
  echo "+ $*" >&2
  "$@"
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed 's/-\+/-/g; s/^-//; s/-$//'
}

# ---------------------------------------------------------------- fixed data
if [[ ! -f data/manifest.json ]]; then
  run uv run opd-prepare-data --config "$CONFIG"
else
  echo "Reusing data/manifest.json" >&2
fi

# ------------------------------------------------- step 1: baseline accuracy
for benchmark in $BENCHMARKS; do
  student_slug="$(slugify "$(grep -oP '(?<=^student = ")[^"]+' "$CONFIG")")"
  if [[ ! -f "artifacts/eval/${student_slug}-bf16-baseline/${benchmark}/report.json" ]]; then
    run uv run opd-eval --config "$CONFIG" --benchmark "$benchmark" --tag baseline $LIMIT_ARGS
  fi
  for condition in $CONDITIONS; do
    teacher="${condition%%:*}"
    precision="${condition##*:}"
    if [[ ! -f "artifacts/eval/$(slugify "$teacher")-${precision}-teacher/${benchmark}/report.json" ]]; then
      run uv run opd-eval --config "$CONFIG" --model "$teacher" --precision "$precision" \
        --benchmark "$benchmark" --tag teacher $LIMIT_ARGS
    fi
  done
done

# ------------------------------- step 2: student trajectories + teacher scoring
if [[ ! -f artifacts/trajectories/baseline/manifest.json ]]; then
  run uv run opd-trajectories --config "$CONFIG" --tag baseline $LIMIT_ARGS
fi
for condition in $CONDITIONS; do
  teacher="${condition%%:*}"
  precision="${condition##*:}"
  if [[ ! -f "artifacts/scoring/baseline/$(slugify "$teacher")-${precision}/report.json" ]]; then
    run uv run opd-score --config "$CONFIG" --teacher "$teacher" --precision "$precision" $LIMIT_ARGS
  fi
done

# ------------------------------------------------------------ step 3: OPD
for condition in $CONDITIONS; do
  teacher="${condition%%:*}"
  precision="${condition##*:}"
  checkpoint="outputs/opd/$(slugify "$teacher")-${precision}/final"
  if [[ ! -f "$checkpoint/config.json" ]]; then
    run uv run opd-train --config "$CONFIG" --teacher "$teacher" --precision "$precision"
  else
    echo "Reusing $checkpoint" >&2
  fi
done

# --------------------------------------------- step 4: post-OPD measurement
for condition in $CONDITIONS; do
  teacher="${condition%%:*}"
  precision="${condition##*:}"
  tag="opd-$(slugify "$teacher")-${precision}"
  checkpoint="outputs/opd/$(slugify "$teacher")-${precision}/final"

  for benchmark in $BENCHMARKS; do
    student_slug="$(slugify "$(grep -oP '(?<=^student = ")[^"]+' "$CONFIG")")"
    if [[ ! -f "artifacts/eval/${student_slug}-bf16-${tag}/${benchmark}/report.json" ]]; then
      run uv run opd-eval --config "$CONFIG" --checkpoint "$checkpoint" --benchmark "$benchmark" \
        --tag "$tag" $LIMIT_ARGS
    fi
  done

  # Did the student actually move toward its teacher?
  if [[ ! -f "artifacts/trajectories/${tag}/manifest.json" ]]; then
    run uv run opd-trajectories --config "$CONFIG" --checkpoint "$checkpoint" --tag "$tag" $LIMIT_ARGS
  fi
  if [[ ! -f "artifacts/scoring/${tag}/$(slugify "$teacher")-${precision}/report.json" ]]; then
    run uv run opd-score --config "$CONFIG" --teacher "$teacher" --precision "$precision" \
      --trajectories "artifacts/trajectories/${tag}/manifest.json" --student-checkpoint "$checkpoint" \
      --tag "$tag" $LIMIT_ARGS
  fi
done

run uv run opd-report --config "$CONFIG"
