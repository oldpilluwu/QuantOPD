# Progress

Last updated: 2026-08-21

## Status

**Phase 0: complete and verified on GPU.** All six teacher configurations passed diagnostics and
two-step OPD smoke training on an NVIDIA RTX A6000 (49,140 MiB, driver 610.43.02) at commit
`d91142a`.

**Jump-ahead pilot: implemented, not yet executed on GPU.** A single vertical slice covering
benchmarking, trajectory scoring, OPD, and re-benchmarking now exists on branch `jump-ahead`. It
passes 59 CPU tests and `ruff`, and every CLI imports and parses arguments, but **no GPU numbers
have been produced yet.** Nothing below the Phase 0 table is a measured result.

An earlier `phase-1` branch and a downloaded trajectory bundle exist but were deliberately
abandoned in favour of this simpler implementation. They are untouched, not merged, and not used.

## Phase 0 results (measured)

| Teacher | Precision | Teacher memory | Diagnostic peak | Training peak | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | BF16 | 7.49 GiB | 15.37 GiB | 22.07 GiB | 52.76 s |
| 4B | INT8 | 4.11 GiB | 12.00 GiB | 18.69 GiB | 53.52 s |
| 4B | INT4 | 2.42 GiB | 10.38 GiB | 17.07 GiB | 52.22 s |
| 14B | BF16 | 27.51 GiB | 35.39 GiB | 42.09 GiB | 52.53 s |
| 14B | INT8 | 15.20 GiB | 23.09 GiB | 29.79 GiB | 58.77 s |
| 14B | INT4 | 9.05 GiB | 17.17 GiB | 23.87 GiB | 55.92 s |

INT8 cut teacher memory by ~45%, INT4 by ~67%. The key comparison is viable: 14B INT4 at 9.05 GiB
teacher / 23.87 GiB peak sits close to 4B BF16 at 7.49 / 22.07 GiB.

The two-step losses are pipeline checks, not measurements of student quality.

## The jump-ahead pilot

Scope was narrowed to the one comparison `PLAN.md` calls critical — **4B BF16 vs 14B INT4**, near
equal teacher memory, very different teacher size — run end to end rather than broadening the grid.

Pipeline: baseline benchmarks → student trajectories scored by each teacher → OPD → re-benchmark
and re-score. Driven by `scripts/run_pilot.sh`, which skips completed stages so it can resume.

### Deviation: thinking mode is now OFF

Phase 0 fixed `enable_thinking=True`. **The pilot sets it to `False`.**

Reason: a trajectory run under the old branch produced 973/1000 completions that hit the
2,048-token cap without emitting an answer. At that budget the benchmark measures truncation, not
reasoning. Non-thinking with a 1,024-token budget makes answers terminate, cuts eval to minutes,
and makes OPD rollouts affordable on a single A6000.

Cost of the deviation: results are not comparable to Qwen3's published thinking-mode numbers, and
the thinking-mode question is deferred rather than answered.

Two consequences worth remembering:

- TRL 1.6.0's `DistillationDataCollator` calls `apply_chat_template` **without**
  `chat_template_kwargs`, so `enable_thinking` cannot be routed through the dataset. It is bound to
  the tokenizer in `opd.models.force_non_thinking`. Verified against the real Qwen3 template.
- vLLM builds its own tokenizer and would bypass that shim, so prompts cross the boundary as token
  ids, not text.

### Other decisions

- **Benchmark harness:** a thin module over vLLM/Transformers plus Math-Verify, rather than
  LightEval or lm-eval-harness. Avoids a dependency conflict with the pinned
  `transformers 5.15` / `torch 2.13` stack, and keeps the quantized teacher on the exact
  bitsandbytes path OPD uses, so the benchmarked teacher is the teacher that supervises.
- **Gold answers are wrapped in `\boxed{}` before parsing.** Probing Math-Verify 0.9.0 showed
  `parse("x+1")` and `parse("\\text{even}")` return empty — bare MATH-500 golds would have been
  silently counted unparseable. Wrapping also routes gold and prediction through one extraction
  path.
- **GSM8K is subsampled to a fixed seeded 500 items**, frozen in the manifest, to bound teacher
  cost while keeping every model on identical items.
- **Diagnostics use the calibration subset**, disjoint from OPD training prompts by construction.
- **Exact and approximate KL are both computed.** The diagnostics are full-vocabulary; TRL
  optimises a `loss_top_k=1`-plus-tail approximation. Measuring both on the same tokens means the
  approximation gap is known before any result is attributed to teacher precision.

### Dependency stack changed on Linux

Adding the `vllm` extra re-resolved the lock. On **Linux** the pinned stack is now
**torch 2.10.0 / transformers 4.57.6**; Windows keeps torch 2.13.0 / transformers 5.15.0. vLLM
0.19 constrains both, and uv's marker is `sys_platform != 'win32'`, so the GPU server gets the
older versions **even when bootstrapping with `INSTALL_VLLM=0`**.

Consequence: the Phase 0 table above was measured on torch 2.13 / transformers 5.15 and may not
reproduce byte-for-byte. Every report written by the new CLIs records its own package versions via
`opd.runtime.runtime_environment`, so a stack mismatch between two reports is detectable rather
than silent. Teacher footprints are re-measured in the pilot rather than assumed from Phase 0.

### Repository changes

`src/opd_phase0/` → `src/opd/`; `configs/phase0.toml` → `configs/experiment.toml` with new
`[eval]`, `[trajectories]`, `[scoring]`, `[opd]`, and `[report]` sections. `train_smoke.py` was
folded into `train_opd.py`, whose invariant checks now run on every training run rather than only
the smoke test. New modules: `prompts`, `grading`, `generate`, `runtime`, `metrics`, `evaluate`,
`trajectories`, `score`, `train_opd`, `report`.

The config loader now refuses to load a config whose `[trajectories]` sampling settings disagree
with `[opd]`, because the scoring diagnostics only mean something if the trajectories were sampled
the way OPD samples.

## Next step

Run on the A6000, smallest first, checking the gates before spending time on the long stages:

1. `uv run opd-check-env`
2. `uv run opd-prepare-data`
3. `uv run opd-eval --benchmark math500 --limit 16` — confirm truncation rate is low. If most
   completions still hit the cap, stop: the budget or the mode is wrong and accuracy is meaningless.
4. `uv run opd-trajectories --limit 8` then `uv run opd-score --teacher Qwen/Qwen3-4B --precision
   bf16 --limit 8` — confirm reverse KL is finite and positive.
5. `uv run opd-train --teacher Qwen/Qwen3-4B --precision bf16 --subset smoke --max-steps 2`
6. `bash scripts/run_pilot.sh`

Watch for OOM on the 14B INT4 training condition: the estimate is ~40 GiB of 48 GiB, with
`--optimizer adamw_8bit` as the first fallback.

GPU results must be added here only after those commands actually complete.
