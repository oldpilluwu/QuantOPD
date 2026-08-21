# Progress

Last updated: 2026-08-22

## Status

**Phase 0: complete and verified on GPU.** All six teacher configurations passed diagnostics and
two-step OPD smoke training on an NVIDIA RTX A6000 (49,140 MiB, driver 610.43.02) at commit
`d91142a`.

**Jump-ahead pilot: implemented; benchmarking verified on GPU, nothing else run yet.** A vertical
slice covering benchmarking, trajectory scoring, OPD, and re-benchmarking exists on branch
`jump-ahead` (61 CPU tests, `ruff` clean). Benchmark evaluation has run on the A6000 for the
student and the 4B BF16 teacher; see "First GPU results" below.

**Not yet run on GPU:** the 14B INT4 teacher, Omni-MATH, trajectory generation, scoring, and OPD
training itself. The only measured numbers in this document are the Phase 0 table and the First
GPU results table.

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
reasoning. Non-thinking makes answers terminate, cuts eval to minutes, and keeps OPD rollouts
affordable on a single A6000. Confirmed in practice: MATH-500 truncation fell to 16% at a
1,024-token budget, versus 97% with thinking on.

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
- **Oversized benchmarks are subsampled to a fixed seeded index set**, frozen in the manifest, to
  bound teacher cost while keeping every model on identical items.
- **Diagnostics use the calibration subset**, disjoint from OPD training prompts by construction.
- **Exact and approximate KL are both computed.** The diagnostics are full-vocabulary; TRL
  optimises a `loss_top_k=1`-plus-tail approximation. Measuring both on the same tokens means the
  approximation gap is known before any result is attributed to teacher precision.

### Dependency stack changed

Adding the `vllm` extra re-resolved the lock to **torch 2.10.0 / transformers 4.57.6** on every
platform (vLLM 0.19 constrains both). This applies on the GPU server **even when bootstrapping
with `INSTALL_VLLM=0`**.

Two uv settings are needed to make `uv sync --extra vllm` work on the server:

- `required-environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]` — the lock
  is regenerated from a non-Linux dev machine, so uv otherwise has no reason to prefer wheels the
  server can install. This also collapsed the resolution to a single fork, so the dev machine and
  the server now run identical versions instead of diverging (Windows previously resolved
  torch 2.13 / transformers 5.15).
- `constraint-dependencies = ["xgrammar!=0.2.4"]` — **xgrammar 0.2.4 is a broken release.** It
  published 13 files where neighbouring versions publish 35, and ships no cp312 Linux x86_64 wheel;
  0.2.3 and 0.2.5-rc do. Its cp311 x86_64 wheel *does* exist, which was enough to satisfy
  `required-environments`, so the first fix alone still failed on the Python 3.12 server. vLLM 0.19
  asks only for `xgrammar>=0.1.32,<1.0.0`, so pinning away from 0.2.4 costs nothing.

When regenerating the lock, verify wheel coverage rather than assuming it: every ABI-specific
package must have a `manylinux ... x86_64` wheel for **each** `cpXY` it targets, not just one.

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

## First GPU results (MATH-500, first 100 items, non-thinking, 1024-token budget)

| Model | Precision | accuracy | on finished | truncation | mean tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B (student) | BF16 | 0.70 | 0.821 | 0.16 | 532.4 |
| Qwen3-4B (teacher) | BF16 | 0.81 | 0.940 | 0.16 | 531.8 |

The pipeline works end to end and the `dtype=` kwarg is fine on transformers 4.57.6. Parse-failure
rate was 0.0 for both. Truncated completions are near-uniformly wrong: 69 of the student's 70
correct answers came from finished completions.

Caveats: n=100 gives roughly a +/-9 point interval, so the 11-point gap is only just outside noise.
Identical 0.16 truncation and near-identical mean length for both models is unexplained and worth
one overlap check on the per-item files; accuracy differs enough that the models are clearly
distinct, so it is most likely difficulty-driven.

## Benchmark change after those results

MATH-500 compresses the comparison: a 1.7B student at 0.70 leaves the teachers little room, and
GSM8K would be worse (a model this size ceilings there at ~85-90%) while still costing INT4 teacher
generation time.

- **GSM8K is no longer run by the pilot.** It stays in the config, one flag away
  (`BENCHMARKS="math500 omnimath gsm8k"`).
- **Omni-MATH replaces it** as the discriminating benchmark: olympiad-level, in-domain with the
  OpenR1-Math training prompts, subsampled to a fixed seeded 300 items.
- **Eval budget raised to 2,048 tokens** for every benchmark, with the vLLM context raised to 4,096
  so prompt plus completion cannot overflow. **This invalidates the MATH-500 numbers above**; they
  must be regenerated before being compared with anything.
- OPD rollouts stay at 1,024 to bound training cost. Evaluating with more headroom than training
  does not bias the comparison, since baseline and post-OPD students are measured identically.

Omni-MATH was chosen over OlympiadBench, which was the original intent. OlympiadBench stores
`final_answer` as a list with multi-answer rows and units, needing grading logic MATH-500 does not
use, and its difficulty is the constant "Competition". Omni-MATH has a plain string `answer` (so it
reuses the existing code path unchanged) and a numeric 1-10 `difficulty`, which locates the
measurable band if the student floors out.

Accuracy is now sliced by declared `group_fields` (`level`/`subject` for MATH-500, `difficulty` for
Omni-MATH). The slice is free and shows whether a headline score is ceilinged.

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
