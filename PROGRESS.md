# Progress

Last updated: 2026-08-22

## Status

**Phase 0: complete and verified on GPU.** All six teacher configurations passed diagnostics and
two-step OPD smoke training on an NVIDIA RTX A6000 (49,140 MiB, driver 610.43.02) at commit
`d91142a`.

**Jump-ahead pilot: implemented; benchmarking verified on GPU, nothing else run yet.** A vertical
slice covering benchmarking, trajectory scoring, OPD, and re-benchmarking exists on branch
`jump-ahead` (67 CPU tests, `ruff` clean). Benchmark evaluation has run on the A6000 for the
student and both teachers; see "First GPU results" below, which already contains a result worth
acting on.

**Not yet run on GPU:** Omni-MATH, trajectory generation, scoring, and OPD training itself. The
only measured numbers in this document are the Phase 0 table and the First GPU results table.

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

| Model | Precision | accuracy | on finished | truncation | mean tokens | backend |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-1.7B (student) | BF16 | 0.70 | 0.821 | 0.16 | 532.4 | vLLM |
| Qwen3-4B (teacher) | BF16 | 0.81 | 0.940 | 0.16 | 531.8 | vLLM |
| Qwen3-14B (teacher) | INT4 | 0.81 | 0.942 | 0.14 | 526.2 | Transformers |
| Qwen3-14B (teacher) | BF16 | 0.82 | 0.964 | 0.17 | 527.8 | vLLM |

**MATH-500 is saturated.** Going 4B -> 14B at full precision, a 3.5x size increase, buys one
point (0.81 -> 0.82). INT4 costs ~1 point raw and ~2 on finished completions, about two problems
out of 83. At n=100 the interval is roughly +/-9 points, so all three teachers are statistically
indistinguishable. The benchmark stopped measuring teacher capability somewhere below 4B.

Consequences:

- **A harder benchmark is mandatory, not optional.** Omni-MATH replaces GSM8K for this reason.
- **AWQ cannot be evaluated on MATH-500 either.** If NF4 costs only ~1 point here, a better
  quantizer has no room to show an improvement. Quantizer comparisons have to run on the hard set.
- **The teacher-quality question is not settled.** Benchmark accuracy is an argmax-only summary
  while the OPD objective consumes full token-level distributions, so teachers that tie on accuracy
  can still supervise differently. Trajectory scoring is now the load-bearing measurement rather
  than a diagnostic.

Caveats: the teachers ran on different backends (vLLM for BF16, Transformers for INT4), though
greedy decoding makes that a small effect. Near-identical truncation and mean length across all
four models is unexplained; accuracy differs enough that the models are clearly distinct, so it is
most likely difficulty-driven.

The pipeline works end to end and the `dtype=` kwarg is fine on transformers 4.57.6. Parse-failure
rate was 0.0 throughout. Truncated completions are near-uniformly wrong: 69 of the student's 70
correct answers came from finished completions.

## Omni-MATH results (100 items, 50 for the student; 2048-token budget)

| Model | Precision | accuracy | 95% CI | on finished | truncation | backend |
| --- | --- | ---: | :---: | ---: | ---: | --- |
| Qwen3-1.7B (student) | BF16 | 0.17 | [0.08, 0.29] | 0.242 | 0.34 | vLLM |
| Qwen3-4B | BF16 | 0.20 | [0.13, 0.29] | 0.339 | 0.44 | vLLM |
| Qwen3-14B | BF16 | 0.23 | [0.16, 0.32] | 0.390 | 0.41 | vLLM |
| Qwen3-14B | INT4 NF4 | 0.27 | [0.19, 0.36] | 0.403 | 0.38 | Transformers |
| Qwen3-14B-AWQ | AWQ | 0.22 | [0.15, 0.31] | 0.328 | 0.39 | vLLM |

**The difficulty gate passed:** the student at 0.17 is inside the measurable band, unlike MATH-500
where it sat at 0.70 against a saturated ceiling. Omni-MATH is the right benchmark.

**Nothing here is statistically separated.** Every Wilson interval overlaps every other. At n=100
and p ~ 0.2 the interval is about +/-8 points, and the entire observed spread is 7 points.

Two signals that we are reading noise rather than effects:

- **INT4 (0.27) scoring above BF16 (0.23)** for the same 14B model. Quantization improving a
  teacher is not a plausible effect at this size; it is sampling variance, and it is also the only
  cross-backend comparison in the table.
- **Truncation is 34-44%** with mean completions ~1500 against a 2048 cap. Roughly two fifths of
  the score is measuring whether a model finishes in budget rather than whether it can solve the
  problem.

The one comparison deliberately built to be controlled does behave: **AWQ 0.22 vs BF16 0.23, both
on vLLM, same model, same backend.** That suggests AWQ is close to lossless, though the intervals
are far too wide to claim it.

### Two measurement bugs found here

- **Math-Verify treats a timeout as a wrong answer.** A "Timeout during comparison" appeared on the
  14B BF16 run; with `raise_on_error=False` (the default) `verify` logs a warning and returns
  `False`, so grading failures were silently counted as model failures, biased toward the hardest
  problems where sympy has the most work to do. Every call now sets `raise_on_error=True`, and the
  timeout is 15s rather than 5s.

  Note `TimeoutException` subclasses **`BaseException`, not `Exception`**, so `except Exception`
  does not catch it. The first attempt at this fix therefore turned a silent wrong answer into an
  aborted run: one pathological expression killed a full 300-item evaluation. Both `parse` and
  `verify` failures are now caught by name and recorded, and `verification_timeouts` is reported
  separately from `verification_errors`. A non-trivial timeout count biases *comparisons*, not just
  absolute numbers, because it concentrates on the hardest problems.
- **Accuracy was reported without any interval**, which invites over-reading a 7-point spread.
  Reports now carry a Wilson `accuracy_ci95`.

## Omni-MATH at full size (student only so far)

| Model | Precision | n | accuracy | 95% CI | on finished | truncation | verification errors |
| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |
| Qwen3-1.7B (student) | BF16 | 300 | 0.253 | [0.207, 0.305] | 0.354 | 0.31 | 1 |

The student moved 0.17 (n=50) -> 0.253 (n=300), inside the earlier [0.08, 0.29] interval. That is
how much the first N items differ from the whole subsample, and it is the reason the n=50 and n=100
numbers above cannot be read as an ordering.

`verification_errors` is 1 in 300, so Math-Verify timeouts are not distorting the score.

**The teacher numbers in the table above are NOT comparable to this one:** they ran with
`--limit 100`, which takes the *first* 100 frozen indices, while the student ran all 300. Read
naively it says a 1.7B student beats both 14B teachers. It does not; the item sets differ. Every
condition must be re-run at the full 300 before anything is compared.

`opd-report` now detects this: `find_item_count_mismatches` groups evaluations by benchmark and
emits a `comparability_warnings` block plus a loud terminal warning when conditions within one
benchmark were scored on different item counts.

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
