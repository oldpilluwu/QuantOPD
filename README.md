# Quantized Teachers for On-Policy Distillation

Experiments for `PLAN.md`: how teacher **size** and **precision** trade off when distilling
on-policy into a fixed BF16 student.

The current code implements one vertical slice end to end — benchmark the models, score student
trajectories against each teacher, run OPD, then re-benchmark — for the comparison `PLAN.md` calls
critical:

| Teacher | Precision | Teacher memory |
| --- | --- | ---: |
| `Qwen/Qwen3-4B` | BF16 | ~7.5 GiB |
| `Qwen/Qwen3-14B` | INT4 (bitsandbytes NF4) | ~9.1 GiB |

Near-equal teacher memory, very different teacher size. Student is `Qwen/Qwen3-1.7B` in BF16
throughout.

The project reuses Transformers, bitsandbytes, Datasets, vLLM, TRL's `DistillationTrainer`, and
Math-Verify. Project code only handles reproducibility, condition wiring, measurement, and
reporting.

## Fixed decisions

- **Student:** `Qwen/Qwen3-1.7B`, full-weight BF16 training, identical initial checkpoint in every
  condition.
- **Objective:** fully on-policy (`lmbda=1`) reverse KL (`beta=1`), `loss_top_k=1` with a tail
  bucket — carried forward unchanged from the Phase 0 smoke run.
- **Data:** prompt-only examples from `open-r1/OpenR1-Math-220k`. Supplied reasoning traces are
  never trained on; OPD supervises the student on states the student itself visits.
- **Thinking mode: off.** See the note below — this is a deliberate deviation.
- **Benchmarks:** MATH-500 (all 500) and Omni-MATH (a fixed seeded 300-item subsample),
  greedy pass@1 at 2,048 tokens. Omni-MATH is olympiad-level: a 1.7B student already scores ~0.70
  on MATH-500, which compresses the student-teacher gap. GSM8K stays configured but is excluded
  from the pilot -- a model this size ceilings there, so it costs INT4 teacher time for no signal.
- **Runtime:** one Linux NVIDIA GPU with BF16 support. Developed against an A6000 48 GB.

All scientific settings live in `configs/experiment.toml`.

### Why thinking mode is disabled

Qwen3 defaults to thinking mode. An earlier run with thinking on and a 2,048-token budget produced
973/1000 completions that hit the cap without emitting an answer — the accuracy measured was
truncation, not reasoning. This pilot runs `enable_thinking=False` so answers terminate, eval
takes minutes rather than hours, and OPD rollouts stay affordable on one GPU.

Budgets differ by stage on purpose: benchmarks get 2,048 tokens (olympiad problems need room),
while OPD rollouts stay at 1,024 to bound training cost. Evaluating with more headroom than
training does not bias anything, because baseline and post-OPD students are measured identically.

Two implementation notes follow from that:

- TRL 1.6.0's `DistillationDataCollator` calls `apply_chat_template` with **no**
  `chat_template_kwargs`, so the flag cannot be passed through the dataset. `opd.models
  .force_non_thinking` binds it to the tokenizer instead, which is the only way evaluation,
  trajectory generation, scoring, and training all render prompts identically.
- vLLM builds its own tokenizer internally and would not see that shim, so prompts are rendered
  once with the shimmed tokenizer and passed to vLLM as **token ids**, never as text.

## Setup

Python 3.11 or 3.12 on a Linux GPU server. Install `uv`, clone, then:

```bash
bash scripts/bootstrap.sh
```

This installs the Linux-only vLLM extra by default; use `INSTALL_VLLM=0` for a CPU-only
environment. Expose a single GPU (`CUDA_VISIBLE_DEVICES=0`) — both models are placed on the first
visible device. Set `HF_TOKEN` if Hub authentication is needed.

## Running the pilot

```bash
bash scripts/run_pilot.sh
```

Each stage skips work already on disk, so an interrupted run resumes with the same command.
Restrict the grid with environment variables:

```bash
CONDITIONS="Qwen/Qwen3-14B:int4" BENCHMARKS=math500 bash scripts/run_pilot.sh
```

### Or one stage at a time

```bash
uv run opd-prepare-data
```

Resolves every dataset branch to an immutable Hub commit, builds disjoint calibration/training
subsets, freezes the benchmark item indices, and writes `data/manifest.json`. Refuses to overwrite
without `--force`.

```bash
uv run opd-eval --model Qwen/Qwen3-14B --precision int4 --benchmark math500 --tag teacher
```

Greedy pass@1. BF16 models run through vLLM; quantized teachers run through Transformers with the
**same bitsandbytes loader OPD uses**, so the benchmarked teacher is the teacher that supervises.
Reports accuracy, accuracy-on-finished, parse-failure rate, truncation rate, throughput, footprint,
and peak VRAM. Use `--checkpoint` to evaluate a trained student and `--limit` to smoke test.

The Transformers path is throughput-bound on **batch size**, not attention: bitsandbytes
dequantizes weights on every forward pass. Qwen3-14B's KV cache is ~160 KiB/token, so at a
2,048-token budget batch 32 needs ~19 GiB and batch 64 ~29 GiB including 9 GiB of INT4 weights.
Push it with `--batch-size 64` on a 48 GB card. `--attn-implementation flash_attention_2` is
available too, but requires the `flash-attn` package and helps far less than batching here.

`--backend vllm` is rejected for quantized precisions: the engine loads unquantized weights, so it
would report INT4 while measuring BF16.

Accuracy is also sliced by whatever grouping columns a benchmark declares (`level` and `subject`
for MATH-500, `difficulty` for Omni-MATH) into `accuracy_by_group`. That slice is free — the
completions already exist — and it shows whether a headline score is ceilinged on easy items, and
where the student-teacher gap actually lives.

```bash
uv run opd-trajectories --tag baseline
uv run opd-score --teacher Qwen/Qwen3-14B --precision int4
```

Trajectories are sampled from the **calibration** subset, which is disjoint from the OPD training
subset by construction, at the same temperature/top-p/top-k/length OPD uses (the config loader
rejects a mismatch). Scoring then measures, per student-generated token:

- `KL(student‖teacher)` — the OPD objective — and `KL(teacher‖student)`
- student and teacher entropy, and the entropy shift
- top-{1,5,10,50} agreement
- teacher probability mass on the student's top-k support
- teacher log-probability of each sampled token
- first position where the teacher's argmax differs from the sampled token
- the `loss_top_k=1`-plus-tail approximation TRL actually optimises, on the same tokens, so the gap
  between the diagnostic and the objective is measured rather than assumed

Both models are resident at once and each trajectory is scored at batch 1, so there is no padding
to mask. Log-softmax runs in float32 with float64 accumulation, chunked over positions. **No logits
are persisted.** Results are reported prompt-weighted and token-weighted, each with a seeded
bootstrap 95% CI resampled over prompts (never tokens — tokens within a trajectory are dependent).

```bash
uv run opd-train --teacher Qwen/Qwen3-14B --precision int4
```

TRL `DistillationTrainer` with colocated vLLM student generation. `--max-steps 2` reproduces the
original Phase 0 smoke test; the invariant checks (finite loss, teacher frozen and gradient-free,
student fully trainable, `lmbda==1`, `beta==1`) run on every training run.

If the 14B INT4 condition runs out of memory on a 48 GB card, in order: `--optimizer adamw_8bit`
(drops optimizer state ~13.6 → ~3.4 GiB), then `--vllm-gpu-memory-utilization 0.15`, then
`--no-vllm`.

```bash
uv run opd-report
```

Joins every stage's reports into `artifacts/summary/{summary.json,headline.csv,...}`.

## Local checks

Model commands need the GPU server; everything else runs on CPU with no downloads:

```bash
uv run pytest
```

```bash
uv run ruff check .
```

## Reading the results

Before trusting any number:

- truncation rate should be low; a high rate means the budget is being measured, not the model.
  Compare `accuracy` against `accuracy_on_finished`, because Math-Verify extracts a bare number
  when there is no `\boxed{}`, so a truncated completion still "parses" and truncation hides
  inside the parse-failure rate;
- `KL(student‖teacher)` must be finite and positive at baseline, and should fall over training
  (`opd-score` fails loudly if it is not);
- measured teacher footprints should match Phase 0: 4B BF16 ≈ 7.49 GiB, 14B INT4 ≈ 9.05 GiB.

## Scope

Not yet implemented: the 8B teacher, INT8 conditions, 14B BF16, GPTQ/AWQ, multiple seeds, the
temperature ablation, a second student size, and the off-policy KD baseline. These are later
`PLAN.md` phases and none are needed to decide whether this slice works.
