# PLAN.md

# Quantized Teachers for On-Policy Distillation

## Project Goal

Study how **teacher model size** and **teacher quantization level** affect the quality and efficiency of **on-policy distillation (OPD)** into a fixed full-precision student.

The central practical question is:

> At a fixed teacher-memory or teacher-serving budget, is it better to use a **smaller BF16 teacher** or a **larger quantized teacher**?

The central scientific question is:

> How do teacher size and teacher precision interact to determine how useful, learnable, and efficient the teacher supervision is for a fixed BF16 student?

The desired final contribution is a **teacher size–precision Pareto frontier for OPD**.

A strong result would show that, at equal or lower teacher memory:

- a larger INT8/INT4 teacher matches or outperforms a smaller BF16 teacher;
- the best teacher precision depends on teacher size and/or student size;
- the observed trade-off is reproducible across quantizers, datasets, and student scales.

---

# Core Experimental Setup

## Student

Primary student:

- `Qwen/Qwen3-1.7B`
- Student precision: **BF16**
- Keep the same initial student checkpoint across all experimental conditions.

Optional second student for scaling:

- `Qwen/Qwen3-4B`
- Student precision: **BF16**

## Teachers

Primary teacher sizes:

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`

Teacher precision conditions:

- BF16
- 8-bit
- 4-bit

Recommended paper-quality quantization:

- GPTQ 8-bit
- GPTQ 4-bit

Recommended quick pilot quantization:

- bitsandbytes 8-bit
- bitsandbytes 4-bit / NF4

Optional robustness quantizer:

- AWQ 4-bit

## Training Dataset

Primary OPD prompt dataset:

- `open-r1/OpenR1-Math-220k`

Use only the **problem/question** field for the main OPD experiments.

Do **not** train on the supplied reasoning traces in the primary experiment.

Reason:

- OPD should use student-generated trajectories.
- The teacher should supervise the student on the states the student itself visits.

Initial training subset:

- 5,000–10,000 prompts for pilot runs.

Larger experiment:

- 20,000–50,000 prompts if needed after the pilot succeeds.

## Evaluation Datasets

Primary:

- `HuggingFaceH4/MATH-500`

Secondary:

- `openai/gsm8k`

Optional later additions:

- AIME-style benchmark
- OlympiadBench
- code reasoning benchmark
- instruction-following benchmark

Do not use evaluation datasets for teacher calibration or OPD training.

---

# Metrics Used Throughout the Project

## Student Quality Metrics

Primary:

- MATH-500 exact-answer accuracy
- GSM8K exact-answer accuracy

Secondary:

- pass@1
- pass@k where relevant
- average generated tokens
- reasoning length
- invalid / malformed answer rate
- repetition / degeneration rate

## Teacher Quality Metrics

- teacher standalone benchmark accuracy
- teacher entropy
- teacher top-1 agreement with BF16 teacher
- KL divergence from BF16 teacher
- teacher/student KL divergence
- teacher top-k mass on the student's support

## Efficiency Metrics

Measure actual system behavior rather than theoretical bit-width alone.

- model weight memory
- peak GPU VRAM
- teacher inference tokens/sec
- teacher forward latency
- OPD wall-clock training time
- GPU-hours
- GPU count required
- teacher/student co-location feasibility
- communication overhead if teacher and student are on separate devices

## OPD Metrics

- OPD loss
- KL loss over training
- student entropy
- gradient norm
- training stability
- reward / answer accuracy during rollouts
- first divergence position where useful
- percentage of teacher calls producing useful corrective information

---

# Phase 0 — Environment and Reproducibility Setup

## Goal

Build one reproducible pipeline before running any scientific experiment.

The same code path should support:

- BF16 teachers
- 8-bit teachers
- 4-bit teachers
- fixed BF16 student
- identical prompts
- identical student initialization
- identical OPD objective

## Models Needed

- Qwen3-1.7B
- Qwen3-4B
- Qwen3-14B

8B can be added later.

## Dataset Needed

- OpenR1-Math-220k

Use a fixed calibration subset and a fixed OPD training subset.

Save:

- calibration indices
- training indices
- evaluation indices
- random seeds

## Evals Needed

None beyond smoke tests.

## What to Check

- all models load correctly;
- teacher logits are accessible in every precision;
- the same prompt tokenizes identically for every teacher;
- teacher quantization does not change tokenizer behavior;
- student stays BF16;
- student gradients are correct;
- teacher parameters receive no gradients;
- OPD loss is finite;
- quantized teacher inference is reproducible.

## Metrics

- peak VRAM
- forward-pass latency
- logits shape
- probability normalization
- KL sanity checks
- deterministic output under fixed seeds where expected

## Decision

Do not proceed until all precision conditions can be evaluated through the exact same interface.

---

# Phase 1 — Teacher Quantization Sanity Check

## Goal

Determine how much teacher behavior changes under 8-bit and 4-bit quantization before spending compute on OPD.

This phase answers:

> Does teacher quantization preserve enough of the BF16 teacher's behavior to justify training experiments?

## Models Needed

Small teacher:

- Qwen3-4B BF16
- Qwen3-4B 8-bit
- Qwen3-4B 4-bit

Large teacher:

- Qwen3-14B BF16
- Qwen3-14B 8-bit
- Qwen3-14B 4-bit

Student:

- Qwen3-1.7B BF16

## Dataset Needed

Sample:

- 500–1,000 prompts from OpenR1-Math-220k

Use the same prompts for every teacher.

Use the untouched student to generate student trajectories.

Then evaluate every teacher on the exact same student prefixes.

## Evals Needed

No training yet.

Evaluate teacher behavior and resource usage.

## What to Look For

### 1. Teacher fidelity

Compare each quantized teacher against its own BF16 version.

Measure:

- top-1 agreement;
- top-k agreement;
- KL divergence;
- entropy shift;
- answer accuracy.

### 2. Teacher-to-student compatibility

Compare:

- KL(teacher || student)
- probability mass the teacher assigns to tokens within the student's top-k predictions.

Look for whether quantization:

- makes the teacher more similar to the student;
- makes it less similar;
- preserves useful corrections;
- destroys important distinctions.

### 3. Efficiency

Measure whether memory actually decreases as expected.

A nominal 4-bit checkpoint is not enough; record real peak VRAM and throughput.

## Metrics

For each teacher condition:

- standalone accuracy
- BF16-vs-quantized KL
- BF16-vs-quantized top-1 agreement
- teacher entropy
- teacher/student KL
- teacher top-k mass on student support
- VRAM
- tokens/sec
- latency

## Important Insights

Possible outcome A:

- INT8 is extremely close to BF16.
- INT4 degrades moderately.

Interpretation:

- INT8 is likely a very strong efficiency candidate.

Possible outcome B:

- INT4 retains high teacher accuracy but changes the probability distribution.

Interpretation:

- potentially interesting for distillation even if greedy outputs remain similar.

Possible outcome C:

- INT4 becomes unstable or highly divergent.

Interpretation:

- use 8-bit as the main precision reduction;
- reconsider the quantizer;
- test GPTQ vs AWQ before abandoning 4-bit.

## Decision

Continue to Phase 2 if at least one quantized teacher gives:

- substantial memory reduction;
- reasonably preserved teacher quality;
- stable logits.

---

# Phase 2 — Minimal OPD Pilot

## Goal

Test whether teacher precision changes final BF16 student quality.

This is the first experiment that can determine whether the project is scientifically alive.

## Models Needed

Student:

- Qwen3-1.7B BF16

Teachers:

- Qwen3-4B BF16
- Qwen3-4B 8-bit
- Qwen3-4B 4-bit
- Qwen3-14B BF16
- Qwen3-14B 8-bit
- Qwen3-14B 4-bit

Total:

- 6 teacher conditions
- 1 fixed student checkpoint

## Dataset Needed

- 5,000–10,000 OpenR1-Math-220k prompts

Every run must use:

- identical prompt set;
- identical student initialization;
- identical training-token budget;
- identical optimizer;
- identical learning rate;
- identical rollout settings;
- identical OPD objective.

## Evals Needed

Primary:

- MATH-500

Secondary:

- GSM8K

## Main Questions

### Question 1

Does quantizing a teacher hurt final student performance?

### Question 2

Can a quantized large teacher outperform a smaller BF16 teacher?

Example critical comparison:

- 14B 4-bit
- 4B BF16

### Question 3

Does the effect of precision depend on teacher size?

Example:

- 4B may need BF16;
- 14B may tolerate 8-bit or 4-bit.

That interaction is more interesting than a global "INT8 is good" result.

## Metrics

For each run:

- final MATH-500 accuracy
- final GSM8K accuracy
- OPD training loss
- student rollout correctness
- generated token length
- peak teacher VRAM
- peak total VRAM
- teacher throughput
- total wall-clock time
- GPU-hours

## What to Look For

### Strong signal

At similar teacher memory:

- larger quantized teacher > smaller BF16 teacher.

Example:

- 14B INT4 > 4B BF16.

This supports the central efficiency story.

### Very strong signal

Quantized teacher > same-size BF16 teacher.

Example:

- 14B INT8 > 14B BF16.

This suggests quantization may alter supervision in a beneficial way.

### Weak but useful signal

INT8 teacher ≈ BF16 teacher with much lower memory.

This still supports efficient teacher serving, though the paper needs stronger scaling evidence.

### Negative signal

Every quantized teacher consistently harms student quality more than its memory savings justify.

Then:

- change quantizer;
- use less aggressive precision;
- reconsider whether the project should focus on a threshold law rather than an improved frontier.

## Decision

Proceed if one of these happens:

1. larger low-bit teacher beats smaller BF16 teacher at comparable memory;
2. quantized teacher matches BF16 teacher with significant resource savings;
3. teacher size and precision show a clear interaction.

---

# Phase 3 — Equal-Memory Teacher Frontier

## Goal

Convert the pilot into the main practical result.

Study:

> At a fixed teacher-memory budget, which combination of teacher size and precision gives the best BF16 student?

## Models Needed

Student:

- Qwen3-1.7B BF16

Teachers:

- Qwen3-4B
- Qwen3-8B
- Qwen3-14B

Precisions:

- BF16
- 8-bit
- 4-bit

Total:

- 9 primary teacher configurations

## Dataset Needed

Use the same dataset family.

Training:

- 10,000–30,000 OpenR1-Math prompts

Keep a fixed prompt set for all runs.

## Evals Needed

- MATH-500
- GSM8K

Optional:

- one harder reasoning benchmark

## Core Plot

Plot:

**final student quality vs actual teacher memory**

Every point represents:

- teacher size
- teacher precision

Draw the Pareto frontier.

## Additional Plot

Plot:

**final student quality vs teacher GPU-hours**

This distinguishes memory efficiency from compute efficiency.

## What to Look For

### Frontier crossover

Example:

At ~8 GB:

- 4B BF16 student result = 39
- 8B INT8 student result = 42
- 14B INT4 student result = 42.5

This is a strong result.

### Dominated configurations

Determine whether some teacher settings are never sensible.

Example:

- 8B BF16 may be dominated by 14B INT8.

### Precision threshold

Determine how far each teacher size can be compressed before student quality collapses.

### Size-dependent robustness

Larger teachers may tolerate more quantization because of redundancy.

Or the opposite may happen.

Both outcomes are scientifically interesting.

## Metrics

- student accuracy
- teacher memory
- total memory
- throughput
- GPU-hours
- wall time
- final KL
- teacher fidelity
- quantization ratio
- quality per GB
- quality per GPU-hour

## Derived Metrics

Possible practical metric:

`StudentAccuracy / TeacherMemoryGB`

Use with caution; the Pareto plot is more important.

## Decision

If a stable Pareto structure emerges, this becomes the core paper result.

---

# Phase 4 — Temperature and Calibration Ablation

## Goal

Understand whether improvements from quantized teachers come from simple probability softening or something more specific to quantization.

This phase is not required to prove the efficiency claim.

It is needed to support stronger mechanistic claims.

## Models Needed

Focus only on configurations that showed an interesting effect.

Example:

- 14B BF16
- 14B INT8
- 14B INT4

Create BF16 temperature variants:

- temperature 1.0
- temperature 1.1
- temperature 1.2
- temperature 1.3
- possibly entropy-matched temperature

## Dataset Needed

Same OPD dataset subset as the main experiment.

## Evals Needed

Same evaluation suite.

## What to Look For

### If temperature reproduces the quantized-teacher gain

Interpretation:

- quantization may be acting largely through calibration / logit softening.

Still useful:

- quantization provides that behavior while reducing memory.

Do not claim a unique quantization mechanism.

### If quantization beats entropy-matched BF16 teachers

Interpretation:

- quantization changes supervision in ways beyond global temperature.

Investigate:

- token ranking changes;
- tail probability changes;
- local support overlap;
- teacher/student KL;
- logit perturbation structure.

## Metrics

- student accuracy
- teacher entropy
- teacher/student KL
- top-k overlap
- calibration error
- entropy-matching error
- student final accuracy

## Decision

Use this phase to decide how strong the mechanism section can be.

---

# Phase 5 — Quantizer Robustness

## Goal

Show that the result is not an artifact of one quantization implementation.

## Models Needed

Use only the most informative teacher size.

Example:

- Qwen3-14B

Compare:

- BF16
- GPTQ 4-bit
- AWQ 4-bit
- bitsandbytes NF4

Optional:

- GPTQ 8-bit

## Dataset Needed

Same calibration set where applicable.

Same OPD training set.

## Evals Needed

Same main benchmark set.

## What to Look For

### Robust precision effect

If multiple 4-bit methods show similar teacher-student behavior, the claim is stronger.

### Quantizer-specific effect

If only one quantizer works, the paper becomes more about quantizer design choices than bit-width itself.

That is still publishable if analyzed carefully, but the framing must change.

## Metrics

- teacher standalone accuracy
- teacher/BF16 KL
- student final accuracy
- VRAM
- throughput
- quantization time
- calibration sensitivity

## Decision

At least two quantization methods should support the main trend before making broad claims about "teacher quantization."

---

# Phase 6 — Seed and Calibration Robustness

## Goal

Establish statistical reliability.

## Models Needed

Only rerun the most important configurations.

Do not repeat the entire grid.

Recommended:

- best BF16 teacher
- best INT8 teacher
- best INT4 teacher
- key equal-memory comparison

## Dataset Needed

Repeat using:

- multiple OPD seeds;
- multiple GPTQ calibration subsets if calibration affects the quantizer.

## Evals Needed

Same evaluation suite.

## Metrics

Report:

- mean
- standard deviation
- confidence intervals

For pairwise comparisons:

- bootstrap confidence interval
- paired evaluation where appropriate

## What to Look For

The main ordering should survive:

- OPD initialization noise;
- data order;
- calibration set selection.

## Decision

If the core ordering changes frequently across seeds, do not claim a universal frontier.

Instead report instability and investigate why.

---

# Phase 7 — Second Student Scale

## Goal

Determine whether the optimal teacher size–precision trade-off depends on the student.

This turns the project from a single-case efficiency study into a more general scientific result.

## Models Needed

Students:

- Qwen3-1.7B BF16
- Qwen3-4B BF16

Teachers:

Use only a subset based on earlier results.

Example:

- 8B BF16
- 8B INT8
- 14B BF16
- 14B INT8
- 14B INT4

## Dataset Needed

Same training dataset family.

Use a consistent token budget or explicitly study training-budget scaling.

## Evals Needed

Same evaluation suite.

## Main Question

Does the best teacher precision depend on the student size?

For example:

- 1.7B student may benefit most from 8B INT8;
- 4B student may benefit most from 14B INT8;
- the amount of acceptable teacher quantization may change with student capacity.

## Metrics

All previous quality and efficiency metrics.

Additionally:

- teacher/student parameter ratio
- teacher/student KL
- teacher/student support overlap

## What to Look For

A systematic interaction:

`best_teacher = f(student_size, teacher_size, teacher_precision)`

This is much stronger than a one-student observation.

## Decision

If the result generalizes, consider fitting a simple teacher-selection rule.

---

# Phase 8 — Off-Policy KD Baseline

## Goal

Determine whether the teacher size–precision trade-off is specific to OPD or a general property of distillation.

## Models Needed

Use only representative teacher configurations.

Example:

- 4B BF16
- 14B BF16
- 14B INT8
- 14B INT4

Student:

- 1.7B BF16

## Dataset Needed

Use the same source problems.

For off-policy KD:

- teacher-generated trajectories or fixed reference trajectories.

For OPD:

- student-generated trajectories.

## Evals Needed

Same main evaluation suite.

## What to Look For

### Same frontier in KD and OPD

Interpretation:

- contribution is broader:
  "efficient teacher precision scaling for LLM distillation."

### Stronger low-bit advantage in OPD

Interpretation:

- teacher quantization is especially important when the teacher must remain online inside the training loop.

This strengthens the OPD-specific story.

## Metrics

- student accuracy
- teacher calls
- teacher inference cost
- total training cost
- teacher memory
- final performance

---

# Phase 9 — Scaling Rule / Predictive Model

## Goal

Move from an empirical grid to a compact rule for selecting the teacher under resource constraints.

Only do this if the earlier phases show a stable pattern.

## Inputs

Potential variables:

- teacher parameter count
- teacher precision
- teacher memory
- teacher standalone accuracy
- student parameter count
- teacher/student KL
- teacher/student support overlap

## Target

Predict:

- final student accuracy
- gain from OPD
- optimal teacher configuration under a memory budget

Possible model:

`StudentQuality = f(log TeacherParams, TeacherBits, log StudentParams, MemoryBudget)`

## What to Look For

A useful rule such as:

> Under a fixed teacher-memory budget, increasing teacher parameter count and reducing precision is beneficial until teacher fidelity drops below a task-dependent threshold.

Or:

> INT8 consistently dominates BF16 teacher configurations at equal memory, while 4-bit becomes favorable only above a certain teacher size.

## Metrics

- prediction error
- held-out configuration performance
- rank correlation
- Pareto frontier prediction accuracy

## Decision

Only claim a scaling law if it predicts configurations not used to fit it.

Otherwise call it an empirical trend.

---

# Phase 10 — Final Paper Package

## Core Research Question

> At a fixed teacher-serving budget, what teacher size and precision should be used for on-policy distillation into a full-precision student?

## Main Claims to Aim For

### Claim 1 — Efficiency

Teacher quantization substantially reduces OPD memory and/or serving cost.

### Claim 2 — Pareto Frontier

Larger low-precision teachers can outperform smaller high-precision teachers at equal teacher-memory budget.

### Claim 3 — Interaction

Teacher size and precision interact rather than contributing independently.

### Claim 4 — Robustness

The effect persists across:

- multiple teacher sizes;
- at least two quantizers;
- multiple seeds;
- ideally multiple student sizes.

### Optional Claim 5 — Mechanism

Teacher quantization changes supervision in a way that cannot be explained solely by global temperature scaling.

Only make this claim if the ablations support it.

---

# Minimum Publishable Experiment Set

If compute is limited, prioritize:

1. Student: Qwen3-1.7B BF16.
2. Teachers:
   - 4B BF16
   - 4B 8-bit
   - 4B 4-bit
   - 8B BF16
   - 8B 8-bit
   - 8B 4-bit
   - 14B BF16
   - 14B 8-bit
   - 14B 4-bit
3. OpenR1-Math prompts for OPD.
4. MATH-500 + GSM8K evaluation.
5. Actual VRAM, throughput, wall-clock, and GPU-hour measurement.
6. One second quantizer for the strongest 4-bit comparison.
7. Three seeds for the critical equal-memory comparison.
8. Temperature ablation only for the most interesting teacher size.

---

# Kill Criteria

Stop or reframe the project if:

- quantized teachers consistently produce much worse students;
- larger low-bit teachers never outperform smaller BF16 teachers under any comparable resource budget;
- the apparent gain disappears under multiple seeds;
- the result exists only for one quantizer/calibration set;
- teacher memory decreases but actual OPD wall-clock cost becomes much worse;
- the only benefit is a trivial memory reduction already fully expected from existing quantization work.

Possible reframe if the main hypothesis fails:

> Characterize the minimum teacher precision required for stable OPD as a function of teacher size and task difficulty.

A negative threshold result can still be useful if it is systematic and predictive.

---

# Success Criteria

## Pilot Success

At least one quantized teacher:

- retains most teacher quality;
- saves substantial memory;
- produces a student close to or better than the BF16-teacher condition.

## Strong Project Success

At equal teacher memory:

- a larger quantized teacher produces a better BF16 student than a smaller BF16 teacher.

## Main-Conference-Level Success

The result:

- generalizes across teacher sizes;
- survives seeds;
- survives a second quantizer;
- shows a clear size × precision interaction;
- produces a useful teacher-selection Pareto frontier or predictive rule.

---

# Recommended Immediate Order of Work

## First

Implement teacher quantization and Phase 1 diagnostics.

## Second

Run the six-condition minimal OPD pilot:

- 4B × {BF16, 8-bit, 4-bit}
- 14B × {BF16, 8-bit, 4-bit}

## Third

If the pilot is positive, add 8B and construct the equal-memory frontier.

## Fourth

Run the critical robustness checks:

- multiple seeds;
- second quantizer;
- temperature ablation.

## Fifth

Only after the central result is stable:

- add second student size;
- compare OPD against off-policy KD;
- attempt a scaling rule.

---

# Guiding Principle

Keep the project centered on one simple question:

> **What is the best teacher you can afford for OPD?**

Do not add student quantization, SFT, RL, token weighting, adaptive bit-width, or other mechanisms until the core teacher size–precision effect has been established.

Every additional experiment should either:

1. strengthen the efficiency frontier;
2. explain why it happens;
3. test whether it generalizes.
