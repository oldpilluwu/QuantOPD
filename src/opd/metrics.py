"""Exact distribution metrics between a student and a teacher on student-generated tokens.

Three things here are easy to get wrong and are therefore stated explicitly:

**Alignment.** For a sequence ``prompt (P tokens) + completion (C tokens)``, the logits at index
``i`` predict token ``i+1``. The distribution that produced completion token ``j`` therefore lives
at index ``P-1+j``. Everything downstream is meaningless if this is off by one, so
:func:`completion_logits` is the single place it is computed and it is pinned by a unit test.

**Numerics.** Logits arrive in bfloat16, which carries roughly three decimal digits. Summing
151,936 terms in it would be noise, so log-softmax runs in float32 and the vocabulary sums
accumulate in float64.

**Exact, not top-k.** These are full-vocabulary divergences. The TRL training loss uses a
``loss_top_k=1`` plus tail-bucket approximation, which is computed alongside so the gap between
the diagnostic and the objective is visible rather than assumed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class PositionAccumulator:
    """Running sums over the completion positions of one trajectory."""

    tokens: int = 0
    reverse_kl: float = 0.0
    forward_kl: float = 0.0
    student_entropy: float = 0.0
    teacher_entropy: float = 0.0
    teacher_logprob_of_sampled: float = 0.0
    top_1_approx_reverse_kl: float = 0.0
    top_k_agreement: dict[int, float] = field(default_factory=dict)
    teacher_mass_on_student_top_k: dict[int, float] = field(default_factory=dict)
    first_divergence: int | None = None
    position_profile_sums: list[float] = field(default_factory=list)
    position_profile_counts: list[int] = field(default_factory=list)


def completion_logits(model: Any, input_ids: torch.Tensor, prompt_length: int, completion_length: int) -> torch.Tensor:
    """Logits for the distributions that generated each completion token.

    Returns shape ``(completion_length, vocab)``. Index ``j`` is the distribution the model would
    have used to emit completion token ``j``.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("Scoring runs one trajectory at a time to avoid padding effects")
    if input_ids.shape[1] != prompt_length + completion_length:
        raise ValueError(
            f"input_ids has {input_ids.shape[1]} tokens but prompt+completion is "
            f"{prompt_length}+{completion_length}"
        )
    if completion_length < 1:
        raise ValueError("Cannot score an empty completion")

    # logits_to_keep=C+1 keeps the last C+1 positions, i.e. absolute indices P-1 .. P+C-1. Row 0 is
    # index P-1, which predicts completion token 0; the final row predicts the token after the
    # completion and is dropped. This avoids materialising P prompt-position logits.
    kept = completion_length + 1
    try:
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=kept).logits[0]
    except TypeError:
        logits = model(input_ids=input_ids, use_cache=False).logits[0]

    if logits.shape[0] == kept:
        logits = logits[:-1, :]
    elif logits.shape[0] == prompt_length + completion_length:
        # The model accepted but ignored logits_to_keep, or the fallback path ran.
        logits = logits[prompt_length - 1 : prompt_length - 1 + completion_length, :]
    else:
        raise RuntimeError(
            f"Unexpected logit row count {logits.shape[0]}; expected {kept} or "
            f"{prompt_length + completion_length}"
        )
    if logits.shape[0] != completion_length:
        raise RuntimeError(f"Expected {completion_length} completion logit rows, got {logits.shape[0]}")
    return logits


def _top_k_agreement(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, k: int) -> torch.Tensor:
    if k == 1:
        return (student_log_probs.argmax(dim=-1) == teacher_log_probs.argmax(dim=-1)).double()
    student_top = student_log_probs.topk(k, dim=-1).indices
    teacher_top = teacher_log_probs.topk(k, dim=-1).indices
    # Set intersection per row, expressed as a broadcast equality so it stays on the GPU.
    overlap = (student_top.unsqueeze(-1) == teacher_top.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    return overlap.double() / k


def score_trajectory(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_token_ids: list[int],
    top_ks: tuple[int, ...],
    chunk_size: int,
    profile_bins: int,
) -> PositionAccumulator:
    """Reduce one trajectory's logits to scalar sums. No logits are retained."""
    completion_length = len(completion_token_ids)
    if student_logits.shape[0] != completion_length or teacher_logits.shape[0] != completion_length:
        raise ValueError("Logit rows must match the number of completion tokens")
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError(
            f"Student vocab {student_logits.shape[-1]} != teacher vocab {teacher_logits.shape[-1]}; "
            "the distributions are not comparable"
        )

    accumulator = PositionAccumulator()
    accumulator.top_k_agreement = dict.fromkeys(top_ks, 0.0)
    accumulator.teacher_mass_on_student_top_k = dict.fromkeys(top_ks, 0.0)
    accumulator.position_profile_sums = [0.0] * profile_bins
    accumulator.position_profile_counts = [0] * profile_bins

    tokens = torch.tensor(completion_token_ids, dtype=torch.long, device=student_logits.device)

    for start in range(0, completion_length, chunk_size):
        stop = min(start + chunk_size, completion_length)
        student_log_probs = F.log_softmax(student_logits[start:stop].float(), dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits[start:stop].float(), dim=-1)
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        difference = (student_log_probs - teacher_log_probs).double()
        reverse = (student_probs.double() * difference).sum(dim=-1)
        forward = (teacher_probs.double() * -difference).sum(dim=-1)
        accumulator.reverse_kl += float(reverse.sum())
        accumulator.forward_kl += float(forward.sum())
        accumulator.student_entropy += float(-(student_probs.double() * student_log_probs.double()).sum())
        accumulator.teacher_entropy += float(-(teacher_probs.double() * teacher_log_probs.double()).sum())

        chunk_tokens = tokens[start:stop]
        sampled_teacher_logprob = teacher_log_probs.gather(-1, chunk_tokens.unsqueeze(-1)).squeeze(-1)
        sampled_student_logprob = student_log_probs.gather(-1, chunk_tokens.unsqueeze(-1)).squeeze(-1)
        accumulator.teacher_logprob_of_sampled += float(sampled_teacher_logprob.double().sum())

        # TRL's objective at loss_top_k=1 with a tail bucket: a two-outcome reverse KL over
        # {sampled token, everything else}. Same tokens, so the approximation gap is measurable.
        student_point = sampled_student_logprob.double().exp().clamp(1e-12, 1 - 1e-12)
        teacher_point = sampled_teacher_logprob.double().exp().clamp(1e-12, 1 - 1e-12)
        approx = student_point * (student_point.log() - teacher_point.log()) + (1 - student_point) * (
            (1 - student_point).log() - (1 - teacher_point).log()
        )
        accumulator.top_1_approx_reverse_kl += float(approx.sum())

        for k in top_ks:
            accumulator.top_k_agreement[k] += float(_top_k_agreement(student_log_probs, teacher_log_probs, k).sum())
            student_top = student_log_probs.topk(k, dim=-1).indices
            accumulator.teacher_mass_on_student_top_k[k] += float(
                teacher_probs.double().gather(-1, student_top).sum()
            )

        teacher_argmax = teacher_log_probs.argmax(dim=-1)
        if accumulator.first_divergence is None:
            mismatches = (teacher_argmax != chunk_tokens).nonzero()
            if mismatches.numel() > 0:
                accumulator.first_divergence = start + int(mismatches[0].item())

        # Where in the completion does the disagreement live?
        for offset, value in enumerate(reverse.tolist()):
            position = start + offset
            bucket = min(profile_bins - 1, position * profile_bins // completion_length)
            accumulator.position_profile_sums[bucket] += value
            accumulator.position_profile_counts[bucket] += 1

        accumulator.tokens += stop - start

    return accumulator


def trajectory_means(accumulator: PositionAccumulator, top_ks: tuple[int, ...]) -> dict[str, float]:
    """Per-token means within one trajectory."""
    n = accumulator.tokens
    if n == 0:
        raise ValueError("Cannot average over zero tokens")
    means = {
        "reverse_kl": accumulator.reverse_kl / n,
        "forward_kl": accumulator.forward_kl / n,
        "student_entropy": accumulator.student_entropy / n,
        "teacher_entropy": accumulator.teacher_entropy / n,
        "entropy_shift": (accumulator.teacher_entropy - accumulator.student_entropy) / n,
        "teacher_logprob_of_sampled": accumulator.teacher_logprob_of_sampled / n,
        "top_1_approx_reverse_kl": accumulator.top_1_approx_reverse_kl / n,
    }
    for k in top_ks:
        means[f"top_{k}_agreement"] = accumulator.top_k_agreement[k] / n
        means[f"teacher_mass_on_student_top_{k}"] = accumulator.teacher_mass_on_student_top_k[k] / n
    return means


def _bootstrap_ci(
    values: list[float],
    weights: list[float] | None,
    samples: int,
    seed: int,
) -> dict[str, float] | None:
    """Percentile bootstrap over trajectories.

    Resampling is over *prompts*, never tokens: tokens within a trajectory are highly dependent,
    so a token-level bootstrap would report a confidence interval several times too narrow.
    """
    if len(values) < 2 or samples < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        picks = [rng.randrange(n) for _ in range(n)]
        if weights is None:
            means.append(sum(values[i] for i in picks) / n)
        else:
            total_weight = sum(weights[i] for i in picks)
            if total_weight == 0:
                continue
            means.append(sum(values[i] * weights[i] for i in picks) / total_weight)
    if not means:
        return None
    means.sort()
    lower = means[max(0, int(0.025 * len(means)) - 1)]
    upper = means[min(len(means) - 1, int(0.975 * len(means)))]
    return {"lower": lower, "upper": upper}


def aggregate(
    per_trajectory: list[dict[str, float]],
    token_counts: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Prompt-weighted and token-weighted means, each with a bootstrap CI.

    The two can disagree -- token-weighted is dominated by long completions, prompt-weighted gives
    every problem one vote -- so both are reported rather than picking one.
    """
    if not per_trajectory:
        raise ValueError("No trajectories to aggregate")

    metric_names = list(per_trajectory[0])
    total_tokens = sum(token_counts)
    weights = [float(count) for count in token_counts]
    result: dict[str, Any] = {}

    for index, name in enumerate(metric_names):
        values = [row[name] for row in per_trajectory]
        prompt_mean = sum(values) / len(values)
        token_mean = sum(v * w for v, w in zip(values, weights, strict=True)) / total_tokens
        result[name] = {
            "prompt_weighted": {
                "mean": prompt_mean,
                "ci95": _bootstrap_ci(values, None, bootstrap_samples, bootstrap_seed + index),
            },
            "token_weighted": {
                "mean": token_mean,
                "ci95": _bootstrap_ci(values, weights, bootstrap_samples, bootstrap_seed + index),
            },
        }

    return result


def merge_position_profile(accumulators: list[PositionAccumulator], bins: int) -> list[dict[str, float | None]]:
    sums = [0.0] * bins
    counts = [0] * bins
    for accumulator in accumulators:
        for bucket in range(bins):
            sums[bucket] += accumulator.position_profile_sums[bucket]
            counts[bucket] += accumulator.position_profile_counts[bucket]
    return [
        {
            "bin": bucket,
            "fraction_through_completion": (bucket + 0.5) / bins,
            "mean_reverse_kl": (sums[bucket] / counts[bucket]) if counts[bucket] else None,
            "tokens": counts[bucket],
        }
        for bucket in range(bins)
    ]


def summarize_divergence(accumulators: list[PositionAccumulator]) -> dict[str, Any]:
    positions = [a.first_divergence for a in accumulators]
    diverged = [p for p in positions if p is not None]
    return {
        "trajectories": len(positions),
        "trajectories_with_divergence": len(diverged),
        "mean_first_divergence_position": (sum(diverged) / len(diverged)) if diverged else None,
        "median_first_divergence_position": (sorted(diverged)[len(diverged) // 2] if diverged else None),
    }
