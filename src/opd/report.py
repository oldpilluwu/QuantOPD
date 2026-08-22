"""`opd-report` -- collect every report into one JSON and one CSV table.

The headline question is whether the student improved, and whether the larger INT4 teacher matched
the smaller BF16 teacher at comparable teacher memory. This walks the artifact tree rather than
being told what ran, so a partial pilot still produces a readable table.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

from .common import atomic_write_json
from .config import DEFAULT_CONFIG, ExperimentConfig, load_config

EVAL_COLUMNS = ["accuracy", "accuracy_on_parseable", "prediction_parse_failure_rate"]


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_evaluations(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(config.eval.output_dir.glob("*/*/report.json")):
        report = _read(report_path)
        if report is None:
            continue
        rows.append(
            {
                "stage": "eval",
                "tag": report["tag"],
                "model": report["model"]["model"],
                "precision": report["model"]["precision"],
                "weights_source": report["model"]["weights_source"],
                "backend": report["model"]["backend"],
                "benchmark": report["benchmark"]["name"],
                "items": report["benchmark"]["items"],
                **{name: report["accuracy"][name] for name in EVAL_COLUMNS},
                "mean_completion_tokens": report["generation"]["mean_completion_tokens"],
                "truncation_rate": report["generation"]["truncation_rate"],
                "teacher_memory_gib": report["model"]["footprint_gib"],
                "peak_vram_gib": report["peak_vram_gib"],
                "tokens_per_second": report["generation"]["completion_tokens_per_second"],
            }
        )
    return rows


def _mean(aggregates: dict[str, Any], metric: str, weighting: str = "token_weighted") -> float | None:
    entry = aggregates.get(metric)
    return None if entry is None else entry[weighting]["mean"]


def collect_scoring(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(config.scoring.output_dir.glob("*/*/report.json")):
        report = _read(report_path)
        if report is None:
            continue

        aggregates = report["aggregates"]
        rows.append(
            {
                "stage": "scoring",
                "tag": report["tag"],
                "model": report["teacher"]["model"],
                "precision": report["teacher"]["precision"],
                "teacher_memory_gib": report["teacher"]["footprint_gib"],
                "trajectories": report["trajectories"]["count"],
                "tokens": report["trajectories"]["tokens"],
                "reverse_kl": _mean(aggregates, "reverse_kl"),
                "reverse_kl_prompt_weighted": _mean(aggregates, "reverse_kl", "prompt_weighted"),
                "forward_kl": _mean(aggregates, "forward_kl"),
                "top_1_agreement": _mean(aggregates, "top_1_agreement"),
                "entropy_shift": _mean(aggregates, "entropy_shift"),
                "teacher_logprob_of_sampled": _mean(aggregates, "teacher_logprob_of_sampled"),
                "top_1_approx_reverse_kl": _mean(aggregates, "top_1_approx_reverse_kl"),
                "mean_first_divergence": report["first_divergence"]["mean_first_divergence_position"],
                "peak_vram_gib": report["efficiency"]["cuda_peak_during_paired_scoring_gib"],
            }
        )
    return rows


def collect_training(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(config.opd.output_dir.glob("*/training-report.json")):
        report = _read(report_path)
        if report is None:
            continue
        rows.append(
            {
                "stage": "training",
                "tag": "opd",
                "model": report["teacher"]["model"],
                "precision": report["teacher"]["precision"],
                "teacher_memory_gib": report["teacher"]["footprint_gib"],
                "max_steps": report["budget"]["max_steps"],
                "prompts_consumed": report["budget"]["prompts_consumed"],
                "training_loss": report["result"]["training_loss"],
                "first_logged_loss": report["result"]["first_logged_loss"],
                "last_logged_loss": report["result"]["last_logged_loss"],
                "runtime_seconds": report["result"]["runtime_seconds"],
                "gpu_hours": (report["result"]["runtime_seconds"] or 0) / 3600,
                "peak_vram_gib": report["result"]["cuda_peak_allocated_gib"],
                "checkpoint": report["checkpoint"],
                "all_checks_passed": all(report["checks"].values()),
            }
        )
    return rows


def find_item_count_mismatches(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag benchmarks whose conditions were scored on different numbers of items.

    `--limit N` takes the *first* N frozen indices, so a run at 100 and a run at 300 cover
    different problems. Comparing them looks like a result and is an artifact -- the kind of
    mistake that reads as "the 1.7B student beat the 14B teacher".
    """
    by_benchmark: dict[str, dict[int, list[str]]] = {}
    for row in evaluations:
        by_benchmark.setdefault(row["benchmark"], {}).setdefault(row["items"], []).append(
            f"{row['model']}:{row['precision']}:{row['tag']}"
        )
    return [
        {
            "benchmark": benchmark,
            "item_counts": {count: sorted(names) for count, names in sorted(counts.items())},
        }
        for benchmark, counts in sorted(by_benchmark.items())
        if len(counts) > 1
    ]


def _mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact binomial p-value over the discordant pairs."""
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(comb(discordant, i) for i in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def paired_comparisons(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Compare every pair of conditions on the items they both saw.

    Conditions are scored on the same frozen indices, so the comparison is paired and McNemar's
    exact test on the discordant items is far more powerful than asking whether two marginal
    confidence intervals overlap. At n=300 with ~10-point effects that difference decides whether
    a real result is visible at all -- and "did the OPD student beat the baseline student" is
    precisely this comparison.
    """
    per_benchmark: dict[str, dict[str, dict[int, bool]]] = {}
    for path in sorted(config.eval.output_dir.glob("*/*/per-item.json")):
        benchmark = path.parent.name
        condition = path.parent.parent.name
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        per_benchmark.setdefault(benchmark, {})[condition] = {
            int(row["item_index"]): bool(row["correct"]) for row in rows
        }

    results: list[dict[str, Any]] = []
    for benchmark, conditions in sorted(per_benchmark.items()):
        for left, right in combinations(sorted(conditions), 2):
            shared = sorted(set(conditions[left]) & set(conditions[right]))
            if not shared:
                continue
            only_left = sum(1 for i in shared if conditions[left][i] and not conditions[right][i])
            only_right = sum(1 for i in shared if conditions[right][i] and not conditions[left][i])
            p_value = _mcnemar_exact(only_left, only_right)
            accuracy_left = sum(conditions[left][i] for i in shared) / len(shared)
            accuracy_right = sum(conditions[right][i] for i in shared) / len(shared)
            results.append(
                {
                    "benchmark": benchmark,
                    "a": left,
                    "b": right,
                    "paired_items": len(shared),
                    "accuracy_a": accuracy_left,
                    "accuracy_b": accuracy_right,
                    "difference": accuracy_left - accuracy_right,
                    "only_a_correct": only_left,
                    "only_b_correct": only_right,
                    "mcnemar_p": p_value,
                    "significant_at_05": p_value < 0.05,
                }
            )
    return results


def build_headline(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (condition, benchmark): the table the pilot exists to produce."""
    headline: dict[tuple[str, str], dict[str, Any]] = {}
    for row in evaluations:
        key = (row["tag"], row["benchmark"])
        headline[key] = {
            "condition": row["tag"],
            "benchmark": row["benchmark"],
            "model": row["model"],
            "precision": row["precision"],
            "accuracy": row["accuracy"],
            "truncation_rate": row["truncation_rate"],
            "teacher_memory_gib": row["teacher_memory_gib"],
        }
    return [headline[key] for key in sorted(headline)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join every stage's reports into one summary table.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or config.report.output_dir

    evaluations = collect_evaluations(config)
    scoring = collect_scoring(config)
    training = collect_training(config)

    mismatches = find_item_count_mismatches(evaluations)
    comparisons = paired_comparisons(config)
    summary = {
        "schema_version": 1,
        "comparability_warnings": mismatches,
        "paired_comparisons": comparisons,
        "headline": build_headline(evaluations),
        "evaluations": evaluations,
        "scoring": scoring,
        "training": training,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "evaluations.csv", evaluations)
    write_csv(output_dir / "scoring.csv", scoring)
    write_csv(output_dir / "training.csv", training)
    write_csv(output_dir / "headline.csv", summary["headline"])
    write_csv(output_dir / "paired-comparisons.csv", comparisons)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "evaluations": len(evaluations),
                "scoring": len(scoring),
                "training": len(training),
            "paired_comparisons": len(comparisons),
            },
            indent=2,
        )
    )
    for row in summary["headline"]:
        accuracy = row["accuracy"]
        print(f"  {row['condition']:<28} {row['benchmark']:<8} accuracy={accuracy:.4f}")

    for mismatch in mismatches:
        print("")
        print(
            f"WARNING: {mismatch['benchmark']} conditions were scored on different item counts; "
            "these accuracies are NOT comparable. --limit N takes the first N frozen indices, so "
            "different N means different problems. Re-run the short ones at the full size."
        )
        for count, names in mismatch["item_counts"].items():
            print(f"    {count:>5} items: {', '.join(names)}")


if __name__ == "__main__":
    main()
