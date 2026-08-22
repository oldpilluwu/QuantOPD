"""`opd-plot` -- figures from the evaluation reports.

Reads the same artifacts `opd-report` does, so figures regenerate as runs land rather than being
hand-maintained. Four figures, each answering one question:

1. ``omnimath-accuracy``   -- the headline: does anything beat the student, and by how much?
2. ``accuracy-vs-memory``  -- PLAN.md's core plot: quality against teacher memory budget.
3. ``benchmark-saturation``-- why MATH-500 was replaced by Omni-MATH.
4. ``difficulty-profile``  -- where the teacher advantage lives, and where truncation eats it.

Design notes, since charts are read by people:

- Confidence intervals are drawn on every accuracy mark. The whole story here is that most of
  these differences are inside the noise, and a bare bar chart would hide exactly that.
- Dot-plus-interval rather than bars: a bar implies the area between zero and the value is
  meaningful, and invites truncating the axis to exaggerate small differences.
- Emphasis colouring (one accent, the rest recessive) rather than a categorical hue per model.
  The student is the reference every other mark is read against; giving five models five hues
  would bury that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json
from .config import DEFAULT_CONFIG, ExperimentConfig, load_config

# Palette roles. Light surface only: these are figures for a paper, not a themed web page.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"  # categorical slot 1, the emphasised mark
RECESSIVE = "#898781"  # de-emphasis gray for context marks
# Slots 1-8 in the documented fixed order. Hues are assigned by position and NEVER cycled: a
# repeated hue would silently give two models the same identity, which is worse than failing.
SERIES = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)


def series_colour(index: int) -> str:
    if index >= len(SERIES):
        raise ValueError(
            f"{index + 1} series exceeds the {len(SERIES)}-hue palette. Fold the tail into "
            "'Other' or facet into small multiples rather than reusing a hue."
        )
    return SERIES[index]

# Teacher weight footprints in GiB. bitsandbytes numbers are measured (Phase 0); the pre-quantized
# checkpoints are their published safetensors size.
TEACHER_MEMORY_GIB = {
    ("Qwen/Qwen3-4B", "bf16"): 7.49,
    ("Qwen/Qwen3-14B", "int4"): 9.05,
    ("Qwen/Qwen3-14B-AWQ", "awq"): 9.29,
    ("JunHowie/Qwen3-14B-GPTQ-Int4", "gptq"): 9.30,
    ("Qwen/Qwen3-14B", "bf16"): 27.51,
    ("Qwen/Qwen3-8B", "bf16"): 15.26,
}

DIFFICULTY_BANDS = (
    ("easy\n(≤3.5)", lambda d: d is not None and d <= 3.5),
    ("mid\n(4–5.5)", lambda d: d is not None and 3.5 < d <= 5.5),
    ("hard\n(6–7)", lambda d: d is not None and 5.5 < d <= 7.0),
    ("very hard\n(7.5+)", lambda d: d is not None and d > 7.0),
)


def short_name(model: str, precision: str) -> str:
    base = model.split("/")[-1].replace("Qwen3-", "").replace("-AWQ", "").replace("-GPTQ-Int4", "")
    label = {"bf16": "BF16", "int4": "NF4", "awq": "AWQ", "gptq": "GPTQ", "int8": "INT8"}[precision]
    return f"{base} {label}"


def load_conditions(config: ExperimentConfig, benchmark: str) -> list[dict[str, Any]]:
    """Every completed evaluation for one benchmark, largest item count only.

    Runs at different `--limit` cover different problems, so mixing them would compare models on
    different questions. Keeping only the largest n is the honest default.
    """
    rows: list[dict[str, Any]] = []
    for report_path in sorted(config.eval.output_dir.glob("*/*/report.json")):
        if report_path.parent.name != benchmark:
            continue
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        model, precision = report["model"]["model"], report["model"]["precision"]
        interval = report["accuracy"].get("accuracy_ci95")
        rows.append(
            {
                "slug": report_path.parent.parent.name,
                "label": short_name(model, precision),
                "model": model,
                "precision": precision,
                "tag": report["tag"],
                "is_student": model == config.models.student,
                "items": report["benchmark"]["items"],
                "accuracy": report["accuracy"]["accuracy"],
                "lower": (interval or {}).get("lower"),
                "upper": (interval or {}).get("upper"),
                "truncation": report["generation"]["truncation_rate"],
                "memory": TEACHER_MEMORY_GIB.get((model, precision)),
                "per_item": report_path.parent / "per-item.json",
            }
        )
    if not rows:
        return []
    widest = max(row["items"] for row in rows)
    return [row for row in rows if row["items"] == widest]


def _style_axes(axes, value_axis: str = "x") -> None:
    """Recessive chrome: hairline grid on the value axis only, no box around the plot."""
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    axes.grid(axis=value_axis, color=GRIDLINE, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)


def _title(axes, title: str, subtitle: str | None = None) -> None:
    axes.set_title(
        title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left",
        pad=26 if subtitle else 10,
    )
    if subtitle:
        axes.text(
            0.0, 1.015, subtitle, transform=axes.transAxes,
            color=INK_SECONDARY, fontsize=9, va="bottom", ha="left",
        )


def figure_accuracy(plt, rows: list[dict[str, Any]], benchmark: str, out: Path) -> Path | None:
    """Dot-and-interval, sorted. The interval is the point: most gaps here are inside it."""
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["accuracy"])
    figure, axes = plt.subplots(figsize=(7.4, 0.46 * len(rows) + 1.9))
    figure.patch.set_facecolor(SURFACE)
    _style_axes(axes, value_axis="x")

    for index, row in enumerate(rows):
        colour = ACCENT if row["is_student"] else RECESSIVE
        if row["lower"] is not None:
            axes.plot(
                [row["lower"], row["upper"]],
                [index, index],
                color=colour,
                linewidth=2.0,
                solid_capstyle="round",
                alpha=0.55,
                zorder=2,
            )
        axes.plot(
            row["accuracy"], index, "o", markersize=9, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3,
        )
        axes.text(
            row["upper"] + 0.012 if row["upper"] is not None else row["accuracy"] + 0.012,
            index,
            f"{row['accuracy']:.3f}",
            va="center", ha="left", fontsize=9,
            color=INK_PRIMARY if row["is_student"] else INK_SECONDARY,
        )

    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels(
        [r["label"] + ("  (student)" if r["is_student"] else "") for r in rows],
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    axes.set_ylim(-0.6, len(rows) - 0.4)
    axes.set_xlim(0, max(r["upper"] or r["accuracy"] for r in rows) + 0.085)
    axes.set_xlabel("accuracy (greedy pass@1)", color=INK_SECONDARY, fontsize=9.5)
    _title(
        axes,
        f"{benchmark}: accuracy with 95% confidence interval",
        f"n = {rows[0]['items']} items, identical for every condition · bars are Wilson intervals",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return out


def figure_accuracy_vs_memory(plt, rows: list[dict[str, Any]], benchmark: str, out: Path) -> Path | None:
    """PLAN.md's core plot: student quality against the teacher-memory budget.

    The x axis is **categorical, ordered by memory**, with the actual GiB printed under each tick,
    rather than a continuous memory axis. Three of these teachers sit inside a 1.24x memory range
    while the BF16 point is 3x further out; on a continuous axis -- linear or log -- their labels
    are wider than the gap between them and collide unreadably. With a handful of discrete
    configurations there is nothing to interpolate, so even spacing costs no information and the
    near-equal budgets are stated in the tick labels and the subtitle instead.
    """
    teachers = [r for r in rows if not r["is_student"] and r["memory"] is not None]
    students = [r for r in rows if r["is_student"]]
    if not teachers:
        return None
    teachers = sorted(teachers, key=lambda r: r["memory"])

    figure, axes = plt.subplots(figsize=(8.0, 4.9))
    figure.patch.set_facecolor(SURFACE)
    _style_axes(axes, value_axis="y")

    # Shade the configurations that share a memory budget: that comparison is the study's point.
    budget = [i for i, r in enumerate(teachers) if r["memory"] <= teachers[0]["memory"] * 1.35]
    if len(budget) > 1:
        axes.axvspan(
            min(budget) - 0.42, max(budget) + 0.42,
            color=INK_MUTED, alpha=0.07, zorder=0,
        )
        axes.text(
            (min(budget) + max(budget)) / 2, 0.985,
            f"≈ equal teacher memory ({teachers[budget[0]]['memory']:.1f}–{teachers[budget[-1]]['memory']:.1f} GiB)",
            transform=axes.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color=INK_MUTED,
        )

    if students:
        student = students[0]
        if student["lower"] is not None:
            axes.axhspan(student["lower"], student["upper"], color=ACCENT, alpha=0.10, zorder=1)
        axes.axhline(student["accuracy"], color=ACCENT, linewidth=2.0, zorder=2)
        axes.text(
            0.012, student["accuracy"] + 0.004, f"student {student['accuracy']:.3f}",
            transform=axes.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=9, color=ACCENT, fontweight="bold",
        )

    for index, row in enumerate(teachers):
        if row["lower"] is not None:
            axes.plot(
                [index, index], [row["lower"], row["upper"]],
                color=RECESSIVE, linewidth=2.0, alpha=0.55, solid_capstyle="round", zorder=3,
            )
        axes.plot(
            index, row["accuracy"], "o", markersize=9, color=RECESSIVE,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4,
        )
        axes.annotate(
            f"{row['accuracy']:.3f}",
            (index, row["upper"] or row["accuracy"]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=9, color=INK_SECONDARY, zorder=5,
        )

    axes.set_xticks(range(len(teachers)))
    axes.set_xticklabels(
        [f"{r['label']}\n{r['memory']:.2f} GiB" for r in teachers],
        fontsize=9, color=INK_SECONDARY,
    )
    axes.set_xlim(-0.6, len(teachers) - 0.4)
    axes.set_ylabel("accuracy (greedy pass@1)", color=INK_SECONDARY, fontsize=9.5)

    floor_candidates = [r["lower"] or r["accuracy"] for r in teachers]
    floor_candidates += [r["lower"] or r["accuracy"] for r in students]
    top = max(r["upper"] or r["accuracy"] for r in teachers)
    axes.set_ylim(max(0.0, min(floor_candidates) - 0.05), top + 0.075)
    _title(
        axes,
        f"{benchmark}: teacher quality against teacher memory",
        "teachers ordered by memory · shaded band is the student's 95% interval",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return out


def figure_saturation(plt, per_benchmark: dict[str, list[dict[str, Any]]], out: Path) -> Path | None:
    """Small multiples, independent axes.

    The two benchmarks were run at different item counts and token budgets, so a shared axis would
    invite a comparison that is not valid. What transfers is the *spread* within each panel.
    """
    names = [b for b in ("math500", "omnimath") if per_benchmark.get(b)]
    if len(names) < 2:
        return None

    figure, axes_pair = plt.subplots(1, 2, figsize=(10.6, 3.9))
    figure.patch.set_facecolor(SURFACE)

    for axes, benchmark in zip(axes_pair, names, strict=True):
        rows = sorted(per_benchmark[benchmark], key=lambda r: r["accuracy"])
        _style_axes(axes, value_axis="x")
        for index, row in enumerate(rows):
            colour = ACCENT if row["is_student"] else RECESSIVE
            if row["lower"] is not None:
                axes.plot(
                    [row["lower"], row["upper"]], [index, index],
                    color=colour, linewidth=2.0, alpha=0.55, solid_capstyle="round", zorder=2,
                )
            axes.plot(
                row["accuracy"], index, "o", markersize=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3,
            )
            anchor = row["upper"] if row["upper"] is not None else row["accuracy"]
            axes.text(
                anchor + 0.022, index, f"{row['accuracy']:.2f}",
                va="center", ha="left", fontsize=8.5, color=INK_SECONDARY,
            )
        spread = max(r["accuracy"] for r in rows) - min(r["accuracy"] for r in rows)
        teachers = [r for r in rows if not r["is_student"]]
        teacher_spread = (
            max(r["accuracy"] for r in teachers) - min(r["accuracy"] for r in teachers) if teachers else 0.0
        )
        axes.set_yticks(range(len(rows)))
        axes.set_yticklabels([r["label"] for r in rows], fontsize=9, color=INK_SECONDARY)
        axes.set_ylim(-0.6, len(rows) - 0.4)
        axes.set_xlim(0, 1.0)
        axes.set_xlabel("accuracy", color=INK_SECONDARY, fontsize=9)
        _title(
            axes,
            benchmark,
            f"n={rows[0]['items']} · teacher spread {teacher_spread:.3f} · student-to-best {spread:.3f}",
        )

    figure.suptitle(
        "MATH-500 is saturated; Omni-MATH separates the student from the teachers",
        color=INK_PRIMARY, fontsize=12, fontweight="bold", x=0.008, ha="left", y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return out


def figure_difficulty(plt, rows: list[dict[str, Any]], out: Path) -> Path | None:
    """Accuracy and truncation by difficulty band, stacked panels sharing an x axis.

    Two panels rather than two y-scales on one plot: a dual axis would let the reader infer a
    relationship from crossing lines that is an artefact of arbitrary scaling.
    """
    usable = [r for r in rows if r["per_item"].exists()]
    if not usable:
        return None

    difficulty: dict[int, float | None] = {}
    per_condition: dict[str, list[dict[str, Any]]] = {}
    for row in usable:
        with row["per_item"].open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        per_condition[row["label"]] = items
        for item in items:
            difficulty.setdefault(int(item["item_index"]), item.get("difficulty"))
    if not any(v is not None for v in difficulty.values()):
        return None

    ordered = sorted(per_condition, key=lambda label: "student" not in label.lower())
    band_names = [name for name, _ in DIFFICULTY_BANDS]
    counts = [sum(1 for d in difficulty.values() if test(d)) for _, test in DIFFICULTY_BANDS]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.2, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.55, 1.0]}
    )
    figure.patch.set_facecolor(SURFACE)
    for axes in (top, bottom):
        _style_axes(axes, value_axis="y")

    for index, label in enumerate(ordered):
        items = per_condition[label]
        correct_by_index = {int(i["item_index"]): bool(i["correct"]) for i in items}
        length_by_index = {int(i["item_index"]): i["finish_reason"] == "length" for i in items}
        accuracies, truncations = [], []
        for _, test in DIFFICULTY_BANDS:
            ids = [i for i, d in difficulty.items() if test(d) and i in correct_by_index]
            accuracies.append(sum(correct_by_index[i] for i in ids) / len(ids) if ids else None)
            truncations.append(sum(length_by_index[i] for i in ids) / len(ids) if ids else None)
        colour = series_colour(index)
        width = 2.6 if index == 0 else 1.7
        for axes, values in ((top, accuracies), (bottom, truncations)):
            axes.plot(
                range(len(band_names)), values, marker="o", markersize=7,
                color=colour, linewidth=width,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=label, zorder=3,
            )

    top.set_ylabel("accuracy", color=INK_SECONDARY, fontsize=9.5)
    top.set_ylim(0, None)
    top.legend(
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncols=2,
        loc="upper right", handlelength=1.6,
    )
    _title(
        top,
        "Where the teacher advantage lives",
        "above difficulty 6 every model collapses together and truncation takes over",
    )

    bottom.set_ylabel("truncation rate", color=INK_SECONDARY, fontsize=9.5)
    bottom.set_ylim(0, 1.0)
    bottom.set_xticks(range(len(band_names)))
    bottom.set_xticklabels(
        [f"{name}\nn={count}" for name, count in zip(band_names, counts, strict=True)],
        fontsize=9, color=INK_SECONDARY,
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark results from the evaluation reports.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark", default="omnimath", help="Benchmark for the per-benchmark figures.")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - depends on the local environment
        raise SystemExit("matplotlib is not installed. Run: uv sync --group viz") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = args.output_dir or (config.report.output_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_benchmark = {
        name: load_conditions(config, name)
        for name in (item.name for item in config.evaluation)
    }
    rows = per_benchmark.get(args.benchmark, [])

    written = [
        figure_accuracy(plt, rows, args.benchmark, output_dir / f"{args.benchmark}-accuracy.png"),
        figure_accuracy_vs_memory(plt, rows, args.benchmark, output_dir / f"{args.benchmark}-accuracy-vs-memory.png"),
        figure_saturation(plt, per_benchmark, output_dir / "benchmark-saturation.png"),
        figure_difficulty(plt, rows, output_dir / f"{args.benchmark}-difficulty-profile.png"),
    ]
    written = [path for path in written if path is not None]

    manifest = {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "conditions": [
            {k: v for k, v in row.items() if k != "per_item"} for row in rows
        ],
        "figures": [str(path) for path in written],
    }
    atomic_write_json(output_dir / "figures.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "figures": [p.name for p in written]}, indent=2))


if __name__ == "__main__":
    main()
