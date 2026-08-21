"""Answer grading. Math-Verify does the equivalence checking; this module only feeds it.

Two things here are load-bearing and were established by probing Math-Verify 0.9.0 directly:

1. Gold answers are wrapped in ``\\boxed{}`` before parsing. MATH-500 stores bare LaTeX, and
   ``parse("x+1")`` / ``parse("\\text{even}")`` return *empty* -- the answer would be silently
   counted as unparseable gold. Wrapping also routes gold and prediction through the same
   extraction path, since the prompt asks the model for ``\\boxed{}``.
2. Math-Verify's timeout uses signals on Linux and multiprocessing elsewhere, and the latter is
   unusable on Windows. Production runs on Linux and keeps the timeout; other platforms disable it
   so the CPU tests can run.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any

from math_verify import parse, verify

# A pathological prediction can send sympy into a very long simplification. On Linux this is
# bounded; see the module docstring for why other platforms cannot be.
PARSE_TIMEOUT_SECONDS: int | None = 5 if sys.platform == "linux" else None
VERIFY_TIMEOUT_SECONDS: int | None = 5 if sys.platform == "linux" else None

GSM8K_ANSWER = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


@dataclass(frozen=True)
class Grade:
    correct: bool
    gold_parseable: bool
    prediction_parseable: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "gold_parseable": self.gold_parseable,
            "prediction_parseable": self.prediction_parseable,
            "error": self.error,
        }


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K stores a full worked solution ending in ``#### <answer>``."""
    match = GSM8K_ANSWER.search(answer_field)
    if match is None:
        raise ValueError(f"No '####' answer marker in GSM8K answer: {answer_field!r}")
    return match.group(1).replace(",", "").replace("$", "").strip()


def extract_gold(benchmark: str, answer_field: str) -> str:
    if benchmark == "gsm8k":
        return gsm8k_gold(answer_field)
    return answer_field.strip()


def grade(gold: str, prediction: str) -> Grade:
    """Grade one prediction against one gold answer.

    ``correct`` is always a bool: an unparseable prediction is a wrong answer, not a missing
    measurement. ``prediction_parseable`` is reported separately so a model that reasons well but
    formats badly is distinguishable from one that is simply wrong.
    """
    gold_parsed = parse(r"\boxed{" + gold.strip() + "}", parsing_timeout=PARSE_TIMEOUT_SECONDS)
    if not gold_parsed:
        # A gold answer we cannot parse is a dataset/harness problem, not a model failure.
        return Grade(correct=False, gold_parseable=False, prediction_parseable=False)

    prediction_parsed = parse(prediction, parsing_timeout=PARSE_TIMEOUT_SECONDS)
    if not prediction_parsed:
        return Grade(correct=False, gold_parseable=True, prediction_parseable=False)

    try:
        correct = bool(verify(gold_parsed, prediction_parsed, timeout_seconds=VERIFY_TIMEOUT_SECONDS))
    except Exception as error:  # noqa: BLE001 - sympy raises a wide variety of exceptions
        return Grade(
            correct=False,
            gold_parseable=True,
            prediction_parseable=True,
            error=f"{type(error).__name__}: {error}",
        )
    return Grade(correct=correct, gold_parseable=True, prediction_parseable=True)


def summarize(grades: list[Grade]) -> dict[str, Any]:
    total = len(grades)
    if total == 0:
        raise ValueError("Cannot summarize an empty list of grades")
    correct = sum(1 for item in grades if item.correct)
    parseable = [item for item in grades if item.prediction_parseable]
    unparseable_gold = sum(1 for item in grades if not item.gold_parseable)
    return {
        "count": total,
        "correct": correct,
        "accuracy": correct / total,
        # Separates "cannot do the maths" from "cannot follow the output format".
        "accuracy_on_parseable": (
            sum(1 for item in parseable if item.correct) / len(parseable) if parseable else None
        ),
        "prediction_parse_failure_rate": 1.0 - len(parseable) / total,
        "unparseable_gold": unparseable_gold,
        "verification_errors": sum(1 for item in grades if item.error is not None),
    }
