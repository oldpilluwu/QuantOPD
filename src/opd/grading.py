"""Answer grading. Math-Verify does the equivalence checking; this module only feeds it.

Two things here are load-bearing and were established by probing Math-Verify 0.9.0 directly:

1. Gold answers are wrapped in ``\\boxed{}`` before parsing. MATH-500 stores bare LaTeX, and
   ``parse("x+1")`` / ``parse("\\text{even}")`` return *empty* -- the answer would be silently
   counted as unparseable gold. Wrapping also routes gold and prediction through the same
   extraction path, since the prompt asks the model for ``\\boxed{}``.
2. Math-Verify's timeout uses signals on Linux and multiprocessing elsewhere, and the latter is
   unusable on Windows. Production runs on Linux and keeps the timeout; other platforms disable it
   so the CPU tests can run.
3. ``TimeoutException`` subclasses ``BaseException`` rather than ``Exception``, so it must be
   caught by name. Missing that turns one pathological olympiad expression into an aborted run.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from typing import Any

from math_verify import parse, verify
from math_verify.errors import TimeoutException

# A pathological prediction can send sympy into a very long simplification. On Linux this is
# bounded; see the module docstring for why other platforms cannot be. The default of 5s was too
# short for olympiad-level answers (nested radicals, large products), and Math-Verify treats a
# timeout as a *wrong answer* rather than an error, so a too-short budget silently depresses
# accuracy on exactly the hardest problems.
PARSE_TIMEOUT_SECONDS: int | None = 15 if sys.platform == "linux" else None
VERIFY_TIMEOUT_SECONDS: int | None = 15 if sys.platform == "linux" else None

GSM8K_ANSWER = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> dict[str, float] | None:
    """95% confidence interval for a proportion.

    Wilson rather than normal-approximation: it stays inside [0, 1] and behaves sensibly at the
    small sample sizes and low accuracies this project reads (n=100 at p=0.2 gives roughly +/-8
    points, which is wide enough that headline differences are easy to over-read without it).
    """
    if total == 0:
        return None
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return {"lower": max(0.0, centre - margin), "upper": min(1.0, centre + margin)}


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


def _parse_or_fail(text: str, stage: str) -> tuple[list[Any], str | None]:
    """Parse one expression, converting any failure into a message instead of an exception."""
    try:
        return parse(text, parsing_timeout=PARSE_TIMEOUT_SECONDS, raise_on_error=True), None
    except TimeoutException:
        return [], f"TimeoutException during {stage} parsing"
    except Exception as error:  # noqa: BLE001 - sympy raises a wide variety of exceptions
        return [], f"{type(error).__name__} during {stage} parsing: {error}"


def grade(gold: str, prediction: str) -> Grade:
    """Grade one prediction against one gold answer.

    Grading failures are *counted*, never raised and never silently scored as wrong. Both matter:

    - Math-Verify's ``TimeoutException`` subclasses ``BaseException``, not ``Exception``, so a bare
      ``except Exception`` does not catch it and one pathological expression aborts the whole run.
    - With ``raise_on_error=False`` the same timeout instead returns ``False``, which is
      indistinguishable from a wrong answer and biases against the hardest problems, where sympy
      has the most work to do.

    So every call sets ``raise_on_error=True`` and every failure is caught here explicitly.
    """
    gold_parsed, error = _parse_or_fail(r"\boxed{" + gold.strip() + "}", "gold")
    if error is not None or not gold_parsed:
        # Unparseable gold is a dataset or harness problem, not a model failure.
        return Grade(correct=False, gold_parseable=False, prediction_parseable=False, error=error)

    prediction_parsed, error = _parse_or_fail(prediction, "prediction")
    if error is not None or not prediction_parsed:
        return Grade(correct=False, gold_parseable=True, prediction_parseable=False, error=error)

    try:
        correct = bool(
            verify(
                gold_parsed,
                prediction_parsed,
                timeout_seconds=VERIFY_TIMEOUT_SECONDS,
                raise_on_error=True,
            )
        )
    except TimeoutException:
        return Grade(
            correct=False,
            gold_parseable=True,
            prediction_parseable=True,
            error="TimeoutException during comparison",
        )
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
        "accuracy_ci95": wilson_interval(correct, total),
        # Separates "cannot do the maths" from "cannot follow the output format".
        "accuracy_on_parseable": (
            sum(1 for item in parseable if item.correct) / len(parseable) if parseable else None
        ),
        "prediction_parse_failure_rate": 1.0 - len(parseable) / total,
        "unparseable_gold": unparseable_gold,
        # Grading failures, not model failures. A non-trivial timeout count means the score is
        # depressed by sympy giving up rather than by the model being wrong, and it concentrates
        # on the hardest problems -- so it biases comparisons, not just the absolute number.
        "verification_errors": sum(1 for item in grades if item.error is not None),
        "verification_timeouts": sum(1 for item in grades if item.error and "Timeout" in item.error),
    }
