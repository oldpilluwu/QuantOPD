from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitIndices:
    calibration: tuple[int, ...]
    training: tuple[int, ...]
    smoke: tuple[int, ...]


def make_split_indices(
    population_size: int,
    calibration_size: int,
    training_size: int,
    smoke_size: int,
    seed: int,
) -> SplitIndices:
    requested = calibration_size + training_size
    if requested > population_size:
        raise ValueError(f"Requested {requested} rows from a dataset with {population_size} rows")
    if smoke_size > training_size:
        raise ValueError("smoke_size cannot exceed training_size")

    indices = list(range(population_size))
    random.Random(seed).shuffle(indices)
    calibration = tuple(indices[:calibration_size])
    training = tuple(indices[calibration_size:requested])
    smoke = training[:smoke_size]
    return SplitIndices(calibration=calibration, training=training, smoke=smoke)


def select_prompt_field(column_names: list[str], candidates: tuple[str, ...]) -> str:
    for field in candidates:
        if field in column_names:
            return field
    raise ValueError(f"None of the configured prompt fields {candidates} exist in columns {column_names}")
