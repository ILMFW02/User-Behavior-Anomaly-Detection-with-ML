"""Leakage-safe time-series folds for UEBA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


@dataclass(frozen=True)
class TemporalFold:
    """One chronological outer evaluation fold with an inner calibration period."""

    fold_id: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


def expanding_folds(
    first_day: date,
    last_day: date,
    *,
    initial_train_days: int = 180,
    validation_days: int = 60,
    test_days: int = 90,
    step_days: int = 90,
    embargo_days: int = 0,
) -> list[TemporalFold]:
    """Return feasible expanding-window folds without silently using future days.

    ``embargo_days`` removes observations at each boundary. It is useful when labels
    include a pre-incident horizon. Causal rolling features themselves do not require
    an embargo because their statistics are shifted to prior observations.
    """
    values = (initial_train_days, validation_days, test_days, step_days)
    if any(value <= 0 for value in values) or embargo_days < 0:
        raise ValueError("Temporal window lengths must be positive; embargo_days cannot be negative.")
    folds: list[TemporalFold] = []
    train_end = first_day + timedelta(days=initial_train_days - 1)
    fold_id = 1
    while True:
        validation_start = train_end + timedelta(days=embargo_days + 1)
        validation_end = validation_start + timedelta(days=validation_days - 1)
        test_start = validation_end + timedelta(days=embargo_days + 1)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_end > last_day:
            break
        folds.append(
            TemporalFold(
                fold_id=fold_id,
                train_start=first_day,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        train_end += timedelta(days=step_days)
    return folds


def select_period(frame: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Select an inclusive time range after validating the canonical ``day`` column."""
    if "day" not in frame.columns:
        raise ValueError("Frame must contain a canonical 'day' column.")
    return frame.filter((pl.col("day") >= start) & (pl.col("day") <= end))


def folds_from_frame(frame: pl.DataFrame, **kwargs: int) -> list[TemporalFold]:
    """Generate folds from the observed data range, rather than hard-coding a count."""
    if frame.is_empty():
        return []
    days = frame.select(pl.col("day").cast(pl.Date).min().alias("min"), pl.col("day").cast(pl.Date).max().alias("max")).row(0, named=True)
    if days["min"] is None or days["max"] is None:
        return []
    return expanding_folds(days["min"], days["max"], **kwargs)
