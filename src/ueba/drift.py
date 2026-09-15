"""Online drift monitors for UEBA risk streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl


class DriftDetector(Protocol):
    def update(self, value: float) -> bool: ...


@dataclass
class PageHinkley:
    """One-sided Page-Hinkley monitor for sustained upward risk/residual drift."""

    delta: float = 0.005
    threshold: float = 20.0
    minimum_instances: int = 30
    _count: int = 0
    _mean: float = 0.0
    _cumulative: float = 0.0
    _minimum: float = 0.0

    def update(self, value: float) -> bool:
        self._count += 1
        self._mean += (value - self._mean) / self._count
        self._cumulative += value - self._mean - self.delta
        self._minimum = min(self._minimum, self._cumulative)
        changed = (
            self._count >= self.minimum_instances
            and self._cumulative - self._minimum > self.threshold
        )
        if changed:
            # Start the next regime fresh; the caller records the date and policy action.
            self._count, self._mean, self._cumulative, self._minimum = 0, 0.0, 0.0, 0.0
        return changed


class ADWIN:
    """Thin optional wrapper around River's ADWIN detector."""

    def __init__(self, delta: float = 0.002) -> None:
        try:
            from river import drift
        except ImportError as error:
            raise ImportError(
                "ADWIN requires the optional dependency. Install with `pip install -e '.[drift]'`."
            ) from error
        self._model = drift.ADWIN(delta=delta)

    def update(self, value: float) -> bool:
        self._model.update(value)
        return bool(self._model.drift_detected)


def detect_drift(
    series: pl.DataFrame,
    *,
    value_column: str = "risk_score",
    time_column: str = "day",
    method: str = "page_hinkley",
    delta: float = 0.005,
    threshold: float = 20.0,
    minimum_instances: int = 30,
) -> pl.DataFrame:
    """Detect drift on an already aggregated, chronologically sorted metric stream."""
    if value_column not in series.columns or time_column not in series.columns:
        raise ValueError(f"Series needs {time_column!r} and {value_column!r} columns.")
    if method == "page_hinkley":
        detector: DriftDetector = PageHinkley(delta, threshold, minimum_instances)
    elif method == "adwin":
        detector = ADWIN(delta)
    else:
        raise ValueError("method must be 'page_hinkley' or 'adwin'.")
    changes: list[dict[str, object]] = []
    for observation, row in enumerate(
        series.select(time_column, value_column).drop_nulls().sort(time_column).iter_rows(named=True),
        start=1,
    ):
        value = float(row[value_column])
        if detector.update(value):
            changes.append(
                {
                    "drift_time": row[time_column],
                    "method": method,
                    "monitored_series": value_column,
                    "observations_at_detection": observation,
                    "value": value,
                }
            )
    return pl.DataFrame(
        changes,
        schema={
            "drift_time": series.schema[time_column],
            "method": pl.String,
            "monitored_series": pl.String,
            "observations_at_detection": pl.Int64,
            "value": pl.Float64,
        },
    )


def adaptive_threshold(
    scores: pl.DataFrame,
    *,
    score_column: str = "risk_score",
    quantile: float = 0.99,
) -> float:
    """Return a training/history-only quantile threshold for a fixed alert rate."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1).")
    values = scores.get_column(score_column).cast(pl.Float64, strict=False).drop_nulls().to_numpy()
    if not len(values):
        raise ValueError("Cannot calibrate an adaptive threshold from no scores.")
    return float(np.quantile(values, quantile))
