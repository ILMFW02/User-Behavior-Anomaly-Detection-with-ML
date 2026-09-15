"""Causal, per-user behavioural baselines."""

from __future__ import annotations

import polars as pl

from .config import BaselineSettings

IDENTITY_COLUMNS = {"user", "day"}


def raw_feature_columns(frame: pl.DataFrame) -> list[str]:
    """Return numeric behavioural features, excluding identity and derived columns."""
    return [
        column
        for column, dtype in frame.schema.items()
        if column not in IDENTITY_COLUMNS
        and not column.startswith(("baseline_", "z_", "peer_"))
        and dtype.is_numeric()
    ]


def add_causal_baselines(
    frame: pl.DataFrame, settings: BaselineSettings
) -> tuple[pl.DataFrame, list[str]]:
    """Add z-score deviations using only earlier observations for that user.

    The input contains a complete daily calendar per user, so the rolling window represents
    calendar days rather than only days on which an event was recorded.
    """
    features = raw_feature_columns(frame)
    ordered = frame.sort(["user", "day"])
    history_count = (pl.col("day").cum_count().over("user") - 1).alias("history_days")

    baseline_expressions: list[pl.Expr] = [history_count]
    for feature in features:
        prior = pl.col(feature).shift(1)
        baseline_expressions.extend(
            [
                prior.rolling_mean(
                    window_size=settings.rolling_window_days,
                    min_samples=1,
                )
                .over("user")
                .alias(f"baseline_{feature}"),
                prior.rolling_std(
                    window_size=settings.rolling_window_days,
                    min_samples=2,
                )
                .over("user")
                .alias(f"baseline_std_{feature}"),
            ]
        )

    with_baselines = ordered.with_columns(baseline_expressions)
    z_features = [f"z_{feature}" for feature in features]
    z_expressions: list[pl.Expr] = []
    for feature, z_feature in zip(features, z_features, strict=True):
        mean = pl.col(f"baseline_{feature}")
        std = pl.col(f"baseline_std_{feature}")
        z_expressions.append(
            pl.when((pl.col("history_days") >= settings.min_history_days) & (std > 0))
            .then((pl.col(feature) - mean) / std)
            .otherwise(None)
            .alias(z_feature)
        )

    return (
        with_baselines.with_columns(z_expressions).with_columns(
            (pl.col("history_days") >= settings.min_history_days).alias("baseline_ready")
        ),
        z_features,
    )
