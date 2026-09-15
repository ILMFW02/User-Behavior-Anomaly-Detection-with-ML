"""Feature selection helpers that never inspect an outer test fold."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def _rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic ordinal ranks without adding a scipy dependency."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def correlation_filter(
    train: pl.DataFrame,
    features: Sequence[str],
    *,
    threshold: float = 0.90,
    priority: Sequence[str] = (),
) -> tuple[list[str], list[dict[str, float | str]]]:
    """Keep a non-constant, low-redundancy feature subset from training data only.

    Features in ``priority`` win a correlated pair. Other pairs prefer the feature
    with greater train-fold variance, avoiding any answer-key labels.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    priority_index = {feature: index for index, feature in enumerate(priority)}
    candidates: list[tuple[str, np.ndarray, float]] = []
    for feature in features:
        values = train.get_column(feature).cast(pl.Float64, strict=False).fill_null(
            train.get_column(feature).cast(pl.Float64, strict=False).median()
        ).fill_null(0.0).to_numpy()
        if len(values) >= 2 and np.nanstd(values) > 0:
            candidates.append((feature, _rank(values), float(np.nanvar(values))))
    candidates.sort(key=lambda item: (priority_index.get(item[0], len(priority_index)), -item[2]))
    kept: list[tuple[str, np.ndarray, float]] = []
    dropped: list[dict[str, float | str]] = []
    for feature, ranks, variance in candidates:
        correlated_with: tuple[str, float] | None = None
        for kept_feature, kept_ranks, _ in kept:
            correlation = float(np.corrcoef(ranks, kept_ranks)[0, 1])
            if abs(correlation) > threshold:
                correlated_with = (kept_feature, correlation)
                break
        if correlated_with is None:
            kept.append((feature, ranks, variance))
        else:
            dropped.append(
                {
                    "dropped_feature": feature,
                    "kept_feature": correlated_with[0],
                    "spearman_correlation": correlated_with[1],
                }
            )
    return [feature for feature, _, _ in kept], dropped


def permutation_importance(
    validation: pl.DataFrame,
    labels: Sequence[int],
    features: Sequence[str],
    score: Callable[[pl.DataFrame], np.ndarray],
    *,
    repetitions: int = 5,
    random_state: int = 42,
) -> pl.DataFrame:
    """Estimate validation-only score sensitivity as AP loss after permutation.

    This is model-agnostic and is deliberately not described as native Isolation
    Forest feature importance. Call it only after the detector has been fit on
    earlier training data; never call it against the final test fold.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    y = np.asarray(labels, dtype=int)
    if y.size != validation.height or np.unique(y).size < 2:
        raise ValueError("Validation labels must align with rows and include both classes.")
    baseline = float(average_precision_score(y, score(validation)))
    random = np.random.default_rng(random_state)
    records: list[dict[str, float | str]] = []
    for feature in features:
        original = validation.get_column(feature).to_numpy()
        losses: list[float] = []
        for _ in range(repetitions):
            shuffled = validation.with_columns(pl.Series(feature, random.permutation(original)))
            losses.append(baseline - float(average_precision_score(y, score(shuffled))))
        records.append(
            {
                "feature": feature,
                "baseline_pr_auc": baseline,
                "mean_pr_auc_drop": float(np.mean(losses)),
                "std_pr_auc_drop": float(np.std(losses, ddof=0)),
            }
        )
    return pl.DataFrame(records).sort(["mean_pr_auc_drop", "feature"], descending=[True, False])


def top_features(importance: pl.DataFrame, minimum: int = 20, maximum: int = 30) -> list[str]:
    """Freeze a bounded feature set after validation-only permutation ranking."""
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Expected 0 < minimum <= maximum.")
    if importance.is_empty():
        return []
    positive = importance.filter(pl.col("mean_pr_auc_drop") > 0)
    candidate = positive if positive.height >= minimum else importance
    return candidate.head(min(maximum, candidate.height))["feature"].to_list()
