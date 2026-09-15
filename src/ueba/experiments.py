"""Fold runner for reproducible feature-selection and detector comparisons."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from .baseline import raw_feature_columns
from .config import ModelSettings, SelectionSettings
from .evaluation import evaluate_scores
from .models import fit_models, score_features
from .selection import correlation_filter, permutation_importance, top_features
from .temporal import TemporalFold, select_period


def _label_vector(frame: pl.DataFrame, labels_path: Path) -> np.ndarray:
    labels = pl.read_csv(labels_path, try_parse_dates=True).select(
        pl.col("user").cast(pl.Utf8), pl.col("day").cast(pl.Date), pl.col("label").cast(pl.Int8)
    )
    return (
        frame.select("user", "day")
        .join(labels, on=["user", "day"], how="left")
        .with_columns(pl.col("label").fill_null(0))["label"]
        .to_numpy()
    )


def run_fold(
    features: pl.DataFrame,
    labels_path: Path,
    fold: TemporalFold,
    model: ModelSettings,
    selection: SelectionSettings,
    *,
    alert_k: int,
    enable_selection: bool = True,
) -> tuple[dict[str, float | int | str], pl.DataFrame, list[str]]:
    """Run one outer fold; answer-key labels influence only inner validation selection."""
    train = select_period(features, fold.train_start, fold.train_end).filter(pl.col("baseline_ready"))
    validation = select_period(features, fold.validation_start, fold.validation_end)
    candidate_features = [
        column
        for column in [*raw_feature_columns(features), *[c for c in features.columns if c.startswith("z_")]]
        if column in train.columns and train.get_column(column).null_count() < train.height
    ]
    filtered, _ = correlation_filter(
        train, candidate_features, threshold=selection.correlation_threshold
    )
    selected = filtered
    if enable_selection:
        peer, anomaly = fit_models(features, fold.train_end, model, selected_features=filtered)
        validation_labels = _label_vector(validation, labels_path)
        # Sparse incidents can leave a validation fold with a single class. It cannot
        # support AP-based selection, so retain the label-free train-only filter.
        if np.unique(validation_labels).size == 2:
            importance = permutation_importance(
                validation,
                validation_labels,
                filtered,
                lambda frame: score_features(frame, peer, anomaly)
                .get_column("risk_score")
                .fill_null(0)
                .to_numpy(),
                repetitions=selection.permutation_repetitions,
                random_state=model.random_state,
            )
            selected = top_features(
                importance, minimum=selection.minimum_features, maximum=selection.maximum_features
            )
            if not selected:
                selected = filtered
    # Include validation behavior without labels for the final unsupervised detector fit.
    refit = features.filter(pl.col("day") <= fold.validation_end)
    peer, anomaly = fit_models(refit, fold.validation_end, model, selected_features=selected)
    scored_test = score_features(select_period(features, fold.test_start, fold.test_end), peer, anomaly)
    try:
        metrics = evaluate_scores(scored_test, labels_path, alert_k=alert_k)
        metrics["evaluation_status"] = "evaluable"
    except ValueError as error:
        metrics = {
            "evaluation_status": f"not_evaluable: {error}",
            "scored_user_days": scored_test.filter(pl.col("risk_score").is_not_null()).height,
        }
    metrics.update(
        {
            "fold_id": fold.fold_id,
            "model_family": model.model_family,
            "selected_feature_count": len(selected),
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
        }
    )
    return metrics, scored_test, selected


def run_model_families(
    features: pl.DataFrame,
    labels_path: Path,
    fold: TemporalFold,
    model: ModelSettings,
    selection: SelectionSettings,
    *,
    families: tuple[str, ...] = ("isolation_forest", "lof", "ocsvm"),
    alert_k: int = 10,
) -> pl.DataFrame:
    """Run comparable model families with identical chronological folds and selection."""
    records: list[dict[str, float | int | str]] = []
    for family in families:
        metrics, _, _ = run_fold(
            features,
            labels_path,
            fold,
            replace(model, model_family=family),
            selection,
            alert_k=alert_k,
        )
        records.append(metrics)
    return pl.DataFrame(records)
