"""Leakage-aware evaluation at user-day, user, and incident levels."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED_LABEL_COLUMNS = {"user", "day", "label"}


def _rank_metrics(truth: list[int], risk: list[float], alert_k: int) -> dict[str, float | int]:
    """Calculate alert-budget ranking metrics with deterministic tie-breaking."""
    if alert_k <= 0:
        raise ValueError("alert_k must be positive.")
    ranked = sorted(zip(risk, truth, strict=True), key=lambda item: item[0], reverse=True)
    selected = ranked[: min(alert_k, len(ranked))]
    selected_positive = sum(label for _, label in selected)
    positives = sum(truth)
    return {
        "alert_k": len(selected),
        "true_positives_at_k": selected_positive,
        "precision_at_k": selected_positive / len(selected) if selected else 0.0,
        "recall_at_k": selected_positive / positives if positives else 0.0,
        "f1_at_k": (
            2
            * (selected_positive / len(selected))
            * (selected_positive / positives)
            / ((selected_positive / len(selected)) + (selected_positive / positives))
            if selected and positives and selected_positive
            else 0.0
        ),
    }


def evaluate_scores(
    scored: pl.DataFrame,
    labels_path: Path,
    test_start: date | None = None,
    alert_k: int = 10,
) -> dict[str, float | int | str]:
    """Evaluate scored days against a held-out ``user,day,label`` file.

    CERT releases use different ``answers/`` layouts. Label conversion therefore remains
    explicit and auditable rather than guessing dates or treating every day of an insider's
    employment as malicious.
    """
    labels = pl.read_csv(labels_path, try_parse_dates=True)
    missing = REQUIRED_LABEL_COLUMNS.difference(labels.columns)
    if missing:
        raise ValueError(
            f"Label file must contain {sorted(REQUIRED_LABEL_COLUMNS)}; missing {sorted(missing)}."
        )
    labels = labels.select(
        pl.col("user").cast(pl.Utf8),
        pl.col("day").cast(pl.Date),
        pl.col("label").cast(pl.Int8),
    )
    scored_test = scored.filter(pl.col("risk_score").is_not_null())
    if test_start is not None:
        scored_test = scored_test.filter(pl.col("day") >= test_start)
    joined = scored_test.join(labels, on=["user", "day"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )
    if joined["label"].n_unique() < 2:
        raise ValueError("Evaluation needs at least one positive and one negative labelled day.")

    truth = joined["label"].to_list()
    risk = joined["risk_score"].to_list()
    positives = int(sum(truth))
    result: dict[str, float | int | str] = {
        "test_start": test_start.isoformat() if test_start else "all_scored_days",
        "scored_user_days": joined.height,
        "positive_user_days": positives,
        "positive_prevalence": positives / joined.height,
        "roc_auc": float(roc_auc_score(truth, risk)),
        "pr_auc": float(average_precision_score(truth, risk)),
    }
    result.update(_rank_metrics(truth, risk, alert_k))
    # Retain the old prevalence-sized K only for comparison with already reported results.
    result.update(
        {
            f"precision_at_positive_count_{positives}": _rank_metrics(truth, risk, max(positives, 1))[
                "precision_at_k"
            ],
            f"recall_at_positive_count_{positives}": _rank_metrics(truth, risk, max(positives, 1))[
                "recall_at_k"
            ],
        }
    )
    return result


def aggregate_user_scores(
    scored: pl.DataFrame, start: date, end: date, top_q: int = 3
) -> pl.DataFrame:
    """Aggregate only scores visible within an explicit alert-review window."""
    if top_q <= 0:
        raise ValueError("top_q must be positive.")
    window = scored.filter(
        pl.col("risk_score").is_not_null() & (pl.col("day") >= start) & (pl.col("day") <= end)
    )
    if window.is_empty():
        return pl.DataFrame(schema={"user": pl.String, "max_risk": pl.Float64, "top_q_mean_risk": pl.Float64})
    return (
        window.sort(["user", "risk_score"], descending=[False, True])
        .group_by("user")
        .agg(
            pl.col("risk_score").max().alias("max_risk"),
            pl.col("risk_score").head(top_q).mean().alias("top_q_mean_risk"),
            pl.len().alias("scored_days"),
        )
    )


def evaluate_user_level(
    scored: pl.DataFrame,
    incidents: pl.DataFrame,
    start: date,
    end: date,
    *,
    alert_k: int = 10,
    top_q: int = 3,
    score_column: str = "top_q_mean_risk",
) -> dict[str, float | int]:
    """Evaluate a fixed review window at the analyst's user ranking level."""
    users = aggregate_user_scores(scored, start, end, top_q)
    if users.is_empty():
        raise ValueError("No scored users exist in this evaluation window.")
    required = {"user", "attack_start"}
    missing = required.difference(incidents.columns)
    if missing:
        raise ValueError(f"Incidents missing required columns: {sorted(missing)}")
    positives = (
        incidents.with_columns(pl.col("attack_start").cast(pl.Datetime).dt.date().alias("_incident_day"))
        .filter((pl.col("_incident_day") >= start) & (pl.col("_incident_day") <= end))
        .select("user")
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    )
    joined = users.join(positives, on="user", how="left").with_columns(pl.col("label").fill_null(0))
    truth = joined["label"].to_list()
    risk = joined[score_column].to_list()
    metrics = _rank_metrics(truth, risk, alert_k)
    ranked_labels = [
        label
        for _, label in sorted(zip(risk, truth, strict=True), key=lambda item: item[0], reverse=True)[:alert_k]
    ]
    dcg = sum(label / __import__("math").log2(index + 2) for index, label in enumerate(ranked_labels))
    ideal = sum(1 / __import__("math").log2(index + 2) for index in range(min(sum(truth), alert_k)))
    metrics.update(
        {
            "scored_users": joined.height,
            "positive_users": int(sum(truth)),
            "user_prevalence": sum(truth) / joined.height,
            "map_at_k": (
                sum(
                    sum(ranked_labels[: index + 1]) / (index + 1)
                    for index, label in enumerate(ranked_labels)
                    if label
                )
                / min(sum(truth), alert_k)
                if sum(truth)
                else 0.0
            ),
            "ndcg_at_k": dcg / ideal if ideal else 0.0,
        }
    )
    return metrics


def daily_alerts(scored: pl.DataFrame, alert_k: int) -> pl.DataFrame:
    """Select a fixed per-day analyst budget before incident matching."""
    if alert_k <= 0:
        raise ValueError("alert_k must be positive.")
    return (
        scored.filter(pl.col("risk_score").is_not_null())
        .sort(["day", "risk_score"], descending=[False, True])
        .group_by("day", maintain_order=True)
        .head(alert_k)
        .select("user", "day", "risk_score")
    )


def evaluate_incident_level(
    scored: pl.DataFrame,
    incidents: pl.DataFrame,
    *,
    alert_k_per_day: int = 10,
    early_warning_days: int = 0,
) -> dict[str, float | int | None]:
    """Measure detection and lead time after enforcing an alert budget each day."""
    required = {"incident_id", "user", "attack_start", "attack_end"}
    missing = required.difference(incidents.columns)
    if missing:
        raise ValueError(f"Incidents missing required columns: {sorted(missing)}")
    alerts = daily_alerts(scored, alert_k_per_day)
    incident_rows = incidents.with_columns(
        pl.col("attack_start").cast(pl.Datetime).dt.date().alias("_start_day"),
        pl.col("attack_end").cast(pl.Datetime).dt.date().alias("_end_day"),
    ).select("incident_id", "user", "_start_day", "_end_day")
    detected = 0
    early = 0
    lead_times: list[int] = []
    matched_alerts: set[tuple[str, date]] = set()
    for incident in incident_rows.iter_rows(named=True):
        user_alerts = alerts.filter(pl.col("user") == incident["user"])
        detection = user_alerts.filter(
            (pl.col("day") >= incident["_start_day"]) & (pl.col("day") <= incident["_end_day"])
        )
        if detection.height:
            detected += 1
            matched_alerts.update((row["user"], row["day"]) for row in detection.iter_rows(named=True))
        if early_warning_days:
            early_alerts = user_alerts.filter(
                (pl.col("day") >= incident["_start_day"] - timedelta(days=early_warning_days))
                & (pl.col("day") < incident["_start_day"])
            )
            if early_alerts.height:
                first_day = early_alerts["day"].min()
                early += 1
                lead_times.append((incident["_start_day"] - first_day).days)
                matched_alerts.update(
                    (row["user"], row["day"]) for row in early_alerts.iter_rows(named=True)
                )
    total = incident_rows.height
    all_alerts = {(row["user"], row["day"]) for row in alerts.iter_rows(named=True)}
    scored_days = scored.filter(pl.col("risk_score").is_not_null()).height
    return {
        "incidents": total,
        "detected_incidents": detected,
        "incident_detection_rate": detected / total if total else 0.0,
        "early_warned_incidents": early,
        "early_warning_rate": early / total if total else 0.0,
        "median_lead_time_days": float(__import__("statistics").median(lead_times)) if lead_times else None,
        "alerts_per_detected_incident": len(all_alerts) / detected if detected else None,
        "false_alerts_per_100_user_days": 100 * len(all_alerts - matched_alerts) / scored_days if scored_days else 0.0,
    }
