"""Evidence-preserving event correlation for UEBA research alerts."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import polars as pl

REQUIRED_EVENT_COLUMNS = {"event_id", "user", "timestamp", "event_type", "action", "object"}


def _event_text(event: dict[str, Any]) -> str:
    return f"{event.get('action', '')} {event.get('object', '')}".lower()


def _alert(
    rule_id: str,
    events: list[dict[str, Any]],
    technique_ids: list[str],
    confidence: float,
    rationale: str,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "user": events[-1]["user"],
        "host": events[-1]["host"],
        "alert_timestamp": events[-1]["timestamp"],
        "mitre_technique_ids": technique_ids,
        "confidence": confidence,
        "rationale": rationale,
        "evidence_event_ids": [str(event["event_id"]) for event in events],
        "evidence_timestamps": [event["timestamp"] for event in events],
    }


def correlate_events(
    events: pl.DataFrame,
    *,
    sensitive_object_pattern: str = r"(?i)(secret|confidential|sensitive|\.pst$|\.zip$)",
    window_minutes: int = 60,
) -> pl.DataFrame:
    """Correlate normalized events without turning association into proof of exfiltration.

    The USB chain emits a *candidate* mapping to ATT&CK T1052/T1567.002. It proves
    only the time-bounded sequence observed in the evidence fields, not that a file
    was copied or sent. Privilege and destructive-action rules map direct observed
    command/action evidence to T1098, T1485, or T1489.
    """
    missing = REQUIRED_EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"Events missing required normalized columns: {sorted(missing)}")
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive.")
    normalized = (
        events.with_columns(
            pl.col("event_id").cast(pl.Utf8),
            pl.col("user").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("event_type").cast(pl.Utf8).str.to_lowercase(),
            pl.col("action").cast(pl.Utf8).fill_null("").str.to_lowercase(),
            pl.col("object").cast(pl.Utf8).fill_null("").str.to_lowercase(),
            (
                pl.col("host").cast(pl.Utf8).fill_null("_unknown")
                if "host" in events.columns
                else pl.lit("_unknown", dtype=pl.Utf8)
            ).alias("host"),
            (
                pl.col("is_external").cast(pl.Boolean, strict=False).fill_null(False)
                if "is_external" in events.columns
                else pl.lit(False)
            ).alias("is_external"),
        )
        .filter(pl.col("timestamp").is_not_null())
        .sort(["user", "host", "timestamp"])
    )
    pattern = re.compile(sensitive_object_pattern)
    window = timedelta(minutes=window_minutes)
    alerts: list[dict[str, Any]] = []
    for (user, host), group in normalized.group_by(["user", "host"], maintain_order=True):
        latest_usb: dict[str, Any] | None = None
        latest_sensitive_file: tuple[dict[str, Any], dict[str, Any]] | None = None
        for event in group.iter_rows(named=True):
            text = _event_text(event)
            if event["event_type"] == "device" and re.search(
                r"\b(connect|insert|mount)\b|device_connect", text
            ):
                latest_usb = event
            elif event["event_type"] == "file" and pattern.search(event["object"]):
                if latest_usb and event["timestamp"] - latest_usb["timestamp"] <= window:
                    latest_sensitive_file = (latest_usb, event)
            elif event["event_type"] == "email" and event["is_external"] and latest_sensitive_file:
                usb, file_event = latest_sensitive_file
                if event["timestamp"] - file_event["timestamp"] <= window:
                    alerts.append(
                        _alert(
                            "UEBA-CORR-001",
                            [usb, file_event, event],
                            ["T1052", "T1567.002"],
                            0.85,
                            "Candidate chain: USB connected, sensitive file accessed, then external email "
                            "within the configured window. Evidence does not prove copying or exfiltration.",
                        )
                    )
                    latest_sensitive_file = None
            if re.search(r"\b(useradd|usermod|net user|add-user|sudoers|groupadd)\b", text):
                alerts.append(
                    _alert(
                        "UEBA-CORR-002",
                        [event],
                        ["T1098"],
                        0.75,
                        "Observed account or group-administration command/action requiring investigation.",
                    )
                )
            if re.search(r"\b(delete|del |rm |format|wipe|shred|service stop|systemctl stop)\b", text):
                techniques = ["T1489"] if re.search(r"service stop|systemctl stop", text) else ["T1485"]
                alerts.append(
                    _alert(
                        "UEBA-CORR-003",
                        [event],
                        techniques,
                        0.75,
                        "Observed destructive file or service-stop command/action requiring investigation.",
                    )
                )
    if not alerts:
        return pl.DataFrame(
            schema={
                "rule_id": pl.String,
                "user": pl.String,
                "host": pl.String,
                "alert_timestamp": pl.Datetime,
                "mitre_technique_ids": pl.List(pl.String),
                "confidence": pl.Float64,
                "rationale": pl.String,
                "evidence_event_ids": pl.List(pl.String),
                "evidence_timestamps": pl.List(pl.Datetime),
            }
        )
    return pl.DataFrame(alerts).sort("alert_timestamp")


def apply_rule_risk(
    scored: pl.DataFrame, alerts: pl.DataFrame, *, rule_weight: float = 0.20
) -> pl.DataFrame:
    """Blend evidence-backed rule confidence into an existing 0--100 daily risk score.

    This performs no label-dependent tuning. Select ``rule_weight`` only on the inner
    validation period, then freeze it for the outer test. Rule IDs remain attached so
    analysts can distinguish a correlated alert from a purely statistical anomaly.
    """
    if not 0 <= rule_weight <= 1:
        raise ValueError("rule_weight must be between zero and one.")
    if "risk_score" not in scored.columns:
        raise ValueError("Scored data must contain risk_score.")
    if alerts.is_empty():
        return scored.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("rule_component"),
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("correlation_rule_ids"),
        )
    daily_rules = (
        alerts.with_columns(pl.col("alert_timestamp").cast(pl.Datetime).dt.date().alias("day"))
        .group_by(["user", "day"])
        .agg(
            pl.col("confidence").max().alias("rule_component"),
            pl.col("rule_id").unique().alias("correlation_rule_ids"),
        )
    )
    return (
        scored.join(daily_rules, on=["user", "day"], how="left")
        .with_columns(pl.col("rule_component").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("risk_score").is_not_null())
            .then(
                100
                * (
                    (1 - rule_weight) * pl.col("risk_score") / 100
                    + rule_weight * pl.col("rule_component")
                )
            )
            .otherwise(None)
            .alias("risk_score")
        )
        .with_columns(pl.col("correlation_rule_ids").fill_null([]))
    )
