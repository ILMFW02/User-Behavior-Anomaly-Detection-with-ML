"""Convert CERT answer keys and lab ground truth into auditable labels."""

from __future__ import annotations

import csv
import re
from datetime import date, datetime, timedelta
from itertools import chain
from pathlib import Path

import polars as pl

_RELEASE = re.compile(r"r\d+\.\d+")
_TIMESTAMP_FORMATS = (
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def _parse_timestamp(value: str) -> datetime | None:
    value = value.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)  # noqa: DTZ007 - CERT timestamps are local-naive.
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _release_for(path: Path) -> str:
    match = _RELEASE.search(str(path).replace("\\", "/"))
    return match.group(0) if match else "unknown"


def _answer_files(answers_dir: Path, release: str | None) -> list[Path]:
    files = sorted(path for path in answers_dir.rglob("*.csv") if path.is_file())
    if release is None:
        return files
    return [path for path in files if release in str(path).replace("\\", "/")]


def build_insider_events(answers_dir: Path, release: str | None = "r4.2") -> pl.DataFrame:
    """Parse answer-key events without assuming one CERT release's file layout.

    Legacy r4.2 rows are headerless ``event_type,event_id,timestamp,user,...``.
    Headered answer files must expose a timestamp/date and user/username column.
    The returned event provenance is retained for incident-level evaluation.
    """
    if not answers_dir.is_dir():
        raise FileNotFoundError(f"Answers directory does not exist: {answers_dir}")

    events: list[dict[str, str | datetime]] = []
    invalid_rows = 0
    answer_files = _answer_files(answers_dir, release)
    if not answer_files:
        selected = release or "any release"
        raise FileNotFoundError(f"No answer CSV files found for {selected} beneath {answers_dir}.")

    for path in answer_files:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = csv.reader(source)
            first = next(rows, None)
            if first is None:
                continue
            normalized = [cell.strip().lower() for cell in first]
            is_headered = bool({"user", "username", "timestamp", "date", "time"} & set(normalized))
            if is_headered:
                user_index = next(
                    (index for index, value in enumerate(normalized) if value in {"user", "username", "user_id"}),
                    None,
                )
                time_index = next(
                    (index for index, value in enumerate(normalized) if value in {"timestamp", "date", "time"}),
                    None,
                )
                event_type_index = next(
                    (index for index, value in enumerate(normalized) if value in {"event_type", "type", "activity"}),
                    None,
                )
                event_id_index = next(
                    (index for index, value in enumerate(normalized) if value in {"event_id", "id"}),
                    None,
                )
                iterable = rows
            else:
                user_index, time_index, event_type_index, event_id_index = 3, 2, 0, 1
                iterable = chain((first,), rows)

            for row_number, row in enumerate(iterable, start=2 if is_headered else 1):
                if user_index is None or time_index is None or max(user_index, time_index) >= len(row):
                    invalid_rows += 1
                    continue
                timestamp = _parse_timestamp(row[time_index])
                user = row[user_index].strip()
                if timestamp is None or not user:
                    invalid_rows += 1
                    continue
                events.append(
                    {
                        "event_type": row[event_type_index] if event_type_index is not None and event_type_index < len(row) else "unknown",
                        "event_id": row[event_id_index] if event_id_index is not None and event_id_index < len(row) else f"{path.name}:{row_number}",
                        "timestamp": timestamp,
                        "user": user,
                        "source_file": str(path.relative_to(answers_dir)),
                        "release": _release_for(path),
                    }
                )

    if not events:
        raise ValueError("No valid malicious events found in answer files.")
    if invalid_rows:
        print(f"Warning: skipped {invalid_rows} malformed answer-key rows.")
    return pl.DataFrame(events).with_columns(pl.col("timestamp").dt.date().alias("day")).sort(
        ["user", "timestamp"]
    )


def build_insider_user_day_labels(
    answers_dir: Path, release: str | None = "r4.2", early_warning_days: int = 0
) -> pl.DataFrame:
    """Create event-day or pre-incident labels from answer events.

    When ``early_warning_days`` is positive, only the preceding days are labelled. The
    incident day remains excluded, so early-warning AP cannot be mistaken for detection AP.
    """
    events = build_insider_events(answers_dir, release)
    event_days = events.select("user", "day").unique()
    if early_warning_days == 0:
        return event_days.with_columns(pl.lit(1, dtype=pl.Int8).alias("label")).sort(["user", "day"])
    windows: list[tuple[str, date]] = []
    for row in event_days.iter_rows(named=True):
        for offset in range(1, early_warning_days + 1):
            windows.append((row["user"], row["day"] - timedelta(days=offset)))
    return (
        pl.DataFrame(windows, schema=["user", "day"], orient="row")
        .unique()
        .join(event_days, on=["user", "day"], how="anti")
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
        .sort(["user", "day"])
    )


def build_cert_incidents(answers_dir: Path, release: str | None = "r4.2") -> pl.DataFrame:
    """Derive auditable CERT incident intervals per answer file and user.

    CERT answer keys list malicious events, not canonical incident intervals. This
    deterministic approximation preserves the source file so reports cannot overstate it
    as independently supplied ground truth.
    """
    events = build_insider_events(answers_dir, release)
    return (
        events.group_by(["release", "source_file", "user"])
        .agg(
            pl.col("timestamp").min().alias("attack_start"),
            pl.col("timestamp").max().alias("attack_end"),
            pl.col("event_id").alias("evidence_event_ids"),
        )
        .with_row_index("incident_number")
        .with_columns(
            (pl.col("release") + ":" + pl.col("source_file") + ":" + pl.col("incident_number").cast(pl.Utf8))
            .alias("incident_id")
        )
        .select("incident_id", "user", "attack_start", "attack_end", "release", "source_file", "evidence_event_ids")
        .sort(["attack_start", "user"])
    )


LAB_INCIDENT_COLUMNS = {
    "incident_id",
    "user",
    "attack_start",
    "attack_end",
    "scenario_id",
    "technique_id",
}


def load_lab_incidents(path: Path) -> pl.DataFrame:
    """Read lab ground truth with an explicit, reproducible incident schema."""
    incidents = pl.read_csv(path, try_parse_dates=True)
    missing = LAB_INCIDENT_COLUMNS.difference(incidents.columns)
    if missing:
        raise ValueError(f"Ground truth is missing required columns: {sorted(missing)}")
    result = incidents.with_columns(
        pl.col("user").cast(pl.Utf8),
        pl.col("attack_start").cast(pl.Datetime),
        pl.col("attack_end").cast(pl.Datetime),
    )
    if result.filter(pl.col("attack_end") < pl.col("attack_start")).height:
        raise ValueError("Every attack_end must be at or after attack_start.")
    return result
