"""Feature extraction from CERT event CSV files.

The implementation streams each source CSV through Polars' lazy engine, then materialises
only its daily aggregates. This keeps the large HTTP log out of Python row-by-row processing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from .config import FeatureSettings
from .dataset import EVENT_FILES, inspect_dataset
from .organization import attach_organization_context, load_organization_context

KEY_COLUMNS = ["user", "day"]


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def _contains_any(column: str | None, terms: tuple[str, ...]) -> pl.Expr:
    if column is None or not terms:
        return pl.lit(False)
    pattern = "(?i)" + "|".join(re.escape(term) for term in terms)
    return pl.col(column).cast(pl.Utf8).str.contains(pattern).fill_null(False)


def _event_base(path: Path) -> tuple[pl.LazyFrame, set[str]]:
    source = pl.scan_csv(path, infer_schema_length=10_000, ignore_errors=True)
    columns = set(source.collect_schema().names())
    user_column = _first_present(columns, ("user", "username", "user_id"))
    timestamp_column = _first_present(columns, ("date", "timestamp", "time"))
    if user_column is None or timestamp_column is None:
        raise ValueError(f"{path.name} must have a user and date/timestamp column.")

    event = (
        source.with_columns(
            pl.col(user_column).cast(pl.Utf8).alias("user"),
            pl.col(timestamp_column)
            .cast(pl.Utf8)
            # CERT r4.2 uses a US-formatted timestamp. Automatic parsing interprets
            # ambiguous dates as day/month and drops dates such as 09/18/2010.
            .str.to_datetime(format="%m/%d/%Y %H:%M:%S", strict=False)
            .alias("timestamp"),
        )
        .filter(pl.col("user").is_not_null() & pl.col("timestamp").is_not_null())
        .with_columns(
            pl.col("timestamp").dt.date().alias("day"),
            pl.col("timestamp").dt.hour().alias("_hour"),
            pl.col("timestamp").dt.weekday().alias("_weekday"),
        )
    )
    return event, columns


def _off_hours(settings: FeatureSettings) -> pl.Expr:
    return (
        (pl.col("_hour") < settings.workday_start_hour)
        | (pl.col("_hour") >= settings.workday_end_hour)
        | (pl.col("_weekday") >= 6)
    )


def _aggregate_logon(
    event: pl.LazyFrame, columns: set[str], settings: FeatureSettings
) -> pl.LazyFrame:
    activity = _first_present(columns, ("activity", "action", "event"))
    pc = _first_present(columns, ("pc", "computer", "host"))
    activity_text = pl.col(activity).cast(pl.Utf8) if activity else pl.lit("")
    aggregate = event.group_by(KEY_COLUMNS).agg(
        pl.len().alias("logon_events"),
        activity_text.str.contains("(?i)logon").sum().alias("logon_count"),
        activity_text.str.contains("(?i)logoff").sum().alias("logoff_count"),
        _off_hours(settings).sum().alias("logon_off_hours_events"),
        (pl.col(pc).n_unique() if pc else pl.lit(0)).alias("logon_unique_pcs"),
    )
    if pc is None:
        return aggregate
    pc_days = (
        event.select(["user", "day", pl.col(pc).cast(pl.String).alias("_pc")])
        .unique()
        .with_columns(
            pl.col("day").min().over(["user", "_pc"]).alias("_first_pc_day"),
            pl.col("day").min().over("user").alias("_first_user_day"),
        )
        .group_by(KEY_COLUMNS)
        .agg(
            (
                (pl.col("day") == pl.col("_first_pc_day"))
                & (pl.col("day") > pl.col("_first_user_day"))
            )
            .sum()
            .alias("logon_new_pc_count")
        )
    )
    return aggregate.join(pc_days, on=KEY_COLUMNS, how="left")


def _aggregate_device(
    event: pl.LazyFrame, columns: set[str], settings: FeatureSettings
) -> pl.LazyFrame:
    activity = _first_present(columns, ("activity", "action", "event"))
    activity_text = pl.col(activity).cast(pl.Utf8) if activity else pl.lit("")
    return event.group_by(KEY_COLUMNS).agg(
        pl.len().alias("device_events"),
        activity_text.str.contains("(?i)connect|insert|mount").sum().alias("device_connect_count"),
        _off_hours(settings).sum().alias("device_off_hours_events"),
    )


def _aggregate_http(
    event: pl.LazyFrame, columns: set[str], settings: FeatureSettings
) -> pl.LazyFrame:
    url = _first_present(columns, ("url", "uri", "website"))
    return event.group_by(KEY_COLUMNS).agg(
        pl.len().alias("http_events"),
        _off_hours(settings).sum().alias("http_off_hours_events"),
        _contains_any(url, settings.suspicious_url_terms).sum().alias("http_watchlist_events"),
        _contains_any(url, settings.job_search_url_terms)
        .sum()
        .alias("http_job_search_events"),
        _contains_any(url, settings.data_leak_url_terms)
        .sum()
        .alias("http_data_leak_events"),
        _contains_any(url, settings.monitoring_tool_url_terms)
        .sum()
        .alias("http_monitoring_tool_events"),
    )


def _recipient_list(columns: set[str]) -> pl.Expr:
    recipient_columns = [
        column for column in ("to", "cc", "bcc") if column in columns
    ]
    if not recipient_columns:
        return pl.lit("").str.split(";")
    return pl.concat_str(
        [pl.col(column).cast(pl.String).fill_null("") for column in recipient_columns],
        separator=";",
    ).str.split(";")


def _external_recipient_count(
    columns: set[str], internal_domains: tuple[str, ...]
) -> pl.Expr:
    """Count recipient addresses whose domain is not present in LDAP."""
    if not internal_domains:
        return pl.lit(0)
    recipients = _recipient_list(columns)
    populated = pl.element().str.strip_chars().str.len_chars() > 0
    recipient_domain = pl.element().str.strip_chars().str.to_lowercase().str.extract(
        r"@(.+)$", 1
    )
    return (
        recipients.list.eval(
            (
                populated & ~recipient_domain.is_in(internal_domains).fill_null(False)
            ).cast(pl.Int64)
        )
        .list.sum()
        .fill_null(0)
    )


def _aggregate_email(
    event: pl.LazyFrame,
    columns: set[str],
    settings: FeatureSettings,
    internal_domains: tuple[str, ...],
) -> pl.LazyFrame:
    attachments = _first_present(columns, ("attachments", "attachment", "attachment_count"))
    size = _first_present(columns, ("size", "bytes", "message_size"))
    recipients = _recipient_list(columns)
    populated_recipient = pl.element().str.strip_chars().str.len_chars() > 0
    recipient_count = recipients.list.eval(
        populated_recipient.cast(pl.Int64)
    ).list.sum().fill_null(0)
    external_recipient_count = _external_recipient_count(columns, internal_domains)
    has_attachment = (
        pl.col(attachments).cast(pl.Int64, strict=False).fill_null(0) > 0
        if attachments
        else pl.lit(False)
    )
    return event.group_by(KEY_COLUMNS).agg(
        pl.len().alias("email_events"),
        has_attachment.sum().alias("email_with_attachment_count"),
        (
            pl.col(attachments).cast(pl.Int64, strict=False).fill_null(0).sum()
            if attachments
            else pl.lit(0)
        ).alias("email_attachment_total"),
        (pl.col(size).cast(pl.Float64, strict=False).fill_null(0).sum() if size else pl.lit(0)).alias(
            "email_total_bytes"
        ),
        recipient_count.sum().alias("email_recipient_count"),
        external_recipient_count.sum().alias("email_external_recipient_count"),
        (external_recipient_count > 0).sum().alias("email_external_events"),
        _off_hours(settings).sum().alias("email_off_hours_events"),
    )


def _aggregate_file(
    event: pl.LazyFrame, columns: set[str], settings: FeatureSettings
) -> pl.LazyFrame:
    filename = _first_present(columns, ("filename", "file_name", "path"))
    return event.group_by(KEY_COLUMNS).agg(
        pl.len().alias("file_events"),
        _off_hours(settings).sum().alias("file_off_hours_events"),
        _contains_any(filename, settings.sensitive_file_terms).sum().alias("sensitive_file_events"),
    )


AGGREGATORS = {
    "logon.csv": _aggregate_logon,
    "device.csv": _aggregate_device,
    "http.csv": _aggregate_http,
    "file.csv": _aggregate_file,
}


def _aggregate_usb_file_correlation(
    device_path: Path, file_path: Path, settings: FeatureSettings
) -> pl.DataFrame:
    """Count file events occurring while the same user's USB is connected on that PC."""
    device, device_columns = _event_base(device_path)
    file_events, file_columns = _event_base(file_path)
    device_pc = _first_present(device_columns, ("pc", "computer", "host"))
    file_pc = _first_present(file_columns, ("pc", "computer", "host"))
    device_activity = _first_present(device_columns, ("activity", "action", "event"))
    file_name = _first_present(file_columns, ("filename", "file_name", "path"))
    if not (device_pc and file_pc and device_activity):
        return pl.DataFrame(schema={column: pl.String for column in KEY_COLUMNS})

    device_state = (
        device.filter(
            pl.col(device_activity)
            .cast(pl.String)
            .str.contains(r"(?i)^(connect|disconnect)$")
        )
        .with_columns(
            pl.col(device_pc).cast(pl.String).alias("_pc"),
            pl.when(pl.col(device_activity).cast(pl.String).str.contains(r"(?i)^connect$"))
            .then(1)
            .otherwise(0)
            .alias("_usb_connected"),
        )
        .select(["user", "_pc", "timestamp", "_usb_connected"])
        .sort(["user", "_pc", "timestamp"])
    )
    files = (
        file_events.with_columns(pl.col(file_pc).cast(pl.String).alias("_pc"))
        .select(["user", "day", "_hour", "_weekday", "_pc", "timestamp", *( [file_name] if file_name else [] )])
        .sort(["user", "_pc", "timestamp"])
    )
    correlated = (
        files.join_asof(
            device_state,
            on="timestamp",
            by=["user", "_pc"],
            strategy="backward",
        )
        .filter(pl.col("_usb_connected") == 1)
        .group_by(KEY_COLUMNS)
        .agg(
            pl.len().alias("usb_connected_file_events"),
            _off_hours(settings).sum().alias("usb_connected_file_off_hours_events"),
            _contains_any(file_name, settings.sensitive_file_terms)
            .sum()
            .alias("usb_connected_sensitive_file_events"),
        )
        .collect(engine="streaming")
    )
    return correlated


def _aggregate_external_email_after_file(
    email_path: Path,
    file_path: Path,
    settings: FeatureSettings,
    internal_domains: tuple[str, ...],
) -> pl.DataFrame:
    """Identify external emails sent within an hour after a same-PC file event."""
    if not internal_domains:
        return pl.DataFrame(schema={column: pl.String for column in KEY_COLUMNS})
    emails, email_columns = _event_base(email_path)
    files, file_columns = _event_base(file_path)
    email_pc = _first_present(email_columns, ("pc", "computer", "host"))
    file_pc = _first_present(file_columns, ("pc", "computer", "host"))
    file_name = _first_present(file_columns, ("filename", "file_name", "path"))
    if not email_pc or not file_pc:
        return pl.DataFrame(schema={column: pl.String for column in KEY_COLUMNS})

    external_emails = (
        emails.filter(_external_recipient_count(email_columns, internal_domains) > 0)
        .with_columns(pl.col(email_pc).cast(pl.String).alias("_pc"))
        .select(["user", "day", "_pc", "timestamp"])
        .sort(["user", "_pc", "timestamp"])
    )
    file_events = (
        files.with_columns(
            pl.col(file_pc).cast(pl.String).alias("_pc"),
            pl.col("timestamp").alias("_file_timestamp"),
            _contains_any(file_name, settings.sensitive_file_terms).alias("_sensitive_file"),
        )
        .select(["user", "_pc", "timestamp", "_file_timestamp", "_sensitive_file"])
        .sort(["user", "_pc", "timestamp"])
    )
    return (
        external_emails.join_asof(
            file_events,
            on="timestamp",
            by=["user", "_pc"],
            strategy="backward",
        )
        .filter(
            pl.col("_file_timestamp").is_not_null()
            & ((pl.col("timestamp") - pl.col("_file_timestamp")) <= pl.duration(minutes=60))
        )
        .group_by(KEY_COLUMNS)
        .agg(
            pl.len().alias("external_email_after_file_events"),
            pl.col("_sensitive_file")
            .sum()
            .alias("external_email_after_sensitive_file_events"),
        )
        .collect(engine="streaming")
    )


def build_user_day_features(raw_dir: Path, settings: FeatureSettings) -> pl.DataFrame:
    """Aggregate each required log into a user-day feature table."""
    manifest = inspect_dataset(raw_dir)
    if not manifest.is_usable:
        raise ValueError(
            "CERT dataset validation failed. Run `ueba validate-data` for the exact problem."
        )

    organization = load_organization_context(raw_dir / "LDAP")
    aggregates: list[pl.DataFrame] = []
    for filename in EVENT_FILES:
        event, columns = _event_base(raw_dir / filename)
        aggregate = (
            _aggregate_email(event, columns, settings, organization.internal_domains)
            if filename == "email.csv"
            else AGGREGATORS[filename](event, columns, settings)
        ).collect(engine="streaming")
        aggregates.append(aggregate)
    aggregates.append(
        _aggregate_usb_file_correlation(
            raw_dir / "device.csv", raw_dir / "file.csv", settings
        )
    )
    aggregates.append(
        _aggregate_external_email_after_file(
            raw_dir / "email.csv",
            raw_dir / "file.csv",
            settings,
            organization.internal_domains,
        )
    )

    # Stack sparse event-specific aggregates, then combine all rows belonging to one user-day.
    combined = (
        pl.concat(aggregates, how="diagonal_relaxed")
        .group_by(KEY_COLUMNS)
        .agg(pl.exclude(KEY_COLUMNS).sum())
        .fill_null(0)
        .sort(KEY_COLUMNS)
    )
    all_users = combined.select("user").unique()
    all_days = pl.date_range(
        combined["day"].min(),
        combined["day"].max(),
        interval="1d",
        eager=True,
    ).to_frame("day")
    user_days = (
        all_users.join(all_days, how="cross")
        .join(combined, on=KEY_COLUMNS, how="left")
        .fill_null(0)
        .sort(KEY_COLUMNS)
    )
    return attach_organization_context(user_days, organization)
