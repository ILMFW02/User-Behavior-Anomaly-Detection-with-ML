"""Time-aware organisation context derived from CERT LDAP snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class OrganizationContext:
    """LDAP snapshots and internal email domains available in the dataset."""

    snapshots: pl.DataFrame
    internal_domains: tuple[str, ...]


def load_organization_context(ldap_dir: Path) -> OrganizationContext:
    """Load monthly LDAP snapshots without carrying employee names into ML data."""
    if not ldap_dir.is_dir():
        return OrganizationContext(
            snapshots=pl.DataFrame(
                schema={
                    "user": pl.String,
                    "snapshot_day": pl.Date,
                    "role": pl.String,
                    "department": pl.String,
                    "team": pl.String,
                }
            ),
            internal_domains=(),
        )

    snapshots: list[pl.DataFrame] = []
    domains: set[str] = set()
    for path in sorted(ldap_dir.glob("*.csv")):
        try:
            snapshot_day = date.fromisoformat(f"{path.stem}-01")
        except ValueError:
            continue
        ldap = pl.read_csv(path)
        required = {"user_id", "email", "role", "department", "team"}
        if not required.issubset(ldap.columns):
            continue
        normalized_email = ldap["email"].cast(pl.String).str.to_lowercase()
        domains.update(
            domain
            for domain in normalized_email.str.extract(r"@(.+)$", 1).drop_nulls().to_list()
        )
        snapshots.append(
            ldap.select(
                pl.col("user_id").cast(pl.String).alias("user"),
                pl.lit(snapshot_day).alias("snapshot_day"),
                pl.col("role").cast(pl.String),
                pl.col("department").cast(pl.String),
                pl.col("team").cast(pl.String),
            )
        )

    if not snapshots:
        raise ValueError(f"No compatible LDAP snapshot found in {ldap_dir}.")
    return OrganizationContext(
        snapshots=pl.concat(snapshots).sort(["user", "snapshot_day"]),
        internal_domains=tuple(sorted(domains)),
    )


def attach_organization_context(
    user_days: pl.DataFrame, organization: OrganizationContext
) -> pl.DataFrame:
    """Attach the most recent LDAP context available on each user-day."""
    if organization.snapshots.is_empty():
        return user_days.with_columns(
            pl.lit(None, dtype=pl.String).alias("role"),
            pl.lit(None, dtype=pl.String).alias("department"),
            pl.lit(None, dtype=pl.String).alias("team"),
            pl.lit(None, dtype=pl.String).alias("org_peer_key"),
        )

    joined = user_days.sort(["user", "day"]).join_asof(
        organization.snapshots,
        left_on="day",
        right_on="snapshot_day",
        by="user",
        strategy="backward",
    )
    return joined.with_columns(
        pl.concat_str(
            [pl.col("role").fill_null("unknown"), pl.col("department").fill_null("unknown")],
            separator=" | ",
        ).alias("org_peer_key")
    )
