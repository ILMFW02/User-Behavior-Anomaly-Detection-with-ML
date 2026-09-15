"""CERT dataset discovery without reading whole CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

EVENT_FILES = ("logon.csv", "device.csv", "http.csv", "email.csv", "file.csv")
REQUIRED_EVENT_COLUMNS = {"user", "date"}


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    event_headers: dict[str, tuple[str, ...]]
    missing_event_files: tuple[str, ...]
    ldap_files: tuple[Path, ...]
    answer_files: tuple[Path, ...]

    @property
    def is_usable(self) -> bool:
        return not self.missing_event_files and all(
            REQUIRED_EVENT_COLUMNS.issubset(headers)
            for headers in self.event_headers.values()
        )


def _header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        return tuple(name.strip() for name in next(reader))


def inspect_dataset(root: Path) -> DatasetManifest:
    """Return expected files and headers, without loading their data rows."""
    root = root.resolve()
    event_headers = {
        filename: _header(root / filename)
        for filename in EVENT_FILES
        if (root / filename).is_file()
    }
    missing = tuple(filename for filename in EVENT_FILES if filename not in event_headers)
    ldap_dir = root / "LDAP"
    answers_dir = root / "answers"

    return DatasetManifest(
        root=root,
        event_headers=event_headers,
        missing_event_files=missing,
        ldap_files=tuple(sorted(ldap_dir.glob("*.csv"))) if ldap_dir.is_dir() else (),
        answer_files=tuple(sorted(answers_dir.rglob("*.csv"))) if answers_dir.is_dir() else (),
    )


def format_manifest(manifest: DatasetManifest) -> str:
    """Format a concise, copyable validation report."""
    lines = [f"Dataset root: {manifest.root}"]
    for filename in EVENT_FILES:
        headers = manifest.event_headers.get(filename)
        if headers is None:
            lines.append(f"[MISSING] {filename}")
            continue
        missing_columns = REQUIRED_EVENT_COLUMNS.difference(headers)
        status = "OK" if not missing_columns else f"MISSING COLUMNS: {sorted(missing_columns)}"
        lines.append(f"[{status}] {filename}: {', '.join(headers)}")

    lines.append(f"LDAP CSV files: {len(manifest.ldap_files)}")
    lines.append(f"Answer CSV files: {len(manifest.answer_files)}")
    lines.append(f"Dataset usable for feature extraction: {manifest.is_usable}")
    return "\n".join(lines)
