"""Run manifests for deterministic, auditable UEBA experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Calculate a full SHA-256 digest; call deliberately for large raw inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_record(path: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    """Record a file identity without forcing expensive hashing of multi-GB log sources."""
    resolved = path.resolve()
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }
    if hash_contents:
        record["sha256"] = file_sha256(resolved)
    return record


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_manifest(
    report_dir: Path,
    *,
    command: str,
    config_path: Path,
    inputs: list[Path],
    seed: int,
    hash_inputs: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write immutable run metadata beside metrics, models, and scored results."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = report_dir / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "seed": seed,
        "git_commit": _git_commit(config_path.parent),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            package: version(package)
            for package in ("numpy", "polars", "scikit-learn")
        },
        "config": input_record(config_path, hash_contents=True),
        "inputs": [input_record(path, hash_contents=hash_inputs) for path in inputs if path.is_file()],
        "extra": extra or {},
    }
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
