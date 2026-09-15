"""Normalize offline Wazuh JSON alerts for reproducible lab replays.

This module deliberately does not call the Wazuh API.  It accepts alerts exported from a
lab (usually ``alerts.json`` / JSON Lines) and produces a small, stable event contract that
an integration can write to its own queue, file, or feature-building process.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO, TypedDict

UNKNOWN = "unknown"


class CanonicalAlert(TypedDict):
    """Fields emitted by :func:`normalize_wazuh_alert`."""

    event_id: str
    user: str
    host: str
    timestamp: str
    event_type: str
    action: str
    object: str
    is_external: bool


class WazuhReplayError(ValueError):
    """Raised when an alert replay file is not valid Wazuh JSON."""


def _lookup(alert: Mapping[str, Any], path: str) -> Any | None:
    """Return a dotted path from nested or flattened Wazuh alert data."""
    if path in alert:
        return alert[path]
    current: Any = alert
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first(alert: Mapping[str, Any], *paths: str) -> str:
    for path in paths:
        value = _lookup(alert, path)
        if value is not None and str(value).strip():
            return str(value).strip()
    return UNKNOWN


def _groups(alert: Mapping[str, Any]) -> set[str]:
    groups = _lookup(alert, "rule.groups")
    if isinstance(groups, list):
        return {str(group).lower() for group in groups}
    if isinstance(groups, str):
        return {part.strip().lower() for part in groups.split(",") if part.strip()}
    return set()


def _event_kind(alert: Mapping[str, Any]) -> str:
    groups = _groups(alert)
    provider = _first(alert, "data.win.system.providerName", "win.system.providerName").lower()
    if "syscheck" in groups or isinstance(_lookup(alert, "data.syscheck"), Mapping):
        return "syscheck"
    if any("sysmon" in group for group in groups) or "sysmon" in provider:
        return "sysmon"
    if "audit" in groups or isinstance(_lookup(alert, "data.audit"), Mapping):
        return "auditd"
    if isinstance(_lookup(alert, "data.win.system"), Mapping) or "windows" in groups:
        return "windows"
    return "wazuh"


def _sysmon_action(alert: Mapping[str, Any]) -> str:
    event_id = _first(alert, "data.win.system.eventID", "win.system.eventID")
    return {
        "1": "process_create",
        "3": "network_connection",
        "11": "file_create",
        "12": "registry_create_delete",
        "13": "registry_value_set",
        "22": "dns_query",
    }.get(event_id, "event")


def _windows_action(alert: Mapping[str, Any]) -> str:
    event_id = _first(alert, "data.win.system.eventID", "win.system.eventID")
    return {
        "4624": "logon",
        "4625": "failed_logon",
        "4634": "logoff",
        "4648": "explicit_credential_logon",
        "4672": "privileged_logon",
        "4776": "credential_validation",
        "6416": "device_connect",
    }.get(event_id, "event")


def _action(alert: Mapping[str, Any], kind: str) -> str:
    if kind == "syscheck":
        value = _first(alert, "data.type", "data.event", "data.syscheck.event").lower()
        if value != UNKNOWN:
            for action in ("added", "modified", "deleted"):
                if action in value:
                    return action
        return "changed"
    if kind == "sysmon":
        return _sysmon_action(alert)
    if kind == "windows":
        return _windows_action(alert)
    if kind == "auditd":
        return _first(alert, "data.audit.type", "audit.type").lower()
    return "alert"


def _object(alert: Mapping[str, Any], kind: str) -> str:
    supplied = _first(alert, "data.ueba.object", "data.object")
    if supplied != UNKNOWN:
        return supplied
    if kind == "syscheck":
        return _first(alert, "data.path", "data.file", "data.syscheck.path", "syscheck.path")
    if kind == "sysmon":
        return _first(
            alert,
            "data.win.eventdata.TargetFilename",
            "data.win.eventdata.Image",
            "data.win.eventdata.DestinationIp",
        )
    if kind == "windows":
        return _first(
            alert,
            "data.win.eventdata.WorkstationName",
            "data.win.eventdata.IpAddress",
            "data.win.eventdata.ProcessName",
        )
    if kind == "auditd":
        return _first(alert, "data.audit.exe", "data.audit.path", "data.audit.key", "audit.exe")
    return _first(alert, "rule.description", "full_log")


def _is_public_address(value: str) -> bool:
    """Return whether *value* is a globally routable IP address."""
    candidate = value.strip().strip("[]")
    if ":" in candidate and candidate.count(":") == 1 and "." in candidate:
        candidate = candidate.rsplit(":", 1)[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return address.is_global


def _is_external(alert: Mapping[str, Any]) -> bool:
    ip_paths = (
        "data.srcip",
        "data.dstip",
        "data.audit.addr",
        "data.win.eventdata.IpAddress",
        "data.win.eventdata.SourceNetworkAddress",
        "data.win.eventdata.DestinationIp",
    )
    return any(_is_public_address(_first(alert, path)) for path in ip_paths)


def _fallback_event_id(alert: Mapping[str, Any]) -> str:
    encoded = json.dumps(alert, sort_keys=True, separators=(",", ":"), default=str)
    return f"wazuh:{hashlib.sha256(encoded.encode()).hexdigest()[:24]}"


def normalize_wazuh_alert(alert: Mapping[str, Any]) -> CanonicalAlert:
    """Map one Wazuh alert to the adapter's canonical, dependency-free schema.

    Missing source values are represented by ``"unknown"``.  If Wazuh did not assign an
    identifier, a deterministic hash of the alert is used, allowing replay deduplication.
    """
    kind = _event_kind(alert)
    supplied_type = _first(alert, "data.ueba.event_type", "data.event_type")
    supplied_action = _first(alert, "data.ueba.action", "data.action")
    action = supplied_action.lower() if supplied_action != UNKNOWN else _action(alert, kind)
    if supplied_type != UNKNOWN:
        event_type = supplied_type.lower()
    elif kind == "syscheck" or (kind == "sysmon" and action == "file_create"):
        event_type = "file"
    elif kind == "windows" and action == "device_connect":
        event_type = "device"
    elif kind == "windows" and action != "event":
        event_type = "windows_auth"
    else:
        event_type = kind
    event_id = _first(
        alert,
        "id",
        "_id",
        "data.id",
        "data.win.system.eventRecordID",
        "data.audit.sequence",
    )
    return {
        "event_id": _fallback_event_id(alert) if event_id == UNKNOWN else event_id,
        "user": _first(
            alert,
            "data.win.eventdata.TargetUserName",
            "data.win.eventdata.SubjectUserName",
            "data.win.eventdata.User",
            "data.audit.acct",
            "data.audit.user",
            "data.user",
            "user",
        ),
        "host": _first(
            alert,
            "agent.name",
            "data.win.system.computer",
            "data.hostname",
            "predecoder.hostname",
            "host.name",
        ),
        "timestamp": _first(alert, "timestamp", "@timestamp", "data.timestamp"),
        "event_type": event_type,
        "action": action,
        "object": _object(alert, kind),
        "is_external": _is_external(alert),
    }


def _records_from_stream(source: TextIO) -> Iterator[Mapping[str, Any]]:
    prefix = source.read(4096)
    if not prefix:
        return
    source.seek(0)
    if prefix.lstrip().startswith("["):
        try:
            records = json.load(source)
        except json.JSONDecodeError as error:
            raise WazuhReplayError(f"Invalid JSON array: {error}") from error
        if not isinstance(records, list):
            raise WazuhReplayError("A JSON replay must contain an array of alert objects.")
        for number, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise WazuhReplayError(f"Array item {number} is not an alert object.")
            yield record
        return

    for number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise WazuhReplayError(f"Invalid JSON on line {number}: {error}") from error
        if not isinstance(record, Mapping):
            raise WazuhReplayError(f"Line {number} is not an alert object.")
        yield record


def iter_wazuh_alerts(path: str | Path) -> Iterator[CanonicalAlert]:
    """Stream a JSON Lines export or JSON array export as canonical Wazuh events."""
    with Path(path).open(encoding="utf-8-sig") as source:
        for alert in _records_from_stream(source):
            yield normalize_wazuh_alert(alert)
