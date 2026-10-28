"""The cutover probe line: one closed JSON schema (TASK-1476 §3). STANDARD LIBRARY ONLY.

The cutover plan verifies credentials by running a probe inside a target
container (a fixed-argv ``docker exec``) and reading one line of its stdout.
The ops worker has no network and no DB, so the probe is the only thing that
talks to an upstream; this module is the whole contract between the two:

* the app runs ``python -m llm_bawt.service.cutover_probe <id>``;
* the claude-code bridge runs ``python -m claude_code_bridge.cutover_probe broker``;
* each prints exactly one :func:`emit` line, and the worker reads it with
  :func:`parse_line`. Anything else is unverifiable, never a guess.

A line is ``{"probe", "ok", "status", "code"}`` and nothing else: no value,
no body, no message. Only the ``records`` probe adds a ``records`` key, and
only booleans, and only when that probe succeeded: for each of
:data:`RECORD_KEYS`, whether its stored record exists (``configured``) and is
connected and readable (``connected``).

This file is loaded three ways, so it imports nothing outside the standard
library and nothing of llm_bawt: as ``llm_bawt.ops.probe_contract`` in the
app, by bare module name in the standalone ops worker, and by file path in
the claude-code bridge (which can't import the llm_bawt package).
"""
from __future__ import annotations

import json
from typing import Any

#: Every probe the cutover plan may run.
PROBE_IDS: frozenset[str] = frozenset({
    "records",
    "crawl4ai_bearer",
    "broker",
    "anthropic_models",
    "zai_completion",
    "kimi_completion",
    "brave_search",
    "newsapi_search",
    "s3_head",
    "sonarr_status",
    "radarr_status",
    "sabnzbd_status",
    "web_fetch",
})

OK = "ok"
NOT_CONFIGURED = "not_configured"      # no usable stored record / token for this probe
UNAUTHORIZED = "unauthorized"          # the upstream rejected the credential
HTTP_ERROR = "http_error"              # any other non-2xx answer
UNREACHABLE = "unreachable"            # transport failure: DNS, refused, TLS
TIMEOUT = "timeout"
UNEXPECTED_RESPONSE = "unexpected_response"  # 2xx, but not the answer that proves it works
ERROR = "error"                        # the probe itself failed (bad argv, store down, bug)

#: Closed outcome codes. ``ok`` is true exactly when the code is :data:`OK`.
CODES: frozenset[str] = frozenset({
    OK, NOT_CONFIGURED, UNAUTHORIZED, HTTP_ERROR, UNREACHABLE, TIMEOUT, UNEXPECTED_RESPONSE, ERROR,
})

#: The probe that proves each env_import destination works from its stored
#: record. Keys equal ``{spec.destination for spec in llm_bawt.env_import.SPECS}``
#: (pinned by a test; this file can't import SPECS).
DESTINATION_PROBES: dict[str, str] = {
    "zai": "zai_completion",
    "kimi-coding": "kimi_completion",
    "brave": "brave_search",
    "newsapi": "newsapi_search",
    "object-store": "s3_head",
    "sonarr": "sonarr_status",
    "radarr": "radarr_status",
    "sabnzbd": "sabnzbd_status",
}

#: The stored records the ``records`` probe reports: every env_import
#: destination plus the two Anthropic credentials.
RECORD_KEYS: frozenset[str] = frozenset(DESTINATION_PROBES) | {"anthropic-api", "claude"}

RECORDS_PROBE = "records"
_BASE_KEYS = ("probe", "ok", "status", "code")
_RECORD_FIELDS = frozenset({"configured", "connected"})


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_status(value: Any) -> bool:
    """``None`` or an HTTP status (100-599). bool is an int subclass: never a status."""
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599)


def _check_records(records: Any) -> None:
    if not isinstance(records, dict) or set(records) != RECORD_KEYS:
        raise ValueError("records: keys must be exactly RECORD_KEYS")
    for state in records.values():
        if not isinstance(state, dict) or set(state) != _RECORD_FIELDS:
            raise ValueError("records: each entry is exactly {configured, connected}")
        if not (_is_bool(state["configured"]) and _is_bool(state["connected"])):
            raise ValueError("records: configured/connected must be booleans")
        if state["connected"] and not state["configured"]:
            raise ValueError("records: connected implies configured")


def validate(line: Any) -> dict[str, Any]:
    """``line`` itself when it is a schema-exact probe object; ValueError otherwise."""
    if not isinstance(line, dict):
        raise ValueError("probe line must be a JSON object")
    probe = line.get("probe")
    # Only a successful records probe carries records: a failed one has none to trust.
    wanted = {*_BASE_KEYS, "records"} if probe == RECORDS_PROBE and line.get("ok") is True else set(_BASE_KEYS)
    if set(line) != wanted:
        raise ValueError("probe line keys are off-schema")
    if not isinstance(probe, str) or probe not in PROBE_IDS:
        raise ValueError("unknown probe id")
    if not _is_bool(line["ok"]):
        raise ValueError("ok must be a boolean")
    if not is_status(line["status"]):
        raise ValueError("status must be an HTTP status int or null")
    code = line["code"]
    if not isinstance(code, str) or code not in CODES:
        raise ValueError("unknown code")
    if line["ok"] != (code == OK):
        raise ValueError("ok must be true exactly when code is ok")
    if "records" in line:
        _check_records(line["records"])
    return line


def emit(probe: str, ok: bool, status: int | None, code: str,
         records: dict[str, dict[str, bool]] | None = None) -> str:
    """The one stdout line for a probe result. Raises ValueError rather than emit off-schema."""
    line: dict[str, Any] = {"probe": probe, "ok": ok, "status": status, "code": code}
    if records is not None:
        line["records"] = dict(sorted(records.items())) if isinstance(records, dict) else records
    validate(line)
    return json.dumps(line, separators=(",", ":"))


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise ValueError(f"non-JSON constant {name}")


def parse_line(stdout: str) -> dict[str, Any]:
    """The probe object on the FIRST line of ``stdout``; the rest is discarded.

    Raises ValueError on anything that is not exactly the schema: not one JSON
    object, duplicate keys, NaN/Infinity, an extra or missing key, a wrong type.
    """
    if not isinstance(stdout, str):
        raise ValueError("stdout must be text")
    first = stdout.split("\n", 1)[0].strip()
    if not first:
        raise ValueError("no probe line")
    try:
        line = json.loads(first, object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except json.JSONDecodeError:
        raise ValueError("probe line is not JSON") from None
    return validate(line)


__all__ = [
    "CODES",
    "DESTINATION_PROBES",
    "ERROR",
    "HTTP_ERROR",
    "NOT_CONFIGURED",
    "OK",
    "PROBE_IDS",
    "RECORDS_PROBE",
    "RECORD_KEYS",
    "TIMEOUT",
    "UNAUTHORIZED",
    "UNEXPECTED_RESPONSE",
    "UNREACHABLE",
    "emit",
    "is_status",
    "parse_line",
    "validate",
]
