"""Post-restart health probe for plain container actions (TASK-1697). STANDARD LIBRARY ONLY.

A plain ``restart``/``start`` used to succeed as soon as Docker accepted the
call, even when the service never came back (the 10-29 dev API outage). A spec
may now name the same in-container health check the service-config kinds use
(``health_url``/``health_field``/``health_value``/``health_timeout_seconds``);
the worker then polls it after the action and fails the job when the service
isn't healthy in time. The restart itself happened, so that failure is a known
outcome, never ``side_effect_unknown``.

Imported by the app (validation, worker capability check) and copied into the
ops worker image, which carries :data:`WORKER_CAPABILITY_LABEL`.
"""
from __future__ import annotations

import json
import re
import struct

WORKER_CAPABILITY_LABEL = "llm-bawt.ops.restart-health"
HEALTH_ACTIONS = frozenset({"restart", "start"})
HEALTH_KEYS = frozenset({"health_url", "health_field", "health_value", "health_timeout_seconds"})
_URL = re.compile(r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/[A-Za-z0-9._/-]{0,128}")
_FIELD = re.compile(r"[a-z_]{1,32}")
_VALUE = re.compile(r"[A-Za-z0-9._-]{1,32}")
POLL_SECONDS = 2


class RestartUnhealthy(RuntimeError):
    """The action ran; the service did not become healthy. A known outcome."""

    outcome_known = True


def has_health(spec: dict) -> bool:
    """A plain restart/start that asks for the probe. Service-config kinds carry
    their own health_* fields and their own worker capability; never these."""
    return spec.get("action") in HEALTH_ACTIONS and bool(HEALTH_KEYS & set(spec))


def validate_health(spec: dict) -> None:
    """All four keys or none, only on restart/start, in the service-config grammar."""
    present = HEALTH_KEYS & set(spec)
    if not present:
        return
    if spec.get("action") not in HEALTH_ACTIONS:
        raise ValueError("health_* fields apply only to restart/start")
    if present != HEALTH_KEYS:
        raise ValueError("health_url, health_field, health_value and health_timeout_seconds go together")
    if not isinstance(spec["health_url"], str) or not _URL.fullmatch(spec["health_url"]):
        raise ValueError("health_url must be an in-container http://127.0.0.1:<port>/<path>")
    if not isinstance(spec["health_field"], str) or not _FIELD.fullmatch(spec["health_field"]):
        raise ValueError("health_field must be a short lowercase name")
    value = spec["health_value"]
    if value is not True and not (isinstance(value, str) and _VALUE.fullmatch(value)):
        raise ValueError("health_value must be a short token or true")
    timeout = spec["health_timeout_seconds"]
    if type(timeout) is not int or not 10 <= timeout <= 600:
        raise ValueError("health_timeout_seconds must be an integer in 10..600")


def _demux_stdout(raw: bytes) -> str:
    out, i = bytearray(), 0
    while i + 8 <= len(raw):
        stream, size = raw[i], struct.unpack(">I", raw[i + 4:i + 8])[0]
        if stream == 1:
            out += raw[i + 8:i + 8 + size]
        i += 8 + size
    return out.decode(errors="replace")


def _probe_once(api, container_id: str, spec: dict) -> str | None:
    """None when healthy, else a short reason. Never echoes the response body."""
    state = api.request("GET", f"/containers/{container_id}/json").get("State") or {}
    if not state.get("Running"):
        return f"container not running (status {state.get('Status')})"
    try:
        exec_id = api.request("POST", f"/containers/{container_id}/exec", body={
            "AttachStdout": True, "AttachStderr": True,
            "Cmd": ["curl", "-fsS", "--max-time", "5", spec["health_url"]]})["Id"]
        raw = api.request("POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, raw=True)
        code = api.request("GET", f"/exec/{exec_id}/json").get("ExitCode")
    except RuntimeError as exc:
        return f"health exec failed: {str(exc)[:200]}"
    if code != 0:
        return f"health request exited {code}"
    try:
        payload = json.loads(_demux_stdout(raw))
    except ValueError:
        return "health response is not JSON"
    field, value = spec["health_field"], spec["health_value"]
    found = payload.get(field) if isinstance(payload, dict) else None
    # JSON true matches by identity: 1 and "true" are not true.
    if found is True if value is True else (type(found) is str and found == value):
        return None
    return f"health {field} did not match expected {value!r}"


def wait_healthy(api, container_id: str, spec: dict, *, clock, sleep) -> str:
    """Poll until healthy or ``health_timeout_seconds``; raise RestartUnhealthy."""
    deadline = clock() + spec["health_timeout_seconds"]
    while True:
        reason = _probe_once(api, container_id, spec)
        if reason is None:
            return f"health {spec['health_field']} matched"
        if clock() >= deadline:
            raise RestartUnhealthy(
                f"service not healthy within {spec['health_timeout_seconds']}s after {spec['action']}: {reason}")
        sleep(POLL_SECONDS)
