"""Catalog settings for a quiesce operation (TASK-1233): validation and scope.

An operation with ``quiesce_enabled`` holds its scope's agent turns and waits
for them to drain before its action runs (:mod:`.quiesce`). The settings are
validated with the rest of the catalog row and resolved once, into the
approval snapshot, so the approved scope is the one the driver holds.

``quiesce_scope_json`` forms (null means every agent bot)::

    {"all_agents": true}
    {"backends": ["claude-code", ...]}
    {"bots": ["byte", ...]}
    {"backend_from_arg": "<declared string enum arg>",
     "map": {"<every enum value>": "<agent backend>", ...}}

The last form scopes the hold to the backend the action touches, e.g. the
bridge a restart names. Its map must cover every value the argument accepts,
so no approved invocation can resolve to "no scope".
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from agent_bridge.run_contract import QUIESCE_TIMEOUT_MAX_S

from ..service.turn_gates import GateScope
from .models import EXECUTOR_DOCKER

_STATIC_KEYS = ("all_agents", "backends", "bots")
_FROM_ARG_KEYS = {"backend_from_arg", "map"}


def _names(value: Any, field: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or not all(isinstance(v, str) and v.strip() for v in value)):
        raise ValueError(f"quiesce scope {field} must be a non-empty list of names")
    return value


def _parse_scope(raw: str | None) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("quiesce_scope_json must be JSON") from exc
    if not isinstance(data, Mapping):
        raise ValueError("quiesce_scope_json must be an object or null")
    return data


def validate_quiesce_settings(op: Any, schema: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``op``'s quiesce columns are coherent.

    Checked whether or not quiesce is enabled, so a row can't be enabled
    later onto settings nobody validated.
    """
    if type(op.quiesce_enabled) is not bool:
        raise ValueError("quiesce_enabled must be boolean")
    timeout = op.quiesce_timeout_seconds
    if type(timeout) is not int or not 1 <= timeout <= QUIESCE_TIMEOUT_MAX_S:
        raise ValueError(f"quiesce_timeout_seconds must be an integer in 1..{QUIESCE_TIMEOUT_MAX_S}")
    if op.quiesce_enabled and op.executor_kind != EXECUTOR_DOCKER:
        raise ValueError("quiesce requires executor_kind 'docker'")
    data = _parse_scope(op.quiesce_scope_json)
    if data is None:
        return
    if set(data) == _FROM_ARG_KEYS:
        arg = data["backend_from_arg"]
        prop = schema.get("properties", {}).get(arg) if isinstance(arg, str) else None
        if not isinstance(prop, Mapping) or prop.get("type") != "string" or not prop.get("enum"):
            raise ValueError("quiesce backend_from_arg must reference a declared string enum argument")
        if arg not in schema.get("required", ()):
            raise ValueError("quiesce backend_from_arg must reference a required argument")
        mapping = data["map"]
        if (not isinstance(mapping, Mapping) or set(mapping) != set(prop["enum"])
                or not all(isinstance(v, str) and v.strip() for v in mapping.values())):
            raise ValueError("quiesce map must name a backend for exactly every value the argument accepts")
        return
    if len(data) != 1 or next(iter(data)) not in _STATIC_KEYS:
        raise ValueError("quiesce scope must be one of all_agents, backends, bots or backend_from_arg+map")
    if "all_agents" in data:
        if data["all_agents"] is not True:
            raise ValueError("quiesce scope all_agents must be true")
        return
    key = next(iter(data))
    _names(data[key], key)


def resolve_scope(scope_json: str | None, args: Mapping[str, Any]) -> GateScope:
    """The gate scope one invocation holds, from validated settings and args."""
    data = _parse_scope(scope_json)
    if data is None or data.get("all_agents"):
        return GateScope.every_agent()
    if "backend_from_arg" in data:
        value = args.get(data["backend_from_arg"])
        backend = data["map"].get(value) if isinstance(value, str) else None
        if not backend:
            raise ValueError(f"quiesce scope has no backend for {data['backend_from_arg']}={value!r}")
        return GateScope.on_backends(backend)
    if "backends" in data:
        return GateScope.on_backends(*_names(data["backends"], "backends"))
    return GateScope.for_bots(*_names(data["bots"], "bots"))


def quiesce_binding(operation: Mapping[str, Any], args: Mapping[str, Any]) -> dict[str, Any] | None:
    """The snapshot's ``quiesce`` entry for an operation (``to_api`` dict) and
    resolved args; ``None`` when the operation does not quiesce."""
    if not operation.get("quiesce_enabled"):
        return None
    scope = resolve_scope(operation.get("quiesce_scope_json"), args)
    return {"enabled": True, "timeout_seconds": int(operation["quiesce_timeout_seconds"]),
            "scope": json.loads(scope.to_json())}


__all__ = ["quiesce_binding", "resolve_scope", "validate_quiesce_settings"]
