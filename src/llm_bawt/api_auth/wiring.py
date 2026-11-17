"""Attach API authentication to the FastAPI app (TASK-2380).

Builds the existing-credential checkers for :class:`ApiAuthMiddleware`, the
route-template lookup used for bounded observations, and the mode resolver
backed by the global runtime settings. Shipped with the mode ``off``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from .middleware import TURN_HEADER, ApiAuthMiddleware, Principal
from .mode import DENYLIST_KEY, ModeResolver

logger = logging.getLogger(__name__)


def _first(headers: dict[bytes, list[bytes]], name: bytes) -> str | None:
    values = headers.get(name)
    return values[0].decode("latin-1") if values else None


def turn_capability(headers) -> Principal | None:
    """A verified, unexpired turn capability identifies a bot (not an admin)."""
    token = _first(headers, TURN_HEADER)
    if not token:
        return None
    from ..task_turn_context import TaskTurnContextError, open_task_turn_context

    try:
        context = open_task_turn_context(token)
    except (TaskTurnContextError, Exception):
        return None
    return Principal(context.bot_id, "bot")


def service_bearer(headers) -> Principal | None:
    from ..bawthub_auth.service_token import TokenCheck, service_token_source

    if service_token_source().check(_first(headers, b"authorization")) is TokenCheck.OK:
        return Principal("bawthub-service", "service")
    return None


def bridge_token(headers) -> Principal | None:
    from ..service.bridge_gate import GATES, HEADER, Outcome, bridge_gates

    header = _first(headers, HEADER.lower().encode())
    if header is None:
        return None
    gates = bridge_gates()
    for gate in GATES:
        if gates.classify(gate, header) in (Outcome.DERIVED, Outcome.LEGACY):
            return Principal(f"bridge-{gate.slug}", "bridge")
    return None


CHECKERS = (turn_capability, service_bearer, bridge_token)


def route_template(app):
    """Map a scope to its route path template, so observations stay bounded."""
    from starlette.routing import Match

    def lookup(scope) -> str:
        for route in getattr(app, "routes", ()):
            try:
                match, _ = route.matches(scope)
            except Exception:
                continue
            if match is Match.FULL:
                return getattr(route, "path", "<route>")
        return "<unmatched>"

    return lookup


class _GlobalSettings:
    """Reads global runtime settings; raises until the store is attached."""

    def __init__(self) -> None:
        self.store = None

    def __call__(self) -> Mapping[str, Any]:
        store = self.store
        if store is None or getattr(store, "engine", None) is None:
            raise RuntimeError("runtime settings store not attached")
        # Read the rows directly: get_scope_settings() silently drops a row
        # whose JSON is invalid, which would turn a garbled enforced mode into
        # "absent" (the default). An undecodable auth row is a read failure,
        # so the resolver keeps its last-known-good mode (or stays unresolved).
        import json

        from sqlmodel import Session, select

        from ..runtime_settings import RuntimeSetting

        with Session(store.engine) as session:
            rows = session.exec(select(RuntimeSetting).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*")).all()
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[row.key] = json.loads(row.value_json)
            except (TypeError, ValueError):
                if _auth_key(row.key):
                    raise ValueError(f"runtime setting {row.key} is not valid JSON") from None
        return out


def _auth_key(key: str) -> bool:
    return key.endswith("_auth_mode") or key == DENYLIST_KEY


SETTINGS = _GlobalSettings()
RESOLVER = ModeResolver(SETTINGS)


def use_store(store) -> None:
    """App startup: resolve the mode from this ``RuntimeSettingsStore``."""
    SETTINGS.store = store


def install(app) -> None:
    app.add_middleware(ApiAuthMiddleware, resolver=RESOLVER, checkers=CHECKERS,
                       route_template=route_template(app))
