"""The ``X-Bridge-Token`` gate on the app's internal bridge routes (TASK-1477).

Bridges call the app's internal broker routes (provider tokens, the ChatGPT
bundle, GitHub git credentials). Two gates protect them, each with its own
token:

* ``broker``: every provider token/bundle broker route.
* ``git_credential``: the GitHub git-credential route the bridge's
  ``credential.helper`` calls.

Each gate's token is **derived** from the one bootstrap root
(``crypto.derive_key(<label>)``, hex), never stored. The app writes it to a
shared-volume file that the bridges read (``*_TOKEN_FILE`` on their side);
see :func:`export_tokens`. Legacy tenants still carry a per-gate env secret
(``BRIDGE_CLAUDE_TOKEN_SECRET`` / ``BRIDGE_GIT_CREDENTIAL_TOKEN``); until the
removal release both values are accepted.

Every request is classified as ``derived``, ``legacy``, ``missing`` or
``bad``:

* ``bad`` (a header matching nothing) is always refused with 401.
* ``missing`` is refused only while the gate is enforced.

Enforcement is the global runtime setting ``bridge_gate_enforced:<gate>``
(bool), read through a short cache so flipping it needs no restart. With no
setting row the default reproduces today's behaviour: enforced when the
legacy env secret is set (or bootstrap marked a fresh tenant with
``BRIDGE_GATE_DEFAULT_ENFORCED=1``), open otherwise. Counts of each class feed
``/v1/health/checks``; token values are never logged or reported.
"""

from __future__ import annotations

import enum
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

HEADER = "X-Bridge-Token"
DEFAULT_ENFORCED_ENV = "BRIDGE_GATE_DEFAULT_ENFORCED"
EXPORT_UID_ENV = "BRIDGE_TOKEN_EXPORT_UID"
SETTING_PREFIX = "bridge_gate_enforced:"
SETTINGS_TTL_SECONDS = 5.0
# A class "moved recently" when it was counted within this window.
RECENT_SECONDS = 3600.0


@dataclass(frozen=True)
class Gate:
    slug: str
    label: str  # derive_key label; versioned so a later scheme can't collide
    legacy_env: str
    export_env: str

    @property
    def setting_key(self) -> str:
        return SETTING_PREFIX + self.slug


BROKER = Gate("broker", "bridge-broker-token/v1",
              "BRIDGE_CLAUDE_TOKEN_SECRET", "BRIDGE_BROKER_TOKEN_EXPORT_FILE")
GIT_CREDENTIAL = Gate("git_credential", "bridge-git-credential-token/v1",
                      "BRIDGE_GIT_CREDENTIAL_TOKEN", "BRIDGE_GIT_CREDENTIAL_TOKEN_EXPORT_FILE")
GATES: tuple[Gate, ...] = (BROKER, GIT_CREDENTIAL)


class Outcome(enum.Enum):
    DERIVED = "derived"
    LEGACY = "legacy"
    MISSING = "missing"
    BAD = "bad"


class State(enum.Enum):
    ENFORCED = "enforced"
    OPEN_DERIVED_SEEN = "open_derived_seen"  # a bridge can authenticate; the gate is still open
    OPEN_UNAUTHENTICATED = "open_unauthenticated"  # network-only traffic, no derived yet
    IDLE = "idle"


def _derive(label: str) -> str:
    from .providers import crypto

    return crypto.derive_key(label).hex()


def _matches(presented: bytes, expected: str | None) -> bool:
    return bool(expected) and hmac.compare_digest(presented, expected.encode("utf-8"))


class BridgeGates:
    """Classifier, enforcement, counters and export for every gate.

    ``settings`` returns the global runtime settings mapping (or raises; a
    failed read falls back to the default rule). ``derive`` maps a label to
    its hex token. ``clock`` is monotonic seconds.
    """

    def __init__(
        self,
        settings: Callable[[], Mapping[str, Any]],
        *,
        environ: Mapping[str, str] | None = None,
        derive: Callable[[str], str] = _derive,
        clock: Callable[[], float] = time.monotonic,
        settings_ttl: float = SETTINGS_TTL_SECONDS,
    ):
        self._settings = settings
        self._environ = os.environ if environ is None else environ
        self._derive = derive
        self._clock = clock
        self._ttl = settings_ttl
        self._lock = threading.Lock()
        self._derived: dict[str, str] = {}
        self._underivable_reported: set[str] = set()
        self._cached: Mapping[str, Any] | None = None
        self._cached_at: float | None = None
        self._settings_failing = False
        self._counts = {g.slug: {o: 0 for o in Outcome} for g in GATES}
        self._last = {g.slug: {o: None for o in Outcome} for g in GATES}

    # --- tokens ---------------------------------------------------------
    def derived_token(self, gate: Gate) -> str | None:
        """The derived token, or None while the root can't derive one."""
        token = self._derived.get(gate.slug)
        if token is not None:
            return token
        try:
            token = self._derive(gate.label)
        except Exception as error:  # noqa: BLE001 - any key problem means "no derived token"
            if gate.slug not in self._underivable_reported:
                self._underivable_reported.add(gate.slug)
                logger.warning("No derived %s bridge token: the master key can't derive one (%s)",
                               gate.slug, type(error).__name__)
            return None
        # The root is fixed for the process lifetime (crypto caches it too).
        self._derived[gate.slug] = token
        return token

    def _legacy(self, gate: Gate) -> str | None:
        return (self._environ.get(gate.legacy_env) or "").strip() or None

    # --- enforcement ----------------------------------------------------
    def _global_settings(self) -> Mapping[str, Any] | None:
        now = self._clock()
        with self._lock:
            if self._cached_at is not None and now - self._cached_at < self._ttl:
                return self._cached
        try:
            values: Mapping[str, Any] | None = self._settings()
            failing = False
        except Exception as error:  # noqa: BLE001 - fall back to the default rule
            values, failing = None, True
            if not self._settings_failing:
                logger.warning("Bridge gate settings unreadable (%s); using the default rule",
                               type(error).__name__)
        with self._lock:
            self._cached, self._cached_at, self._settings_failing = values, now, failing
        return values

    def default_enforced(self, gate: Gate) -> bool:
        flag = (self._environ.get(DEFAULT_ENFORCED_ENV) or "").strip() == "1"
        return flag or self._legacy(gate) is not None

    def enforced(self, gate: Gate) -> bool:
        value = (self._global_settings() or {}).get(gate.setting_key)
        if isinstance(value, bool):
            return value
        return self.default_enforced(gate)

    # --- checking -------------------------------------------------------
    def classify(self, gate: Gate, header: str | None) -> Outcome:
        # Only an absent header is MISSING. A present-but-empty one matches
        # nothing, so it is BAD and refused even while the gate is open.
        if header is None:
            return Outcome.MISSING
        if not header:
            return Outcome.BAD
        presented = header.encode("utf-8")
        # Compare against both before deciding, so timing doesn't say which
        # value a near miss resembled.
        derived = _matches(presented, self.derived_token(gate))
        legacy = _matches(presented, self._legacy(gate))
        if derived:
            return Outcome.DERIVED
        if legacy:
            return Outcome.LEGACY
        return Outcome.BAD

    def check(self, gate: Gate, header: str | None) -> Outcome:
        """Classify, count and refuse (401) when the request may not pass."""
        outcome = self.classify(gate, header)
        self._count(gate, outcome)
        if outcome is Outcome.BAD or (outcome is Outcome.MISSING and self.enforced(gate)):
            raise HTTPException(status_code=401, detail="bad bridge token")
        return outcome

    def _count(self, gate: Gate, outcome: Outcome) -> None:
        now = self._clock()
        with self._lock:
            self._counts[gate.slug][outcome] += 1
            self._last[gate.slug][outcome] = now

    # --- health ---------------------------------------------------------
    def counts(self, gate: Gate) -> dict[str, int]:
        with self._lock:
            return {o.value: n for o, n in self._counts[gate.slug].items()}

    def state(self, gate: Gate) -> State:
        if self.enforced(gate):
            return State.ENFORCED
        now = self._clock()

        def recent(outcome: Outcome) -> bool:
            seen = self._last[gate.slug][outcome]
            return seen is not None and now - seen <= RECENT_SECONDS

        with self._lock:
            if recent(Outcome.DERIVED):
                return State.OPEN_DERIVED_SEEN
            if recent(Outcome.MISSING) and self._counts[gate.slug][Outcome.DERIVED] == 0:
                return State.OPEN_UNAUTHENTICATED
        return State.IDLE

    def health_checks(self) -> list[dict]:
        from .health_checks import DEGRADED, OK, _check

        rows = []
        for gate in GATES:
            state = self.state(gate)
            if state is State.OPEN_DERIVED_SEEN:
                status, reason = DEGRADED, (
                    f"{gate.slug} bridge gate is open while bridges send the derived token; "
                    f"set {gate.setting_key}=true")
            elif state is State.OPEN_UNAUTHENTICATED:
                # Warning, not degraded: this is a network-only install's normal state
                # until its bridges send the derived token.
                status, reason = OK, (
                    f"warning: {gate.slug} bridge gate is open and bridges send no token")
            else:
                status, reason = OK, None
            row = _check(f"bridge_gate.{gate.slug}", status, reason)
            row.update(state=state.value, enforced=state is State.ENFORCED, counts=self.counts(gate))
            rows.append(row)
        return rows

    # --- export ---------------------------------------------------------
    def export_tokens(self) -> dict[str, str]:
        """Write each gate's derived token to its export file (app startup).

        Unset path = disabled. With no derived token the stale file is removed,
        so a bridge never sends a token the app no longer holds. Never raises;
        never logs a value. Returns ``{gate: result}`` for the log line.
        """
        from llm_bawt.bawthub_auth.token_export import publish

        results: dict[str, str] = {}
        for gate in GATES:
            raw = (self._environ.get(gate.export_env) or "").strip()
            if not raw:
                results[gate.slug] = "disabled"
                continue
            try:
                result = publish(Path(raw), self.derived_token(gate), self._environ,
                                 uid_env=EXPORT_UID_ENV)
                results[gate.slug] = result.value
            except (OSError, ValueError) as error:
                logger.error("Could not export the %s bridge token to %s (%s: %s)",
                             gate.slug, raw, type(error).__name__, error)
                results[gate.slug] = "failed"
        return results


def seed_enforced_defaults(engine, environ: Mapping[str, str] | None = None) -> list[str]:
    """Bootstrap marked this tenant fresh: record enforcement in the DB.

    Runs at startup only when ``BRIDGE_GATE_DEFAULT_ENFORCED=1``. Create-if-
    absent per gate, so an operator's existing row (including a ``false``
    rollback) is never overwritten, and from then on the row is the source of
    truth even if the env marker is lost. Returns the keys it inserted.
    """
    environ = os.environ if environ is None else environ
    if (environ.get(DEFAULT_ENFORCED_ENV) or "").strip() != "1" or engine is None:
        return []
    from sqlalchemy.exc import IntegrityError
    from sqlmodel import Session, select

    from llm_bawt.runtime_settings import RuntimeSetting

    inserted: list[str] = []
    for gate in GATES:
        with Session(engine) as session:
            exists = session.exec(select(RuntimeSetting.id).where(
                RuntimeSetting.scope_type == "global",
                RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == gate.setting_key,
            )).first()
            if exists is not None:
                continue
            session.add(RuntimeSetting(scope_type="global", scope_id="*",
                                       key=gate.setting_key, value_json=json.dumps(True)))
            try:
                session.commit()
            except IntegrityError:  # another process seeded it first
                session.rollback()
                continue
        inserted.append(gate.setting_key)
    if inserted:
        logger.info("Seeded bridge gate enforcement for a fresh tenant: %s", ", ".join(inserted))
    return inserted


# --- process-wide instance ----------------------------------------------
class _SettingsSource:
    """The runtime-settings store, attached by app startup (:func:`use_store`).

    Until then (or with no database) reads raise, and the gates fall back to
    the default rule, which is never weaker than before TASK-1477.
    """

    def __init__(self) -> None:
        self.store = None

    def __call__(self) -> Mapping[str, Any]:
        store = self.store
        if store is None or store.engine is None:
            raise RuntimeError("runtime settings store not attached")
        return store.get_scope_settings("global", "*")


_SOURCE = _SettingsSource()
_GATES = BridgeGates(_SOURCE)


def use_store(store) -> None:
    """App startup: read enforcement from this ``RuntimeSettingsStore``."""
    _SOURCE.store = store


def bridge_gates() -> BridgeGates:
    return _GATES


def check(request: Request, gate: Gate) -> Outcome:
    """Route guard: refuse (401) a request this gate must not let through."""
    return _GATES.check(gate, request.headers.get(HEADER))


def reject_bridge_token_for_preview(request: Request) -> None:
    """A broad production bridge grant must never authenticate a Preview request."""
    if request.headers.get(HEADER) is not None:
        raise HTTPException(status_code=403, detail="Preview broker does not accept bridge tokens")
