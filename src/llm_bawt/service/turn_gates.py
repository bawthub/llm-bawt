"""System gates: the hold/deny primitive shared by freeze and quiesce (TASK-1233).

One gate row per owner activation (``system_gates``), never overwritten. The
row id IS the generation. A gate is live while ``released_at IS NULL AND
expires_at > now()`` on the database clock, so the app and the ops reconciler
share one time source. Gates from several owners coexist; an owner releases
only its own.

``gate_blocks`` is the one predicate every admission edge asks. The caller
holds ``turn_admission.lock_target(conn, bot_id)`` and evaluates it in the
same transaction, so each check is linearized with ``activate_hold`` (which
takes the same per-bot locks before it inserts the gate): either the producer
commits first and the roster captures its row, or the gate commits first and
the producer sees it.

Scope is stored as a predicate and resolved at check time against the bot's
current profile (S3): a bot that moves onto a held backend is held, one that
moves off is released. Only agent bots are ever gated; chat bots (Mira among
them) never are. Gates are PostgreSQL-only, like the lock that linearizes them.

``turn_admission`` imports this module; never the reverse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Mapping

from sqlalchemy import text as sa_text

from ..bot_types import normalize_bot_type
from ..utils.schema import SchemaBootstrapGuard, ensure_columns, ensure_relations

HOLD = "hold"
DENY = "deny"
GATE_MODES = frozenset({HOLD, DENY})

OWNER_JOB = "job"
OWNER_OPERATOR = "operator"
OWNER_KINDS = frozenset({OWNER_JOB, OWNER_OPERATOR})

# The closed set of admission edges that ask a gate.
ACTOR_KINDS = frozenset({"reserve", "save_turn", "enqueue", "delivery_claim", "schedule", "steer"})
# Edges that may re-attach to a run already accepted before the gate (C1).
_REATTACH_ACTORS = frozenset({"delivery_claim", "save_turn"})

# Roster dispositions, assigned by ``activate_hold`` (B4).
REVOKED_UNSTARTED = "revoked_unstarted"  # reservation revoked; counts as drained
ACCEPTED = "accepted"  # reserved with execution evidence; drains on exit
RUNNING = "running"  # pending / streaming / cancelling
DISPOSITIONS = frozenset({REVOKED_UNSTARTED, ACCEPTED, RUNNING})

# Control states on a member row (Al's control lane, B3).
CONTROL_STATES = frozenset({
    "pending", "steering", "steered", "unsupported", "not_needed", "self", "unconfirmed", "cancelled",
})


class GateScopeError(ValueError):
    """A gate scope or spec that cannot be stored."""


@dataclass(frozen=True)
class GateScope:
    """Which agent bots a gate covers, as a predicate evaluated at check time.

    Exactly one of: every agent bot, agent bots on the named backends, or the
    named agent bots. Chat bots are never covered.
    """

    all_agents: bool = False
    backends: frozenset[str] = frozenset()
    bots: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if sum((bool(self.all_agents), bool(self.backends), bool(self.bots))) != 1:
            raise GateScopeError("a gate scope names exactly one of all_agents, backends or bots")

    @classmethod
    def every_agent(cls) -> "GateScope":
        return cls(all_agents=True)

    @classmethod
    def on_backends(cls, *backends: str) -> "GateScope":
        return cls(backends=frozenset(b.strip().lower() for b in backends if b and b.strip()))

    @classmethod
    def for_bots(cls, *bots: str) -> "GateScope":
        return cls(bots=frozenset(b.strip().lower() for b in bots if b and b.strip()))

    def covers(self, bot_id: str, *, bot_type: str | None, agent_backend: str | None) -> bool:
        if normalize_bot_type(bot_type, agent_backend) != "agent":
            return False
        if self.all_agents:
            return True
        if self.backends:
            return (agent_backend or "").strip().lower() in self.backends
        return bot_id.strip().lower() in self.bots

    def to_json(self) -> str:
        if self.all_agents:
            return json.dumps({"all_agents": True})
        if self.backends:
            return json.dumps({"backends": sorted(self.backends)})
        return json.dumps({"bots": sorted(self.bots)})

    @classmethod
    def from_json(cls, raw: Any) -> "GateScope":
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, Mapping):
            raise GateScopeError(f"unreadable gate scope: {raw!r}")
        if data.get("all_agents"):
            return cls.every_agent()
        if data.get("backends"):
            return cls.on_backends(*data["backends"])
        return cls.for_bots(*(data.get("bots") or ()))


@dataclass(frozen=True)
class GateActor:
    """Which admission edge is asking, and for which turn (no authority of its own)."""

    kind: str
    turn_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ACTOR_KINDS:
            raise ValueError(f"unknown gate actor kind {self.kind!r}")


@dataclass(frozen=True)
class GateBlock:
    """The live gate that refused an admission. ``id`` is the generation."""

    id: int
    mode: str
    owner_kind: str
    owner_id: str
    reason: str | None
    expires_at: datetime

    @property
    def refusal(self) -> str:
        """The wire code: a deny gate freezes (503), a hold gate holds (409)."""
        return "system_frozen" if self.mode == DENY else "system_held"


@dataclass(frozen=True)
class GateSpec:
    """What ``activate_hold`` inserts. ``ttl_seconds`` is added to the DB clock."""

    owner_kind: str
    owner_id: str
    mode: str
    scope: GateScope
    ttl_seconds: float
    reason: str | None = None
    allow_actors: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.owner_kind not in OWNER_KINDS:
            raise GateScopeError(f"unknown gate owner kind {self.owner_kind!r}")
        if not (self.owner_id or "").strip():
            raise GateScopeError("a gate needs an owner_id")
        if self.mode not in GATE_MODES:
            raise GateScopeError(f"unknown gate mode {self.mode!r}")
        if not self.ttl_seconds or self.ttl_seconds <= 0:
            raise GateScopeError("a gate needs a positive ttl")
        unknown = set(self.allow_actors) - ACTOR_KINDS
        if unknown:
            raise GateScopeError(f"unknown allow_actors: {sorted(unknown)}")


_schema_guard = SchemaBootstrapGuard()


def ensure_gate_schema(engine) -> None:
    """Create ``system_gates`` and ``gate_members`` once per engine (PostgreSQL only)."""
    if _is_postgres(engine):
        _schema_guard.run(engine, "system-gates", _bootstrap)


def _bootstrap(conn) -> None:
    ensure_relations(conn, _GATE_RELATIONS)
    ensure_columns(conn, "gate_members", _MEMBER_ADDED_COLUMNS)


# Created only when the catalog lacks them (``ensure_relations``): even a no-op
# ``CREATE INDEX IF NOT EXISTS`` takes SHARE on its table and would queue every
# writer (binds pin members, ``pin_member_request``) behind an open transaction.
_GATE_RELATIONS = {
    "system_gates": """
        CREATE TABLE IF NOT EXISTS system_gates (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            owner_kind VARCHAR(16) NOT NULL CHECK (owner_kind IN ('job','operator')),
            owner_id VARCHAR(128) NOT NULL,
            mode VARCHAR(8) NOT NULL CHECK (mode IN ('hold','deny')),
            scope JSONB NOT NULL,
            reason TEXT,
            allow_actors JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL,
            released_at TIMESTAMPTZ
        )
    """,
    "ix_system_gates_unreleased": """
        CREATE INDEX IF NOT EXISTS ix_system_gates_unreleased
        ON system_gates (expires_at) WHERE released_at IS NULL
    """,
    "ix_system_gates_owner": """
        CREATE INDEX IF NOT EXISTS ix_system_gates_owner ON system_gates (owner_id)
    """,
    "gate_members": """
        CREATE TABLE IF NOT EXISTS gate_members (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            gate_id BIGINT NOT NULL REFERENCES system_gates(id),
            bot_id VARCHAR(128) NOT NULL,
            turn_id VARCHAR(128) NOT NULL,
            turn_status VARCHAR(32) NOT NULL,
            disposition VARCHAR(24) NOT NULL
              CHECK (disposition IN ('revoked_unstarted','accepted','running')),
            backend VARCHAR(64),
            agent_session_key VARCHAR(128),
            agent_request_id VARCHAR(128),
            producer_kind VARCHAR(16) NOT NULL DEFAULT 'ordinary',
            producer_id VARCHAR(128),
            producer_claim VARCHAR(128),
            control_state VARCHAR(16) NOT NULL DEFAULT 'pending'
              CHECK (control_state IN ('pending','steering','steered','unsupported',
                                       'not_needed','self','unconfirmed','cancelled')),
            control_request_id VARCHAR(255),
            control_claim_token VARCHAR(128),
            control_lease_expires_at TIMESTAMPTZ,
            control_attempts INTEGER NOT NULL DEFAULT 0,
            control_error TEXT,
            ack_status VARCHAR(32),
            ack_note TEXT,
            acked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (gate_id, turn_id)
        )
    """,
    "ix_gate_members_turn": """
        CREATE INDEX IF NOT EXISTS ix_gate_members_turn ON gate_members (turn_id)
    """,
}

# Every gate_members column added after the table shipped; ``ensure_columns``
# alters only what the catalog lacks, so a migrated schema takes no lock.
_MEMBER_ADDED_COLUMNS = {
    # (f): when control_state last changed. Every UPDATE that changes it sets
    # it in the same statement (no trigger). NULL on pre-existing rows, never
    # the migration's own time; readers use COALESCE(control_changed_at, created_at).
    "control_changed_at": ("TIMESTAMPTZ", "now()"),
    # test_Z (b): the exit evidence home when the member's turn row is gone
    # (``stamp_member_exit``); the same record shape as the turn-row stamp.
    "execution_exit": ("JSONB", None),
}


def _is_postgres(conn) -> bool:
    return getattr(getattr(conn, "dialect", None), "name", None) == "postgresql"


_LIVE = "released_at IS NULL AND expires_at > now()"


def _live_gates(conn) -> list[Mapping[str, Any]]:
    return conn.execute(sa_text(
        "SELECT id, mode, owner_kind, owner_id, reason, expires_at, scope, allow_actors"
        f" FROM system_gates WHERE {_LIVE}"
    )).mappings().all()


def _bot_profile(conn, bot_id: str) -> Mapping[str, Any] | None:
    return conn.execute(sa_text(
        "SELECT bot_type, agent_backend FROM bot_profiles WHERE slug=:bot"
    ), {"bot": bot_id}).mappings().first()


def _jsonb(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) else raw


def _block(row: Mapping[str, Any]) -> GateBlock:
    return GateBlock(
        id=int(row["id"]), mode=row["mode"], owner_kind=row["owner_kind"],
        owner_id=row["owner_id"], reason=row["reason"], expires_at=row["expires_at"],
    )


def gate_blocks(conn, bot_id: str, actor: GateActor) -> GateBlock | None:
    """The strictest live gate refusing ``actor`` on ``bot_id``, else None.

    Call after ``lock_target(conn, bot_id)`` in the admission's transaction.
    A gate does not block when: it doesn't cover the bot (scope resolved now,
    agent bots only); ``actor.kind`` is in its ``allow_actors``; or the actor
    re-attaches (``delivery_claim``/``save_turn``) to a turn that gate's
    roster recorded as ``accepted`` (C1). Strictest: deny before hold, then
    the latest ``expires_at``. Always None off PostgreSQL.
    """
    if not _is_postgres(conn):
        return None
    gates = _live_gates(conn)
    if not gates:
        return None
    profile = _bot_profile(conn, bot_id)
    if profile is None:
        return None  # not a known bot, so not an agent bot
    blocking = []
    for row in gates:
        scope = GateScope.from_json(row["scope"])
        if not scope.covers(bot_id, bot_type=profile["bot_type"], agent_backend=profile["agent_backend"]):
            continue
        if actor.kind in set(_jsonb(row["allow_actors"]) or ()):
            continue
        if actor.kind in _REATTACH_ACTORS and actor.turn_id and _accepted_member(conn, row["id"], actor.turn_id):
            continue
        blocking.append(row)
    if not blocking:
        return None
    strictest = max(blocking, key=lambda row: (row["mode"] == DENY, row["expires_at"], row["id"]))
    return _block(strictest)


def _accepted_member(conn, gate_id: int, turn_id: str) -> bool:
    return conn.execute(sa_text(
        "SELECT 1 FROM gate_members WHERE gate_id=:gate AND turn_id=:turn AND disposition='accepted'"
    ), {"gate": gate_id, "turn": turn_id}).first() is not None


def gate_is_live(conn, gate_id: int) -> bool:
    """Is generation ``gate_id`` unreleased and unexpired on the DB clock?"""
    if not _is_postgres(conn):
        return False
    return conn.execute(sa_text(
        f"SELECT 1 FROM system_gates WHERE id=:id AND {_LIVE}"
    ), {"id": gate_id}).first() is not None


def extend_gate(conn, gate_id: int, *, owner_id: str, ttl_seconds: float) -> bool:
    """Renew a live gate's lease; extends only, never shortens.

    False when the gate is released, already expired, or not ``owner_id``'s:
    a lapsed gate is never revived (the caller opens a new generation).
    """
    return bool(conn.execute(sa_text(f"""
        UPDATE system_gates
        SET expires_at=GREATEST(expires_at, now() + (:ttl * interval '1 second'))
        WHERE id=:id AND owner_id=:owner AND {_LIVE}
    """), {"id": gate_id, "owner": owner_id, "ttl": float(ttl_seconds)}).rowcount)


def release_gates(conn, *, owner_id: str, gate_id: int | None = None) -> list[int]:
    """Release ``owner_id``'s unreleased gates (one generation, or all of them).

    Owner-matched: no owner can release another owner's gate. Releasing an
    expired gate still stamps ``released_at`` (idempotent). Returns the ids
    released by this call.
    """
    clause = " AND id=:id" if gate_id is not None else ""
    rows = conn.execute(sa_text(
        "UPDATE system_gates SET released_at=now()"
        f" WHERE owner_id=:owner AND released_at IS NULL{clause} RETURNING id"
    ), {"owner": owner_id, "id": gate_id}).all()
    return sorted(int(row[0]) for row in rows)


def has_execution_evidence(turn: Mapping[str, Any]) -> bool:
    """May this reserved turn already be executing at a bridge? (C1)

    ``agent_request_id`` is bound before every chat.send publish, and the
    expired-lease re-arm keeps it, so its absence proves nothing was sent.
    A crash between bind and publish over-counts, the conservative direction.
    """
    return bool(turn.get("agent_request_id"))


# B1: the execution_exit record, a top-level per-attempt key of a turn's
# request_json (turn_termination.ATTEMPT_KEYS): late writes keep it, a re-arm drops it.
EXECUTION_EXIT_KEY = "execution_exit"
EXIT_SOURCES = frozenset({"a", "b", "c"})


@dataclass(frozen=True)
class DrainStatus:
    """Is one roster member drained? ``reason`` names the evidence or the blocker."""

    member_id: int
    turn_id: str
    drained: bool
    reason: str


def execution_exit_drains(record: Any, *, request_id: str | None, backend: str | None) -> tuple[bool, str]:
    """The one B1 drain rule for an ``execution_exit`` record.

    Drained only on ``exited is True`` from a known source, for exactly this
    request id (a null id never matches) and, when the member has one, its
    backend. Everything else is a named blocker, never a guess.
    """
    if not request_id:
        return False, "no_request_id"
    if not isinstance(record, Mapping):
        return False, "no_execution_exit"
    if record.get("request_id") != request_id:
        return False, "request_mismatch"
    if backend and str(record.get("backend") or "").lower() != backend.lower():
        return False, "backend_mismatch"
    if record.get("source") not in EXIT_SOURCES:
        return False, "unknown_exit_source"
    if record.get("exited") is not True:
        return False, "exit_unconfirmed"
    return True, f"execution_exit:{record['source']}"


def _pinned_request(row: Mapping[str, Any]) -> str | None:
    """A member's request: the roster's pin, else the one its turn bound after capture (same attempt)."""
    return row["agent_request_id"] or row["turn_request_id"]


def _drain_status(row: Mapping[str, Any]) -> DrainStatus:
    from .turn_termination import turn_payload  # keeps this module's import closure DB-only

    if row["disposition"] == REVOKED_UNSTARTED:
        drained, reason = True, REVOKED_UNSTARTED
    elif row["turn_present"] is None:
        # test_Z: the turn row is gone, so the member row is the evidence home.
        # A request id is bound (and pinned here, ``pin_member_request``)
        # before every publish, so a member that never got one never ran.
        if not row["agent_request_id"]:
            drained, reason = True, "never_bound"
        else:
            drained, reason = execution_exit_drains(
                row["member_exit"], request_id=row["agent_request_id"], backend=row["backend"])
            reason = reason if drained else f"turn_missing:{reason}"
    else:
        drained, reason = execution_exit_drains(
            turn_payload(row["request_json"]).get(EXECUTION_EXIT_KEY),
            request_id=_pinned_request(row),
            backend=row["backend"],
        )
    return DrainStatus(int(row["id"]), row["turn_id"], drained, reason)


_DRAIN_ROWS = """
    SELECT m.id, m.turn_id, m.disposition, m.backend, m.agent_request_id,
           m.execution_exit AS member_exit, t.id AS turn_present, t.agent_request_id AS turn_request_id, t.request_json
    FROM gate_members m LEFT JOIN turn_logs t ON t.id = m.turn_id
    WHERE m.gate_id = :gate {extra} ORDER BY m.id
"""


def gate_drain_status(conn, gate_id: int) -> list[DrainStatus]:
    """Every member of generation ``gate_id`` with its drain verdict, in one query."""
    rows = conn.execute(sa_text(_DRAIN_ROWS.format(extra="")), {"gate": gate_id}).mappings().all()
    return [_drain_status(row) for row in rows]


def member_drained(conn, gate_id: int, turn_id: str) -> DrainStatus | None:
    """One member's drain verdict; None when ``turn_id`` is not a member of ``gate_id``."""
    row = conn.execute(sa_text(_DRAIN_ROWS.format(extra="AND m.turn_id = :turn")),
                       {"gate": gate_id, "turn": turn_id}).mappings().first()
    return _drain_status(row) if row is not None else None


def pin_member_request(conn, turn_id: str, request_id: str) -> int:
    """test_Z (a) pin-at-bind: record a newly bound request on every member row of the turn.

    The binder calls this in its own transaction, after it has locked the turn
    row; ``activate_hold`` locks the captured turn rows too, so a bind racing a
    capture is either read by the capture or pins the committed member. Only an
    unpinned member is filled: the roster's pinned request decides (B1). Every
    gate, live or not (Snark z-r1): ``never_bound`` is sound only if every bind
    pins, and a member of a lost generation may still be carried.
    """
    if not _is_postgres(conn):
        return 0
    return conn.execute(sa_text(
        "UPDATE gate_members SET agent_request_id=:request WHERE turn_id=:turn AND agent_request_id IS NULL"
    ), {"turn": turn_id, "request": request_id}).rowcount


def stamp_member_exit(conn, member_id: int, record: Mapping[str, Any]) -> bool:
    """test_Z (b): stamp exit evidence on a member whose turn row is gone.

    ``record`` has the turn-row ``execution_exit`` shape and is judged by the
    same ``execution_exit_drains``. While the turn row exists it stays the one
    evidence home, so this writes nothing and returns False.
    """
    return bool(conn.execute(sa_text(
        "UPDATE gate_members m SET execution_exit=CAST(:record AS jsonb) WHERE m.id=:id"
        " AND NOT EXISTS (SELECT 1 FROM turn_logs t WHERE t.id = m.turn_id)"
    ), {"id": member_id, "record": json.dumps(dict(record), default=str)}).rowcount)


def bots_in_scope(conn, scope: GateScope) -> list[str]:
    """Every agent bot the scope covers right now, in lock order (sorted)."""
    rows = conn.execute(sa_text("SELECT slug, bot_type, agent_backend FROM bot_profiles")).mappings().all()
    return sorted(
        row["slug"] for row in rows
        if scope.covers(row["slug"], bot_type=row["bot_type"], agent_backend=row["agent_backend"])
    )


def insert_gate(conn, spec: GateSpec) -> GateBlock:
    """Insert one gate row (``activate_hold`` calls this under the target locks)."""
    row = conn.execute(sa_text("""
        INSERT INTO system_gates (owner_kind, owner_id, mode, scope, reason, allow_actors, expires_at)
        VALUES (:owner_kind, :owner_id, :mode, CAST(:scope AS jsonb), :reason,
                CAST(:allow AS jsonb), now() + (:ttl * interval '1 second'))
        RETURNING id, mode, owner_kind, owner_id, reason, expires_at
    """), {
        "owner_kind": spec.owner_kind, "owner_id": spec.owner_id, "mode": spec.mode,
        "scope": spec.scope.to_json(), "reason": spec.reason,
        "allow": json.dumps(sorted(spec.allow_actors)), "ttl": float(spec.ttl_seconds),
    }).mappings().one()
    return _block(row)


@dataclass(frozen=True)
class GateMember:
    """One roster row: an open turn captured at activation, with its disposition."""

    gate_id: int
    bot_id: str
    turn_id: str
    turn_status: str
    disposition: str
    backend: str | None = None
    agent_session_key: str | None = None
    agent_request_id: str | None = None
    producer_kind: str = "ordinary"
    producer_id: str | None = None
    producer_claim: str | None = None
    control_state: str = "pending"

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise ValueError(f"unknown disposition {self.disposition!r}")
        if self.control_state not in CONTROL_STATES:
            raise ValueError(f"unknown control state {self.control_state!r}")


def initial_control_state(owner_kind: str, disposition: str) -> str:
    """Only a job-owned gate steers its members, and never a revoked one (nothing runs)."""
    return "pending" if owner_kind == OWNER_JOB and disposition != REVOKED_UNSTARTED else "not_needed"


_MEMBER_FIELDS = tuple(f.name for f in fields(GateMember))
_MEMBER_COLUMNS = ", ".join(_MEMBER_FIELDS)


def insert_member(conn, member: GateMember, *, skip_existing: bool = False) -> bool:
    """Write one roster row. ``skip_existing`` (carry-forward only) yields to a fresh row."""
    conflict = " ON CONFLICT (gate_id, turn_id) DO NOTHING" if skip_existing else ""
    return bool(conn.execute(sa_text(
        f"INSERT INTO gate_members ({_MEMBER_COLUMNS})"
        f" VALUES ({', '.join(':' + name for name in _MEMBER_FIELDS)}){conflict}"
    ), vars(member)).rowcount)


def carry_forward(conn, *, predecessor: int, gate: GateBlock) -> list[GateMember]:
    """Carry ``predecessor``'s undrained members into ``gate`` (fresh roster already written).

    A fresh capture sees only open rows, but a member whose row ended without
    a confirmed exit (B1) may still be running; dropping it on a rebuild would
    let the action kill it. It is carried as ``running``, exit-tracked only.
    Scope governs admission only (``gate_blocks``); the roster governs drain.
    Carried members are kept regardless of current scope (r2 S3): a bot that
    left the scope is admitted again, but its old run still sits on the held
    backend and must drain. Over-waiting is bounded by the quiesce timeout. A
    turn the fresh capture already has keeps the fresh row (UNIQUE(gate_id,
    turn_id)); acknowledgements carry onto it either way, so a rebuild never
    erases a bot's ack. A member-row ``execution_exit`` is not carried: one that
    drains is never carried, and the exit sweep stamps the live generation.

    A carried member takes ``_pinned_request``, the drain's own rule: a second
    guard behind pin-at-bind for a request bound without ``bind_agent_bridge``
    (test_Z z-r1). Runs in the caller's transaction.
    """
    undrained = [status.member_id for status in gate_drain_status(conn, predecessor) if not status.drained]
    rows = conn.execute(sa_text(
        f"SELECT {', '.join('m.' + name for name in _MEMBER_FIELDS)}, t.status AS current_status,"
        " t.agent_request_id AS turn_request_id"
        " FROM gate_members m LEFT JOIN turn_logs t ON t.id = m.turn_id"
        " WHERE m.gate_id = :old AND m.id = ANY(:ids) ORDER BY m.id"
    ), {"old": predecessor, "ids": undrained}).mappings().all() if undrained else []
    carried: list[GateMember] = []
    for row in rows:
        # Never ``accepted``: its row is no longer an open reservation, and the
        # C1 re-attach exemption must not let a re-armed retry start under hold.
        member = GateMember(**{**{name: row[name] for name in _MEMBER_FIELDS}, "gate_id": gate.id,
                               "turn_status": row["current_status"] or row["turn_status"],
                               "agent_request_id": _pinned_request(row),
                               "disposition": RUNNING,
                               "control_state": initial_control_state(gate.owner_kind, RUNNING)})
        if insert_member(conn, member, skip_existing=True):
            carried.append(member)
    # One ack carry for the whole new roster, fresh rows and carried rows alike.
    conn.execute(sa_text("""
        UPDATE gate_members AS fresh
        SET ack_status = prior.ack_status, ack_note = prior.ack_note, acked_at = prior.acked_at
        FROM gate_members AS prior
        WHERE fresh.gate_id = :new AND prior.gate_id = :old AND prior.turn_id = fresh.turn_id
          AND prior.ack_status IS NOT NULL
    """), {"new": gate.id, "old": predecessor})
    return carried


__all__ = [
    "ACCEPTED",
    "ACTOR_KINDS",
    "DENY",
    "DISPOSITIONS",
    "DrainStatus",
    "EXECUTION_EXIT_KEY",
    "EXIT_SOURCES",
    "GateActor",
    "GateBlock",
    "GateMember",
    "GateScope",
    "GateScopeError",
    "GateSpec",
    "HOLD",
    "OWNER_JOB",
    "OWNER_OPERATOR",
    "REVOKED_UNSTARTED",
    "RUNNING",
    "bots_in_scope",
    "carry_forward",
    "ensure_gate_schema",
    "execution_exit_drains",
    "extend_gate",
    "gate_blocks",
    "gate_drain_status",
    "gate_is_live",
    "has_execution_evidence",
    "initial_control_state",
    "insert_gate",
    "insert_member",
    "member_drained",
    "pin_member_request",
    "release_gates",
    "stamp_member_exit",
]
