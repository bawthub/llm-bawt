"""B1 sources (b) and (c): judge a held member's ``run.status`` into exit evidence (TASK-1233).

The quiesce driver drains a member only on a confirmed ``execution_exit``
(:func:`..service.turn_gates.execution_exit_drains`). Source (a) is written
from the run's own terminal. This module is the pure rule and the DB half of
the sweep (:mod:`..service.quiesce_exit_sweep`) that writes sources (b) and (c)
for members (a) never covered:

* **(b) restart.** ``not_found``/``tombstoned`` from a boot that held none of
  the run's durable holder records, and :func:`agent_bridge.run_status.proves_restart`
  holds against every one of them: a backend whose runs die with the bridge,
  PID 1 on both sides. openclaw never proves. ``boot_id`` is the answering boot.
* **(c) same boot.** The run's compact terminal record (``RUN_TERMINAL_RETENTION_S``)
  says ``teardown.exited``: ``true`` stamps a confirmed exit, ``false`` stamps
  ``exit_unconfirmed`` (re-probed every pass, never drained). A terminal with
  ``exited: null``, or an absence (``not_found`` or ``tombstoned``) with no
  terminal, is no evidence: a tombstone only says the request is no longer
  registered, and every bridge unregisters in ``finally`` whatever its
  teardown did (Vex r1 B1).

**Holders.** (b) reads two durable boot records per ``(backend, request_id)``,
both independent of the turn row: the accept first, then the sighting.

* The accept (``bridge_run_accepts``, :mod:`..service.run_accepts`) is the
  boot that admitted the send, written by the stream worker from the run's
  ``RUN_ACCEPTED``. A bridge that dies before the sweep ever sees the run live
  still left it.
* The sighting (``quiesce_run_sightings``) is the latest boot that answered a
  live state. The sweep writes it; it covers a run accepted before the accept
  record existed, or whose worker could not write it.

A member with neither cannot get (b) and falls to the quiesce timeout
(fail-safe). Rows past their retention are deleted.

**Abort.** A member bound to a request its bridge reports ``not_found``, with
no holder record and no terminal, may be a send still queued on the command
stream. The judgement asks for one backend-matched ``chat.abort``
(:attr:`Judgement.abort`) so the run produces evidence through the paths that
already prove it: a send read after the abort is refused as tombstoned by a
process that never accepted it (``REFUSED``, so (c) on the next pass), and a
send read before it is cancelled and publishes its own terminal (source (a)).
Either way the abort itself is never evidence.

Every write is its own short transaction. Nothing here locks a member row and
then a turn row: a turn stamp (``stamp_turn_exit``, turn row only) and a
member stamp (``stamp_member_exit``, member row only) are separate
transactions. PostgreSQL only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import text

from agent_bridge.run_contract import RunTerminal
from agent_bridge.run_status import (ABSENT_STATES, LIVE_STATES, RunSighting, RunState, RunStatusReply,
                                     proves_restart)

from ..service.run_accepts import ensure_accept_schema
from ..service.turn_gates import (ensure_gate_schema, execution_exit_record, gate_drain_status, stamp_member_exit,
                                  stamp_turn_exit)
from ..utils.schema import SchemaBootstrapGuard, ensure_relations
from .models import JOB_QUIESCING

log = logging.getLogger(__name__)

SIGHTING_RETENTION_S = 86400
"""A sighting idle this long is deleted; far past any quiesce wait (``QUIESCE_TIMEOUT_MAX_S``)."""

_SIGHTING_RELATIONS = {
    "quiesce_run_sightings": """
        CREATE TABLE IF NOT EXISTS quiesce_run_sightings (
            backend VARCHAR(64) NOT NULL,
            request_id VARCHAR(128) NOT NULL,
            boot_id VARCHAR(128) NOT NULL,
            pid1 BOOLEAN,
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (backend, request_id)
        )
    """,
    "ix_quiesce_run_sightings_last_seen": """
        CREATE INDEX IF NOT EXISTS ix_quiesce_run_sightings_last_seen
        ON quiesce_run_sightings (last_seen_at)
    """,
}

_schema_guard = SchemaBootstrapGuard()


def ensure_sighting_schema(engine) -> None:
    """Create the gate tables, ``bridge_run_accepts`` and ``quiesce_run_sightings`` once per engine."""
    ensure_gate_schema(engine)
    ensure_accept_schema(engine)
    _schema_guard.run(engine, "quiesce-run-sightings", lambda conn: ensure_relations(conn, _SIGHTING_RELATIONS))


@dataclass(frozen=True)
class SweepMember:
    """One undrained member of a live job gate, addressed for ``run.status``."""

    member_id: int
    gate_id: int
    turn_id: str
    turn_present: bool
    backend: str | None
    session_key: str | None
    request_id: str | None
    sighting: RunSighting | None
    drain_reason: str = ""
    """The driver's current blocker for this member (``gate_drain_status``)."""
    turn_request_id: str | None = None
    """The request the turn row is bound to now; differs from ``request_id`` (the pin) after a rebind."""

    accepted: RunSighting | None = None
    """The durable accept record (``bridge_run_accepts``): the boot that admitted the send."""

    @property
    def holders(self) -> tuple[RunSighting, ...]:
        """Every durable record of a boot that held the run: the accept first, then the sighting."""
        return tuple(h for h in (self.accepted, self.sighting) if h is not None)

    @property
    def unstampable(self) -> bool:
        """The turn row is the evidence home but bound to another request: no stamp can land."""
        return self.turn_present and bool(self.turn_request_id) and self.turn_request_id != self.request_id


@dataclass(frozen=True)
class Judgement:
    """What one reply means: a record to stamp, a sighting to keep, or neither (``reason`` says why)."""

    reason: str
    record: dict[str, Any] | None = None
    sighting: RunSighting | None = None
    abort: bool = False
    """Send one ``chat.abort`` for the member's request (see the module docstring)."""


def _record(source: str, request_id: str, backend: str, boot_id: str, exited: bool) -> dict[str, Any]:
    return execution_exit_record(source=source, request_id=request_id, backend=backend, boot_id=boot_id,
                                 exited=exited)


def judge(member: SweepMember, reply: RunStatusReply | None, terminal: RunTerminal | None) -> Judgement:
    """The one (b)/(c) rule. Pure: no I/O, so every case is unit-tested."""
    if reply is None:
        return Judgement("ambiguous")
    if not member.request_id or not member.backend:
        return Judgement("unaddressed")
    if reply.request_id != member.request_id or reply.backend.lower() != member.backend.lower():
        return Judgement("reply_mismatch")
    if reply.state in LIVE_STATES:
        return Judgement(f"live:{reply.state.value}", sighting=RunSighting.from_reply(reply))
    if reply.state not in ABSENT_STATES:
        return Judgement(f"no_evidence:{reply.state.value}")
    holders = member.holders
    if holders and all(proves_restart(holder, reply) for holder in holders):
        return Judgement("restart", _record("b", reply.request_id, reply.backend, reply.boot_id, True))
    if terminal is not None and (terminal.request_id != reply.request_id
                                 or terminal.backend.lower() != reply.backend.lower()):
        terminal = None
    if terminal is not None and terminal.teardown.exited is not None:
        exited = terminal.teardown.exited
        if not exited and member.drain_reason == "exit_unconfirmed":
            return Judgement("terminal_survivor")  # already the named blocker; don't rewrite it every pass
        return Judgement("terminal_exited" if exited else "terminal_survivor",
                         _record("c", reply.request_id, reply.backend, terminal.boot_id, exited))
    # A tombstone alone is never evidence (Vex r1 B1): every bridge releases a
    # request in ``finally`` whatever its teardown did, and openclaw tombstones
    # as an abort starts, so the run may still be alive in the boot that
    # answers ``tombstoned``.
    if terminal is not None:
        return Judgement("terminal_unattempted")
    if reply.state is RunState.NOT_FOUND and not holders and member.session_key:
        return Judgement("absent_unaccepted", abort=True)
    return Judgement("absent_unproven")


_MEMBERS = """
    SELECT m.id AS member_id, m.gate_id, m.turn_id, t.id IS NOT NULL AS turn_present,
           COALESCE(m.backend, t.agent_backend) AS backend,
           COALESCE(m.agent_session_key, t.agent_session_key) AS session_key,
           COALESCE(m.agent_request_id, t.agent_request_id) AS request_id,
           t.agent_request_id AS turn_request_id,
           s.backend AS sighted_backend, s.boot_id AS sighted_boot, s.pid1 AS sighted_pid1,
           a.backend AS accepted_backend, a.boot_id AS accepted_boot, a.pid1 AS accepted_pid1
    FROM system_gates g
    JOIN ops_jobs j ON j.id = g.owner_id AND j.quiesce_gate_id = g.id
    JOIN gate_members m ON m.gate_id = g.id
    LEFT JOIN turn_logs t ON t.id = m.turn_id
    LEFT JOIN quiesce_run_sightings s
      ON s.backend = lower(COALESCE(m.backend, t.agent_backend))
     AND s.request_id = COALESCE(m.agent_request_id, t.agent_request_id)
    LEFT JOIN bridge_run_accepts a
      ON a.backend = lower(COALESCE(m.backend, t.agent_backend))
     AND a.request_id = COALESCE(m.agent_request_id, t.agent_request_id)
    WHERE g.owner_kind = 'job' AND j.state = :quiescing
      AND g.released_at IS NULL AND g.expires_at > now()
    ORDER BY m.id
"""


def _member(row: Mapping[str, Any], drain_reason: str) -> SweepMember:
    def holder(prefix: str) -> RunSighting | None:
        if not row[f"{prefix}_boot"]:
            return None
        return RunSighting(row["request_id"], row[f"{prefix}_backend"], row[f"{prefix}_boot"], row[f"{prefix}_pid1"])

    return SweepMember(int(row["member_id"]), int(row["gate_id"]), row["turn_id"], bool(row["turn_present"]),
                       row["backend"], row["session_key"], row["request_id"], holder("sighted"), drain_reason,
                       row["turn_request_id"], holder("accepted"))


def undrained_members(engine) -> list[SweepMember]:
    """Every member of a live QUIESCING job's current gate that the drain rule does not yet drain.

    The verdict is :func:`gate_drain_status`, the driver's own, so the sweep
    probes exactly the members the driver waits on (``exit_unconfirmed`` included).
    """
    with engine.connect() as conn:
        rows = conn.execute(text(_MEMBERS), {"quiescing": JOB_QUIESCING}).mappings().all()
        undrained = {status.member_id: status.reason for gate in sorted({int(r["gate_id"]) for r in rows})
                     for status in gate_drain_status(conn, gate) if not status.drained}
    return [_member(row, undrained[int(row["member_id"])]) for row in rows if int(row["member_id"]) in undrained]


def record_sighting(engine, sighting: RunSighting) -> None:
    """Keep the latest live boot of a request (its own transaction)."""
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO quiesce_run_sightings (backend, request_id, boot_id, pid1)"
            " VALUES (:backend, :request, :boot, :pid1)"
            " ON CONFLICT (backend, request_id) DO UPDATE"
            " SET boot_id = EXCLUDED.boot_id, pid1 = EXCLUDED.pid1, last_seen_at = now()"
        ), {"backend": sighting.backend.lower(), "request": sighting.request_id, "boot": sighting.boot_id,
            "pid1": sighting.pid1})


def stamp(engine, member: SweepMember, record: Mapping[str, Any]) -> bool:
    """Stamp ``record`` on the turn row, else (row gone) on the member row.

    Two transactions, never one: ``stamp_turn_exit`` locks only the turn row,
    ``stamp_member_exit`` only the member row (and writes only when the turn
    row is gone), so a turn deleted between them is still covered.
    """
    if member.turn_present:
        with engine.begin() as conn:
            if stamp_turn_exit(conn, member.turn_id, record):
                return True
    with engine.begin() as conn:
        return stamp_member_exit(conn, member.member_id, record)


def purge_sightings(engine, *, retention_s: int = SIGHTING_RETENTION_S) -> int:
    with engine.begin() as conn:
        return conn.execute(text(
            "DELETE FROM quiesce_run_sightings WHERE last_seen_at < now() - make_interval(secs => :s)"
        ), {"s": retention_s}).rowcount


__all__ = [
    "Judgement", "SIGHTING_RETENTION_S", "SweepMember", "ensure_sighting_schema", "judge", "purge_sightings",
    "record_sighting", "stamp", "undrained_members",
]
