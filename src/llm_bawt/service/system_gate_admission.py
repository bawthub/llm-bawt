"""Edge policy for the producers and steers that ask a system gate (TASK-1233 C3).

``turn_gates`` owns the gate rows and ``gate_blocks``; ``turn_admission`` owns
``lock_target`` and ``SystemGateRefused``. This module holds the per-edge
decisions that are more than a plain refusal. Every check runs after
``lock_target(conn, bot_id)`` in the caller's transaction, so it is linearized
with ``activate_hold``: either the producer commits first and the roster
captures it, or the gate commits first and the check sees it.

What a live gate does at each edge:

* enqueue: a deny gate (freeze) refuses ``deny``-disposition producers
  (external sends, composer starts) with ``system_frozen``. ``hold`` producers
  (internal callbacks, scheduled work, restart retries, a trusted self-model
  continuation) are accepted and wait at the claim. A hold gate (quiesce)
  refuses nothing at enqueue.
* delivery claim: any live gate leaves the head QUEUED with no attempt spent
  (``inter_bot_delivery_claim``).
* scheduled prompt: any live gate leaves the occurrence due, a held no-op.
* steer: any live gate refuses a steer into a running turn, except the gate
  owner's own stop request: a quiesce control steer carrying a ``ControlGrant``
  that matches its member row (B3, ``check_steer``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import HTTPException
from sqlalchemy import text as sa_text

from .turn_admission import SystemGateRefused, lock_target
from .turn_gates import DENY, GateActor, GateBlock, gate_blocks

ON_GATE_DENY = "deny"
ON_GATE_HOLD = "hold"
GATE_DISPOSITIONS = frozenset({ON_GATE_DENY, ON_GATE_HOLD})


def admit_enqueue(conn, bot_id: str, *, on_gate: str) -> None:
    """Refuse a ``deny`` producer under a deny gate (call under lock_target)."""
    if on_gate not in GATE_DISPOSITIONS:
        raise ValueError(f"on_gate must be one of {sorted(GATE_DISPOSITIONS)}")
    if on_gate == ON_GATE_HOLD:
        return
    block = gate_blocks(conn, bot_id, GateActor("enqueue"))
    if block is not None and block.mode == DENY:
        raise SystemGateRefused(block)


def schedule_held(conn, bot_id: str) -> bool:
    """True when a live gate holds a due scheduled occurrence (under lock_target)."""
    return gate_blocks(conn, bot_id, GateActor("schedule")) is not None


class SteerGateRefused(HTTPException):
    """A live system gate refused a steer before any bridge RPC.

    An HTTPException so the public steer route reports it unchanged (503
    ``system_frozen`` / 409 ``system_held``); typed so the durable dispatcher
    can wait it out without spending an attempt.
    """

    def __init__(self, block: GateBlock):
        refused = SystemGateRefused(block)
        self.block = block
        self.code = refused.code
        super().__init__(status_code=refused.status_code, detail=str(refused))


@dataclass(frozen=True)
class ControlGrant:
    """A quiesce control steer's claim on one member row (B3). Internal only.

    The public steer route has no field that can carry one. It authorizes
    nothing by itself: ``check_steer`` re-reads the member under
    ``lock_target`` and admits only the live claim it names.
    """

    gate_id: int
    bot_id: str
    turn_id: str
    claim_token: str


@dataclass(frozen=True)
class ControlTarget:
    """The one run a control steer may address: the member's pinned request.

    Never the bot's current turn or backend (Snark, carried out-of-scope
    member). ``backend``/``session_key`` are None when the pinned run's
    binding is unknown, and such a run is never steered.
    """

    request_id: str
    backend: str | None
    session_key: str | None


def control_target(member: Mapping[str, Any]) -> ControlTarget | None:
    """Resolve a member row's pinned run (None until a request is pinned).

    ``member`` carries the member's ``agent_request_id``/``backend``/
    ``agent_session_key`` and its turn row's ``turn_request_id``/
    ``turn_backend``/``turn_session_key``. The turn row's binding is used only
    when it is the binding of the pinned request (``bind_agent_bridge`` writes
    all three together); otherwise the roster's capture is the only record.
    The request id is only an address, never proof the member is live.
    """
    request_id = member["agent_request_id"]
    if not request_id:
        return None
    if member["turn_request_id"] == request_id:
        backend = member["turn_backend"] or member["backend"]
        session_key = member["turn_session_key"] or member["agent_session_key"]
    else:
        backend, session_key = member["backend"], member["agent_session_key"]
    return ControlTarget(request_id=request_id, backend=backend, session_key=session_key)


# The live-member join (agreed with Byte): a member of its job's current
# generation, the gate unreleased and unexpired, the job still quiescing.
LIVE_MEMBER_SQL = """
    SELECT m.id, m.gate_id, m.bot_id, m.turn_id, m.disposition, m.control_state,
           m.control_attempts, m.backend, m.agent_session_key, m.agent_request_id,
           j.caller_turn_id, t.id AS turn_present, t.ended_at IS NOT NULL AS turn_ended,
           t.user_id AS turn_user_id, t.agent_request_id AS turn_request_id,
           t.agent_backend AS turn_backend, t.agent_session_key AS turn_session_key
    FROM gate_members m
    JOIN system_gates g ON g.id = m.gate_id
    JOIN ops_jobs j ON j.id = g.owner_id AND j.quiesce_gate_id = m.gate_id
    LEFT JOIN turn_logs t ON t.id = m.turn_id
    WHERE g.owner_kind = 'job' AND g.released_at IS NULL AND g.expires_at > now()
      AND j.state = 'quiescing' AND {where}
"""


class ControlGrantRefused(HTTPException):
    """A control grant that names no live claim (forged, stale or lapsed)."""

    def __init__(self, grant: ControlGrant):
        self.grant = grant
        super().__init__(status_code=409, detail="control_grant_refused")


def _admit_grant(conn, grant: ControlGrant) -> ControlTarget:
    """The grant's pinned run, under the caller's ``lock_target``; else refuse.

    Admits only when the gate is its job's live generation, the member row is
    (gate, turn, bot), it is ``steering`` under this exact claim token, and
    the control lease is unexpired. Reads only: the member row is not locked,
    and no turn row is ever locked here.
    """
    row = conn.execute(sa_text(LIVE_MEMBER_SQL.format(where=(
        "m.gate_id = :gate AND m.turn_id = :turn AND m.bot_id = :bot"
        " AND m.control_state = 'steering' AND m.control_claim_token = :token"
        " AND m.control_lease_expires_at > now()"
    ))), {"gate": grant.gate_id, "turn": grant.turn_id, "bot": grant.bot_id,
          "token": grant.claim_token}).mappings().first()
    target = control_target(row) if row is not None else None
    if target is None:
        raise ControlGrantRefused(grant)
    return target


def check_steer(engine, bot_id: str, *, turn_id: str, grant: ControlGrant | None = None) -> ControlTarget | None:
    """Raise ``SteerGateRefused`` when a live gate covers ``bot_id``.

    Its own short transaction under ``lock_target``. Only this check is
    linearized with gate activation; the bridge RPC that follows is not. A
    gate activated after the check is harmless: its roster already holds the
    running turn, and stopping that turn is the quiesce owner's job (S1).

    With a ``grant`` (the quiesce control pump only) the gates are not asked:
    the steer is a gate owner's own request that the run stop, and it is
    admitted only on a live claim (``_admit_grant``). The return value is
    the pinned run it may address; any mismatch raises ``ControlGrantRefused``.
    """
    if grant is not None and (grant.bot_id != bot_id or grant.turn_id != turn_id):
        raise ControlGrantRefused(grant)
    if engine is None or engine.dialect.name != "postgresql":
        if grant is not None:
            raise ControlGrantRefused(grant)  # no gate, so no claim to honour
        return None  # system gates are PostgreSQL-only
    with engine.begin() as conn:
        lock_target(conn, bot_id)
        if grant is not None:
            return _admit_grant(conn, grant)
        block = gate_blocks(conn, bot_id, GateActor("steer", turn_id))
    if block is not None:
        raise SteerGateRefused(block)
    return None


__all__ = [
    "ControlGrant",
    "ControlGrantRefused",
    "ControlTarget",
    "GATE_DISPOSITIONS",
    "LIVE_MEMBER_SQL",
    "ON_GATE_DENY",
    "ON_GATE_HOLD",
    "SteerGateRefused",
    "admit_enqueue",
    "check_steer",
    "control_target",
    "schedule_held",
]
