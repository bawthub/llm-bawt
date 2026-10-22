"""The quiesce control pump: ask each held running turn to stop (TASK-1233 B3).

The member row is the message. A quiesce job's gate roster (``gate_members``)
is the queue: the pump never writes an ``inter_bot_deliveries`` row, so a
control steer cannot queue behind a held FIFO head, and no outcome ever
enqueues or starts a turn. It runs in the app, from the inter-bot dispatcher's
sweep (leader-only), as one single-flight task (``QuiesceControlRunner``).

Per due member of a live generation (``LIVE_MEMBER_SQL``):

* txn1 (``claim``): ``lock_target(bot)``, then the member row ``FOR UPDATE
  SKIP LOCKED``, re-validated in the same statement. Turn rows are only read,
  never locked: ``bind_agent_bridge`` locks the turn row and then updates its
  members, so a pump that locked a turn row while holding a member could
  deadlock with a bind (Caid, df30f61).

  - revoked, turn gone or ended -> ``not_needed`` (the drain predicate
    settles it; an ended member's bot is never steered);
  - the job's caller's own turn -> ``self`` (never steered; still drains);
  - no pinned request yet -> retried later, nothing changes state;
  - pinned run on a backend that cannot steer -> ``unsupported``; a pinned
    run whose binding is unknown -> ``unconfirmed``;
  - attempts exhausted -> ``unconfirmed``;
  - another member row (another job or generation) for the same pinned run
    already ``steered`` -> ``steered`` with no RPC; one ``steering`` under a
    live lease -> retried later (a lapsed one does not hold it: a dead
    generation's claim is never reclaimed). A pinned run is asked once,
    ever: a later job that holds a run some earlier job, live or not,
    already steered waits on the drain without asking again. The bridge
    would dedupe the same request id anyway;
  - otherwise -> ``steering`` with a claim token and lease.

* the RPC: ``steer_active_turn`` with ``control=ControlGrant``, priority
  ``next`` (never kills running tools, TASK-934), text from the prompt
  registry (``ops.quiesce_request``), addressed only to the member's pinned
  run. ``control_request_id`` = ``steer_quiesce_<turn>_<request>`` names the
  run (the sibling dedupe key); the RPC id is that plus the member's steer
  generation (``ControlClaim.rpc_request_id``). The bridge dedupes on the RPC
  id and replays its first result, so a retry of one generation re-reads it
  instead of steering twice (the publisher keeps ``steer_quiesce_`` results as
  long as the dedupe key). The generation advances only on a definitive
  rejection before the bridge accepted it (TASK-1251's rule), which proves it
  never reached a run; otherwise every replay of a cached ``no_active_run``
  or ``maintenance_active`` would come back the same forever. The timeline
  message id stays one per run.

* txn2 (``settle``): a compare-and-set on the claim token. ok -> ``steered``;
  the turn row gone -> ``not_needed``; anything else -> ``pending`` with
  backoff, or ``unconfirmed`` once attempts run out. A run that isn't
  registered at the bridge (``no_active_run`` / ``active_run_mismatch``) is
  retried, not settled: it may still be starting or queued in its session,
  and once its turn ends the next claim settles it ``not_needed``. A claim
  cancelled meanwhile (``cancel_pending_controls`` on release) is left as it
  is.

Every UPDATE that changes ``control_state`` stamps ``control_changed_at`` in
the same statement; backoff bookkeeping does not.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from sqlalchemy import text as sa_text

from .inter_bot_dispatcher_steer import DEFINITIVE_NOT_ACCEPTED
from .system_gate_admission import LIVE_MEMBER_SQL, ControlGrant, control_target
from .turn_admission import lock_target
from .turn_gates import REVOKED_UNSTARTED, ensure_gate_schema

log = logging.getLogger(__name__)

STEERABLE_BACKEND = "claude-code"

# One steer RPC's outcome, as the pump settles it.
STEERED = "steered"
GONE = "gone"  # the pinned run is no longer active: nothing to stop
RETRY = "retry"  # not ready or ambiguous: same id, later


@dataclass(frozen=True)
class ControlClaim:
    """One claimed member: the grant the steer carries, its run's id and the RPC generation."""

    member_id: int
    grant: ControlGrant
    request_id: str  # the run: control_request_id
    user_id: str
    attempts: int
    generation: int = 0

    @property
    def rpc_request_id(self) -> str:
        """The bridge RPC id: the run's id for generation 0, then ``_g<k>``."""
        return self.request_id if self.generation == 0 else f"{self.request_id}_g{self.generation}"


@dataclass(frozen=True)
class SteerOutcome:
    kind: str
    detail: str | None = None


Steer = Callable[[ControlClaim], Awaitable[SteerOutcome]]


def control_request_id(turn_id: str, agent_request_id: str) -> str:
    """The id of one pinned run: the same for every job and gate generation (Caid).

    It is the sibling dedupe key and the RPC id of steer generation 0.
    """
    return f"steer_quiesce_{turn_id}_{agent_request_id}"


# Lease and gate expiry are compared on clock_timestamp(), never now(): the
# reads run after a lock_target wait, and now() is the transaction's start.
_DUE = (
    "((m.control_state = 'pending' AND (m.control_next_attempt_at IS NULL"
    " OR m.control_next_attempt_at <= now()))"
    " OR (m.control_state = 'steering' AND (m.control_lease_expires_at IS NULL"
    " OR m.control_lease_expires_at <= clock_timestamp())))"
)
_OPEN = "m.control_state IN ('pending', 'steering')"
_TABLES = "SELECT to_regclass('ops_jobs') IS NOT NULL AND to_regclass('gate_members') IS NOT NULL"


class QuiesceControlPump:
    """Claims due members, steers each once, and settles the result."""

    LEASE_SECONDS = 60.0  # bounds one steer RPC (the bridge RPC waits 10s)
    MAX_ATTEMPTS = 6
    BACKOFF_BASE_SECONDS = 2.0
    BACKOFF_CAP_SECONDS = 30.0
    UNBOUND_RETRY_SECONDS = 2.0  # no pinned request yet
    SIBLING_RETRY_SECONDS = 5.0  # the same run is being steered for another generation
    BATCH = 32

    def __init__(self, engine, steer: Steer):
        ensure_gate_schema(engine)  # once per engine; the delivery store already ran it
        self.engine = engine
        self.steer = steer

    # --- reads -------------------------------------------------------------

    def _ready(self, conn) -> bool:
        return bool(conn.execute(sa_text(_TABLES)).scalar())

    def due(self) -> list[tuple[int, str]]:
        """(member id, bot id) of every member due a claim, oldest first. No locks."""
        with self.engine.connect() as conn:
            if not self._ready(conn):
                return []
            rows = conn.execute(sa_text(
                LIVE_MEMBER_SQL.format(where=_DUE) + " ORDER BY m.id LIMIT :limit"
            ), {"limit": self.BATCH}).mappings().all()
        return [(int(row["id"]), row["bot_id"]) for row in rows]

    def has_open_work(self) -> bool:
        """Is any live member still pending or steering (due now or later)?"""
        with self.engine.connect() as conn:
            if not self._ready(conn):
                return False
            return conn.execute(sa_text(
                LIVE_MEMBER_SQL.format(where=_OPEN) + " LIMIT 1"
            )).first() is not None

    # --- txn1 --------------------------------------------------------------

    def claim(self, member_id: int, bot_id: str) -> ControlClaim | None:
        """Re-validate one member under its bot's lock; a claim only when a steer is due."""
        with self.engine.begin() as conn:
            lock_target(conn, bot_id)
            row = conn.execute(sa_text(
                LIVE_MEMBER_SQL.format(where=f"m.id = :id AND m.bot_id = :bot AND {_DUE}")
                + " FOR UPDATE OF m SKIP LOCKED"
            ), {"id": member_id, "bot": bot_id}).mappings().first()
            if row is None:
                return None  # released, settled, not due, or held by another pump
            return self._decide(conn, row)

    def _decide(self, conn, row: Mapping[str, Any]) -> ControlClaim | None:
        if row["disposition"] == REVOKED_UNSTARTED or row["turn_present"] is None or row["turn_ended"]:
            self._set_state(conn, row["id"], "not_needed", error="turn_not_running")
            return None
        if row["caller_turn_id"] and row["caller_turn_id"] == row["turn_id"]:
            self._set_state(conn, row["id"], "self")
            return None
        target = control_target(row)
        if target is None:
            self._defer(conn, row, self.UNBOUND_RETRY_SECONDS)
            return None
        if target.backend and target.backend != STEERABLE_BACKEND:
            self._set_state(conn, row["id"], "unsupported", error=f"backend:{target.backend}")
            return None
        if not target.backend or not target.session_key:
            self._set_state(conn, row["id"], "unconfirmed", error="pinned_run_unaddressable")
            return None
        if row["control_attempts"] >= self.MAX_ATTEMPTS:
            self._set_state(conn, row["id"], "unconfirmed", error="attempts_exhausted")
            return None
        request_id = control_request_id(row["turn_id"], target.request_id)
        sibling = conn.execute(sa_text("""
            SELECT bool_or(control_state = 'steered') AS steered,
                   bool_or(control_state = 'steering' AND control_lease_expires_at > clock_timestamp()) AS steering,
                   max(COALESCE(control_generation, 0)) AS generation
            FROM gate_members WHERE control_request_id = :rid AND id <> :id
        """), {"rid": request_id, "id": row["id"]}).mappings().one()
        if sibling["steered"]:
            self._set_state(conn, row["id"], "steered", request_id=request_id)
            return None
        if sibling["steering"]:
            self._defer(conn, row, self.SIBLING_RETRY_SECONDS)
            return None
        # A sibling's generation k means every earlier one was definitively
        # rejected, so starting there skips replays that could only say so again.
        generation = max(int(row["control_generation"] or 0), int(sibling["generation"] or 0))
        token = uuid.uuid4().hex
        conn.execute(sa_text("""
            UPDATE gate_members SET control_state = 'steering', control_changed_at = now(),
                control_request_id = :rid, control_claim_token = :token, control_generation = :gen,
                control_lease_expires_at = clock_timestamp() + (:lease * interval '1 second'),
                control_attempts = control_attempts + 1, control_next_attempt_at = NULL
            WHERE id = :id
        """), {"id": row["id"], "rid": request_id, "token": token, "gen": generation,
               "lease": self.LEASE_SECONDS})
        return ControlClaim(
            member_id=int(row["id"]),
            grant=ControlGrant(gate_id=int(row["gate_id"]), bot_id=row["bot_id"],
                               turn_id=row["turn_id"], claim_token=token),
            request_id=request_id, user_id=row["turn_user_id"] or "nick",
            attempts=int(row["control_attempts"]) + 1, generation=generation,
        )

    @staticmethod
    def _set_state(conn, member_id: int, state: str, *, error: str | None = None,
                   request_id: str | None = None) -> None:
        """A state change under the member lock: stamps control_changed_at, drops any claim."""
        conn.execute(sa_text("""
            UPDATE gate_members SET control_state = :state, control_changed_at = now(),
                control_claim_token = NULL, control_lease_expires_at = NULL,
                control_next_attempt_at = NULL, control_error = COALESCE(:error, control_error),
                control_request_id = COALESCE(:rid, control_request_id)
            WHERE id = :id
        """), {"id": member_id, "state": state, "error": error, "rid": request_id})

    def _defer(self, conn, row: Mapping[str, Any], seconds: float) -> None:
        """Retry later. Only bookkeeping for a pending member; a lapsed claim goes back to pending."""
        if row["control_state"] == "pending":
            conn.execute(sa_text(
                "UPDATE gate_members SET control_next_attempt_at = now() + (:s * interval '1 second')"
                " WHERE id = :id"
            ), {"id": row["id"], "s": seconds})
            return
        conn.execute(sa_text("""
            UPDATE gate_members SET control_state = 'pending', control_changed_at = now(),
                control_claim_token = NULL, control_lease_expires_at = NULL,
                control_next_attempt_at = now() + (:s * interval '1 second')
            WHERE id = :id
        """), {"id": row["id"], "s": seconds})

    # --- txn2 --------------------------------------------------------------

    def backoff(self, attempts: int) -> float:
        return min(self.BACKOFF_CAP_SECONDS, self.BACKOFF_BASE_SECONDS * 2 ** max(0, attempts - 1))

    def settle(self, claim: ControlClaim, outcome: SteerOutcome) -> bool:
        """CAS the claim to its result; False when the claim was cancelled or reclaimed meanwhile."""
        delay: float | None = None
        generation = claim.generation
        if outcome.kind == STEERED:
            state, error = "steered", None
        elif outcome.kind == GONE:
            state, error = "not_needed", outcome.detail or "turn_not_found"
        else:
            exhausted = claim.attempts >= self.MAX_ATTEMPTS
            state, error = ("unconfirmed" if exhausted else "pending"), outcome.detail or "steer_failed"
            delay = None if exhausted else self.backoff(claim.attempts)
            if outcome.detail in DEFINITIVE_NOT_ACCEPTED:
                generation += 1  # this generation never reached a run: the next one is fresh
        with self.engine.begin() as conn:
            return bool(conn.execute(sa_text("""
                UPDATE gate_members SET control_state = :state, control_changed_at = now(),
                    control_claim_token = NULL, control_lease_expires_at = NULL, control_error = :error,
                    control_generation = :gen,
                    control_next_attempt_at = CASE WHEN CAST(:delay AS double precision) IS NULL THEN NULL
                        ELSE now() + (CAST(:delay AS double precision) * interval '1 second') END
                WHERE id = :id AND control_state = 'steering' AND control_claim_token = :token
            """), {"id": claim.member_id, "token": claim.grant.claim_token, "state": state,
                   "error": error, "delay": delay, "gen": generation}).rowcount)

    # --- one pass ----------------------------------------------------------

    async def run_pass(self) -> int:
        """Claim every due member, steer the claims concurrently, settle each. Returns claims made."""
        claims: list[ControlClaim] = []
        for member_id, bot_id in await asyncio.to_thread(self.due):
            claim = await asyncio.to_thread(self.claim, member_id, bot_id)
            if claim is not None:
                claims.append(claim)
        await asyncio.gather(*(self._steer_one(claim) for claim in claims))
        return len(claims)

    async def _steer_one(self, claim: ControlClaim) -> None:
        try:
            outcome = await self.steer(claim)
        except asyncio.CancelledError:
            raise  # the claim's lease lapses and the member is reclaimed
        except Exception as exc:
            log.exception("quiesce control steer failed: member=%s turn=%s", claim.member_id, claim.grant.turn_id)
            outcome = SteerOutcome(RETRY, str(exc) or exc.__class__.__name__)
        settled = await asyncio.to_thread(self.settle, claim, outcome)
        log.info("quiesce control member=%s turn=%s bot=%s outcome=%s settled=%s",
                 claim.member_id, claim.grant.turn_id, claim.grant.bot_id, outcome.kind, settled)


def service_steer(service) -> Steer:
    """The production RPC: ``steer_active_turn`` under the claim's grant, from ``system``."""
    async def steer(claim: ControlClaim) -> SteerOutcome:
        from fastapi import HTTPException

        from ..bot_identities import SYSTEM_BOT_ID
        from ..message_authorship import AuthorReference
        from ..ops.quiesce_prompts import QUIESCE_REQUEST_KEY, resolve
        from .routes.chat import ChatSteerRequest, steer_active_turn

        grant = claim.grant
        message = await asyncio.to_thread(resolve, QUIESCE_REQUEST_KEY, grant.bot_id)
        request = ChatSteerRequest(
            turn_id=grant.turn_id, message=message, bot_id=grant.bot_id, user_id=claim.user_id,
            # One timeline message per pinned run, however many attempts.
            message_id=str(uuid.uuid5(uuid.NAMESPACE_URL, claim.request_id)), priority="next",
        )
        try:
            response = await steer_active_turn(
                service, request, steer_request_id=claim.rpc_request_id,
                author=AuthorReference.bot(SYSTEM_BOT_ID), control=grant,
            )
        except HTTPException as exc:
            return classify_refusal(exc)
        return SteerOutcome(STEERED if response.ok else RETRY, response.detail)

    return steer


def classify_refusal(exc: Any) -> SteerOutcome:
    """A refused steer: only the turn row gone is final, everything else is retried.

    ``no_active_run`` / ``active_run_mismatch`` are retried too: the bridge
    hasn't registered the pinned run (yet), and the next claim settles the
    member ``not_needed`` once its turn has ended.
    """
    detail = str(getattr(exc, "detail", "") or "")
    if getattr(exc, "status_code", None) == 404:
        return SteerOutcome(GONE, detail or "turn_not_found")
    return SteerOutcome(RETRY, detail or exc.__class__.__name__)


class QuiesceControlRunner:
    """Single-flight pump task for the dispatcher's sweep, gated on elapsed time.

    ``maybe_start`` is called every dispatcher pass (leader-only). It starts
    one drain task, at most once per ``INTERVAL_SECONDS``; the task runs passes
    every ``INTERVAL_SECONDS`` while any live member is still pending or
    steering, then exits. A new gate is picked up on the next dispatcher pass
    (every 15s at most). ``pump_factory`` returns None where no pump can run.
    """

    INTERVAL_SECONDS = 2.0

    def __init__(self, pump_factory: Callable[[], QuiesceControlPump | None], *,
                 clock: Callable[[], float] = time.monotonic):
        self._factory = pump_factory
        self._clock = clock
        self._pump: QuiesceControlPump | None = None
        self._task: asyncio.Task | None = None
        self._last_start: float | None = None

    def maybe_start(self) -> bool:
        if self._task is not None and not self._task.done():
            return False
        now = self._clock()
        if self._last_start is not None and now - self._last_start < self.INTERVAL_SECONDS:
            return False
        self._pump = self._pump or self._factory()
        if self._pump is None:
            return False
        self._last_start = now
        self._task = asyncio.create_task(self._drain(self._pump), name="quiesce-control-pump")
        return True

    async def _drain(self, pump: QuiesceControlPump) -> None:
        while True:
            try:
                await pump.run_pass()
                if not await asyncio.to_thread(pump.has_open_work):
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("quiesce control pass failed; retrying on the next dispatcher pass")
                return
            await asyncio.sleep(self.INTERVAL_SECONDS)

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def build_pump(service, engine) -> QuiesceControlPump | None:
    """The app's pump, or None off PostgreSQL (gates are PostgreSQL-only)."""
    if getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None
    return QuiesceControlPump(engine, service_steer(service))


__all__ = [
    "ControlClaim",
    "GONE",
    "QuiesceControlPump",
    "QuiesceControlRunner",
    "RETRY",
    "STEERED",
    "SteerOutcome",
    "build_pump",
    "classify_refusal",
    "control_request_id",
    "service_steer",
]
