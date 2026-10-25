"""PostgreSQL tests for the TASK-1233 system gate primitive and activate_hold.

Every gate here is scoped to this test's own bots (or a backend no other bot
uses), so parallel PG tests are never held by it.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.schemas import ChatCompletionRequest
from llm_bawt.service.turn_admission import (
    AdmissionClaim,
    AdmissionFenceLost,
    SystemGateRefused,
    TargetBusy,
    activate_hold,
    lock_target,
)
from llm_bawt.service.turn_gates import (
    GateActor,
    GateScope,
    GateSpec,
    extend_gate,
    gate_blocks,
    gate_drain_status,
    gate_is_live,
    member_drained,
    release_gates,
    stamp_member_exit,
)
from tests.pg_isolation import isolated_pg_store
from tests.pg_lock_wait import wait_blocked_by
from tests.test_turn_admission import _mcp_claim

pytestmark = pytest.mark.usefixtures("private_app_pg")


@pytest.fixture
def gated():
    """A TurnLogStore, one agent bot profile, and cleanup of every gate this test owns."""
    from llm_bawt.runtime_settings import BotProfileStore
    from llm_bawt.service.turn_logs import TurnLogStore

    store = isolated_pg_store(TurnLogStore)
    if store.engine is None:
        pytest.skip("PostgreSQL unavailable")
    profiles = BotProfileStore(store.config)
    assert profiles.engine is not None
    tag = uuid.uuid4().hex
    ctx = SimpleNamespace(store=store, tag=tag, bots=[], owners=[], senders=[])

    def bot(*, bot_type="agent", backend="claude-code"):
        slug = f"test-gate-{tag}-{len(ctx.bots)}"
        now = datetime.now(timezone.utc)
        with store.engine.begin() as conn:
            conn.execute(sa_text("""
                INSERT INTO bot_profiles (slug, name, description, system_prompt, requires_memory,
                    voice_optimized, uses_tools, uses_search, uses_home_assistant, bot_type,
                    agent_backend, created_at, updated_at)
                VALUES (:slug, :slug, '', '', false, false, false, false, false, :type, :backend,
                    :now, :now)
            """), {"slug": slug, "type": bot_type, "backend": backend, "now": now})
        ctx.bots.append(slug)
        return slug

    def owner():
        owner_id = f"job-{tag}-{len(ctx.owners)}"
        ctx.owners.append(owner_id)
        return owner_id

    ctx.bot, ctx.owner = bot, owner
    yield ctx
    with store.engine.begin() as conn:
        for owner_id in ctx.owners:
            conn.execute(sa_text(
                "DELETE FROM gate_members WHERE gate_id IN (SELECT id FROM system_gates WHERE owner_id=:o)"
            ), {"o": owner_id})
            conn.execute(sa_text("DELETE FROM system_gates WHERE owner_id=:o"), {"o": owner_id})
        for sender in ctx.senders:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:s"), {"s": sender})
        for slug in ctx.bots:
            conn.execute(sa_text("DELETE FROM turn_logs WHERE bot_id=:b"), {"b": slug})
            conn.execute(sa_text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})


def _spec(owner_id, scope, *, mode="hold", owner_kind="job", ttl=600, allow=frozenset()):
    return GateSpec(owner_kind=owner_kind, owner_id=owner_id, mode=mode, scope=scope,
                    ttl_seconds=ttl, reason="test gate", allow_actors=allow)


def _activate(store, spec):
    with store.engine.begin() as conn:
        return activate_hold(conn, spec)


def _blocks(store, bot, kind="save_turn", turn_id=None):
    from llm_bawt.service.turn_admission import lock_target

    with store.engine.begin() as conn:
        lock_target(conn, bot)
        return gate_blocks(conn, bot, GateActor(kind, turn_id))


def _open(store, bot, status="streaming", *, request_id=None):
    turn_id = f"turn-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, ended_at, agent_request_id)
            VALUES (:id, now(), '/test', true, :bot, 'nick', :status, '', '', NULL, :req)
        """), {"id": turn_id, "bot": bot, "status": status, "req": request_id})
    return turn_id


def _save(store, bot, *, turn_id=None, admission=None, claim=None):
    payload = {"messages": []}
    if admission:
        payload["admission"] = admission
    turn_id = turn_id or f"turn-{uuid.uuid4().hex}"
    store.save_turn(
        turn_id=turn_id, request_id="r", path="/v1/chat/completions", stream=True, model="m",
        bot_id=bot, user_id="nick", status="streaming", latency_ms=None, user_prompt="hi",
        request_payload=payload, response_text="", tool_calls=None, admission_claim=claim,
    )
    return turn_id


def _exit(store, turn_id, *, ended=False, **record):
    """Stamp a turn's execution_exit (the writer's top-level ATTEMPT_KEY); optionally end the row."""
    with store.engine.begin() as conn:
        payload = json.loads(conn.execute(sa_text("SELECT request_json FROM turn_logs WHERE id=:id"),
                                          {"id": turn_id}).scalar_one() or "{}")
        if record:
            payload["execution_exit"] = {"backend": None, "boot_id": "boot-1", "source": "a",
                                         "at": "2026-10-22T15:00:00+00:00", **record}
        conn.execute(sa_text(
            "UPDATE turn_logs SET request_json=:json, ended_at=CASE WHEN :ended THEN now() END WHERE id=:id"
        ), {"json": json.dumps(payload), "ended": ended, "id": turn_id})


def _members(store, gate_id):
    with store.engine.connect() as conn:
        return {row["turn_id"]: dict(row) for row in conn.execute(sa_text(
            "SELECT * FROM gate_members WHERE gate_id=:g"
        ), {"g": gate_id}).mappings()}


def _claimed_delivery(ctx, bot, **enqueue):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    deliveries = InterBotDeliveryStore(ctx.store.config)
    sender = f"test-gate-sender-{uuid.uuid4().hex}"
    ctx.senders.append(sender)
    record, _ = deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}],
                 "bot_id": bot, "user": "nick", "stream": True},
        **enqueue,
    )
    claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=False)
    assert claimed and claimed.id == record.id and claimed.status == "DISPATCHING"
    return deliveries, claimed


def _delivery_claim(deliveries, claimed):
    from llm_bawt.service.inter_bot_claims import turn_admission_claim

    request = ChatCompletionRequest.model_validate({
        **deliveries.payload(claimed.id), "inter_bot_claim_token": claimed.claim_token,
    })
    service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=deliveries))
    return turn_admission_claim(service, request)


def _approval_reservation(store, approvals, bot, request_id):
    """An approval continuation claimed and holding a reservation on ``bot``."""
    row = _mcp_claim(approvals, request_id=request_id)
    fence = approvals.continuation_claim_fence(row.id, row.continuation_claim_token)
    turn_id = f"turn-approval-{uuid.uuid4().hex}"
    store.reserve_idle_turn(turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="c",
                            request_json={"admission": "idle_only"},
                            trigger_message_id=str(uuid.uuid4()),
                            owner=row.continuation_claim_token, fence=fence)
    return row, turn_id


@pytest.mark.integration
def test_lock_wait_observer_sees_a_waiter_that_arrives_after_its_first_poll(gated, monkeypatch):
    """TASK-1451: pg_stat_activity is snapshotted per transaction. An observer that
    polls inside one transaction keeps its first, empty view and misses this waiter."""
    from tests import pg_lock_wait

    bot = gated.bot()
    first_poll_missed, errors = threading.Event(), []

    def pause():
        first_poll_missed.set()
        time.sleep(pg_lock_wait.POLL_INTERVAL_S)

    monkeypatch.setattr(pg_lock_wait, "_pause", pause)

    def waiter():
        try:
            assert first_poll_missed.wait(5), "observer never polled"
            with gated.store.engine.begin() as conn:
                lock_target(conn, bot)
        except Exception as exc:  # surfaced by the assert below
            errors.append(repr(exc))

    with gated.store.engine.begin() as holder:
        lock_target(holder, bot)
        pid = holder.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        worker = threading.Thread(target=waiter)
        worker.start()
        wait_blocked_by(gated.store.engine, pid, worker, detail=lambda: errors)
    worker.join(10)
    assert not worker.is_alive() and not errors, errors


# ---------------------------------------------------------------------------
# Inert: with no live gate for the bot, admission is exactly today's
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_live_gate_leaves_admission_unchanged(gated):
    store, bot = gated.store, gated.bot()
    assert _blocks(store, bot) is None
    released = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))
    with store.engine.begin() as conn:
        assert release_gates(conn, owner_id=released.gate.owner_id) == [released.gate.id]
    expired = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE system_gates SET expires_at=now() - interval '1 second' WHERE id=:id"),
                     {"id": expired.gate.id})
        assert not gate_is_live(conn, expired.gate.id)
    _activate(store, _spec(gated.owner(), GateScope.for_bots(gated.bot())))  # another bot's gate
    assert _blocks(store, bot) is None
    turn_id = _save(store, bot)
    assert store.get_turn(turn_id).status == "streaming"


@pytest.mark.integration
def test_chat_bots_are_never_gated(gated):
    store = gated.store
    chat = gated.bot(bot_type="chat", backend=None)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(chat), mode="deny"))
    assert gate.members == ()
    assert _blocks(store, chat) is None
    assert store.get_turn(_save(store, chat)).status == "streaming"


# ---------------------------------------------------------------------------
# B2: activation is linearized with admission under the per-bot lock
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_b2_admission_paused_after_the_predicate_lands_in_the_roster(gated, monkeypatch):
    import llm_bawt.service.turn_logs as turn_logs_module

    store, bot = gated.store, gated.bot()
    checked, proceed, holder = threading.Event(), threading.Event(), {}
    real = turn_logs_module.gate_blocks

    def paused(conn, bot_id, actor):
        block = real(conn, bot_id, actor)
        holder["pid"] = conn.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        checked.set()
        assert proceed.wait(10)
        return block

    monkeypatch.setattr(turn_logs_module, "gate_blocks", paused)
    turn_id = f"turn-{uuid.uuid4().hex}"
    outcomes = {}
    admission = threading.Thread(target=lambda: outcomes.setdefault("saved", _save(store, bot, turn_id=turn_id)))
    admission.start()
    assert checked.wait(10)
    activation = threading.Thread(target=lambda: outcomes.setdefault(
        "gate", _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))))
    activation.start()
    try:
        wait_blocked_by(store.engine, holder["pid"], activation)
    finally:
        proceed.set()
        admission.join(10)
        activation.join(10)
    assert outcomes["saved"] == turn_id
    members = _members(store, outcomes["gate"].gate.id)
    assert members[turn_id]["disposition"] == "running"
    assert members[turn_id]["control_state"] == "pending"


@pytest.mark.integration
def test_b2_activation_first_refuses_the_producer(gated):
    store, bot = gated.store, gated.bot()
    _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))
    turn_id = f"turn-{uuid.uuid4().hex}"
    with pytest.raises(SystemGateRefused) as refused:
        _save(store, bot, turn_id=turn_id)
    assert (refused.value.code, refused.value.status_code) == ("system_held", 409)
    assert isinstance(refused.value, TargetBusy)  # durable producers defer it
    assert store.get_turn(turn_id) is None
    with pytest.raises(SystemGateRefused):
        store.reserve_idle_turn(turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="x",
                                request_json={}, trigger_message_id=str(uuid.uuid4()), owner="o")
    assert store.get_turn(turn_id) is None

    frozen_bot = gated.bot()
    _activate(store, _spec(gated.owner(), GateScope.for_bots(frozen_bot), mode="deny"))
    with pytest.raises(SystemGateRefused) as frozen:
        _save(store, frozen_bot)
    assert (frozen.value.code, frozen.value.status_code) == ("system_frozen", 503)


@pytest.mark.integration
def test_b2_every_open_row_is_in_the_roster(gated):
    store, bot = gated.store, gated.bot()
    first, second = _open(store, bot, "streaming"), _open(store, bot, "cancelling")
    ended = _open(store, bot, "completed")
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET ended_at=now() WHERE id=:id"), {"id": ended})
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))
    members = _members(store, gate.gate.id)
    assert set(members) == {first, second}  # an ended row is never a member
    assert {m["disposition"] for m in members.values()} == {"running"}
    assert members[second]["turn_status"] == "cancelling"


# ---------------------------------------------------------------------------
# B4: reservations at activation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_b4_hold_before_acceptance_revokes_and_refunds(gated):
    store, bot = gated.store, gated.bot()
    deliveries, claimed = _claimed_delivery(gated, bot)
    assert claimed.attempt_count == 1
    old_claim = _delivery_claim(deliveries, claimed)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))

    member = _members(store, gate.gate.id)[claimed.turn_id]
    assert (member["disposition"], member["control_state"]) == ("revoked_unstarted", "not_needed")
    assert (member["producer_kind"], member["producer_id"]) == ("delivery", claimed.id)
    record = deliveries.get(claimed.id)
    assert (record.status, record.attempt_count, record.claim_token) == ("QUEUED", 0, None)
    assert store.get_turn(claimed.turn_id) is None  # reservation released
    with pytest.raises(AdmissionFenceLost):  # the stalled claimant can never start it
        _save(store, bot, turn_id=claimed.turn_id, admission="idle_only", claim=old_claim)


@pytest.mark.integration
def test_b4_held_reset_after_transport_accept_redispatches(gated):
    """The dispatcher stamps transport_accepted_at before save_turn. A hold in
    that window must clear it, or claim_next fails the reset as an ambiguous
    accepted replay (Loopy P1 on e26e3cd)."""
    store, bot = gated.store, gated.bot()
    deliveries, claimed = _claimed_delivery(gated, bot, session_policy="reset_retain_history")
    assert deliveries.mark_transport_accepted(claimed.id, claimed.claim_token)
    owner_id = gated.owner()
    gate = _activate(store, _spec(owner_id, GateScope.for_bots(bot)))

    assert _members(store, gate.gate.id)[claimed.turn_id]["disposition"] == "revoked_unstarted"
    held = deliveries.get(claimed.id)
    assert (held.status, held.transport_accepted_at, held.delivery_mode) == ("QUEUED", None, None)
    with store.engine.begin() as conn:
        release_gates(conn, owner_id=owner_id)
    reclaimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=False)
    assert reclaimed is not None and (reclaimed.id, reclaimed.status) == (claimed.id, "DISPATCHING")


@pytest.mark.integration
def test_b4_hold_of_a_rearmed_delivery_releases_only_its_reservation(gated):
    store, bot = gated.store, gated.bot()
    deliveries, claimed = _claimed_delivery(gated, bot)
    with store.engine.begin() as conn:  # recover_expired's re-arm of an expired lease
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='QUEUED', claim_token=NULL, claim_owner=NULL,
                lease_expires_at=NULL, available_at=now(), next_retry_at=now() WHERE id=:id
        """), {"id": claimed.id})
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))

    member = _members(store, gate.gate.id)[claimed.turn_id]
    assert (member["disposition"], member["producer_id"]) == ("revoked_unstarted", claimed.id)
    assert store.get_turn(claimed.turn_id) is None  # reservation released
    record = deliveries.get(claimed.id)
    assert (record.status, record.attempt_count) == ("QUEUED", claimed.attempt_count)


@pytest.mark.integration
def test_activation_takes_target_locks_in_sorted_order(gated):
    from llm_bawt.service.turn_admission import lock_target

    store = gated.store
    first_bot, second_bot = gated.bot(), gated.bot()
    with store.engine.connect() as holder:
        txn = holder.begin()
        pid = holder.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        lock_target(holder, first_bot)
        activation = threading.Thread(target=lambda: _activate(
            store, _spec(gated.owner(), GateScope.for_bots(second_bot, first_bot))))
        activation.start()
        try:
            wait_blocked_by(store.engine, pid, activation)
            # Activation waits on the first bot before it locks any other.
            assert holder.execute(sa_text("SELECT pg_try_advisory_xact_lock(hashtext(:t))"),
                                  {"t": second_bot}).scalar_one() is True
        finally:
            txn.rollback()
            activation.join(10)


@pytest.mark.integration
def test_activation_locks_producers_in_ordinal_order_like_recover_expired(gated):
    """recover_expired locks lease-expired deliveries in ordinal order with no
    target lock; activation must take the same order or the two deadlock."""
    store = gated.store
    first_bot, second_bot = gated.bot(), gated.bot()  # sorted: first_bot locks first
    deliveries, low = _claimed_delivery(gated, second_bot)  # lower ordinal, later bot
    _, high = _claimed_delivery(gated, first_bot)
    outcomes = {}
    with store.engine.connect() as recover:
        txn = recover.begin()
        pid = recover.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        recover.execute(sa_text("SELECT 1 FROM inter_bot_deliveries WHERE id=:id FOR UPDATE"),
                        {"id": low.id})
        activation = threading.Thread(target=lambda: outcomes.setdefault("gate", _activate(
            store, _spec(gated.owner(), GateScope.for_bots(first_bot, second_bot)))))
        activation.start()
        try:
            wait_blocked_by(store.engine, pid, activation)
            # recover_expired's next row: free, because activation waits on the
            # lower ordinal before it locks any higher one.
            recover.execute(sa_text("SELECT 1 FROM inter_bot_deliveries WHERE id=:id FOR UPDATE NOWAIT"),
                            {"id": high.id})
        finally:
            txn.rollback()
            activation.join(10)
    members = _members(store, outcomes["gate"].gate.id)
    assert {members[c.turn_id]["disposition"] for c in (low, high)} == {"revoked_unstarted"}
    assert {deliveries.get(c.id).status for c in (low, high)} == {"QUEUED"}


@pytest.mark.integration
def test_activation_locks_approvals_in_id_order_like_the_operator_purge(gated):
    """purge_ops locks in-flight approvals ORDER BY id with no target lock;
    activation must take approval producers in the same order."""
    from llm_bawt.approval_policies import ToolApprovalPolicyStore

    store = gated.store
    first_bot, second_bot = gated.bot(), gated.bot()
    approvals = ToolApprovalPolicyStore(None, engine=store.engine)
    approvals._ensure_tables_exist()
    prefix = f"req-gate-{uuid.uuid4().hex}"
    low_id, high_id = f"{prefix}-a", f"{prefix}-b"
    outcomes = {}
    try:
        low, low_turn = _approval_reservation(store, approvals, second_bot, low_id)
        high, high_turn = _approval_reservation(store, approvals, first_bot, high_id)
        with store.engine.connect() as purge:
            txn = purge.begin()
            pid = purge.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
            purge.execute(sa_text("SELECT 1 FROM tool_approval_requests WHERE id=:id FOR UPDATE"),
                          {"id": low_id})
            activation = threading.Thread(target=lambda: outcomes.setdefault("gate", _activate(
                store, _spec(gated.owner(), GateScope.for_bots(first_bot, second_bot)))))
            activation.start()
            try:
                wait_blocked_by(store.engine, pid, activation)
                purge.execute(sa_text("SELECT 1 FROM tool_approval_requests WHERE id=:id FOR UPDATE NOWAIT"),
                              {"id": high_id})
            finally:
                txn.rollback()
                activation.join(10)
        members = _members(store, outcomes["gate"].gate.id)
        assert {members[t]["disposition"] for t in (low_turn, high_turn)} == {"revoked_unstarted"}
        for row in (low, high):
            returned = approvals.get_request(row.id)
            assert (returned.continuation_state, returned.continuation_claim_token) == ("pending", None)
    finally:
        with store.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM tool_approval_requests WHERE id = ANY(:ids)"),
                         {"ids": [low_id, high_id]})


@pytest.mark.integration
def test_b4_hold_after_acceptance_keeps_the_member_and_exempts_reattach(gated):
    store, bot = gated.store, gated.bot()
    deliveries, claimed = _claimed_delivery(gated, bot)
    with store.engine.begin() as conn:  # the bridge request was bound before chat.send
        conn.execute(sa_text("UPDATE turn_logs SET agent_request_id='req-bound' WHERE id=:id"),
                     {"id": claimed.turn_id})
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot)))

    member = _members(store, gate.gate.id)[claimed.turn_id]
    assert (member["disposition"], member["agent_request_id"]) == ("accepted", "req-bound")
    assert deliveries.get(claimed.id).status == "DISPATCHING"
    assert _blocks(store, bot, "save_turn", claimed.turn_id) is None  # C1 re-attach
    assert _blocks(store, bot, "delivery_claim", claimed.turn_id) is None
    assert _blocks(store, bot, "steer", claimed.turn_id) is not None  # steers are not re-attach
    _save(store, bot, turn_id=claimed.turn_id, admission="idle_only",
          claim=_delivery_claim(deliveries, claimed))
    assert store.get_turn(claimed.turn_id).status == "streaming"
    with pytest.raises(SystemGateRefused):  # any other start is still held
        _save(store, bot)


@pytest.mark.integration
def test_b4_held_approval_refunds_and_reclaims_after_thaw(gated):
    from llm_bawt.approval_policies import ToolApprovalPolicyStore

    store, bot = gated.store, gated.bot()
    approvals = ToolApprovalPolicyStore(None, engine=store.engine)
    approvals._ensure_tables_exist()
    request_id = f"req-gate-{uuid.uuid4().hex}"
    try:
        row, turn_id = _approval_reservation(store, approvals, bot, request_id)
        attempts = row.continuation_attempts
        owner_id = gated.owner()
        gate = _activate(store, _spec(owner_id, GateScope.for_bots(bot)))

        member = _members(store, gate.gate.id)[turn_id]
        assert (member["disposition"], member["producer_kind"], member["producer_id"]) == (
            "revoked_unstarted", "approval", row.id)
        held = approvals.get_request(row.id)
        assert held.continuation_state == "pending" and held.continuation_claim_token is None
        assert held.continuation_attempts == attempts - 1
        assert store.get_turn(turn_id) is None

        with store.engine.begin() as conn:
            release_gates(conn, owner_id=owner_id)
        approvals_row = approvals.claim_continuation(row.id, lease_seconds=0)
        assert approvals_row.continuation_claim_token not in (None, row.continuation_claim_token)
        store.reserve_idle_turn(
            turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="c",
            request_json={"admission": "idle_only"}, trigger_message_id=str(uuid.uuid4()),
            owner=approvals_row.continuation_claim_token,
            fence=approvals.continuation_claim_fence(row.id, approvals_row.continuation_claim_token),
        )
        assert store.get_turn(turn_id).status == "reserved"
    finally:
        with store.engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM tool_approval_requests WHERE id=:id"), {"id": request_id})


# ---------------------------------------------------------------------------
# B1 drain status and the gate-loss rebuild (a union, never a drop)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_drain_status_reads_execution_exit_for_the_pinned_request(gated):
    store, bot = gated.store, gated.bot()
    _, claimed = _claimed_delivery(gated, bot)  # revoked at activation
    exited = _open(store, bot, request_id="req-exited")
    unconfirmed = _open(store, bot, request_id="req-unconfirmed")
    other = _open(store, bot, request_id="req-other")
    late = _open(store, bot)  # request bound only after capture
    bare = _open(store, bot)
    rebound = _open(store, bot, request_id="req-pinned")  # row later bound to another request
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    _exit(store, exited, request_id="req-exited", exited=True)
    _exit(store, unconfirmed, request_id="req-unconfirmed", exited=False)
    _exit(store, other, request_id="req-someone-else", exited=True)
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET agent_request_id='req-late' WHERE id=:id"), {"id": late})
    _exit(store, late, request_id="req-late", exited=True, source="b")
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET agent_request_id='req-new' WHERE id=:id"), {"id": rebound})
    _exit(store, rebound, request_id="req-new", exited=True)  # the pinned request decides

    with store.engine.connect() as conn:
        status = {s.turn_id: (s.drained, s.reason) for s in gate_drain_status(conn, gate.id)}
        assert member_drained(conn, gate.id, exited).reason == "execution_exit:a"
        assert member_drained(conn, gate.id, "turn-not-a-member") is None
    assert status == {
        claimed.turn_id: (True, "revoked_unstarted"),
        exited: (True, "execution_exit:a"),
        unconfirmed: (False, "exit_unconfirmed"),
        other: (False, "request_mismatch"),
        late: (True, "execution_exit:b"),
        bare: (False, "no_request_id"),
        rebound: (False, "request_mismatch"),
    }


@pytest.mark.integration
def test_rebuild_carries_an_ended_member_with_a_live_run_and_drops_a_drained_one(gated):
    store, bot = gated.store, gated.bot()
    owner_id = gated.owner()
    spec = _spec(owner_id, GateScope.for_bots(bot))
    running = _open(store, bot, request_id="req-running")
    live_run = _open(store, bot, request_id="req-live")  # row will end, run not confirmed exited
    exited = _open(store, bot, request_id="req-exited")
    accepted = _open(store, bot, "reserved", request_id="req-accepted")  # accepted, then ends
    first = _activate(store, spec).gate
    assert _members(store, first.id)[accepted]["disposition"] == "accepted"
    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE gate_members SET ack_status='stopping', ack_note=turn_id, acked_at=now()"
            " WHERE gate_id=:g AND turn_id IN (:a, :b)"
        ), {"g": first.id, "a": running, "b": live_run})
        conn.execute(sa_text("UPDATE gate_members SET control_state='steered' WHERE gate_id=:g"),
                     {"g": first.id})
    _exit(store, live_run, ended=True, request_id="req-live", exited=False)  # abort_unconfirmed
    _exit(store, exited, ended=True, request_id="req-exited", exited=True)
    _exit(store, accepted, ended=True)
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='error' WHERE id=:id"), {"id": accepted})
    with store.engine.begin() as conn:  # gate loss
        release_gates(conn, owner_id=owner_id, gate_id=first.id)

    with store.engine.begin() as conn:
        rebuilt = activate_hold(conn, spec, predecessor=first.id)
    members = _members(store, rebuilt.gate.id)
    assert set(members) == {running, live_run, accepted}  # the drained member is not carried
    assert {m.turn_id for m in rebuilt.members} == {running, live_run, accepted}
    assert members[live_run]["agent_request_id"] == "req-live"  # carried with its identity
    # Carried rows are exit-tracked only: an ended "accepted" row loses the C1 exemption.
    assert {members[t]["disposition"] for t in (live_run, accepted)} == {"running"}
    assert members[accepted]["turn_status"] == "error"  # the row's current status, not the capture's
    assert _blocks(store, bot, turn_id=accepted) is not None
    for turn_id in (running, live_run):  # acks survive (fresh capture or carry); control restarts
        assert (members[turn_id]["ack_status"], members[turn_id]["ack_note"]) == ("stopping", turn_id)
        assert members[turn_id]["control_state"] == "pending"


@pytest.mark.integration
def test_rebuild_carries_a_member_whose_bot_left_the_scope(gated):
    """The roster is pinned at activation (r2 S3): a bot outside the rebuilt
    scope still has an undrained run, so the union keeps it."""
    store = gated.store
    kept_bot, left_bot = gated.bot(), gated.bot()
    owner_id = gated.owner()
    live = _open(store, left_bot, request_id="req-left")
    first = _activate(store, _spec(owner_id, GateScope.for_bots(kept_bot, left_bot))).gate
    _exit(store, live, ended=True, request_id="req-left", exited=False)
    with store.engine.begin() as conn:
        release_gates(conn, owner_id=owner_id, gate_id=first.id)
        rebuilt = activate_hold(conn, _spec(owner_id, GateScope.for_bots(kept_bot)), predecessor=first.id)
    member = _members(store, rebuilt.gate.id)[live]
    assert (member["bot_id"], member["disposition"], member["control_state"]) == (left_bot, "running", "pending")
    assert _blocks(store, left_bot) is None  # scope governs admission: the bot is admitted
    assert _blocks(store, kept_bot) is not None


@pytest.mark.integration
def test_rebuild_refuses_another_owners_generation(gated):
    store, bot = gated.store, gated.bot()
    theirs = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    with store.engine.begin() as conn:
        with pytest.raises(ValueError):
            activate_hold(conn, _spec(gated.owner(), GateScope.for_bots(bot)), predecessor=theirs.id)


# ---------------------------------------------------------------------------
# S3, strictest gate, allow_actors, owners (C6), operator rosters
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_s3_backend_scope_is_resolved_at_check_time(gated):
    store, bot = gated.store, gated.bot(backend="claude-code")
    held_backend = f"test-backend-{gated.tag}"
    _activate(store, _spec(gated.owner(), GateScope.on_backends(held_backend)))
    assert _blocks(store, bot) is None
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE bot_profiles SET agent_backend=:b WHERE slug=:s"),
                     {"b": held_backend, "s": bot})
    assert _blocks(store, bot) is not None  # drifted into the held backend
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE bot_profiles SET agent_backend='claude-code' WHERE slug=:s"),
                     {"s": bot})
    assert _blocks(store, bot) is None  # drifted out


@pytest.mark.integration
def test_strictest_gate_wins_and_allow_actors_pass(gated):
    store, bot = gated.store, gated.bot()
    scope = GateScope.for_bots(bot)
    hold = _activate(store, _spec(gated.owner(), scope, ttl=900))
    deny = _activate(store, _spec(gated.owner(), scope, mode="deny", ttl=60, allow=frozenset({"schedule"})))
    assert _blocks(store, bot).id == deny.gate.id  # deny before hold, despite the earlier expiry
    assert _blocks(store, bot, "schedule").id == hold.gate.id  # deny allows schedule; hold doesn't


@pytest.mark.integration
def test_c6_release_and_extend_are_owner_matched(gated):
    store, bot = gated.store, gated.bot()
    job, other = gated.owner(), gated.owner()
    first = _activate(store, _spec(job, GateScope.for_bots(bot)))
    rebuilt = _activate(store, _spec(job, GateScope.for_bots(bot)))  # a new generation
    foreign = _activate(store, _spec(other, GateScope.for_bots(bot)))
    with store.engine.begin() as conn:
        assert not extend_gate(conn, first.gate.id, owner_id=other, ttl_seconds=900)
        assert extend_gate(conn, first.gate.id, owner_id=job, ttl_seconds=900)
        assert release_gates(conn, owner_id=job) == sorted([first.gate.id, rebuilt.gate.id])
        assert not extend_gate(conn, first.gate.id, owner_id=job, ttl_seconds=900)  # never revived
        assert gate_is_live(conn, foreign.gate.id)  # one job's thaw leaves other gates intact
    assert _blocks(store, bot).id == foreign.gate.id


@pytest.mark.integration
def test_operator_gate_roster_is_never_steered(gated):
    store, bot = gated.store, gated.bot()
    running = _open(store, bot)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot), owner_kind="operator", mode="deny"))
    member = _members(store, gate.gate.id)[running]
    assert (member["disposition"], member["control_state"]) == ("running", "not_needed")
    assert json.loads(json.dumps(member["gate_id"])) == gate.gate.id


@pytest.mark.integration
def test_claim_admission_without_gate_rows_is_unaffected(gated):
    """The ordinary reservation path with an owner claim still works under no gate."""
    store, bot = gated.store, gated.bot()
    turn_id = f"turn-{uuid.uuid4().hex}"
    store.reserve_idle_turn(turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="x",
                            request_json={"admission": "idle_only"},
                            trigger_message_id=str(uuid.uuid4()), owner="owner-a")
    _save(store, bot, turn_id=turn_id, admission="idle_only",
          claim=AdmissionClaim(owner="owner-a", fence=lambda conn: True))
    assert store.get_turn(turn_id).status == "streaming"


# ---------------------------------------------------------------------------
# test_Z: a member whose turn row is gone, and the startup column guard
# ---------------------------------------------------------------------------


def _bind(store, turn_id, request_id):
    store.bind_agent_bridge(turn_id=turn_id, backend="claude-code", session_key="s", request_id=request_id)


def _delete_turns(store, *turn_ids):
    with store.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM turn_logs WHERE id = ANY(:ids)"), {"ids": list(turn_ids)})


def _drain(store, gate_id):
    with store.engine.connect() as conn:
        return {s.turn_id: (s.drained, s.reason) for s in gate_drain_status(conn, gate_id)}


@pytest.mark.integration
@pytest.mark.parametrize("holder_sql", [
    "SELECT count(*) FROM gate_members",  # Vex N2: an open reader
    "UPDATE gate_members SET ack_note = ack_note WHERE false",  # an open writer
])
def test_bootstrap_on_a_migrated_schema_waits_on_no_open_transaction(gated, holder_sql):
    from llm_bawt.service.turn_gates import (
        _GATE_RELATIONS,
        _MEMBER_ADDED_COLUMNS,
        _MEMBER_RETIRED_COLUMNS,
        _bootstrap,
    )
    from llm_bawt.utils.schema import SchemaBootstrapGuard, drop_columns, ensure_columns, ensure_relations

    engine = gated.store.engine
    with engine.begin() as conn:  # the declared defaults are spelled as the catalog reports them
        assert ensure_relations(conn, _GATE_RELATIONS) == []
        assert ensure_columns(conn, "gate_members", _MEMBER_ADDED_COLUMNS) == []
        assert drop_columns(conn, "gate_members", _MEMBER_RETIRED_COLUMNS) == []

    def bootstrap(conn):
        conn.execute(sa_text("SET LOCAL lock_timeout = '300ms'"))
        _bootstrap(conn)

    with engine.connect() as holder:
        txn = holder.begin()
        holder.execute(sa_text(holder_sql))
        try:
            assert SchemaBootstrapGuard().run(engine, "test-gates", bootstrap) is True
        finally:
            txn.rollback()


@pytest.mark.integration
def test_ensure_columns_adds_null_columns_and_repairs_only_a_drifted_default(gated):
    from llm_bawt.utils.schema import ensure_columns, ensure_relations

    columns = {"changed_at": ("TIMESTAMPTZ", "now()"), "evidence": ("JSONB", None)}
    with gated.store.engine.begin() as conn:
        conn.execute(sa_text("CREATE TEMP TABLE caid_ensure (id INT) ON COMMIT DROP"))
        conn.execute(sa_text("INSERT INTO caid_ensure VALUES (1)"))
        assert ensure_columns(conn, "caid_ensure", columns) == [
            "ALTER TABLE caid_ensure ADD COLUMN IF NOT EXISTS changed_at TIMESTAMPTZ",
            "ALTER TABLE caid_ensure ALTER COLUMN changed_at SET DEFAULT now()",
            "ALTER TABLE caid_ensure ADD COLUMN IF NOT EXISTS evidence JSONB",
        ]
        conn.execute(sa_text("INSERT INTO caid_ensure (id) VALUES (2)"))
        rows = dict(conn.execute(sa_text("SELECT id, changed_at IS NULL FROM caid_ensure")).all())
        assert rows == {1: True, 2: False}  # existing rows stay NULL; new rows take the default
        assert ensure_columns(conn, "caid_ensure", columns) == []
        conn.execute(sa_text("ALTER TABLE caid_ensure ALTER COLUMN changed_at DROP DEFAULT"))
        assert ensure_columns(conn, "caid_ensure", columns) == [
            "ALTER TABLE caid_ensure ALTER COLUMN changed_at SET DEFAULT now()"]
        index = {"caid_ensure_ix": "CREATE INDEX IF NOT EXISTS caid_ensure_ix ON caid_ensure (id)"}
        assert ensure_relations(conn, index) == ["caid_ensure_ix"]
        assert ensure_relations(conn, index) == []


@pytest.mark.integration
def test_drop_columns_drops_only_what_the_table_still_has_in_one_statement(gated):
    from llm_bawt.utils.schema import drop_columns

    with gated.store.engine.begin() as conn:
        conn.execute(sa_text("CREATE TEMP TABLE caid_drop (id INT, a INT, b TEXT, keep INT) ON COMMIT DROP"))
        assert drop_columns(conn, "caid_drop", ["a", "gone", "b"]) == [
            "ALTER TABLE caid_drop DROP COLUMN IF EXISTS a, DROP COLUMN IF EXISTS b"]
        left = [r[0] for r in conn.execute(sa_text(
            "SELECT attname FROM pg_attribute WHERE attrelid = 'caid_drop'::regclass"
            " AND attnum > 0 AND NOT attisdropped ORDER BY attnum")).all()]
        assert left == ["id", "keep"]
        assert drop_columns(conn, "caid_drop", ["a", "gone", "b"]) == []


@pytest.mark.integration
def test_drop_columns_behind_an_open_reader_is_skipped_and_retried_next_time(gated):
    """Codex S4 r1 S1: the contract ALTER never queues readers behind it for longer than lock_timeout."""
    from llm_bawt.utils.schema import drop_columns

    engine = gated.store.engine
    table = f"caid_drop_{uuid.uuid4().hex[:12]}"
    with engine.begin() as conn:
        conn.execute(sa_text(f"CREATE TABLE {table} (id INT, retired INT)"))
    try:
        with engine.connect() as holder:
            txn = holder.begin()
            holder.execute(sa_text(f"SELECT count(*) FROM {table}"))
            try:
                with engine.begin() as conn:
                    conn.execute(sa_text("SET LOCAL lock_timeout = '10s'"))
                    started = time.monotonic()
                    assert drop_columns(conn, table, ["retired"], lock_timeout="200ms") == []
                    # timing-ok: real PG lock_timeout: its own 200ms vs the caller's 10s
                    assert time.monotonic() - started < 3  # its own timeout, not the caller's 10s
                    assert conn.execute(sa_text("SELECT current_setting('lock_timeout')")).scalar_one() == "10s"
                    conn.execute(sa_text(f"SELECT retired FROM {table}"))  # the txn is still usable
            finally:
                txn.rollback()
        with engine.begin() as conn:
            assert drop_columns(conn, table, ["retired"]) == [
                f"ALTER TABLE {table} DROP COLUMN IF EXISTS retired"]
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text(f"DROP TABLE IF EXISTS {table}"))


def _member_columns(conn):
    return {r[0] for r in conn.execute(sa_text(
        "SELECT attname FROM pg_attribute WHERE attrelid = 'gate_members'::regclass"
        " AND attnum > 0 AND NOT attisdropped")).all()}


@pytest.mark.integration
def test_bootstrap_contracts_a_pre_s4_roster_and_keeps_its_rows(gated):
    """TASK-1431 S4: a table still carrying the B3 pump's six columns loses them; rows survive."""
    from llm_bawt.service.turn_gates import _MEMBER_RETIRED_COLUMNS, _bootstrap

    store, bot = gated.store, gated.bot()
    turn = _open(store, bot)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    with store.engine.connect() as conn:
        txn = conn.begin()  # DDL is transactional: the rollback restores the shared table
        try:
            assert not set(_MEMBER_RETIRED_COLUMNS) & _member_columns(conn)
            conn.execute(sa_text(
                "ALTER TABLE gate_members ADD COLUMN control_request_id VARCHAR(255),"
                " ADD COLUMN control_claim_token VARCHAR(128),"
                " ADD COLUMN control_lease_expires_at TIMESTAMPTZ,"
                " ADD COLUMN control_attempts INTEGER NOT NULL DEFAULT 0,"
                " ADD COLUMN control_next_attempt_at TIMESTAMPTZ,"
                " ADD COLUMN control_generation INTEGER"))
            _bootstrap(conn)
            assert not set(_MEMBER_RETIRED_COLUMNS) & _member_columns(conn)
            assert conn.execute(sa_text(
                "SELECT control_state FROM gate_members WHERE gate_id=:g AND turn_id=:t"),
                {"g": gate.id, "t": turn}).scalar_one() == "pending"
        finally:
            txn.rollback()


@pytest.mark.integration
def test_bind_pins_the_request_on_every_member_row_of_the_turn(gated):
    store, bot = gated.store, gated.bot()
    turn, prepinned = _open(store, bot), _open(store, bot)
    live = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    expired = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate  # lost, not released
    released_owner = gated.owner()
    released = _activate(store, _spec(released_owner, GateScope.for_bots(bot))).gate
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g"),
                     {"g": expired.id})
        release_gates(conn, owner_id=released_owner)
        conn.execute(sa_text("UPDATE gate_members SET agent_request_id='req-pinned' WHERE turn_id=:t"),
                     {"t": prepinned})
    _bind(store, turn, "req-bound")
    _bind(store, prepinned, "req-rebound")
    for gate in (live, expired, released):  # never_bound is sound only if every bind pins
        assert _members(store, gate.id)[turn]["agent_request_id"] == "req-bound"
        assert _members(store, gate.id)[prepinned]["agent_request_id"] == "req-pinned"  # the pin decides
    _delete_turns(store, turn)
    assert _drain(store, live.id)[turn] == (False, "turn_missing:no_execution_exit")


@pytest.mark.integration
def test_a_missing_turn_drains_on_member_row_evidence_or_never_bound(gated):
    store, bot = gated.store, gated.bot()
    owner_id = gated.owner()
    spec = _spec(owner_id, GateScope.for_bots(bot))
    pinned, unbound = _open(store, bot, request_id="req-pinned"), _open(store, bot)
    mismatched, present = _open(store, bot, request_id="req-mine"), _open(store, bot, request_id="req-present")
    gate = _activate(store, spec).gate
    _delete_turns(store, pinned, unbound, mismatched)
    record = {"backend": "claude-code", "boot_id": "boot-1", "source": "b", "exited": True,
              "at": "2026-10-22T15:00:00+00:00"}
    ids = {row["turn_id"]: row["id"] for row in _members(store, gate.id).values()}
    with store.engine.begin() as conn:
        assert stamp_member_exit(conn, ids[mismatched], {**record, "request_id": "req-other"}) is True
        assert stamp_member_exit(conn, ids[present], {**record, "request_id": "req-present"}) is False
    assert _members(store, gate.id)[present]["execution_exit"] is None  # the turn row stays the home
    assert _drain(store, gate.id) == {
        pinned: (False, "turn_missing:no_execution_exit"),
        unbound: (True, "never_bound"),
        mismatched: (False, "turn_missing:request_mismatch"),
        present: (False, "no_execution_exit"),
    }

    with store.engine.begin() as conn:  # gate loss: the pinned identity is carried, never_bound is not
        release_gates(conn, owner_id=owner_id, gate_id=gate.id)
    with store.engine.begin() as conn:
        rebuilt = activate_hold(conn, spec, predecessor=gate.id).gate
    members = _members(store, rebuilt.id)
    assert set(members) == {pinned, mismatched, present}
    assert members[pinned]["agent_request_id"] == "req-pinned"
    with store.engine.begin() as conn:
        assert stamp_member_exit(conn, members[pinned]["id"], {**record, "request_id": "req-pinned"})
    assert _drain(store, rebuilt.id)[pinned] == (True, "execution_exit:b")


@pytest.mark.integration
def test_a_bind_in_the_gate_loss_window_survives_a_turn_delete_before_the_rebuild(gated):
    """Snark z-r1: capture -> gate lost -> bind -> turn row deleted -> rebuild with predecessor."""
    store, bot = gated.store, gated.bot()
    owner_id = gated.owner()
    spec = _spec(owner_id, GateScope.for_bots(bot))
    turn = _open(store, bot)
    first = _activate(store, spec).gate
    with store.engine.begin() as conn:
        release_gates(conn, owner_id=owner_id, gate_id=first.id)
    _bind(store, turn, "req-lost-window")
    _delete_turns(store, turn)  # e.g. agent_context.py's head-turn delete on recovery
    assert _drain(store, first.id)[turn] == (False, "turn_missing:no_execution_exit")  # not never_bound
    with store.engine.begin() as conn:
        rebuilt = activate_hold(conn, spec, predecessor=first.id).gate
    assert _members(store, rebuilt.id)[turn]["agent_request_id"] == "req-lost-window"
    assert _drain(store, rebuilt.id)[turn] == (False, "turn_missing:no_execution_exit")


@pytest.mark.integration
def test_rebuild_carries_a_request_the_member_never_pinned(gated):
    """Loopy z-r1 second guard: a turn bound outside bind_agent_bridge, ended unconfirmed, carried, deleted."""
    store, bot = gated.store, gated.bot()
    owner_id = gated.owner()
    spec = _spec(owner_id, GateScope.for_bots(bot))
    turn = _open(store, bot)
    first = _activate(store, spec).gate
    with store.engine.begin() as conn:
        release_gates(conn, owner_id=owner_id, gate_id=first.id)
        conn.execute(sa_text("UPDATE turn_logs SET agent_request_id='req-late' WHERE id=:id"), {"id": turn})
    _exit(store, turn, ended=True)  # ended with no confirmed exit: the run may be live
    with store.engine.begin() as conn:
        rebuilt = activate_hold(conn, spec, predecessor=first.id).gate
    assert _members(store, rebuilt.id)[turn]["agent_request_id"] == "req-late"
    _delete_turns(store, turn)
    assert _drain(store, rebuilt.id)[turn] == (False, "turn_missing:no_execution_exit")


@pytest.mark.integration
def test_capture_waits_for_an_uncommitted_bind_and_reads_its_request(gated):
    store, bot = gated.store, gated.bot()
    turn = _open(store, bot)
    result = {}
    with store.engine.connect() as binder:
        txn = binder.begin()
        pid = binder.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        binder.execute(sa_text("UPDATE turn_logs SET agent_request_id='req-race' WHERE id=:id"), {"id": turn})
        activation = threading.Thread(target=lambda: result.setdefault(
            "gate", _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate))
        activation.start()
        try:
            wait_blocked_by(store.engine, pid, activation)
        finally:
            txn.commit()
            activation.join(10)
    assert _members(store, result["gate"].id)[turn]["agent_request_id"] == "req-race"


@pytest.mark.integration
def test_a_bind_behind_an_uncommitted_capture_pins_the_member(gated):
    store, bot = gated.store, gated.bot()
    turn = _open(store, bot)
    with store.engine.connect() as capturer:
        txn = capturer.begin()
        pid = capturer.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        gate = activate_hold(capturer, _spec(gated.owner(), GateScope.for_bots(bot))).gate
        bind = threading.Thread(target=_bind, args=(store, turn, "req-race"))
        bind.start()
        try:
            wait_blocked_by(store.engine, pid, bind)
        finally:
            txn.commit()
            bind.join(10)
    assert _members(store, gate.id)[turn]["agent_request_id"] == "req-race"


def _carried_out_of_scope(gated):
    """A predecessor whose unbound member's bot is outside the rebuilt scope (no target lock covers it)."""
    store = gated.store
    kept_bot, left_bot = gated.bot(), gated.bot()
    owner_id = gated.owner()
    turn = _open(store, left_bot)
    first = _activate(store, _spec(owner_id, GateScope.for_bots(kept_bot, left_bot))).gate
    with store.engine.begin() as conn:
        release_gates(conn, owner_id=owner_id, gate_id=first.id)
    return turn, first, _spec(owner_id, GateScope.for_bots(kept_bot))


@pytest.mark.integration
def test_a_bind_behind_an_uncommitted_rebuild_pins_the_carried_member(gated):
    """Loopy xp-z-r1: carry_forward's read and insert must not let a bind fall between them."""
    store = gated.store
    turn, first, spec = _carried_out_of_scope(gated)
    with store.engine.connect() as rebuilder:
        txn = rebuilder.begin()
        pid = rebuilder.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        rebuilt = activate_hold(rebuilder, spec, predecessor=first.id)
        assert turn in {m.turn_id for m in rebuilt.members}  # carried, still unbound
        rebuilt = rebuilt.gate
        bind = threading.Thread(target=_bind, args=(store, turn, "req-carry-race"))
        bind.start()
        try:
            wait_blocked_by(store.engine, pid, bind)
        finally:
            txn.commit()
            bind.join(10)
    assert _members(store, rebuilt.id)[turn]["agent_request_id"] == "req-carry-race"
    _delete_turns(store, turn)
    assert _drain(store, rebuilt.id)[turn] == (False, "turn_missing:no_execution_exit")  # never never_bound


@pytest.mark.integration
def test_a_rebuild_waits_for_an_uncommitted_bind_and_carries_its_request(gated):
    store = gated.store
    turn, first, spec = _carried_out_of_scope(gated)
    result = {}
    with store.engine.connect() as binder:
        txn = binder.begin()
        pid = binder.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        binder.execute(sa_text("SELECT 1 FROM turn_logs WHERE id=:id FOR UPDATE"), {"id": turn})
        binder.execute(sa_text("UPDATE turn_logs SET agent_request_id='req-carry-race' WHERE id=:id"), {"id": turn})
        binder.execute(sa_text(
            "UPDATE gate_members SET agent_request_id='req-carry-race' WHERE turn_id=:id AND agent_request_id IS NULL"
        ), {"id": turn})

        def rebuild():
            with store.engine.begin() as conn:
                result["gate"] = activate_hold(conn, spec, predecessor=first.id).gate

        rebuilding = threading.Thread(target=rebuild)
        rebuilding.start()
        try:
            wait_blocked_by(store.engine, pid, rebuilding)
        finally:
            txn.commit()
            rebuilding.join(10)
    assert _members(store, result["gate"].id)[turn]["agent_request_id"] == "req-carry-race"
