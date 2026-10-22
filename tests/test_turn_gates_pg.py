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
)
from llm_bawt.service.turn_gates import (
    GateActor,
    GateScope,
    GateSpec,
    extend_gate,
    gate_blocks,
    gate_is_live,
    release_gates,
)
from tests.pg_isolation import isolated_pg_store
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


def _members(store, gate_id):
    with store.engine.connect() as conn:
        return {row["turn_id"]: dict(row) for row in conn.execute(sa_text(
            "SELECT * FROM gate_members WHERE gate_id=:g"
        ), {"g": gate_id}).mappings()}


def _claimed_delivery(ctx, bot):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    deliveries = InterBotDeliveryStore(ctx.store.config)
    sender = f"test-gate-sender-{uuid.uuid4().hex}"
    ctx.senders.append(sender)
    record, _ = deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}],
                 "bot_id": bot, "user": "nick", "stream": True},
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


def _wait_blocked_by(engine, holder_pid, worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    with engine.connect() as observer:
        while time.monotonic() < deadline:
            if observer.execute(sa_text("""
                SELECT EXISTS (SELECT 1 FROM pg_stat_activity
                    WHERE datname=current_database() AND wait_event_type='Lock'
                      AND :pid = ANY(pg_blocking_pids(pid)))
            """), {"pid": holder_pid}).scalar_one():
                assert worker.is_alive()
                return
            time.sleep(0.02)
    pytest.fail("activation did not wait on the holder's lock")


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
        _wait_blocked_by(store.engine, holder["pid"], activation)
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
            _wait_blocked_by(store.engine, pid, activation)
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
            _wait_blocked_by(store.engine, pid, activation)
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
                _wait_blocked_by(store.engine, pid, activation)
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
