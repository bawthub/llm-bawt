"""Private-PostgreSQL regressions for initiating-session failure notices.

No dispatcher workers, model calls, sleeps or live database are involved. Every
origin has a real owned session, trigger message and matching initiating turn.
"""
from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.bot_identities import SYSTEM_BOT_ID
from llm_bawt.bot_session_state import BotSessionState
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_failure import DECISION, NOTICE_KEY_PREFIX, NOTICE_KIND, ORIGIN, FailureOrigin
from llm_bawt.memory.postgresql import (
    PARENT_TABLES, _sanitize_table_name, ensure_bot_partitions, partition_name,
)
from llm_bawt.service.delivery_failure_notice import FailureNoticePump
from llm_bawt.service.delivery_failure_notice_prompt import PROMPT_KEY
from llm_bawt.task_turn_context import TaskTurnContext
from tests.bot_tag_fixtures import PRIVATE_MODES, clear_tags, make_private
from tests.test_turn_gates_pg import gated  # noqa: F401 -- shared fixture

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]
SECRET = "private-secret-do-not-forward"
TOOL_ARG = "tool_args={delete_path:/sensitive/customer-data}"


def _origin(ctx, bot, *, status="archived"):
    caller = TaskTurnContext(
        bot_id=bot, user_id="initiating-owner", session_id=str(uuid.uuid4()),
        turn_id=f"turn-{uuid.uuid4().hex}", trigger_message_id=str(uuid.uuid4()), issued_at=123,
    )
    with ctx.store.engine.begin() as conn:
        ensure_bot_partitions(conn, bot)
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status)
            VALUES (:session, :bot, :user, :status)
        """), {"session": caller.session_id, "bot": bot, "user": caller.user_id, "status": status})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:partition, :message, 'user', 'delegate work', 1, :session)
        """), {"partition": _sanitize_table_name(bot), "message": caller.trigger_message_id,
               "session": caller.session_id})
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, ended_at, path, stream, bot_id,
                user_id, status, user_prompt, response_text, trigger_message_id)
            VALUES (:turn, now(), now(), '/test', true, :bot, :user,
                'completed', 'delegate work', '', :message)
        """), {"turn": caller.turn_id, "bot": bot, "user": caller.user_id,
               "message": caller.trigger_message_id})
    return caller


@pytest.fixture
def notices(gated):
    # The shared fixture removes source rows; notices are system-authored. This
    # module owns its entire private DB, so tracking the system sender is safe.
    gated.senders.append(SYSTEM_BOT_ID)
    gated.deliveries = InterBotDeliveryStore(gated.store.config)
    gated.pump = FailureNoticePump(gated.deliveries)
    gated.sender, gated.target = gated.bot(), gated.bot()
    gated.origin = _origin(gated, gated.sender)
    gated.current = _origin(gated, gated.sender, status="active")
    with gated.store.engine.begin() as conn:
        BotSessionState.accept(conn, bot_id=gated.sender, session_id=gated.current.session_id)
    yield gated
    with gated.store.engine.begin() as conn:
        clear_tags(conn, *gated.bots)
        for bot in gated.bots:
            for parent in PARENT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, bot)}"))
            conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})


def _send(ctx, *, caller=None, sender=None, target=None, metadata=None, **extra):
    caller = caller or ctx.origin
    sender, target = sender or caller.bot_id, target or ctx.target
    if sender not in ctx.senders:
        ctx.senders.append(sender)
    return ctx.deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=target,
        message=f"do work {SECRET} {TOOL_ARG}",
        payload={"bot_id": target, "user": "worker-owner", "stream": True,
                 "messages": [{"role": "user", "content": f"do work {SECRET} {TOOL_ARG}"}]},
        sender_turn_id=caller.turn_id,
        metadata={ORIGIN: FailureOrigin.capture(caller), **(metadata or {})}, **extra,
    )


def _claim(ctx, record, *, steer=False):
    claim = ctx.deliveries.claim_next(record.target_bot_id, claim_owner="test-worker", steer_capable=steer)
    assert claim is not None and claim.id == record.id
    return claim


def _delivered(ctx, claim):
    assert ctx.deliveries.mark_transport_accepted(claim.id, claim.claim_token)
    delivered = ctx.deliveries.mark_delivered(
        claim.id, claim.claim_token, response_model=None, response_chars=0,
    )
    assert delivered is not None and delivered.status == "DELIVERED"
    return delivered


def _end(ctx, turn_id, *, status="error", error=None):
    with ctx.store.engine.begin() as conn:
        assert conn.execute(text("""
            UPDATE turn_logs SET status=:status, ended_at=now(), error_text=:error
            WHERE id=:turn
        """), {"turn": turn_id, "status": status,
               "error": f"ChatGPTEventTimeout: {SECRET} {TOOL_ARG}" if error is None else error}).rowcount == 1


def _row(ctx, delivery_id):
    with ctx.store.engine.connect() as conn:
        return dict(conn.execute(text("SELECT * FROM inter_bot_deliveries WHERE id=:id"),
                                 {"id": delivery_id}).mappings().one())


def _notices_for(ctx, delivery_id):
    with ctx.store.engine.connect() as conn:
        return [dict(row) for row in conn.execute(text(
            "SELECT * FROM inter_bot_deliveries WHERE idempotency_key=:key"
        ), {"key": f"{NOTICE_KEY_PREFIX}{delivery_id}"}).mappings()]


def _assert_notice(ctx, source, notice, *, error_code):
    stored = _row(ctx, notice.id)
    payload = stored["payload_json"]
    assert (stored["sender_bot_id"], stored["target_bot_id"], stored["message_kind"], stored["status"]) == (
        SYSTEM_BOT_ID, ctx.sender, NOTICE_KIND, "QUEUED",
    )
    assert payload["session_id"] == ctx.origin.session_id != ctx.current.session_id
    assert payload["user"] == ctx.origin.user_id != "worker-owner"
    assert payload["bot_id"] == ctx.sender and payload["prefer_steer"] is False
    assert payload["extract_memory"] is False and payload["augment_memory"] is False
    assert stored["metadata_json"]["failed_delivery_id"] == source.id
    assert stored["metadata_json"]["error_code"] == error_code
    assert stored["metadata_json"]["initiating_turn_id"] == ctx.origin.turn_id
    assert ORIGIN not in stored["metadata_json"]  # the callback cannot spawn a callback
    assert _row(ctx, source.id)["metadata_json"][DECISION] == {
        "outcome": "enqueued", "notice_delivery_id": notice.id,
    }
    forwarded = json.dumps({"message": stored["message"], "metadata": stored["metadata_json"],
                            "payload": payload})
    assert SECRET not in forwarded and TOOL_ARG not in forwarded
    assert source.id in stored["message"] and ctx.target in stored["message"]
    return stored


@pytest.mark.parametrize("accepted", [False, True], ids=["before-acceptance", "after-acceptance"])
def test_failed_delivery_notifies_the_owned_archived_initiating_session(notices, accepted):
    source, _ = _send(notices, task_id="TASK-1946", project_id="project-test")
    claim = _claim(notices, source)
    if accepted:
        assert notices.deliveries.mark_transport_accepted(source.id, claim.claim_token)
    failed = notices.deliveries.fail_claim(source.id, claim.claim_token, f"TimeoutError: {SECRET} {TOOL_ARG}")
    assert failed is not None and failed.status == "FAILED"

    result = notices.pump.run_pass()

    assert (len(result.notified), result.skipped, result.errors) == (1, 0, 0)
    stored = _assert_notice(notices, source, result.notified[0], error_code="timeout")
    assert (stored["task_id"], stored["project_id"]) == ("TASK-1946", "project-test")
    assert bool(_row(notices, source.id)["transport_accepted_at"]) is accepted
    with notices.store.engine.connect() as conn:
        assert BotSessionState.resolve(conn, bot_id=notices.sender)["id"] == notices.current.session_id


@pytest.mark.parametrize("status", ["error", "timeout"])
def test_delivered_owning_turn_failure_notifies_only_its_initiator(notices, status):
    source, _ = _send(notices)
    claim = _claim(notices, source)
    _delivered(notices, claim)
    _end(notices, claim.turn_id, status=status)

    result = notices.pump.run_pass()

    assert (len(result.notified), result.skipped, result.errors) == (1, 0, 0)
    stored = _assert_notice(notices, source, result.notified[0], error_code="upstream_event_timeout")
    assert stored["metadata_json"]["failed_turn_id"] == claim.turn_id
    assert stored["metadata_json"]["turn_status"] == status
    assert notices.deliveries.get(source.id).status == "DELIVERED"


def test_shared_delivered_steers_do_not_broadcast_the_owning_turn_failure(notices):
    source, _ = _send(notices)
    claim = _claim(notices, source, steer=True)
    _delivered(notices, claim)
    with notices.store.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": claim.turn_id})
    steers = []
    for _ in range(2):
        caller = _origin(notices, notices.bot())
        steer, _ = _send(notices, caller=caller)
        accepted = _claim(notices, steer, steer=True)
        assert (accepted.delivery_mode, accepted.target_turn_id) == ("steer", claim.turn_id)
        _delivered(notices, accepted)
        steers.append(steer)
    _end(notices, claim.turn_id)

    result = notices.pump.run_pass()

    assert (len(result.notified), result.skipped, result.errors) == (1, 0, 0)
    _assert_notice(notices, source, result.notified[0], error_code="upstream_event_timeout")
    for steer in steers:
        assert DECISION not in _row(notices, steer.id)["metadata_json"]
        assert _notices_for(notices, steer.id) == []


def test_a_steers_own_failed_send_notifies_its_sender_without_blaming_the_shared_turn(notices):
    source, _ = _send(notices)
    claim = _claim(notices, source, steer=True)
    _delivered(notices, claim)
    with notices.store.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": claim.turn_id})
    caller = _origin(notices, notices.bot())
    steer, _ = _send(notices, caller=caller)
    accepted = _claim(notices, steer, steer=True)
    assert accepted.delivery_mode == "steer"
    assert notices.deliveries.mark_transport_accepted(steer.id, accepted.claim_token)
    assert notices.deliveries.fail_claim(steer.id, accepted.claim_token, "delivery persistence failed")
    [notice] = notices.pump.run_pass().notified
    stored = _row(notices, notice.id)
    assert stored["target_bot_id"] == caller.bot_id
    assert stored["payload_json"]["session_id"] == caller.session_id
    assert stored["metadata_json"]["failed_delivery_id"] == steer.id
    assert stored["metadata_json"]["failed_turn_id"] == claim.turn_id
    assert stored["metadata_json"]["turn_status"] == "unattributed"
    assert _notices_for(notices, source.id) == []
    assert notices.store.get_turn(claim.turn_id).ended_at is None


@pytest.mark.parametrize("settlement", ["failed", "delivered", "stopped-after-start"])
def test_executed_stop_audit_never_becomes_a_failure_restart_notice(notices, settlement):
    from llm_bawt.service.turn_termination import TurnAbortCoordinator

    source, _ = _send(notices)
    claim = _claim(notices, source)
    assert notices.deliveries.mark_transport_accepted(source.id, claim.claim_token)
    with notices.store.engine.begin() as conn:
        conn.execute(text("""
            UPDATE turn_logs SET status='streaming', agent_request_id='test-started-request'
            WHERE id=:id
        """), {"id": claim.turn_id})
    assert TurnAbortCoordinator(notices.store).request(claim.turn_id, source="chat_stop", peer=None)
    if settlement == "failed":
        assert notices.deliveries.mark_failed(source.id, "late worker error")
    elif settlement == "delivered":
        _delivered(notices, claim)
    else:
        stopped = notices.deliveries.cancel_stopped_claim(source.id, claim.claim_token)
        assert stopped is not None and stopped.metadata["stopped_after_start"] is True
    # A late error finalizer must not erase the meaning of the executed Stop.
    _end(notices, claim.turn_id)
    audit = json.loads(notices.store.get_turn(claim.turn_id).request_json)["abort_request"]
    assert audit["source"] == "chat_stop"

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped, result.errors) == ([], 1, 0)
    assert _row(notices, source.id)["metadata_json"][DECISION]["outcome"] == "ineligible"
    assert _notices_for(notices, source.id) == []


@pytest.mark.parametrize("excluded", ["user", "self", "scheduled"])
def test_non_delegated_sources_are_not_notified(notices, excluded):
    from llm_bawt.message_authorship import AuthorReference

    extra = {}
    if excluded == "user":
        extra["author"] = AuthorReference.user("initiating-owner")
    elif excluded == "self":
        extra["target"] = notices.sender
    else:
        extra["metadata"] = {"prompt_schedule": {"schedule_id": "fixture"}}
    source, _ = _send(notices, **extra)
    assert notices.deliveries.mark_failed(source.id, "failure")
    assert notices.pump.run_pass().notified == []
    assert DECISION not in _row(notices, source.id)["metadata_json"]


@pytest.mark.parametrize("kind", [NOTICE_KIND, "DELIVERY_CANCELLED"])
def test_failed_notice_kinds_never_create_notice_loops(notices, kind):
    source, _ = _send(notices, message_kind=kind)
    assert notices.deliveries.mark_failed(source.id, "notice failed")

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped, result.errors) == ([], 0, 0)
    assert DECISION not in _row(notices, source.id)["metadata_json"]
    assert _notices_for(notices, source.id) == []


@pytest.mark.parametrize("side", ["sender", "target"])
@pytest.mark.parametrize("mode", [*PRIVATE_MODES, "no-profile"])
def test_private_or_unprofiled_bot_is_never_notified_or_decided(notices, side, mode):
    source, _ = _send(notices)
    assert notices.deliveries.mark_failed(source.id, "failure")
    bot = getattr(notices, side)
    with notices.store.engine.begin() as conn:
        if mode == "no-profile":
            conn.execute(text("DELETE FROM bot_profiles WHERE slug=:bot"), {"bot": bot})
        else:
            make_private(conn, bot, mode)

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped, result.errors) == ([], 0, 0)
    assert DECISION not in _row(notices, source.id)["metadata_json"]
    assert _notices_for(notices, source.id) == []


@pytest.mark.parametrize("broken", [
    "deleted-session", "missing-session", "foreign-session-bot", "foreign-session-user",
    "missing-turn", "foreign-turn-bot", "foreign-turn-user", "mismatched-turn-trigger",
    "missing-trigger", "foreign-trigger-session", "sender-turn-mismatch", "chat-profile",
])
def test_sender_context_requires_matching_turn_trigger_and_owned_session(notices, broken):
    source, _ = _send(notices)
    assert notices.deliveries.mark_failed(source.id, "failure")
    caller = notices.origin
    statements = {
        "deleted-session": "UPDATE sessions SET status='deleted' WHERE id=:session",
        "missing-session": "DELETE FROM sessions WHERE id=:session",
        "foreign-session-bot": "UPDATE sessions SET bot_id=:other WHERE id=:session",
        "foreign-session-user": "UPDATE sessions SET user_id='foreign-owner' WHERE id=:session",
        "missing-turn": "DELETE FROM turn_logs WHERE id=:turn",
        "foreign-turn-bot": "UPDATE turn_logs SET bot_id=:other WHERE id=:turn",
        "foreign-turn-user": "UPDATE turn_logs SET user_id='foreign-owner' WHERE id=:turn",
        "mismatched-turn-trigger": "UPDATE turn_logs SET trigger_message_id=:random WHERE id=:turn",
        "missing-trigger": "DELETE FROM messages WHERE bot_id=:partition AND id=:trigger",
        "foreign-trigger-session":
            "UPDATE messages SET session_id=:current WHERE bot_id=:partition AND id=:trigger",
        "sender-turn-mismatch": "UPDATE inter_bot_deliveries SET sender_turn_id=:random WHERE id=:delivery",
        "chat-profile": "UPDATE bot_profiles SET bot_type='chat', agent_backend='' WHERE slug=:sender",
    }
    sql = statements[broken]
    with notices.store.engine.begin() as conn:
        if broken == "missing-session":
            # Preserve the trigger but release its real FK before removing the
            # recorded origin session. No constraint bypass in the fixture.
            conn.execute(text("UPDATE messages SET session_id=:current WHERE session_id=:session"), {
                "current": notices.current.session_id, "session": caller.session_id,
            })
        conn.execute(text(sql), {
            "session": caller.session_id, "turn": caller.turn_id, "trigger": caller.trigger_message_id,
            "partition": _sanitize_table_name(caller.bot_id), "other": notices.target,
            "current": notices.current.session_id, "random": str(uuid.uuid4()),
            "sender": notices.sender, "delivery": source.id,
        })

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped, result.errors) == ([], 1, 0)
    assert _notices_for(notices, source.id) == []
    assert _row(notices, source.id)["metadata_json"][DECISION]["outcome"] == "ineligible"


def test_legacy_delivery_without_trusted_origin_does_not_guess_a_current_session(notices):
    source, _ = _send(notices)
    with notices.store.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET metadata_json=metadata_json - :origin WHERE id=:id"),
                     {"id": source.id, "origin": ORIGIN})
    assert notices.deliveries.mark_failed(source.id, "failure")

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped, result.errors) == ([], 0, 0)
    assert DECISION not in _row(notices, source.id)["metadata_json"]
    assert _notices_for(notices, source.id) == []


def test_duplicate_admission_and_restarted_pump_keep_one_logical_notice(notices):
    key = f"failure-test:{uuid.uuid4().hex}"
    source, duplicate = _send(notices, idempotency_key=key)
    assert not duplicate
    replay, duplicate = _send(notices, caller=notices.current, idempotency_key=key)
    assert duplicate and replay.id == source.id
    assert notices.deliveries.mark_failed(source.id, "worker error")

    [notice] = notices.pump.run_pass().notified
    _assert_notice(notices, source, notice, error_code="worker_error")
    assert notices.pump.run_pass().notified == []
    assert FailureNoticePump(notices.deliveries).run_pass().notified == []
    assert len(_notices_for(notices, source.id)) == 1


@pytest.mark.parametrize("failure", ["blank", "render", "enqueue"])
def test_render_or_enqueue_failure_rolls_back_the_notice_and_source_decision(notices, monkeypatch, caplog, failure):
    import llm_bawt.prompt_registry as registry

    source, _ = _send(notices)
    assert notices.deliveries.mark_failed(source.id, f"worker error: {SECRET} {TOOL_ARG}")
    before = _row(notices, source.id)["metadata_json"]
    original_enqueue = notices.deliveries.enqueue
    resolver = registry.get_prompt_resolver()
    seen = []

    def broken_render(**kwargs):
        seen.append(kwargs)
        if failure == "render":
            raise RuntimeError(f"registry error: {SECRET} {TOOL_ARG}")
        return "   "

    def broken_enqueue(**kwargs):
        # Raise after inserting inside the supplied transaction: a real rollback
        # must undo the notice, not merely leave the source unstamped.
        original_enqueue(**kwargs)
        raise RuntimeError(f"enqueue error: {SECRET} {TOOL_ARG}")

    with monkeypatch.context() as patch:
        if failure == "enqueue":
            patch.setattr(notices.deliveries, "enqueue", broken_enqueue)
        else:
            patch.setattr(resolver, "render", broken_render)
        failed = notices.pump.run_pass()

    assert (failed.notified, failed.skipped, failed.errors) == ([], 0, 1)
    assert _row(notices, source.id)["metadata_json"] == before
    assert _notices_for(notices, source.id) == []
    assert SECRET not in caplog.text and TOOL_ARG not in caplog.text
    if seen:
        assert seen[0]["key"] == PROMPT_KEY
        assert (seen[0]["scope_type"], seen[0]["scope_id"]) == ("bot", notices.sender)
        assert SECRET not in json.dumps(seen) and TOOL_ARG not in json.dumps(seen)
    # A fresh pump has no process-local backoff and retries durable pending work.
    [notice] = FailureNoticePump(notices.deliveries).run_pass().notified
    _assert_notice(notices, source, notice, error_code="worker_error")
    assert len(_notices_for(notices, source.id)) == 1


def test_concurrent_pumps_commit_only_one_logical_notice(notices):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    source, _ = _send(notices)
    assert notices.deliveries.mark_failed(source.id, "worker error")
    barrier = Barrier(2)
    candidate = {"id": source.id, "sender_bot_id": notices.sender, "target_bot_id": notices.target}

    def decide():
        barrier.wait(timeout=10)
        return FailureNoticePump(notices.deliveries)._decide(candidate)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: decide(), range(2)))
    assert sum(record is not None for record, _ in results) == 1
    assert len(_notices_for(notices, source.id)) == 1


@pytest.mark.parametrize("terminal", [None, "ok", "completed"])
def test_open_or_successful_owning_turn_never_produces_a_failure_notice(notices, terminal):
    source, _ = _send(notices)
    claim = _claim(notices, source)
    _delivered(notices, claim)
    if terminal:
        _end(notices, claim.turn_id, status=terminal, error="")
    assert notices.pump.run_pass().notified == []
    assert DECISION not in _row(notices, source.id)["metadata_json"]


def test_manual_retry_preserves_origin_but_not_the_parent_decision(notices):
    from llm_bawt.inter_bot_delivery_retry import retry_failed_delivery

    source, _ = _send(notices)
    assert notices.deliveries.mark_failed(source.id, "first failure")
    [first_notice] = notices.pump.run_pass().notified
    child, _ = retry_failed_delivery(notices.deliveries, source.id)
    assert child.sender_turn_id == source.sender_turn_id
    assert child.metadata[ORIGIN] == source.metadata[ORIGIN]
    assert DECISION not in child.metadata
    assert notices.deliveries.mark_failed(child.id, "second failure")
    [second_notice] = notices.pump.run_pass().notified
    assert first_notice.id != second_notice.id
    assert _row(notices, second_notice.id)["metadata_json"]["original_delivery_id"] == source.id
    assert len(_notices_for(notices, source.id)) == len(_notices_for(notices, child.id)) == 1


def test_existing_restart_child_keeps_the_original_session_and_reports_its_own_failure(notices):
    from agent_bridge.abort_contract import ABORT_CAUSE_KEY, HOST_SHUTDOWN
    from llm_bawt.inter_bot_delivery_retry import restart_retry_delivery

    source, _ = _send(notices)
    claim = _claim(notices, source)
    _end(notices, claim.turn_id, status="aborted", error="host shutdown")
    with notices.store.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET request_json=:audit WHERE id=:id"), {
            "audit": json.dumps({ABORT_CAUSE_KEY: {"cause": HOST_SHUTDOWN}}), "id": claim.turn_id,
        })
    assert notices.deliveries.fail_claim(source.id, claim.claim_token, "host shutdown")
    # Invoke the existing retry producer explicitly in the fixture. The new
    # failure producer must neither create nor execute another worker attempt.
    child, duplicate = restart_retry_delivery(notices.deliveries, source.id)
    assert not duplicate and child.author_entity_id == SYSTEM_BOT_ID
    [parent_notice] = notices.pump.run_pass().notified
    assert _row(notices, parent_notice.id)["metadata_json"]["existing_retry_delivery_id"] == child.id
    assert notices.deliveries.get(child.id).status == "QUEUED"
    assert notices.deliveries.mark_failed(child.id, "child worker error")
    [child_notice] = notices.pump.run_pass().notified
    stored = _assert_notice(notices, child, child_notice, error_code="worker_error")
    assert stored["metadata_json"]["original_delivery_id"] == source.id
    assert notices.pump.run_pass().notified == []


def test_failure_notice_is_queued_under_a_deny_gate_and_claims_only_after_release(notices):
    from llm_bawt.service.turn_gates import GateScope, release_gates
    from tests.test_turn_gates_pg import _activate, _spec

    source, _ = _send(notices)
    assert notices.deliveries.mark_failed(source.id, "failure")
    owner = notices.owner()
    _activate(notices.store, _spec(owner, GateScope.for_bots(notices.sender), mode="deny"))

    [notice] = notices.pump.run_pass().notified

    assert notice.status == "QUEUED"
    assert notices.deliveries.claim_next(notices.sender, claim_owner="worker", steer_capable=False) is None
    assert notices.deliveries.get(notice.id).attempt_count == 0
    with notices.store.engine.begin() as conn:
        release_gates(conn, owner_id=owner)
    claimed = _claim(notices, notice)
    assert claimed.status == "DISPATCHING" and claimed.attempt_count == 1
    assert len(_notices_for(notices, source.id)) == 1
