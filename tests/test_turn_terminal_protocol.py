"""Terminal snapshot/fencing tests; real concurrency uses private PostgreSQL."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest
from sqlalchemy import create_engine, text, update
from sqlmodel import Session

from agent_bridge.terminal_publish import TerminalPublishResult, TerminalPublishStatus
from llm_bawt.service.turn_log_schema import TurnLog
from llm_bawt.service.turn_terminal_protocol import (
    FollowerFenceLost, FollowerLease, PUBLICATION_HORIZON,
    TurnTerminalProtocol, protocol_insert_fields,
)


ACK = TerminalPublishResult(TerminalPublishStatus.PUBLISHED, "100-0")
EXPIRED = TerminalPublishResult(TerminalPublishStatus.EXPIRED, None)


def insert_turn(engine, *, legacy=False, expired_owner=True, unbound=False, **overrides):
    values = dict(
        id=f"turn-{uuid.uuid4().hex}", status="streaming", bot_id="test-terminal",
        user_id="test-user", response_text="", model="test-model", stream=True,
    )
    with Session(engine) as session:
        if not legacy:
            values.update(protocol_insert_fields(
                connection=session.connection(),
                assistant_message_id=str(uuid.uuid4()),
                history_session_id=None if unbound else str(uuid.uuid4()),
            ))
            if expired_owner:
                # Most tests start with a crashed follower whose lease expired.
                values["follower_lease_until"] = datetime(2000, 1, 1, tzinfo=timezone.utc)
        values.update(overrides)
        row = TurnLog(**values)
        session.add(row)
        session.commit()
    return values["id"]


def read(engine, turn_id):
    with Session(engine) as session:
        return session.get(TurnLog, turn_id)


def expire(engine, turn_id, **extra):
    with engine.begin() as connection:
        connection.execute(update(TurnLog).where(TurnLog.id == turn_id).values(
            follower_lease_until=datetime(2000, 1, 1, tzinfo=timezone.utc), **extra,
        ))


@pytest.fixture
def protocol(monkeypatch):
    engine = create_engine("sqlite://")
    TurnLog.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT)"
        ))
    effects = Mock()
    monkeypatch.setattr(
        "llm_bawt.service.turn_terminal_effects.record_terminal_effects", effects,
    )
    monkeypatch.setattr(
        "llm_bawt.service.usage.balance.balance_update_for_turn",
        lambda *args, **kwargs: {"frozen": "balance-after-charge"},
    )
    store = SimpleNamespace(engine=engine, _encode_token_usage=lambda usage, model: json.dumps(usage))
    result = TurnTerminalProtocol(store)
    result.effects = effects
    yield result
    engine.dispose()


def test_user_stop_commits_checkpoint_and_fences_worker(protocol, monkeypatch):
    from llm_bawt.service.turn_termination import TurnAbortCoordinator

    turn = insert_turn(protocol.engine, expired_owner=False)
    row = read(protocol.engine, turn)
    lease = FollowerLease(turn, row.follower_owner, row.follower_lease_until)
    protocol.write_partial(lease, text_value="seen before Stop", reasoning="thinking")
    monkeypatch.setattr(
        "llm_bawt.service.turn_termination.reconcile_finished_tool_safely", Mock(),
    )
    coordinator = TurnAbortCoordinator(protocol.turn_store)
    assert coordinator.request(turn, source="chat_stop", peer=None)
    assert coordinator.settle(turn, {"bridge_state": "cancelled", "confirmed": True})
    row = read(protocol.engine, turn)
    snapshot = json.loads(row.terminal_snapshot_json)
    assert (row.status, row.end_reason, row.terminal_protocol_state) == (
        "aborted", "aborted", "pending",
    )
    assert snapshot["response_text"] == "seen before Stop"
    assert snapshot["reasoning"] == "thinking"
    assert snapshot["completion"]["status"] == "cancelled"
    assert json.loads(row.request_json)["user_stop"]["confirmed"] is True
    with pytest.raises(FollowerFenceLost):
        protocol.write_partial(lease, text_value="late")
    assert not coordinator.settle(turn, {"bridge_state": "retry"})
    assert json.loads(read(protocol.engine, turn).terminal_snapshot_json) == snapshot


def test_user_stop_loses_to_worker_snapshot_but_records_audit(protocol):
    from llm_bawt.service.turn_termination import TurnAbortCoordinator

    turn = insert_turn(protocol.engine, expired_owner=False)
    row = read(protocol.engine, turn)
    lease = FollowerLease(turn, row.follower_owner, row.follower_lease_until)
    coordinator = TurnAbortCoordinator(protocol.turn_store)
    assert coordinator.request(turn, source="chat_stop", peer=None)
    winner = protocol.commit(lease, status="ok", end_reason="stop", response_text="worker partial")
    assert winner.snapshot["completion"]["status"] == "cancelled"
    assert not coordinator.settle(turn, {"bridge_state": "worker_finalized"})
    row = read(protocol.engine, turn)
    assert json.loads(row.terminal_snapshot_json) == winner.snapshot
    assert json.loads(row.request_json)["user_stop"]["settled"] is False


def test_stop_publishes_protocol_snapshot_not_direct_mutable_event(protocol, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from llm_bawt.service.turn_abort import _publish_abort_snapshot

    turn = insert_turn(protocol.engine, expired_owner=False)
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(), _pub_redis=Mock())
    projection = AsyncMock()
    monkeypatch.setattr(
        "llm_bawt.service.turn_terminal_finalize.project_execution_terminal", projection,
    )
    store = SimpleNamespace(engine=protocol.engine, get_turn=lambda turn_id: read(protocol.engine, turn_id))
    service = SimpleNamespace(_turn_log_store=store, _redis_subscriber=subscriber)
    asyncio.run(_publish_abort_snapshot(service, turn))
    projection.assert_awaited_once()
    assert projection.await_args.args[1:] == (turn, subscriber)
    subscriber.publish_tool_event.assert_not_awaited()


def test_protocol_marker_is_explicit_and_legacy_is_never_adopted(protocol):
    legacy = insert_turn(protocol.engine, legacy=True)
    with pytest.raises(FollowerFenceLost, match="legacy"):
        protocol.claim(legacy)
    assert legacy not in protocol.recoverable_ids()
    admitted = insert_turn(protocol.engine)
    row = read(protocol.engine, admitted)
    assert row.terminal_protocol_state == "open"
    assert row.assistant_message_id and row.history_session_id
    assert protocol.recoverable_ids() == [admitted]


@pytest.mark.parametrize("assistant,session", [
    ("", "session"), ("assistant", ""), (" " * 2, "session"),
    ("x" * 37, "session"), ("assistant", "x" * 37),
])
def test_protocol_requires_storage_compatible_pinned_ids(protocol, assistant, session):
    with pytest.raises(ValueError):
        with protocol.engine.begin() as connection:
            protocol_insert_fields(
                connection=connection, assistant_message_id=assistant, history_session_id=session,
            )


@pytest.mark.parametrize("unbound", [False, True])
def test_initial_admission_lease_excludes_recovery_before_executor_runs(protocol, unbound):
    turn = insert_turn(protocol.engine, expired_owner=False, unbound=unbound)
    row = read(protocol.engine, turn)
    assert row.follower_owner and row.follower_lease_until
    assert turn not in protocol.recoverable_ids()
    assert protocol.claim(turn) is None
    # Admission passes this exact token to the executor; it does not re-claim.
    executor = FollowerLease(turn, row.follower_owner, row.follower_lease_until)
    if not unbound:
        protocol.write_partial(executor, text_value="healthy live answer")
    assert protocol.renew(executor).token == executor.token
    expire(protocol.engine, turn)
    assert turn in protocol.recoverable_ids()
    assert protocol.claim(turn).token != executor.token


def add_session(protocol, *, bot="test-terminal", user="test-user", status="active"):
    session_id = str(uuid.uuid4())
    with protocol.engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO sessions (id, bot_id, user_id, status) VALUES (:id,:bot,:user,:status)"
        ), {"id": session_id, "bot": bot, "user": user, "status": status})
    return session_id


def test_admission_can_bind_only_once_to_real_owned_session(protocol):
    turn = insert_turn(protocol.engine, unbound=True)
    lease = protocol.claim(turn)
    assert read(protocol.engine, turn).terminal_protocol_state == "binding"
    session_id = add_session(protocol, status="archived")
    protocol.bind_history_session(lease, session_id=session_id)
    row = read(protocol.engine, turn)
    assert row.history_session_id == session_id and row.terminal_protocol_state == "open"
    assert row.follower_owner == lease.token
    with pytest.raises(FollowerFenceLost):
        protocol.bind_history_session(lease, session_id=add_session(protocol))
    protocol.write_partial(lease, text_value="bound partial")
    result = protocol.commit(lease, status="ok", end_reason="stop", response_text="bound answer")
    assert result.snapshot["history_disposition"] == "bound"
    assert result.snapshot["completion"]["session_id"] == session_id


@pytest.mark.parametrize("owner", [
    {"bot": "different-bot"}, {"user": "different-user"}, {"user": None}, {"status": "deleted"},
])
def test_bind_rejects_unowned_or_deleted_session(protocol, owner):
    turn = insert_turn(protocol.engine, unbound=True)
    lease = protocol.claim(turn)
    with pytest.raises(ValueError, match="wrong owner"):
        protocol.bind_history_session(lease, session_id=add_session(protocol, **owner))
    row = read(protocol.engine, turn)
    assert row.history_session_id is None and row.terminal_protocol_state == "binding"


def test_bind_does_not_treat_cached_or_missing_uuid_as_a_real_session(protocol):
    turn = insert_turn(protocol.engine, unbound=True)
    lease = protocol.claim(turn)
    with pytest.raises(ValueError, match="does not exist"):
        protocol.bind_history_session(lease, session_id=str(uuid.uuid4()))
    assert read(protocol.engine, turn).history_session_id is None
    with pytest.raises(FollowerFenceLost):
        protocol.write_partial(lease, text_value="must not infer before binding")


@pytest.mark.parametrize("authority", ["stop", "recovery"])
def test_pre_binding_terminal_blocks_late_bind_and_has_explicit_unbound_projection(protocol, authority):
    turn = insert_turn(protocol.engine, unbound=True)
    lease = protocol.claim(turn)
    session_id = add_session(protocol)
    if authority == "stop":
        result = protocol.interrupt(turn, status="aborted", end_reason="aborted")
    else:
        expire(protocol.engine, turn)
        replacement = protocol.claim(turn)
        assert replacement.token != lease.token
        result = protocol.commit(
            replacement, status="error", end_reason="error",
            error_text="Incomplete recovery: admission ended before history binding",
        )
    assert result.won is True
    assert result.snapshot["history_disposition"] == "unbound"
    assert result.snapshot["history_session_id"] is None
    assert result.snapshot["completion"]["session_id"] is None
    assert result.snapshot["completion"]["status"] in {"error", "cancelled"}
    with pytest.raises(FollowerFenceLost):
        protocol.bind_history_session(lease, session_id=session_id)
    with pytest.raises(FollowerFenceLost):
        protocol.renew(lease)
    assert read(protocol.engine, turn).history_session_id is None
    projection = protocol.claim(turn, projection=True)
    protocol.mark_history_projected(projection)
    protocol.record_publication(projection, result=ACK)
    assert turn not in protocol.recoverable_ids()


@pytest.mark.parametrize("candidate", [
    {"status": "ok"}, {"response_text": "invented answer"}, {"reasoning": "unbound thought"},
    {"tool_context": "unbound tool"}, {"attachments": [{"asset_id": "unbound"}]},
    {"token_usage": {"output_tokens": 2}}, {"tts_tail": {"delta": "unbound speech"}},
])
def test_unbound_turn_cannot_commit_success_or_inference_content(protocol, candidate):
    turn = insert_turn(protocol.engine, unbound=True)
    lease = protocol.claim(turn)
    with pytest.raises(ValueError, match="pre-inference failure"):
        protocol.commit(lease, **{"status": "error", "end_reason": "error", **candidate})
    assert read(protocol.engine, turn).terminal_snapshot_json is None
    protocol.effects.assert_not_called()


def test_only_one_live_owner_and_expired_owner_cannot_renew_or_write(protocol):
    turn = insert_turn(protocol.engine)
    first = protocol.claim(turn)
    assert protocol.claim(turn) is None
    renewed = protocol.renew(first)
    assert renewed.token == first.token
    protocol.write_partial(renewed, text_value="durable partial", reasoning="thought")
    expire(protocol.engine, turn)
    with pytest.raises(FollowerFenceLost):
        protocol.renew(renewed)
    second = protocol.claim(turn)
    assert second.token != first.token
    for action in (
        lambda: protocol.write_partial(first, text_value="stale"),
        lambda: protocol.commit(first, status="ok", end_reason="stop", response_text="stale"),
    ):
        with pytest.raises(FollowerFenceLost):
            action()
    row = read(protocol.engine, turn)
    assert row.response_text == "durable partial" and row.reasoning == "thought"
    assert row.ended_at is None


def test_commit_freezes_full_winner_before_any_projection_and_revokes_owner(protocol):
    turn = insert_turn(protocol.engine)
    lease = protocol.claim(turn)
    identity = read(protocol.engine, turn)
    result = protocol.commit(
        lease, status="ok", end_reason="question", response_text="the winning answer",
        reasoning="frozen thought", tool_context="[Read]\nresult",
        attachments=[{"asset_id": "frozen"}], token_usage={"output_tokens": 9},
        completion_fields={
            "session_id": "wrong-session", "assistant_message_id": "wrong-assistant",
            "question_id": "question-1", "approval_id": "approval-1",
            "changed_files": {"files": [{"path": "frozen.py"}]}, "animation": "wave",
            "text_offset": 17, "tts_text": "frozen speech",
        },
        tts_tail={"delta": "tail", "tts_offset": 12, "ts": 123},
    )
    assert result.won is True
    snapshot = result.snapshot
    row = read(protocol.engine, turn)
    assert row.ended_at is not None and row.terminal_protocol_state == "pending"
    assert row.terminal_history_projected_at is None and row.terminal_published_at is None
    assert row.follower_owner is None and row.follower_lease_until is None
    assert row.terminal_publication_deadline - row.ended_at == PUBLICATION_HORIZON
    assert json.loads(row.terminal_snapshot_json) == snapshot
    assert snapshot["history_session_id"] == identity.history_session_id
    assert len(snapshot["tools_message_id"]) == 36
    assert snapshot["tools_message_id"] != identity.assistant_message_id
    completion = snapshot["completion"]
    assert completion["assistant_message_id"] == identity.assistant_message_id
    assert completion["session_id"] == identity.history_session_id
    assert completion["status"] == "completed" and completion["end_reason"] == "question"
    assert completion["approval_id"] == "approval-1"
    assert completion["changed_files"] == {"files": [{"path": "frozen.py"}]}
    assert completion["balance_update"] == {"frozen": "balance-after-charge"}
    assert completion["text_offset"] == 17 and completion["tts_text"] == "frozen speech"
    assert snapshot["tts_tail"]["delta"] == "tail"
    assert snapshot["tts_tail"]["turn_id"] == turn
    protocol.effects.assert_called_once()
    # Even a stale finalizer receives the winner without mutation or effects.
    losing = protocol.commit(
        lease, status="error", end_reason="error", response_text="stale replacement",
    )
    assert losing.snapshot == snapshot and losing.won is False
    protocol.effects.assert_called_once()
    with pytest.raises(FollowerFenceLost):
        protocol.write_partial(lease, text_value="stale")


def test_stop_wins_preserves_partial_and_cannot_be_repaired_to_success(protocol):
    turn = insert_turn(protocol.engine)
    lease = protocol.claim(turn)
    protocol.write_partial(lease, text_value="partial before Stop", reasoning="partial thought")
    result = protocol.interrupt(turn, status="aborted", end_reason="aborted")
    assert result.won is True
    stopped = result.snapshot
    assert stopped["response_text"] == "partial before Stop"
    assert stopped["reasoning"] == "partial thought"
    assert stopped["completion"]["status"] == "cancelled"
    losing = protocol.commit(
        lease, status="ok", end_reason="stop", response_text="late success",
    )
    assert losing.snapshot == stopped and losing.won is False
    with pytest.raises(ValueError):
        protocol.interrupt(turn, status="ok", end_reason="stop")
    assert read(protocol.engine, turn).status == "aborted"


def test_cancelling_intent_overrides_success_candidate_and_suppresses_tts(protocol):
    turn = insert_turn(protocol.engine, status="cancelling")
    lease = protocol.claim(turn)
    snapshot = protocol.commit(
        lease, status="ok", end_reason="stop", response_text="partial",
        tts_tail={"delta": "must not speak", "tts_offset": 0},
    ).snapshot
    assert snapshot["completion"]["status"] == "cancelled"
    assert snapshot["completion"]["end_reason"] == "aborted"
    assert snapshot["tts_tail"] is None


def test_invalid_snapshot_rolls_back_terminal_and_effects(protocol):
    turn = insert_turn(protocol.engine)
    lease = protocol.claim(turn)
    with pytest.raises(ValueError):
        protocol.commit(
            lease, status="ok", end_reason="stop", response_text="candidate",
            completion_fields={"invalid_float": float("nan")},
        )
    row = read(protocol.engine, turn)
    assert row.ended_at is None and row.terminal_snapshot_json is None
    assert row.follower_owner == lease.token


def test_projection_crash_retry_uses_same_snapshot_and_separate_claim(protocol):
    turn = insert_turn(protocol.engine)
    follower = protocol.claim(turn)
    snapshot = protocol.commit(follower, status="ok", end_reason="stop", response_text="answer").snapshot
    projection = protocol.claim(turn, projection=True)
    assert projection is not None and projection.token != follower.token
    assert protocol.claim(turn, projection=True) is None
    with pytest.raises(ValueError, match="history"):
        protocol.record_publication(projection, result=ACK)
    # This marker follows the immutable history writer. A crash after the write
    # but before the marker retries that same snapshot with the same message IDs.
    protocol.mark_history_projected(projection)
    expire(protocol.engine, turn)
    next_projection = protocol.claim(turn, projection=True)
    with pytest.raises(FollowerFenceLost):
        protocol.record_publication(projection, result=ACK)
    protocol.record_publication(next_projection, result=ACK)
    row = read(protocol.engine, turn)
    assert row.terminal_protocol_state == "published"
    assert row.terminal_published_stream_id == "100-0"
    assert row.terminal_published_at is not None
    assert json.loads(row.terminal_snapshot_json) == snapshot
    assert turn not in protocol.recoverable_ids()
    assert protocol.claim(turn, projection=True) is None
    protocol.effects.assert_called_once()


def test_publication_expiry_does_not_expire_history_work(protocol):
    turn = insert_turn(protocol.engine)
    snapshot = protocol.commit(protocol.claim(turn), status="error", end_reason="error").snapshot
    lease = protocol.claim(turn, projection=True)
    # Redis is the expiry authority, even if the DB clock is behind it.
    assert read(protocol.engine, turn).terminal_publication_deadline.year > 2000
    protocol.record_publication(lease, result=EXPIRED)
    assert turn in protocol.recoverable_ids()
    history_lease = protocol.claim(turn, projection=True)
    protocol.mark_history_projected(history_lease)
    row = read(protocol.engine, turn)
    assert row.terminal_protocol_state == "publication_expired"
    assert row.terminal_published_stream_id is None
    assert row.terminal_history_projected_at is not None
    assert json.loads(row.terminal_snapshot_json) == snapshot
    assert turn not in protocol.recoverable_ids()


class TestPostgresProtocol:
    pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

    @pytest.fixture
    def pg_protocol(self):
        from llm_bawt.service.turn_logs import TurnLogStore
        from tests.pg_isolation import isolated_pg_store

        turns = isolated_pg_store(TurnLogStore)
        assert turns.engine is not None, "private PostgreSQL must be available"
        # The binding protocol reads only the canonical sessions owner/lifecycle
        # columns. Other tests may already have provisioned the complete table.
        with turns.engine.begin() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id VARCHAR(36) PRIMARY KEY, bot_id VARCHAR(64) NOT NULL,
                    user_id VARCHAR(64), started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP, archived_at TIMESTAMP,
                    status VARCHAR(16) NOT NULL DEFAULT 'active', session_metadata JSONB
                )
            """))
        return TurnTerminalProtocol(turns)

    def test_real_owned_binding_then_stop_pins_the_same_session(self, pg_protocol):
        bot_id = f"test-terminal-{uuid.uuid4().hex[:8]}"
        turn = insert_turn(pg_protocol.engine, unbound=True, bot_id=bot_id)
        lease = pg_protocol.claim(turn)
        session_id = add_session(pg_protocol, bot=bot_id)
        pg_protocol.bind_history_session(lease, session_id=session_id)
        result = pg_protocol.interrupt(turn, status="aborted", end_reason="aborted")
        assert result.snapshot["history_session_id"] == session_id
        assert result.snapshot["completion"]["session_id"] == session_id
        assert result.snapshot["history_disposition"] == "bound"

    def test_stop_racing_bind_cannot_leave_a_terminal_with_a_late_session(self, pg_protocol):
        bot_id = f"test-terminal-{uuid.uuid4().hex[:8]}"
        turn = insert_turn(pg_protocol.engine, unbound=True, bot_id=bot_id)
        lease = pg_protocol.claim(turn)
        session_id = add_session(pg_protocol, bot=bot_id)
        barrier = threading.Barrier(2)

        def bind():
            barrier.wait(timeout=5)
            try:
                pg_protocol.bind_history_session(lease, session_id=session_id)
                return True
            except FollowerFenceLost:
                return False

        def stop():
            barrier.wait(timeout=5)
            return pg_protocol.interrupt(turn, status="aborted", end_reason="aborted")

        with ThreadPoolExecutor(max_workers=2) as pool:
            bind_future, stop_future = pool.submit(bind), pool.submit(stop)
            bound, result = bind_future.result(timeout=10), stop_future.result(timeout=10)
        row = read(pg_protocol.engine, turn)
        assert result.won is True and row.status == "aborted"
        assert row.history_session_id == (session_id if bound else None)
        assert result.snapshot["history_session_id"] == row.history_session_id
        assert result.snapshot["history_disposition"] == ("bound" if bound else "unbound")
        with pytest.raises(FollowerFenceLost):
            pg_protocol.bind_history_session(lease, session_id=session_id)

    def test_concurrent_claims_have_one_winner(self, pg_protocol):
        turn = insert_turn(pg_protocol.engine)
        barrier = threading.Barrier(2)

        def claim():
            barrier.wait(timeout=5)
            return pg_protocol.claim(turn)

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: claim(), range(2)))
        assert sum(outcome is not None for outcome in outcomes) == 1
        winner = next(outcome for outcome in outcomes if outcome is not None)
        expire(pg_protocol.engine, turn)
        takeover = pg_protocol.claim(turn)
        assert takeover.token != winner.token
        with pytest.raises(FollowerFenceLost):
            pg_protocol.renew(winner)
        with pytest.raises(FollowerFenceLost):
            pg_protocol.commit(winner, status="ok", end_reason="stop")

    def test_stop_finalize_race_has_one_immutable_snapshot(self, pg_protocol):
        turn = insert_turn(pg_protocol.engine)
        lease = pg_protocol.claim(turn)
        pg_protocol.write_partial(lease, text_value="preserved partial", reasoning="thought")
        barrier = threading.Barrier(2)

        def finish():
            barrier.wait(timeout=5)
            return pg_protocol.commit(
                lease, status="ok", end_reason="stop", response_text="complete answer",
            )

        def stop():
            barrier.wait(timeout=5)
            return pg_protocol.interrupt(turn, status="aborted", end_reason="aborted")

        with ThreadPoolExecutor(max_workers=2) as pool:
            one, two = pool.submit(finish), pool.submit(stop)
            first, second = one.result(timeout=10), two.result(timeout=10)
        assert first.snapshot == second.snapshot
        assert sorted([first.won, second.won]) == [False, True]
        row = read(pg_protocol.engine, turn)
        assert json.loads(row.terminal_snapshot_json) == first.snapshot
        assert row.status in {"ok", "aborted"} and row.ended_at is not None
        assert len(first.snapshot["tools_message_id"]) == 36
