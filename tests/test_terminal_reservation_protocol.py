"""Protocol reservation identity and transactional revocation foundations."""
from __future__ import annotations

import json
import uuid

import pytest

from llm_bawt.service.turn_terminal_protocol import (
    FollowerFenceLost,
    protocol_reservation_fields,
)
from tests.session_lifecycle_pg_fixtures import rig  # noqa: F401
from tests.test_turn_terminal_protocol import insert_turn, protocol, read  # noqa: F401


def reserved(store, **overrides):
    return insert_turn(store.engine, legacy=True, status="reserved",
                       **{**protocol_reservation_fields(assistant_message_id=str(uuid.uuid4())), **overrides})


def test_unconsumed_reservation_is_not_follower_recovery_work(protocol):  # noqa: F811
    turn = reserved(protocol)
    row = read(protocol.engine, turn)
    assert row.assistant_message_id and row.history_session_id is None
    assert row.follower_owner is None and row.follower_lease_until is None
    assert protocol.claim(turn) is None
    assert protocol.claim(turn, projection=True) is None
    assert turn not in protocol.recoverable_ids()


def test_revocation_joins_outer_transaction_and_rolls_back_with_producer(protocol):  # noqa: F811
    turn = reserved(protocol)
    with pytest.raises(RuntimeError, match="producer failed"), protocol.engine.begin() as conn:
        result = protocol.interrupt(turn, status="aborted", end_reason="aborted",
                                    audit_key="recovery", audit={"cause": "reservation_revoked"},
                                    connection=conn)
        assert result.won and result.snapshot["history_disposition"] == "unbound"
        assert conn.in_transaction()
        raise RuntimeError("producer failed")
    row = read(protocol.engine, turn)
    assert row.status == "reserved" and row.ended_at is None
    assert row.terminal_snapshot_json is None and row.terminal_protocol_state == "reserved"


def test_revocation_commits_empty_immutable_failure_and_pinned_identity(protocol):  # noqa: F811
    turn = reserved(protocol)
    identity = read(protocol.engine, turn).assistant_message_id
    with protocol.engine.begin() as conn:
        winner = protocol.interrupt(turn, status="cancelled", end_reason="aborted", connection=conn)
    row = read(protocol.engine, turn)
    assert row.assistant_message_id == identity
    assert json.loads(row.terminal_snapshot_json) == winner.snapshot
    assert winner.snapshot["response_text"] == "" and winner.snapshot["history_disposition"] == "unbound"
    assert row.terminal_protocol_state == "pending" and row.ended_at is not None
    assert turn in protocol.recoverable_ids()
    assert protocol.interrupt(turn, status="error", end_reason="error").snapshot == winner.snapshot
    protocol.effects.assert_called_once()


@pytest.mark.parametrize("evidence", [{"agent_request_id": "run"}, {"agent_session_key": "session"},
                                      {"follower_owner": "owner"}])
def test_malformed_executing_reservation_is_not_revoked(protocol, evidence):  # noqa: F811
    turn = reserved(protocol, **evidence)
    with pytest.raises(FollowerFenceLost, match="execution ownership"):
        protocol.interrupt(turn, status="aborted", end_reason="aborted")
    assert read(protocol.engine, turn).ended_at is None


def test_reservation_cannot_commit_success_or_content(protocol):  # noqa: F811
    turn = reserved(protocol)
    with pytest.raises(ValueError, match="pre-inference failure"):
        protocol._commit(turn, lease=None, status="ok", end_reason="stop", response_text="invented")
    assert read(protocol.engine, turn).terminal_snapshot_json is None


def test_join_requires_active_canonical_transaction(protocol):  # noqa: F811
    turn = reserved(protocol)
    with protocol.engine.connect() as conn, pytest.raises(ValueError, match="active outer transaction"):
        protocol.interrupt(turn, status="error", end_reason="error", connection=conn)


@pytest.mark.integration
class TestPostgresReservationHandoff:
    @pytest.fixture
    def handoff(self, rig):  # noqa: F811
        from llm_bawt.service.turn_logs import TurnLogStore
        from llm_bawt.service.turn_terminal_protocol import TurnTerminalProtocol
        from tests.pg_isolation import isolated_pg_store

        store = isolated_pg_store(TurnLogStore)
        return rig, store, TurnTerminalProtocol(store), f"turn-{uuid.uuid4().hex}"

    @staticmethod
    def reserve(handoff, *, enabled):
        from llm_bawt.service.turn_admission import lock_target, write_reservation

        rig, store, _, turn = handoff  # noqa: F811
        with store.engine.begin() as conn:
            lock_target(conn, rig.bot)
            return write_reservation(conn, turn_id=turn, bot_id=rig.bot, user_id=rig.user,
                                     user_prompt="queued prompt", request_json={},
                                     trigger_message_id=rig.message_id, terminal_protocol=enabled,
                                     assistant_message_id=str(uuid.uuid5(uuid.NAMESPACE_URL, turn)))

    @staticmethod
    def consume(handoff, **overrides):
        rig, store, _, turn = handoff  # noqa: F811
        return store.save_turn(**dict(
            turn_id=turn, request_id="request", path="/test", stream=False, model="test",
            bot_id=rig.bot, user_id=rig.user, status="pending", latency_ms=None,
            user_prompt="queued prompt", request_payload={}, response_text="", tool_calls=[],
            assistant_message_id=str(uuid.uuid4()), **overrides,
        ))

    def test_fresh_marker_survives_rearm_and_consumption_pins_its_original_uuid(self, handoff):
        _, store, protocol, turn = handoff  # noqa: F811
        assert self.reserve(handoff, enabled=True)
        initial = read(store.engine, turn)
        assert initial.terminal_protocol_state == "reserved"
        assert initial.follower_owner is None and turn not in protocol.recoverable_ids()
        assert self.reserve(handoff, enabled=True)
        assert self.reserve(handoff, enabled=False)  # rollout rollback cannot demote it
        assert read(store.engine, turn).assistant_message_id == initial.assistant_message_id
        lease = self.consume(handoff)
        row = read(store.engine, turn)
        assert lease.assistant_message_id == initial.assistant_message_id == row.assistant_message_id
        assert row.terminal_protocol_state == "binding" and row.follower_owner == lease.token
        assert row.history_session_id is None
        assert not self.reserve(handoff, enabled=True)
        assert read(store.engine, turn).follower_owner == lease.token

    def test_legacy_reservation_is_not_adopted_when_setting_changes(self, handoff):
        from llm_bawt.service.turn_admission import AdmissionFenceLost

        _, store, _, turn = handoff
        assert self.reserve(handoff, enabled=False)
        assert self.reserve(handoff, enabled=True)
        assert read(store.engine, turn).terminal_protocol_state is None
        with pytest.raises(AdmissionFenceLost, match="legacy"):
            self.consume(handoff, terminal_protocol=True)
        assert self.consume(handoff) is None
        assert read(store.engine, turn).terminal_protocol_state is None

    def test_postgres_revocation_is_atomic_with_outer_rollback(self, handoff):
        _, store, protocol, turn = handoff  # noqa: F811
        assert self.reserve(handoff, enabled=True)
        with pytest.raises(RuntimeError, match="rollback"), store.engine.begin() as conn:
            protocol.interrupt(turn, status="aborted", end_reason="aborted", connection=conn)
            raise RuntimeError("rollback")
        assert read(store.engine, turn).terminal_protocol_state == "reserved"
        assert read(store.engine, turn).ended_at is None
        with store.engine.begin() as conn:
            winner = protocol.interrupt(turn, status="aborted", end_reason="aborted", connection=conn)
        assert json.loads(read(store.engine, turn).terminal_snapshot_json) == winner.snapshot
        assert not self.reserve(handoff, enabled=True)

    def test_approval_stale_release_and_retry_keep_durable_identity(self, handoff):
        from types import SimpleNamespace

        from llm_bawt.service.approval_continuations import _continuation_identity
        from llm_bawt.service.turn_admission import AdmissionClaim

        owner, store, _, turn = handoff
        identity = _continuation_identity(SimpleNamespace(continuation_id=turn))
        kwargs = {
            "turn_id": identity["inter_bot_turn_id"], "bot_id": owner.bot, "user_id": owner.user,
            "user_prompt": "approval reply", "request_json": identity,
            "trigger_message_id": identity["user_message_id"], "terminal_protocol": True,
            "assistant_message_id": identity["assistant_message_id"], "fence": lambda conn: True,
        }
        turn = kwargs["turn_id"]
        assert store.reserve_idle_turn(**kwargs, owner="first")
        assert not store.release_reservation(turn, owner="first", fence=lambda conn: False)
        assert store.release_reservation(turn, owner="first", fence=lambda conn: True)
        assert read(store.engine, turn) is None
        assert store.reserve_idle_turn(**kwargs, owner="second")
        assert not store.release_reservation(turn, owner="first", fence=lambda conn: True)
        assert read(store.engine, turn).assistant_message_id == identity["assistant_message_id"]
        lease = store.save_turn(
            turn_id=turn, request_id="request", path="/test", stream=False, model="test",
            bot_id=owner.bot, user_id=owner.user, status="pending", latency_ms=None,
            user_prompt="approval reply", request_payload={}, response_text="", tool_calls=[],
            assistant_message_id="replacement-must-not-win",
            admission_claim=AdmissionClaim(owner="second", fence=lambda conn: True),
        )
        assert lease.assistant_message_id == identity["assistant_message_id"]


@pytest.mark.parametrize("identity", [None, "", "  ", "x" * 37])
def test_protocol_reservation_requires_producer_identity(identity):
    with pytest.raises(ValueError, match="pinned assistant identity"):
        protocol_reservation_fields(assistant_message_id=identity)


def test_recovery_discovery_prioritizes_unattempted_and_older_leases(protocol):  # noqa: F811
    from datetime import UTC, datetime

    old_failure = insert_turn(protocol.engine, created_at=datetime(1990, 1, 1, tzinfo=UTC),
                              follower_lease_until=datetime(2002, 1, 1, tzinfo=UTC))
    waiting = insert_turn(protocol.engine, follower_lease_until=datetime(2001, 1, 1, tzinfo=UTC))
    unattempted = insert_turn(protocol.engine, follower_lease_until=None)
    assert protocol.recoverable_ids(limit=3) == [unattempted, waiting, old_failure]
