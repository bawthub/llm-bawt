"""TASK-1252: running claims and WRITING survive a replacement bridge safely.

Real disposable PostgreSQL, no SDK/provider calls or live bot traffic.
"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_steer_fence import SteerLifecycleFence
from llm_bawt.inter_bot_inbox_claim import INBOX, InboxRunBinding, prepare_inbox_write
from llm_bawt.service.turn_admission import classify_target
from llm_bawt.service.turn_gates import GateScope
from llm_bawt.steer_capabilities import ALL_STEER_CAPS, INBOX_NEXT, decode_caps, encode_caps
from tests import session_lifecycle_pg_fixtures
from tests.test_turn_gates_pg import _activate, _spec

rig = session_lifecycle_pg_fixtures.rig
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def running(rig):
    store = InterBotDeliveryStore(rig.config)
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE bot_profiles SET bot_type='agent', agent_backend='claude-code' WHERE slug=:bot
        """), {"bot": rig.bot})
        conn.execute(text("""
            UPDATE turn_logs SET status='streaming', ended_at=NULL,
                agent_backend='claude-code', agent_session_key='bound-key',
                agent_request_id='bound-request', agent_steer_caps='inbox_next,next'
            WHERE id=:id
        """), {"id": rig.turn_id})
        binding = InboxRunBinding.read(
            conn, target=rig.bot, active=classify_target(conn, rig.bot, exclude_turn_id=None).executing,
        )
    assert binding is not None
    return store, binding


def send(rig, store, message="callback", **extra):
    row, _ = store.enqueue(
        sender_bot_id="fixture-source", target_bot_id=rig.bot, message=message,
        payload={"messages": [{"role": "user", "content": message}], "user": rig.user, **extra},
    )
    return row


def claim(store, binding, *, owner="bridge"):
    return store.claim_next(binding.bot_id, claim_owner=owner, steer_capable=True, inbox_run=binding)


def expire(store, row):
    with store.engine.begin() as conn:
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET lease_expires_at=now()-interval '1 second' WHERE id=:id
        """), {"id": row.id})


def test_explicit_capability_not_legacy_next_or_default_all():
    assert INBOX_NEXT not in ALL_STEER_CAPS
    assert INBOX_NEXT in decode_caps(encode_caps({"next", INBOX_NEXT}))


def test_app_cannot_race_actual_owner_for_new_plain_row(rig, running):
    store, binding = running
    queued = send(rig, store)
    assert store.claim_next(rig.bot, claim_owner="app", steer_capable=True) is None
    row = claim(store, binding)
    assert row.id == queued.id and row.target_turn_id == rig.turn_id
    assert row.status == "STEERING" and row.delivery_mode == "steer"
    assert row.metadata[INBOX]["phase"] == "claimed"
    assert row.metadata[INBOX]["binding"]["session_id"] == rig.session_id
    assert INBOX not in row.to_api()["metadata"]
    assert claim(store, binding, owner="other-bridge") is None


@pytest.mark.parametrize("key", ["bot_id", "user_id", "turn_id", "backend", "session_key",
                                 "request_id", "trigger_message_id", "session_id"])
def test_claim_requires_full_exact_run_binding(rig, running, key):
    store, binding = running
    queued = send(rig, store)
    assert claim(store, replace(binding, **{key: "wrong"})) is None
    assert store.get(queued.id).attempt_count == 0


@pytest.mark.parametrize("change", [
    "status='reserved'", "ended_at=now()", "agent_request_id=NULL",
    "agent_steer_caps='next'", "agent_steer_caps=NULL", "trigger_message_id=NULL",
])
def test_no_bound_active_run_never_reserves_an_idle_turn(rig, running, change):
    store, binding = running
    queued = send(rig, store)
    with rig.engine.begin() as conn:
        conn.execute(text(f"UPDATE turn_logs SET {change} WHERE id=:id"), {"id": rig.turn_id})
    assert claim(store, binding) is None
    assert store.get(queued.id).attempt_count == 0
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT 1 FROM turn_logs WHERE id=:id"), {"id": queued.turn_id}).first() is None


@pytest.mark.parametrize("extra", [
    {"reply_mode": "waited"}, {"session_id": "pinned"}, {"voice_input": True},
    {"steer_only": True}, {"prefer_steer": False},
])
def test_special_rows_stay_out_of_bridge_claim(rig, running, extra):
    store, binding = running
    queued = send(rig, store, **extra)
    assert claim(store, binding) is None
    assert store.get(queued.id).attempt_count == 0


def test_existing_app_steer_is_not_stolen(rig, running):
    store, binding = running
    queued = send(rig, store)
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET delivery_mode='steer', target_turn_id=:turn WHERE id=:id
        """), {"id": queued.id, "turn": rig.turn_id})
    assert claim(store, binding) is None
    row = store.claim_next(rig.bot, claim_owner="app", steer_capable=True)
    assert row.id == queued.id and INBOX not in row.metadata


def test_bridge_reuses_held_head_jumper_authority(rig, running):
    store, binding = running
    held = send(rig, store, "held", prefer_steer=False)
    queued = send(rig, store, "may steer")
    row = claim(store, binding)
    assert row.id == queued.id
    assert store.get(held.id).attempt_count == 0


def test_idle_turn_remains_app_owned_after_bound_run_ends(rig, running):
    store, binding = running
    queued = send(rig, store)
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET ended_at=now(), status='ok' WHERE id=:id"), {"id": rig.turn_id})
    assert claim(store, binding) is None
    row = store.claim_next(rig.bot, claim_owner="app", steer_capable=True)
    assert row.id == queued.id and row.status == "DISPATCHING" and INBOX not in row.metadata


def test_unwritten_release_revalidates_destination_instead_of_stealing_old_run(rig, running):
    store, binding = running
    send(rig, store)
    row = claim(store, binding)
    released = store.requeue(row.id, row.claim_token, "ownership lost before prepare", delay_seconds=0)
    assert released is not None and released.delivery_mode is None and released.attempt_count == 0
    successor = claim(store, binding, owner="replacement")
    assert successor.claim_token != row.claim_token
    assert not prepare_inbox_write(store, row.id, row.claim_token, binding)
    assert prepare_inbox_write(store, successor.id, successor.claim_token, binding)


def test_prewrite_expiry_reclaims_without_old_token_authority(rig, running):
    store, binding = running
    send(rig, store)
    first = claim(store, binding)
    expire(store, first)
    assert not prepare_inbox_write(store, first.id, first.claim_token, binding)
    assert [r.id for r in store.recover_expired()] == [first.id]
    restarted = InterBotDeliveryStore(rig.config)
    second = claim(restarted, binding, owner="replacement")
    assert second.claim_token != first.claim_token
    assert second.metadata[INBOX]["command_uuid"] == first.metadata[INBOX]["command_uuid"]
    assert not prepare_inbox_write(restarted, second.id, first.claim_token, binding)
    assert prepare_inbox_write(restarted, second.id, second.claim_token, binding)


def test_lost_prepare_response_and_expiry_never_grant_second_write(rig, running):
    store, binding = running
    send(rig, store)
    row = claim(store, binding)
    assert prepare_inbox_write(store, row.id, row.claim_token, binding)
    restarted = InterBotDeliveryStore(rig.config)
    assert not prepare_inbox_write(restarted, row.id, row.claim_token, binding)
    expire(store, row)
    assert restarted.recover_expired() == []
    assert claim(restarted, binding) is None
    assert restarted.claim_next(rig.bot, claim_owner="app", steer_capable=True) is None
    fenced = restarted.get(row.id)
    assert fenced.status == "STEERING"
    assert fenced.metadata[INBOX]["write_token"] == row.claim_token
    assert not restarted.mark_transport_accepted(row.id, row.claim_token)
    assert restarted.mark_delivered(row.id, row.claim_token, response_model=None, response_chars=0) is None
    assert restarted.requeue(row.id, row.claim_token, "local not_written", delay_seconds=0) is None
    fence = SteerLifecycleFence(restarted)
    assert not fence.note_steer_started(row.id, 0, binding.turn_id)
    assert fence.reopen_rejected_steer(row.id, 0, binding.turn_id, "cli_terminal") == ("ignored", None)
    assert restarted.fail_claim(row.id, row.claim_token, "transport went away") is None


@pytest.mark.parametrize("change", ["agent_request_id='successor'", "ended_at=now()",
                                  "agent_session_key='successor'", "trigger_message_id=NULL"])
def test_handoff_between_claim_and_preparation_refuses_write(rig, running, change):
    store, binding = running
    send(rig, store)
    row = claim(store, binding)
    with rig.engine.begin() as conn:
        conn.execute(text(f"UPDATE turn_logs SET {change} WHERE id=:id"), {"id": rig.turn_id})
    assert not prepare_inbox_write(store, row.id, row.claim_token, binding)
    assert store.get(row.id).metadata[INBOX]["phase"] == "claimed"


def test_parallel_preparation_grants_only_one_possible_sdk_write(rig, running):
    store, binding = running
    send(rig, store)
    row = claim(store, binding)
    ready = threading.Barrier(2)

    def prepare(_):
        ready.wait(timeout=5)
        return prepare_inbox_write(store, row.id, row.claim_token, binding)

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(prepare, range(2))) == [False, True]


def test_competing_bridge_claims_have_one_owner(rig, running):
    store, binding = running
    send(rig, store)
    ready = threading.Barrier(2)

    def compete(owner):
        ready.wait(timeout=5)
        return claim(store, binding, owner=owner)

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sum(row is not None for row in workers.map(compete, ["a", "b"])) == 1


def test_caller_metadata_cannot_forge_inbox_execution_state(rig, running):
    store, _ = running
    queued, _ = store.enqueue(
        sender_bot_id="fixture-source", target_bot_id=rig.bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}], "user": rig.user},
        metadata={INBOX: {"phase": "writing", "write_token": "caller"}},
    )
    assert INBOX not in queued.metadata


@pytest.mark.parametrize("after_claim", [False, True])
def test_stop_or_gate_blocks_before_permission(rig, running, after_claim):
    store, binding = running
    queued = send(rig, store)
    row = claim(store, binding) if after_claim else None
    owner = f"fixture-inbox:{rig.bot}"
    _activate(rig, _spec(owner, GateScope.for_bots(rig.bot)))
    try:
        if row:
            assert not prepare_inbox_write(store, row.id, row.claim_token, binding)
        else:
            assert claim(store, binding) is None
        assert store.get(queued.id).metadata.get(INBOX, {}).get("phase") != "writing"
    finally:
        with rig.engine.begin() as conn:
            conn.execute(text("DELETE FROM gate_members WHERE gate_id IN (SELECT id FROM system_gates WHERE owner_id=:o)"),
                         {"o": owner})
            conn.execute(text("DELETE FROM system_gates WHERE owner_id=:o"), {"o": owner})


def test_explicit_stop_after_claim_prevents_preparation(rig, running):
    store, binding = running
    send(rig, store)
    row = claim(store, binding)
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET request_json=:request WHERE id=:id"),
                     {"id": rig.turn_id, "request": json.dumps({"abort_request": {"source": "chat_stop"}})})
    assert not prepare_inbox_write(store, row.id, row.claim_token, binding)


def test_new_maintenance_barrier_after_claim_prevents_preparation(rig, running):
    store, binding = running
    send(rig, store)
    row = claim(store, binding)
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, ended_at)
            VALUES (:id, now(), '/test', true, :bot, :user, 'ok', '/compact', '', now())
        """), {"id": f"{rig.turn_id}-maintenance", "bot": rig.bot, "user": rig.user})
    assert not prepare_inbox_write(store, row.id, row.claim_token, binding)
    assert store.get(row.id).metadata[INBOX]["phase"] == "claimed"


@pytest.mark.parametrize("change", ["status='deleted'", "user_id='foreign-owner'"])
def test_owned_history_destination_rechecked_before_permission(rig, running, change):
    store, binding = running
    send(rig, store)
    row = claim(store, binding)
    with rig.engine.begin() as conn:
        conn.execute(text(f"UPDATE sessions SET {change} WHERE id=:id"), {"id": rig.session_id})
    assert not prepare_inbox_write(store, row.id, row.claim_token, binding)
