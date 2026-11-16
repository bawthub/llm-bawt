"""Real private-PG acceptance/repair/ack with a deterministic, provider-free SDK."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from claude_code_bridge.inbox_input import accept_inbox_input
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_steer_fence import SteerLifecycleFence
from llm_bawt.inter_bot_inbox_claim import INBOX, prepare_inbox_write
from llm_bawt.inter_bot_inbox_settlement import InboxSettlement, InboxWriteProof
from llm_bawt.service.inbox_persistence import repair_inbox_history
from llm_bawt.service.routes.chat import _persist_interrupt_anchor
from llm_bawt.utils.bot_partition import sanitize_table_name
from tests import test_delivery_inbox_claim_pg as claims
from tests.test_claude_inbox_input import InputClient

rig, running = claims.rig, claims.running
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def operation(rig, running, *, prepare=True):
    store, binding = running
    claims.send(rig, store)
    row = claims.claim(store, binding)
    evidence = InboxWriteProof(binding, row.metadata[INBOX]["generation"],
                               row.metadata[INBOX]["command_uuid"], row.claim_token)
    if prepare:
        assert prepare_inbox_write(store, row.id, row.claim_token, binding)
    return store, row, evidence, InboxSettlement(store)


class HistoryClient:
    """The service memory-client surface backed by actual private SQL history."""
    def __init__(self, rig):
        self.rig, self.fail, self.calls, self.on_saved = rig, False, [], None

    def get_short_term_manager(self):
        return SimpleNamespace(_backend=None)

    def add_message(self, role, message, session_id, _unused, message_id, _attachments, _reasoning,
                    author_type, author_id):
        self.calls.append((session_id, message_id, message, author_type, author_id))
        if self.fail:
            self.fail = False
            raise RuntimeError("fixture history unavailable")
        with self.rig.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                      author_entity_type, author_entity_id)
                VALUES (:bot, :id, :role, :message, 2, :session, :type, :author)
                ON CONFLICT (bot_id, id) DO NOTHING
            """), {"bot": sanitize_table_name(self.rig.bot), "id": message_id, "role": role,
                   "message": message, "session": session_id, "type": author_type, "author": author_id})
        if self.on_saved:
            self.on_saved()


def history(rig):
    memory = HistoryClient(rig)
    service = SimpleNamespace(_turn_log_store=SimpleNamespace(engine=rig.engine), get_memory_client=lambda *_: memory)
    subscriber = SimpleNamespace(
        turn_text_offset=AsyncMock(side_effect=AssertionError("repair must not sample a live offset")),
        publish_tool_event=AsyncMock(return_value=True),
    )
    return memory, service, subscriber


def repair(store, row, service, subscriber, **changes):
    options = dict(delivery_id=row.id, claim_token=row.claim_token,
                   subscriber=subscriber, persist_anchor=_persist_interrupt_anchor)
    return asyncio.run(repair_inbox_history(service, store, **(options | changes)))


def test_accept_once_fail_history_replace_owner_repair_and_ack_without_reinput(rig, running):
    store, row, evidence, settlement = operation(rig, running, prepare=False)
    client = InputClient()
    client.run.request_id = evidence.binding.request_id

    async def prepare():
        return await asyncio.to_thread(prepare_inbox_write, store, row.id, row.claim_token, evidence.binding)

    receipt = asyncio.run(accept_inbox_input(
        client.run, command_uuid=evidence.command_uuid, steer_rpc_id="fixture-operation",
        framed_message="NEXT[callback]", interrupt_item="ITEM[callback]",
        owns_run=lambda: True, prepare_write=prepare,
    ))
    assert receipt.acceptance == "queued" and len(client.frames) == 1
    assert settlement.note_lifecycle(row.id, evidence, receipt.acceptance, content_offset=7)
    memory, service, subscriber = history(rig)
    memory.fail = True
    assert repair(store, row, service, subscriber) is None
    assert store.get(row.id).status == "STEERING"
    claims.expire(store, row)
    restarted = InterBotDeliveryStore(rig.config)
    replacement = InboxSettlement(restarted).claim_repair(rig.bot, claim_owner="replacement")
    assert replacement.claim_token != row.claim_token
    assert replacement.metadata[INBOX]["write_token"] == row.claim_token
    assert not prepare_inbox_write(restarted, replacement.id, replacement.claim_token, evidence.binding)
    # The original run can end, and the bot's current personal thread can move.
    # Repair still writes its pinned owned destination, with no active-run lookup.
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET ended_at=now(), status='ok' WHERE id=:id"), {"id": rig.turn_id})
    assert repair(restarted, row, service, subscriber) is None
    delivered = repair(restarted, replacement, service, subscriber)
    assert delivered.status == "DELIVERED"
    assert InboxSettlement(restarted).acknowledge(row.id, replacement.claim_token, evidence) == delivered
    assert InboxSettlement(restarted).acknowledge(row.id, row.claim_token, evidence) is None
    assert memory.calls[0] == memory.calls[1]
    assert len(client.frames) == 1
    assert INBOX not in delivered.to_api()["metadata"]
    with rig.engine.connect() as conn:
        saved = conn.execute(text("SELECT * FROM messages WHERE bot_id=:bot AND id=:id"),
                             {"bot": rig.bot, "id": row.user_message_id}).mappings().one()
    assert saved["session_id"] == rig.session_id
    assert saved["interrupt_source_message_id"] == rig.message_id and saved["interrupt_content_offset"] == 7
    assert saved["author_entity_type"] == "bot" and saved["author_entity_id"] == row.sender_bot_id
    subscriber.turn_text_offset.assert_not_awaited()


@pytest.mark.parametrize("key", ["write_token", "generation", "command_uuid", "bot_id", "user_id",
                                 "turn_id", "backend", "session_key", "request_id", "trigger_message_id", "session_id"])
def test_foreign_or_stale_lifecycle_proof_never_settles_operation(rig, running, key):
    store, row, evidence, settlement = operation(rig, running)
    wrong = (replace(evidence, **{key: 1 if key == "generation" else "foreign"}) if key in {
        "write_token", "generation", "command_uuid",
    } else replace(evidence, binding=replace(evidence.binding, **{key: "foreign"})))
    assert settlement.note_lifecycle(row.id, wrong, "started") is None
    assert settlement.note_lifecycle(row.id, wrong, "refused", proof="cli_terminal") is None
    assert settlement.acknowledge(row.id, row.claim_token, wrong) is None
    assert store.get(row.id).metadata[INBOX]["phase"] == "writing"


def test_unprepared_or_transport_only_input_cannot_be_accepted_or_acknowledged(rig, running):
    store, row, evidence, settlement = operation(rig, running, prepare=False)
    assert settlement.note_lifecycle(row.id, evidence, "queued") is None
    assert prepare_inbox_write(store, row.id, row.claim_token, evidence.binding)
    assert settlement.note_lifecycle(row.id, evidence, "sent") is None
    assert settlement.acknowledge(row.id, row.claim_token, evidence) is None
    assert settlement.history_repair(row.id, row.claim_token) is None


def test_late_started_proof_resolves_uncertainty_but_old_queued_does_not(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    assert settlement.note_lifecycle(row.id, evidence, "uncertain")
    assert settlement.note_lifecycle(row.id, evidence, "queued") is None
    assert settlement.history_repair(row.id, row.claim_token) is None
    claims.expire(store, row)
    assert store.recover_expired() == []
    assert settlement.claim_repair(rig.bot, claim_owner="replacement") is None
    assert settlement.note_lifecycle(row.id, evidence, "started")
    replacement = settlement.claim_repair(rig.bot, claim_owner="replacement")
    assert replacement and replacement.claim_token != row.claim_token
    _, service, subscriber = history(rig)
    assert repair(store, replacement, service, subscriber).status == "DELIVERED"


@pytest.mark.parametrize("acknowledged", [False, True])
def test_queued_can_still_be_proven_rejected_and_reopened(rig, running, acknowledged):
    store, row, evidence, settlement = operation(rig, running)
    assert settlement.note_lifecycle(row.id, evidence, "queued")
    if acknowledged:
        _, service, subscriber = history(rig)
        assert repair(store, row, service, subscriber).status == "DELIVERED"
    assert settlement.note_lifecycle(row.id, evidence, "refused", proof="transport_return") is None
    reopened = settlement.note_lifecycle(row.id, evidence, "refused", proof="cli_terminal")
    assert reopened.status == "QUEUED" and reopened.delivery_mode == "steer_rejected"
    assert reopened.metadata["steer_generation"] == 1 and reopened.metadata[INBOX] is None
    assert settlement.note_lifecycle(row.id, evidence, "started") is None
    assert settlement.acknowledge(row.id, row.claim_token, evidence) is None


def test_started_wins_over_rejection_and_older_uncertainty(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    assert settlement.note_lifecycle(row.id, evidence, "started", content_offset=9)
    assert settlement.note_lifecycle(row.id, evidence, "discarded", proof="sweep_receipt") is None
    assert settlement.note_lifecycle(row.id, evidence, "uncertain") is None
    # Late queued proof cannot downgrade start or move the durable anchor.
    assert settlement.note_lifecycle(row.id, evidence, "queued", content_offset=99)
    repair_context = settlement.history_repair(row.id, row.claim_token)
    assert repair_context.content_offset == 9
    _, service, subscriber = history(rig)
    assert repair(store, row, service, subscriber).status == "DELIVERED"


def test_ack_checks_exact_persisted_history_and_required_anchor(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    assert settlement.note_lifecycle(row.id, evidence, "queued", content_offset=3)
    assert settlement.acknowledge(row.id, row.claim_token, evidence) is None
    memory, _, _ = history(rig)
    memory.add_message("user", "callback", rig.session_id, None, row.user_message_id, None, None, "bot", row.sender_bot_id)
    assert settlement.acknowledge(row.id, row.claim_token, evidence) is None
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE messages SET interrupt_source_message_id=:trigger, interrupt_content_offset=3
            WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": row.user_message_id, "trigger": rig.message_id})
    assert settlement.acknowledge(row.id, row.claim_token, evidence).status == "DELIVERED"


@pytest.mark.parametrize("change", ["content='foreign'", "author_entity_id='foreign'", "session_id=NULL"])
def test_ack_never_substitutes_foreign_history_for_acceptance(rig, running, change):
    store, row, evidence, settlement = operation(rig, running)
    settlement.note_lifecycle(row.id, evidence, "queued")
    memory, _, _ = history(rig)
    memory.add_message("user", "callback", rig.session_id, None, row.user_message_id, None, None, "bot", row.sender_bot_id)
    with rig.engine.begin() as conn:
        conn.execute(text(f"UPDATE messages SET {change} WHERE bot_id=:bot AND id=:id"),
                     {"bot": rig.bot, "id": row.user_message_id})
    assert settlement.acknowledge(row.id, row.claim_token, evidence) is None


def test_history_mutation_serializes_generation_rejection_and_takeover(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    settlement.note_lifecycle(row.id, evidence, "queued")
    memory, service, subscriber = history(rig)

    def assert_write_fenced():
        with rig.engine.begin() as conn:
            assert not conn.execute(text(
                "SELECT pg_try_advisory_xact_lock(hashtext(:bot))"
            ), {"bot": rig.bot}).scalar_one()
        with pytest.raises(OperationalError, match="could not obtain lock"):
            with rig.engine.begin() as conn:
                conn.execute(text(
                    "SELECT id FROM inter_bot_deliveries WHERE id=:id FOR UPDATE NOWAIT"
                ), {"id": row.id})

    memory.on_saved = assert_write_fenced

    def anchor(*args, **kwargs):
        assert_write_fenced()
        return _persist_interrupt_anchor(*args, **kwargs)

    assert repair(store, row, service, subscriber, persist_anchor=anchor).status == "DELIVERED"


def test_stale_repair_context_cannot_mutate_history_after_rejection(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    settlement.note_lifecycle(row.id, evidence, "queued")
    context = settlement.history_repair(row.id, row.claim_token)
    assert context is not None
    assert settlement.note_lifecycle(row.id, evidence, "refused", proof="cli_terminal").status == "QUEUED"
    writes = []
    assert not settlement.persist_history(row.id, row.claim_token, context.evidence, lambda: writes.append(True))
    assert writes == []


def test_rejection_waits_for_history_worker_even_after_caller_cancellation(rig, running, monkeypatch):
    store, row, evidence, settlement = operation(rig, running)
    settlement.note_lifecycle(row.id, evidence, "queued", content_offset=7)
    memory, service, subscriber = history(rig)
    writing, release, rejecting = threading.Event(), threading.Event(), threading.Event()
    original_locked_row = SteerLifecycleFence._locked_row
    rejection_thread = []

    def locked_row(self, conn, delivery_id):
        if threading.get_ident() in rejection_thread:
            rejecting.set()
        return original_locked_row(self, conn, delivery_id)

    monkeypatch.setattr(SteerLifecycleFence, "_locked_row", locked_row)

    def pause_after_message():
        writing.set()
        assert release.wait(10), "fixture did not release history worker"

    memory.on_saved = pause_after_message

    def reject():
        rejection_thread.append(threading.get_ident())
        return settlement.note_lifecycle(row.id, evidence, "refused", proof="cli_terminal")

    async def cancel_repair():
        task = asyncio.create_task(repair_inbox_history(
            service, store, delivery_id=row.id, claim_token=row.claim_token,
            subscriber=subscriber, persist_anchor=_persist_interrupt_anchor,
        ))
        await asyncio.to_thread(writing.wait)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with ThreadPoolExecutor(max_workers=1) as pool:
            rejection = pool.submit(reject)
            try:
                assert await asyncio.to_thread(rejecting.wait, 10)
                # The rejection has entered its lock-taking boundary. A cancelled
                # async caller cannot unlock the still-running message/anchor worker.
                with rig.engine.begin() as conn:
                    assert not conn.execute(text(
                        "SELECT pg_try_advisory_xact_lock(hashtext(:bot))"
                    ), {"bot": rig.bot}).scalar_one()
                assert not rejection.done()
            finally:
                release.set()
            assert (await asyncio.wrap_future(rejection)).status == "QUEUED"

    try:
        asyncio.run(cancel_repair())
    finally:
        release.set()
    assert store.get(row.id).metadata[INBOX] is None
    assert not settlement.persist_history(row.id, row.claim_token, evidence, lambda: pytest.fail("stale write"))
    # Rejection can now advance and a successor can persist a different anchor;
    # no cancelled old worker remains to overwrite it after the fence released.
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE messages SET interrupt_source_message_id=:id, interrupt_content_offset=21
            WHERE bot_id=:bot AND id=:message
        """), {"id": rig.message_id, "bot": rig.bot, "message": row.user_message_id})
    with rig.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT interrupt_content_offset FROM messages WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": row.user_message_id}).scalar_one() == 21


def test_repair_refuses_a_deleted_or_retargeted_owned_destination(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    settlement.note_lifecycle(row.id, evidence, "queued")
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET status='deleted' WHERE id=:id"), {"id": rig.session_id})
    memory, service, subscriber = history(rig)
    assert repair(store, row, service, subscriber) is None
    assert memory.calls == []


def test_anchor_failure_repairs_same_message_before_ack(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    settlement.note_lifecycle(row.id, evidence, "queued", content_offset=13)
    memory, service, subscriber = history(rig)
    assert repair(store, row, service, subscriber, persist_anchor=lambda *_a, **_kw: False) is None
    assert store.get(row.id).status == "STEERING"
    assert settlement.acknowledge(row.id, row.claim_token, evidence) is None
    subscriber.publish_tool_event.assert_not_awaited()
    assert repair(store, row, service, subscriber).status == "DELIVERED"
    assert memory.calls[0] == memory.calls[1]


def test_cancelled_repair_does_not_ack_even_if_history_thread_completed(rig, running):
    store, row, evidence, settlement = operation(rig, running)
    settlement.note_lifecycle(row.id, evidence, "queued")
    _, service, subscriber = history(rig)
    subscriber.publish_tool_event.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        repair(store, row, service, subscriber)
    assert store.get(row.id).status == "STEERING"
    subscriber.publish_tool_event.side_effect = None
    assert repair(store, row, service, subscriber).status == "DELIVERED"
