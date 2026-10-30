"""Durable receipts for inter-bot delivery transitions (TASK-1745).

Each winning status write stages its row; the connection's commit hook appends
one ``system_events`` row per (delivery, revision), after every domain write.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text as sa_text
from sqlmodel import Session

from llm_bawt.inter_bot_delivery_ledger import DeliveryTransitionEvents
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.profiles import EntityProfile, EntityType
from tests.bot_tag_fixtures import clear_tags, make_private
from tests.pg_lock_wait import wait_blocked_by
from tests.test_turn_gates_pg import gated  # noqa: F401  (gated is a fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def ledger(gated, monkeypatch):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.utils.config import config

    gated.deliveries = InterBotDeliveryStore(gated.store.config)
    engine = gated.deliveries.engine
    EntityProfile.__table__.create(engine, checkfirst=True)
    gated.user = f"evt-user-{gated.tag}"
    with Session(engine) as session:
        session.add(EntityProfile(entity_type=EntityType.USER, entity_id=gated.user))
        session.commit()
    monkeypatch.setattr(config, "DEFAULT_USER", gated.user, raising=False)
    yield gated
    with engine.begin() as conn:
        conn.execute(sa_text(
            "DELETE FROM system_events WHERE subject_type='delivery' AND subject_id IN"
            " (SELECT id FROM inter_bot_deliveries WHERE sender_bot_id = ANY(:s))"
        ), {"s": gated.senders})
        conn.execute(sa_text("DELETE FROM entity_profiles WHERE entity_type='USER' AND entity_id=:u"),
                     {"u": gated.user})
        clear_tags(conn, *gated.bots)


def _send(ctx, sender, target, **extra):
    if sender not in ctx.senders:
        ctx.senders.append(sender)
    record, duplicate = ctx.deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=target, message="run the check",
        payload={"messages": [{"role": "user", "content": "run the check"}], "bot_id": target,
                 "user": ctx.user, "stream": True},
        **extra,
    )
    return record, duplicate


def _events(ctx, delivery_id):
    with ctx.deliveries.engine.connect() as conn:
        return [dict(row) for row in conn.execute(sa_text(
            "SELECT * FROM system_events WHERE event_key LIKE :key ORDER BY id"
        ), {"key": f"inter_bot:{delivery_id}:%"}).mappings()]


def _row(conn, delivery_id, *, lock=False):
    return dict(conn.execute(sa_text(
        "SELECT * FROM inter_bot_deliveries WHERE id=:id" + (" FOR UPDATE" if lock else "")
    ), {"id": delivery_id}).mappings().one())


def _cancel(ctx, delivery_id, reason="user_stop"):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    with ctx.deliveries.engine.begin() as conn:
        row = InterBotDeliveryStore.cancel_unaccepted_locked(conn, _row(conn, delivery_id, lock=True), reason)
        assert row is not None
        return row


def test_enqueue_and_terminal_transitions_each_get_one_receipt(ledger):
    sender, target = ledger.bot(), ledger.bot()
    record, duplicate = _send(ledger, sender, target)
    assert not duplicate
    assert ledger.deliveries.mark_failed(record.id, "validation exploded: secret text") is not None
    events = _events(ledger, record.id)
    assert [e["event_key"] for e in events] == [
        f"inter_bot:{record.id}:QUEUED:0",
        f"inter_bot:{record.id}:FAILED:{events[1]['event_key'].rsplit(':', 1)[1]}",
    ]
    assert int(events[1]["event_key"].rsplit(":", 1)[1]) > 0
    queued, failed = events
    assert queued["kind"] == "inter_bot_delivery" and queued["source"] == "delivery"
    assert queued["visibility"] == "user_feed" and queued["user_id"] == ledger.user
    assert queued["bot_id"] == sender and queued["subject_id"] == record.id
    assert queued["payload"] == {"phase": "QUEUED"} and queued["tone"] == "info"
    assert failed["tone"] == "error" and failed["payload"] == {"phase": "FAILED"}
    assert "secret" not in failed["summary"]


@pytest.mark.parametrize(("code", "kept"), [("user_stop", True), ("Stopped by user: secret", False)])
def test_only_a_slug_reason_code_reaches_the_payload(ledger, code, kept):
    record, _ = _send(ledger, ledger.bot(), ledger.bot())
    with ledger.deliveries.engine.begin() as conn:
        row = conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET status='CANCELLED', updated_at=now() WHERE id=:id RETURNING *"
        ), {"id": record.id}).mappings().one()
        DeliveryTransitionEvents.stage(conn, row, reason_code=code)
    (cancelled,) = [e for e in _events(ledger, record.id) if e["payload"]["phase"] == "CANCELLED"]
    assert cancelled["tone"] == "warning"
    assert cancelled["payload"] == ({"phase": "CANCELLED", "reason_code": code} if kept
                                    else {"phase": "CANCELLED"})


def test_a_cancel_appends_one_receipt(ledger):
    record, _ = _send(ledger, ledger.bot(), ledger.bot())
    _cancel(ledger, record.id, "Stopped by user")
    assert [e["payload"]["phase"] for e in _events(ledger, record.id)] == ["QUEUED", "CANCELLED"]


def test_a_replayed_enqueue_appends_nothing_new(ledger):
    sender, target = ledger.bot(), ledger.bot()
    first, _ = _send(ledger, sender, target, idempotency_key=f"evt-{ledger.tag}")
    again, duplicate = _send(ledger, sender, target, idempotency_key=f"evt-{ledger.tag}")
    assert duplicate and again.id == first.id
    assert len(_events(ledger, first.id)) == 1


def test_staging_one_row_twice_appends_once(ledger):
    record, _ = _send(ledger, ledger.bot(), ledger.bot())
    with ledger.deliveries.engine.begin() as conn:
        row = _row(conn, record.id)
        DeliveryTransitionEvents.stage(conn, row, row)
        DeliveryTransitionEvents.stage(conn, row)
    assert len(_events(ledger, record.id)) == 1


def test_a_rolled_back_transition_leaves_no_receipt(ledger):
    record, _ = _send(ledger, ledger.bot(), ledger.bot())
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    conn = ledger.deliveries.engine.connect()
    try:
        tx = conn.begin()
        InterBotDeliveryStore.cancel_unaccepted_locked(conn, _row(conn, record.id, lock=True), "user_stop")
        tx.rollback()
        # The next transaction on the same connection must not inherit the staged row.
        with conn.begin():
            conn.execute(sa_text("SELECT 1"))
    finally:
        conn.close()
    assert [e["payload"]["phase"] for e in _events(ledger, record.id)] == ["QUEUED"]
    with ledger.deliveries.engine.connect() as check:
        assert _row(check, record.id)["status"] == "QUEUED"


def test_a_rolled_back_savepoint_drops_only_its_own_rows(ledger):
    sender, target = ledger.bot(), ledger.bot()
    kept, _ = _send(ledger, sender, target)
    released, _ = _send(ledger, sender, target)
    dropped, _ = _send(ledger, sender, target)
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    def cancel(conn, record):
        InterBotDeliveryStore.cancel_unaccepted_locked(conn, _row(conn, record.id, lock=True), "user_stop")

    with ledger.deliveries.engine.begin() as conn:
        cancel(conn, kept)
        with conn.begin_nested():
            cancel(conn, released)
        outer = conn.begin_nested()
        with conn.begin_nested():  # released inside a savepoint that then rolls back
            cancel(conn, dropped)
        outer.rollback()
    assert [e["payload"]["phase"] for e in _events(ledger, kept.id)] == ["QUEUED", "CANCELLED"]
    assert [e["payload"]["phase"] for e in _events(ledger, released.id)] == ["QUEUED", "CANCELLED"]
    assert [e["payload"]["phase"] for e in _events(ledger, dropped.id)] == ["QUEUED"]


def test_a_missing_default_user_makes_bot_relays_audit_only(ledger, monkeypatch):
    from llm_bawt.utils.config import config

    monkeypatch.setattr(config, "DEFAULT_USER", None, raising=False)
    record, _ = _send(ledger, ledger.bot(), ledger.bot())
    (event,) = _events(ledger, record.id)
    assert event["visibility"] == "audit_only" and event["user_id"] is None
    assert event["bot_id"] is None and event["payload"] == {}


@pytest.mark.parametrize("side", ["sender", "target"])
def test_a_private_party_makes_the_receipt_audit_only(ledger, side):
    sender, target = ledger.bot(), ledger.bot()
    with ledger.deliveries.engine.begin() as conn:
        make_private(conn, sender if side == "sender" else target, "tag")
    record, _ = _send(ledger, sender, target)
    (event,) = _events(ledger, record.id)
    assert event["visibility"] == "audit_only" and event["user_id"] is None
    assert event["summary"] == "Restricted operational event"


def test_an_ordinary_chat_send_belongs_to_its_user(ledger, monkeypatch):
    from llm_bawt.utils.config import config

    # Not the default user: the receipt must follow the author.
    monkeypatch.setattr(config, "DEFAULT_USER", f"someone-else-{ledger.tag}", raising=False)
    sender, target = ledger.bot(), ledger.bot()
    record, _ = _send(ledger, sender, target, author=AuthorReference.user(ledger.user),
                      producer="ordinary", user_message_id=str(uuid.uuid4()),
                      metadata={"ordinary_chat": {"session_id": f"s-{ledger.tag}"}})
    (event,) = _events(ledger, record.id)
    assert event["visibility"] == "user_feed" and event["user_id"] == ledger.user
    assert event["bot_id"] == target


def test_competing_multi_row_cancels_never_deadlock_on_the_event_lock(ledger):
    """A holds X's receipt staged while it waits on Y; B cancels Y and commits.

    Were the receipt appended at the write (taking the event-order lock while
    A still waits on a row lock), B's commit would wait on A's event lock while
    A waits on B's row lock: a deadlock. Appended at commit, B finishes first.
    """
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    sender, target = ledger.bot(), ledger.bot()
    x, _ = _send(ledger, sender, target)
    y, _ = _send(ledger, sender, target)
    engine = ledger.deliveries.engine
    b_locked, b_go = threading.Event(), threading.Event()
    b_pid = {}

    def run_b():
        with engine.begin() as conn:
            conn.execute(sa_text("SET LOCAL lock_timeout = '15s'"))
            b_pid["pid"] = conn.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
            row = _row(conn, y.id, lock=True)
            b_locked.set()
            assert b_go.wait(30)
            assert InterBotDeliveryStore.cancel_unaccepted_locked(conn, row, "user_stop") is not None

    def run_a():
        with engine.begin() as conn:
            conn.execute(sa_text("SET LOCAL lock_timeout = '15s'"))
            row = _row(conn, x.id, lock=True)
            assert InterBotDeliveryStore.cancel_unaccepted_locked(conn, row, "user_stop") is not None
            # Waits on B's row lock with X's transition staged.
            assert _row(conn, y.id, lock=True)["status"] == "CANCELLED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        b = pool.submit(run_b)
        assert b_locked.wait(30)
        a = pool.submit(run_a)
        wait_blocked_by(engine, b_pid["pid"], a)
        b_go.set()
        b.result(timeout=60)
        a.result(timeout=60)
    (x_event,) = [e for e in _events(ledger, x.id) if e["payload"]["phase"] == "CANCELLED"]
    (y_event,) = [e for e in _events(ledger, y.id) if e["payload"]["phase"] == "CANCELLED"]
    assert y_event["id"] < x_event["id"]  # commit order


def test_an_ops_event_never_deadlocks_a_committing_delivery_transition(ledger):
    """A quiesce failure (A) records its job transition, then needs a delivery
    row (``release_job_gates`` → ``cancel_pending_controls``) that B holds while
    B's transition commits.

    Were the ops event appended at the write, A would hold the event-order
    lock while waiting on B's row lock, and B's commit hook would wait on A's
    event lock: a deadlock. Staged for commit, B commits first and A follows.
    """
    from types import SimpleNamespace

    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.ops.events import append_job_transition
    from llm_bawt.system_events import SystemEventEmitter

    y, _ = _send(ledger, ledger.bot(), ledger.bot())
    engine = ledger.deliveries.engine
    emitter = SystemEventEmitter(engine)
    job = SimpleNamespace(id=f"job-{ledger.tag}", operation_slug="test.quiesce", state="failed",
                          caller_bot_id=None, caller_user_id=ledger.user, caller_actor="test")
    b_locked, b_go = threading.Event(), threading.Event()
    b_pid = {}

    def run_b():
        with engine.begin() as conn:
            conn.execute(sa_text("SET LOCAL lock_timeout = '15s'"))
            b_pid["pid"] = conn.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
            row = _row(conn, y.id, lock=True)
            b_locked.set()
            assert b_go.wait(30)
            assert InterBotDeliveryStore.cancel_unaccepted_locked(conn, row, "user_stop") is not None

    def run_a():
        with engine.begin() as conn:
            conn.execute(sa_text("SET LOCAL lock_timeout = '15s'"))
            append_job_transition(emitter, conn, job)
            # Waits on B's row lock with the ops event staged.
            assert _row(conn, y.id, lock=True)["status"] == "CANCELLED"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            b = pool.submit(run_b)
            assert b_locked.wait(30)
            a = pool.submit(run_a)
            wait_blocked_by(engine, b_pid["pid"], a)
            b_go.set()
            b.result(timeout=60)
            a.result(timeout=60)
        (y_event,) = [e for e in _events(ledger, y.id) if e["payload"]["phase"] == "CANCELLED"]
        with engine.connect() as conn:
            ops_id = conn.execute(sa_text("SELECT id FROM system_events WHERE event_key=:k"),
                                  {"k": f"ops:{job.id}:failed"}).scalar_one()
        assert y_event["id"] < ops_id  # commit order
    finally:
        with engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM system_events WHERE event_key=:k"),
                         {"k": f"ops:{job.id}:failed"})
