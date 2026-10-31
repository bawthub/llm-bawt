"""PostgreSQL tests for TASK-1327 D2: one sender notice per cancelled delivery.

The pump scans every pending row in the module's private database, so each
test cleans up after itself (the ``gated`` fixture deletes the deliveries of
every tracked sender, and ``notices`` adds the system sender's notices).
"""

from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.bot_identities import SYSTEM_BOT_ID
from llm_bawt.service import delivery_cancel_notice as cn
from llm_bawt.service.turn_admission import lock_target
from tests.bot_tag_fixtures import PRIVATE_MODES, clear_tags, make_private
from tests.pg_lock_wait import wait_blocked_by
from tests.test_turn_gates_pg import gated  # noqa: F401  (gated is a fixture)

pytestmark = pytest.mark.usefixtures("private_app_pg")


@pytest.fixture
def notices(gated):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    gated.senders.append(SYSTEM_BOT_ID)  # the notices: this module's private database only
    gated.deliveries = InterBotDeliveryStore(gated.store.config)
    gated.pump = cn.CancelNoticePump(gated.deliveries)
    yield gated
    with gated.store.engine.begin() as conn:
        clear_tags(conn, *gated.bots)


def _send(ctx, sender, target, message="please run the migration", **extra):
    if sender not in ctx.senders:
        ctx.senders.append(sender)
    record, _ = ctx.deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=target, message=message,
        payload={"messages": [{"role": "user", "content": message}], "bot_id": target,
                 "user": "nick", "stream": True},
        **extra,
    )
    return record


def _cancel(ctx, record, reason="Stopped by user"):
    with ctx.deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET status='CANCELLED', last_error=:reason, updated_at=now()"
            " WHERE id=:id"
        ), {"id": record.id, "reason": reason})


def _stamp(ctx, delivery_id):
    with ctx.deliveries.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT cancel_notified_at FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).scalar_one()


def _notices_for(ctx, delivery_id):
    with ctx.deliveries.engine.connect() as conn:
        return [dict(row) for row in conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE idempotency_key=:key"
        ), {"key": f"{cn.NOTICE_KEY_PREFIX}{delivery_id}"}).mappings()]


@pytest.mark.integration
def test_each_cancelled_bot_delivery_gets_exactly_one_notice(notices):
    sender, target = notices.bot(), notices.bot()
    plain = _send(notices, sender, target, "please run the migration\n  on staging")
    kinded = _send(notices, sender, target, "task result", message_kind="TASK_RESULT", task_id="TASK-9")
    _cancel(notices, plain)
    _cancel(notices, kinded, reason=None)

    first = notices.pump.run_pass()

    assert sorted(record.target_bot_id for record in first.notified) == [sender, sender]
    assert (first.skipped, first.errors) == (0, 0)
    [notice] = _notices_for(notices, plain.id)
    assert (notice["sender_bot_id"], notice["target_bot_id"]) == (SYSTEM_BOT_ID, sender)
    assert (notice["status"], notice["message_kind"]) == ("QUEUED", cn.NOTICE_KIND)
    assert notice["metadata_json"]["cancelled_delivery_id"] == plain.id
    assert notice["payload_json"]["user"] == "nick" and notice["payload_json"]["bot_id"] == sender
    assert plain.id in notice["message"] and target in notice["message"]
    assert "Stopped by user" in notice["message"]
    assert "please run the migration on staging" in notice["message"]  # whitespace-flattened excerpt
    [kinded_notice] = _notices_for(notices, kinded.id)
    assert kinded_notice["task_id"] == "TASK-9" and "not recorded" in kinded_notice["message"]
    assert _stamp(notices, plain.id) is not None and _stamp(notices, kinded.id) is not None

    second = notices.pump.run_pass()

    assert (second.notified, second.skipped, second.errors) == ([], 0, 0)
    assert len(_notices_for(notices, plain.id)) == 1


@pytest.mark.integration
def test_stopped_started_delivery_notifies_sender_without_claiming_it_never_ran(notices):
    sender, target = notices.bot(), notices.bot()
    started = _send(notices, sender, target, "please check the migration")
    with notices.deliveries.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='DELIVERED', delivered_at=now(),
                last_error='Stopped by user after it ran',
                metadata_json=jsonb_set(metadata_json, '{stopped_after_start}', 'true'::jsonb),
                updated_at=now()
            WHERE id=:id
        """), {"id": started.id})

    first = notices.pump.run_pass()
    assert len(first.notified) == 1 and first.errors == 0
    [notice] = _notices_for(notices, started.id)
    assert "stopped by user after it ran" in notice["message"].lower()
    assert "nothing was done" not in notice["message"].lower()
    assert "before it ran" not in notice["message"].lower()
    assert _stamp(notices, started.id) is not None
    assert notices.pump.run_pass().notified == []


@pytest.mark.integration
def test_a_cancelled_notice_produces_no_further_notice(notices):
    sender, target = notices.bot(), notices.bot()
    source = _send(notices, sender, target)
    _cancel(notices, source)
    [notice] = notices.pump.run_pass().notified
    _cancel(notices, notice, reason="Stopped by user")
    looks_like_a_notice = _send(notices, sender, target, message_kind=cn.NOTICE_KIND)
    _cancel(notices, looks_like_a_notice)

    assert notices.pump.run_pass().notified == []
    assert _stamp(notices, notice.id) is None and _stamp(notices, looks_like_a_notice.id) is None
    assert _notices_for(notices, notice.id) == [] and _notices_for(notices, looks_like_a_notice.id) == []


@pytest.mark.integration
def test_user_authored_scheduled_and_self_sent_rows_are_never_read_or_stamped(notices):
    from llm_bawt.message_authorship import AuthorReference

    bot, other = notices.bot(), notices.bot()
    user_chat = _send(notices, bot, bot, "nick's own chat", author=AuthorReference.user("nick"),
                      metadata={"ordinary_chat": {"session_id": None, "stream": True}})
    self_sent = _send(notices, bot, bot, "a note to self")
    scheduled = _send(notices, other, bot, "scheduled prompt")
    with notices.deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET metadata_json=CAST(:m AS jsonb) WHERE id=:id"
        ), {"id": scheduled.id, "m": '{"prompt_schedule": {"schedule_id": "s"}}'})
    for record in (user_chat, self_sent, scheduled):
        _cancel(notices, record)

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped) == ([], 0)
    for record in (user_chat, self_sent, scheduled):
        assert _stamp(notices, record.id) is None
        assert _notices_for(notices, record.id) == []


@pytest.mark.integration
def test_a_sender_that_is_not_an_agent_bot_is_stamped_and_skipped(notices):
    target = notices.bot()
    chat_bot = notices.bot(bot_type="chat", backend="")
    record = _send(notices, chat_bot, target)
    _cancel(notices, record)

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped, result.errors) == ([], 1, 0)
    assert _stamp(notices, record.id) is not None
    assert _notices_for(notices, record.id) == []


@pytest.mark.integration
def test_a_sender_with_no_profile_is_private_and_never_stamped(notices):
    # Unknown bot = private (BOT_TAGS.md §5.3): the row stays pending, unread.
    target = notices.bot()
    no_profile = f"test-gate-unprofiled-{uuid.uuid4().hex}"
    rows = [_send(notices, sender, target) for sender in (no_profile, SYSTEM_BOT_ID)]
    for record in rows:
        _cancel(notices, record)

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped, result.errors) == ([], 0, 0)
    for record in rows:
        assert _stamp(notices, record.id) is None
        assert _notices_for(notices, record.id) == []


@pytest.mark.integration
def test_a_row_with_no_user_scope_is_stamped_and_skipped(notices, monkeypatch):
    sender, target = notices.bot(), notices.bot()
    record, _ = notices.deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=target, message="no user",
        payload={"messages": [{"role": "user", "content": "no user"}], "bot_id": target},
    )
    _cancel(notices, record)
    monkeypatch.setattr(notices.deliveries.config, "DEFAULT_USER", "", raising=False)

    result = notices.pump.run_pass()

    assert (result.notified, result.skipped) == ([], 1)
    assert _stamp(notices, record.id) is not None and _notices_for(notices, record.id) == []


@pytest.mark.integration
@pytest.mark.parametrize("mode", PRIVATE_MODES)
@pytest.mark.parametrize("side", ["sender", "target"])
def test_a_private_bot_is_never_read_stamped_or_notified(notices, mode, side):
    """BOT_TAGS.md §10.11: private sender or target, each of the three private ways."""
    private, other = notices.bot(), notices.bot()
    sender, target = (private, other) if side == "sender" else (other, private)
    hidden = _send(notices, sender, target)
    public_sender, public_target = notices.bot(), notices.bot()
    public = _send(notices, public_sender, public_target)
    for record in (hidden, public):
        _cancel(notices, record)
    with notices.deliveries.engine.begin() as conn:
        make_private(conn, private, mode)

    assert notices.pump.report()["pending"] == 1
    with notices.deliveries.engine.connect() as conn:
        candidates = {row["id"] for row in notices.pump._candidates()}
        locked = conn.execute(sa_text(
            f"SELECT id FROM inter_bot_deliveries WHERE id=:id AND {cn._PENDING} AND {cn._UNPROTECTED}"
        ), notices.pump._params(id=hidden.id)).first()
    assert hidden.id not in candidates and public.id in candidates and locked is None
    result = notices.pump.run_pass()

    # The public pair is still notified: the sql_allowed parameters are wired.
    assert [record.target_bot_id for record in result.notified] == [public_sender]
    assert (result.skipped, result.errors) == (0, 0)
    assert _stamp(notices, hidden.id) is None and _notices_for(notices, hidden.id) == []
    assert notices.pump._decide({"id": hidden.id, "sender_bot_id": sender,
                                 "target_bot_id": target}) == (cn._GONE, None)
    assert _stamp(notices, hidden.id) is None


@pytest.mark.integration
def test_cancel_all_notifies_every_sender_of_the_held_rows(notices):
    from llm_bawt.service.user_stop_hold import HOLD_CANCEL_REASON, cancel_all, hold_owner, open_or_extend

    bot, first, second = notices.bot(), notices.bot(), notices.bot()
    rows = [_send(notices, first, bot, "one"), _send(notices, second, bot, "two")]
    notices.owners.append(hold_owner(bot))
    with notices.store.engine.begin() as conn:
        open_or_extend(conn, bot, 600)
    assert len(cancel_all(notices.store.engine, bot)) == 2

    result = notices.pump.run_pass()

    assert sorted(record.target_bot_id for record in result.notified) == sorted([first, second])
    for record in rows:
        [notice] = _notices_for(notices, record.id)
        assert HOLD_CANCEL_REASON in notice["message"]
    assert notices.pump.run_pass().notified == []


@pytest.mark.integration
def test_the_notice_survives_a_freeze_and_waits_behind_it(notices):
    from llm_bawt.service.turn_gates import GateScope
    from tests.test_turn_gates_pg import _activate, _spec

    sender, target = notices.bot(), notices.bot()
    source = _send(notices, sender, target)
    _cancel(notices, source)
    _activate(notices.store, _spec(notices.owner(), GateScope.for_bots(sender), mode="deny"))

    [notice] = notices.pump.run_pass().notified

    assert notice.target_bot_id == sender and notice.status == "QUEUED"
    assert notices.deliveries.claim_next(sender, claim_owner="worker", steer_capable=False) is None


@pytest.mark.integration
@pytest.mark.parametrize("mode", [cn.REPORT, cn.OFF, "bogus"])
def test_report_and_off_modes_write_nothing(notices, monkeypatch, mode):
    import llm_bawt.runtime_setting_resolution as resolution

    sender, target = notices.bot(), notices.bot()
    source = _send(notices, sender, target)
    _cancel(notices, source)
    monkeypatch.setattr(resolution, "resolve_global_runtime_setting",
                        lambda config, key, fallback=None: mode if key == cn.MODE_SETTING else fallback)

    report = notices.pump.report()
    result = notices.pump.run_pass()

    assert report["pending"] >= 1 and report["oldest"] is not None
    assert (result.notified, result.skipped, result.errors) == ([], 0, 0)
    assert _stamp(notices, source.id) is None and _notices_for(notices, source.id) == []


def _failing_for(monkeypatch, failing: set[str]):
    real = cn.notice_text

    def notice_text(bot_id, variables, *, started=False):
        if variables["delivery_id"] in failing:
            raise RuntimeError("prompt registry down")
        return real(bot_id, variables, started=started)

    monkeypatch.setattr(cn, "notice_text", notice_text)


@pytest.mark.integration
def test_a_prompt_failure_stamps_nothing_and_the_row_retries_after_its_backoff(notices, monkeypatch):
    sender, target = notices.bot(), notices.bot()
    source = _send(notices, sender, target)
    _cancel(notices, source)
    now = [1000.0]
    pump = cn.CancelNoticePump(notices.deliveries, clock=lambda: now[0])
    failing = {source.id}
    _failing_for(monkeypatch, failing)

    failed = pump.run_pass()

    assert (failed.notified, failed.errors) == ([], 1)
    assert _stamp(notices, source.id) is None and _notices_for(notices, source.id) == []
    failing.clear()
    now[0] += cn.RETRY_BASE_S - 1
    assert pump.run_pass() == cn.PassResult()  # still backing off: not read
    now[0] += 2

    [notice] = pump.run_pass().notified

    assert notice.target_bot_id == sender and _stamp(notices, source.id) is not None
    assert pump._backoff == {}


@pytest.mark.integration
def test_rows_that_keep_failing_never_starve_newer_ones(notices, monkeypatch):
    sender, target = notices.bot(), notices.bot()
    old, new = _send(notices, sender, target, "old"), _send(notices, sender, target, "new")
    _cancel(notices, old)
    _cancel(notices, new)
    now = [1000.0]
    pump = cn.CancelNoticePump(notices.deliveries, batch_size=1, clock=lambda: now[0])
    _failing_for(monkeypatch, {old.id})

    assert pump.run_pass().errors == 1  # the oldest row is first in line, and fails
    [notice] = pump.run_pass().notified  # it backs off; the newer row is read next

    assert notice.metadata["cancelled_delivery_id"] == new.id
    assert _stamp(notices, old.id) is None
    now[0] += cn.RETRY_BASE_S + 1
    assert pump.run_pass().errors == 1
    assert pump._backoff[old.id][0] == 2  # the delay doubles per consecutive failure
    assert pump._backoff[old.id][1] == now[0] + 2 * cn.RETRY_BASE_S


@pytest.mark.integration
@pytest.mark.parametrize("mode", PRIVATE_MODES)
def test_the_backfill_runs_only_with_the_column_add(notices, mode):
    sender, target, private = notices.bot(), notices.bot(), notices.bot()
    history = _send(notices, sender, target, "cancelled long ago")
    to_private = _send(notices, sender, private, "cancelled long ago")
    for record in (history, to_private):
        _cancel(notices, record)
    engine = notices.deliveries.engine
    with engine.begin() as conn:
        make_private(conn, private, mode)
        conn.execute(sa_text("ALTER TABLE inter_bot_deliveries DROP COLUMN cancel_notified_at"))
    try:
        with engine.begin() as conn:
            cn._bootstrap(conn)
    finally:
        with engine.begin() as conn:  # never leave the module's database without the column
            cn._bootstrap(conn)

    assert _stamp(notices, history.id) is not None  # history is stamped, never notified
    assert _stamp(notices, to_private.id) is None  # private rows are never touched
    with engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET cancel_notified_at=NULL WHERE id=:id"),
                     {"id": history.id})
        cn._bootstrap(conn)  # column present: no ADD, so no backfill
        index = conn.execute(sa_text("SELECT to_regclass('ix_inter_bot_delivery_cancel_notice')")).scalar_one()
    assert _stamp(notices, history.id) is None
    assert index is not None
    [notice] = notices.pump.run_pass().notified
    assert notice.target_bot_id == sender


@pytest.mark.integration
def test_the_backfill_stamps_nothing_without_bot_tags(notices, monkeypatch):
    sender, target = notices.bot(), notices.bot()
    history = _send(notices, sender, target, "cancelled long ago")
    _cancel(notices, history)
    engine = notices.deliveries.engine
    monkeypatch.setattr(cn, "owned", lambda conn, rel, kinds="rp": rel != "bot_profile_tags")
    with engine.begin() as conn:
        conn.execute(sa_text("ALTER TABLE inter_bot_deliveries DROP COLUMN cancel_notified_at"))
    try:
        with engine.begin() as conn:
            cn._bootstrap(conn)
    finally:
        monkeypatch.undo()
        with engine.begin() as conn:
            cn._bootstrap(conn)
    assert _stamp(notices, history.id) is None


@pytest.mark.integration
def test_target_first_lock_order_never_deadlocks_with_cancel_and_a_multibot_gate(notices):
    """B2 (Loopy's probe): a row-first pump holds the source row and waits on the
    sender's lock, a multi-bot gate holds that lock and waits on the target's, and a
    retried cancel holds the target's lock and waits on the row: a three-party cycle.
    Taking both bots' locks, sorted, before the row lets the cancel finish first."""
    sender, target = notices.bot(), notices.bot()
    assert sender < target  # the gate takes the sender's lock first, as bots_in_scope sorts
    source = _send(notices, sender, target)
    _cancel(notices, source)
    [candidate] = [c for c in notices.pump._candidates() if c["id"] == source.id]
    outcome, errors = [], []

    def run(fn):
        try:
            fn()
        except Exception as exc:  # surfaced by the asserts below
            errors.append(repr(exc))

    pump = threading.Thread(target=run, args=(lambda: outcome.append(notices.pump._decide(candidate)),))
    retried_cancel = threading.Thread(target=run, args=(lambda: outcome.append(notices.deliveries.cancel(source.id)),))
    with notices.store.engine.begin() as gate:
        lock_target(gate, sender)
        pid = gate.execute(sa_text("SELECT pg_backend_pid()")).scalar_one()
        pump.start()
        wait_blocked_by(notices.store.engine, pid, pump, detail=lambda: (outcome, errors))
        retried_cancel.start()
        retried_cancel.join(5)
        assert not retried_cancel.is_alive(), "the cancel waited on a row the pump holds"
        lock_target(gate, target)
    pump.join(10)

    assert not pump.is_alive() and not errors, errors
    assert outcome[0] is None  # the cancel saw a row that is already CANCELLED
    status, notice = outcome[1]
    assert status == cn._NOTIFIED and notice.target_bot_id == sender
    assert len(_notices_for(notices, source.id)) == 1
