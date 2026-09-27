"""Tests for confirmed-start turn reaping (turn_logs.reap_other_open_turns).

On confirmed start, older abandoned ordinary turns are reaped atomically.
TASK-952 excludes registered live workers and pending cancellation intent;
new output is not proof those workers stopped. Reaped tool rows are settled
in the same transaction. Review regressions cover these exclusions.

Engine-backed against in-memory SQLite (RETURNING + partial-predicate UPDATE
both supported) so the real SQL path is exercised without a live Postgres.
"""

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service.turn_logs import TurnLog, TurnLogStore


def _store():
    engine = create_engine("sqlite://")
    from llm_bawt.service.tool_call_store import ToolCallRecord
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__])
    store = TurnLogStore.__new__(TurnLogStore)  # bypass config-driven __init__
    store.engine = engine
    return store


def _add(store, *, turn_id, bot_id, user_id, ended=False, created_at=None):
    with Session(store.engine) as s:
        s.add(TurnLog(
            id=turn_id,
            bot_id=bot_id,
            user_id=user_id,
            created_at=created_at or datetime.now(timezone.utc),
            status="ok" if ended else "streaming",
            ended_at=datetime.now(timezone.utc) if ended else None,
        ))
        s.commit()


def _row(store, turn_id):
    with Session(store.engine) as s:
        return s.exec(select(TurnLog).where(TurnLog.id == turn_id)).first()


def test_reaps_other_open_turns_for_same_bot():
    store = _store()
    _add(store, turn_id="zombie", bot_id="byte", user_id="nick")   # stuck open
    _add(store, turn_id="current", bot_id="byte", user_id="nick")  # the live one

    reaped = store.reap_other_open_turns(bot_id="byte", current_turn_id="current")

    assert [r["id"] for r in reaped] == ["zombie"]
    z = _row(store, "zombie")
    assert z.status == "timeout"
    assert z.end_reason == "timeout"
    assert z.ended_at is not None
    # The triggering turn is never closed by its own reap.
    assert _row(store, "current").ended_at is None


def test_does_not_touch_the_current_turn_or_other_bots():
    store = _store()
    _add(store, turn_id="current", bot_id="byte", user_id="nick")
    _add(store, turn_id="other_bot", bot_id="snark", user_id="nick")  # different bot

    reaped = store.reap_other_open_turns(bot_id="byte", current_turn_id="current")

    assert reaped == []
    assert _row(store, "current").ended_at is None
    assert _row(store, "other_bot").ended_at is None  # snark unaffected


def test_already_closed_turns_are_not_re_reaped():
    store = _store()
    _add(store, turn_id="done", bot_id="byte", user_id="nick", ended=True)
    _add(store, turn_id="current", bot_id="byte", user_id="nick")

    reaped = store.reap_other_open_turns(bot_id="byte", current_turn_id="current")

    assert reaped == []  # already terminal → ended_at NOT NULL → excluded
    assert _row(store, "done").status == "ok"


def test_returns_user_id_for_indicator_clear():
    # The reaped row's own user_id is returned so the caller publishes
    # turn_complete on the correct {bot_id}:{user_id} stream.
    store = _store()
    _add(store, turn_id="zombie", bot_id="byte", user_id="someoneelse")
    _add(store, turn_id="current", bot_id="byte", user_id="nick")

    reaped = store.reap_other_open_turns(bot_id="byte", current_turn_id="current")

    assert reaped == [{"id": "zombie", "user_id": "someoneelse"}]


def test_does_not_reap_newer_queued_turn_when_older_turn_confirms_late():
    store = _store()
    base = datetime(2026, 7, 10, 16, 35, 35, tzinfo=timezone.utc)
    _add(store, turn_id="current", bot_id="byte", user_id="nick", created_at=base)
    _add(
        store,
        turn_id="newer_queued",
        bot_id="byte",
        user_id="nick",
        created_at=base + timedelta(minutes=2),
    )
    _add(
        store,
        turn_id="older_zombie",
        bot_id="byte",
        user_id="nick",
        created_at=base - timedelta(minutes=5),
    )

    reaped = store.reap_other_open_turns(bot_id="byte", current_turn_id="current")

    assert [r["id"] for r in reaped] == ["older_zombie"]
    assert _row(store, "older_zombie").status == "timeout"
    assert _row(store, "current").ended_at is None
    assert _row(store, "newer_queued").ended_at is None
    assert _row(store, "newer_queued").status == "streaming"


def test_missing_current_turn_does_not_reap_anything():
    store = _store()
    _add(store, turn_id="zombie", bot_id="byte", user_id="nick")

    reaped = store.reap_other_open_turns(bot_id="byte", current_turn_id="missing")

    assert reaped == []
    assert _row(store, "zombie").ended_at is None


def test_no_engine_is_safe():
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = None
    assert store.reap_other_open_turns(bot_id="byte", current_turn_id="x") == []
