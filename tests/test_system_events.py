"""Durable user event ledger contract (TASK-1745)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session

from llm_bawt.profiles import EntityProfile, EntityType
from llm_bawt.system_events import SystemEventEmitter


class RecordingRedis:
    def __init__(self, *, fail: bool = False):
        self.events: list[tuple[str, dict]] = []
        self.fail = fail

    def xadd(self, key, values, **_kwargs):
        if self.fail:
            raise ConnectionError("redis down")
        self.events.append((key, json.loads(values["payload"])))
        return "1-0"


def emitter():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    EntityProfile.__table__.create(engine)
    with Session(engine) as session:
        session.add_all(EntityProfile(entity_type=EntityType.USER, entity_id=user)
                        for user in ("user-a", "user-b", "a", "u"))
        session.commit()
    return SystemEventEmitter(engine)


def test_one_transactional_event_owner_scope_and_idempotent_outbox():
    store = emitter()
    with store.engine.begin() as conn:
        event_id = store.append(conn, event_key="ops:job1:queued", kind="ops", source="ops",
                                user_id="user-a", tone="info", summary="job queued",
                                subject_type="ops_job", subject_id="job1")
        assert store.append(conn, event_key="ops:job1:queued", kind="ops", source="ops",
                            user_id="user-a", tone="info", summary="job queued") == event_id
    assert event_id is not None
    assert [row["id"] for row in store.list_user("user-a")] == [str(event_id)]
    assert store.list_page("user-a", limit=1)[1] is False
    assert store.list_user("user-b") == []
    assert store.get_user_event("user-b", event_id) is None
    assert store.get_user_event("user-a", event_id) == store.list_user("user-a")[0]
    redis = RecordingRedis()
    assert store.publish_batch(redis) == 1
    assert store.publish_batch(redis) == 0
    assert redis.events == [("system_events:user-a", store.list_user("user-a")[0])]


def test_newly_private_bot_is_hidden_on_read_and_replayed_sse(monkeypatch):
    import llm_bawt.system_events as module

    class MutableTags:
        private = False
        available = True

        def __init__(self, _engine):
            pass

        def has(self, _bot, _tag, conn=None):
            return self.private

        def snapshot(self, _conn):
            return self

    monkeypatch.setattr(module, "BotTags", MutableTags)
    store = emitter()
    with store.engine.begin() as conn:
        event_id = store.append(conn, event_key="bot:visible", kind="delivery", source="delivery",
                                user_id="user-a", tone="info", summary="safe",
                                bot_id="some-bot")
    assert store.get_user_event("user-a", event_id) is not None
    MutableTags.available = False
    assert store.publish_batch(RecordingRedis()) == 0
    assert store.pending_count() == 1
    MutableTags.available = True
    MutableTags.private = True
    assert store.get_user_event("user-a", event_id) is None
    assert store.list_user("user-a") == []
    assert store.list_page("user-a")[1] is False
    assert store.publish_batch(RecordingRedis()) == 1  # stamps without publishing


def test_ownerless_and_private_events_are_audit_only_without_leaking_payloads():
    store = emitter()
    with store.engine.begin() as conn:
        for index, (owner, visible) in enumerate(((None, True), ("user-a", False))):
            assert store.append(conn, event_key=f"ops:job{index}:failed", kind="ops", source="ops",
                                user_id=owner, tone="error", summary="private job detail",
                                bot_id="private-bot", subject_type="ops_job", subject_id=str(index),
                                payload={"operation": "private operation"}, visible=visible)
    assert store.list_user("user-a") == []
    assert store.publish_batch(RecordingRedis()) == 0
    with store.engine.connect() as conn:
        rows = conn.execute(text("SELECT user_id, bot_id, summary, payload FROM system_events")).all()
    assert all(row.user_id is None and row.bot_id is None and row.summary == "Restricted operational event"
               and "private operation" not in str(row.payload) for row in rows)


def test_domain_transaction_not_poisoned_by_invalid_projection_or_event_error():
    store = emitter()
    with store.engine.begin() as conn:
        conn.execute(text("CREATE TABLE domain_updates (id INTEGER PRIMARY KEY)"))
    with store.engine.begin() as conn:
        conn.execute(text("INSERT INTO domain_updates (id) VALUES (1)"))
        assert store.append(conn, event_key="bad", kind="ops", source="ops", user_id=None,
                            tone="error", summary="bad", payload={"secret": "no"}) is None
        # Deliberately break the event table. The savepoint isolates the error.
        conn.execute(text("DROP TABLE system_events"))
        assert store.append(conn, event_key="ops:2", kind="ops", source="ops", user_id="u",
                            tone="info", summary="fine") is None
        conn.execute(text("INSERT INTO domain_updates (id) VALUES (2)"))
    with store.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM domain_updates")).scalar_one() == 2


def test_rollback_does_not_publish_and_scoped_cursors_follow_ids():
    store = emitter()
    try:
        with store.engine.begin() as conn:
            store.append(conn, event_key="rolled", kind="ops", source="ops", user_id="a",
                         tone="info", summary="rolled")
            raise RuntimeError("abort")
    except RuntimeError:
        pass
    ids = []
    for index in range(3):
        with store.engine.begin() as conn:
            ids.append(store.append(conn, event_key=f"ops:{index}", kind="ops", source="ops",
                                    user_id="a", tone="info", summary="fine"))
    assert store.list_user("a", limit=2) == store.list_user("a", after=ids[0])
    assert store.list_page("a", limit=2)[1] is True
    assert store.list_page("a", after=ids[0], limit=2)[1] is False
    assert [r["id"] for r in store.list_user("a", before=ids[2])] == [str(ids[0]), str(ids[1])]
    redis = RecordingRedis(fail=True)
    assert store.publish_batch(redis) == 0
    assert store.pending_count() == 3
    redis.fail = False
    assert store.publish_batch(redis) == 3
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE system_events SET created_at = :then"),
                     {"then": datetime.now(timezone.utc) - timedelta(days=8)})
    assert store.prune(report_only=True) == 3
    assert store.prune() == 3


def test_unknown_owner_is_audit_only_and_same_transition_key_never_duplicates():
    store = emitter()
    with store.engine.begin() as conn:
        unknown = store.append(conn, event_key="ops:job:queued", kind="ops_job", source="ops",
                               user_id="nonexistent", tone="info", summary="sensitive details")
        assert unknown is not None
        assert store.append(conn, event_key="ops:job:queued", kind="ops_job", source="ops",
                            user_id="user-a", tone="info", summary="different retry") == unknown
    assert store.list_user("user-a") == []
    assert store.list_user("nonexistent") == []
    with store.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM system_events")).scalar_one() == 1


def test_uncommitted_event_not_visible_to_a_second_reader(tmp_path):
    # A file-backed SQLite database gives each reader its own connection.
    engine = create_engine(f"sqlite:///{tmp_path / 'ledger.db'}")
    EntityProfile.__table__.create(engine)
    with Session(engine) as session:
        session.add(EntityProfile(entity_type=EntityType.USER, entity_id="reader"))
        session.commit()
    store = SystemEventEmitter(engine)
    with engine.begin() as conn:
        first = store.append(conn, event_key="first", kind="ops", source="ops",
                             user_id="reader", tone="info", summary="pending")
        assert first is not None
        assert store.list_user("reader", after=0) == []
    assert [row["id"] for row in store.list_user("reader", after=0)] == [str(first)]
