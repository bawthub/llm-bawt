"""TASK-2213: extraction on a shared bot never attributes another user's facts."""
from types import SimpleNamespace

import pytest
from sqlalchemy import Boolean, Column, Float, MetaData, String, Table, create_engine, select

from llm_bawt.memory.extraction.service import ExtractedFact, MemoryExtractionService
from llm_bawt.memory.postgresql_messages import PostgreSQLMessageMixin
from llm_bawt.service.background_tasks import BackgroundTasksMixin


class MessageBackend(PostgreSQLMessageMixin):
    def __init__(self):
        self.engine = create_engine("sqlite://")
        self.bot_id = "shared-test-bot"
        metadata = MetaData()
        self.messages_table = Table(
            "messages", metadata,
            Column("id", String, primary_key=True), Column("role", String),
            Column("content", String), Column("timestamp", Float),
            Column("session_id", String), Column("processed", Boolean, default=False),
            Column("author_entity_type", String), Column("author_entity_id", String),
        )
        self.sessions_table = Table(
            "sessions", metadata, Column("id", String, primary_key=True),
            Column("bot_id", String), Column("user_id", String), Column("status", String),
        )
        metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(self.sessions_table.insert(), [
                {"id": "session-a", "bot_id": self.bot_id, "user_id": "user-a", "status": "active"},
                {"id": "session-b", "bot_id": self.bot_id, "user_id": "user-b", "status": "archived"},
            ])
            conn.execute(self.messages_table.insert(), [
                {"id": "message-a", "role": "user", "content": "I own a dog named Apple.",
                 "timestamp": 1., "session_id": "session-a", "processed": False,
                 "author_entity_type": "user", "author_entity_id": "user-a"},
                {"id": "message-b", "role": "user", "content": "I own a cat named Berry.",
                 "timestamp": 2., "session_id": "session-b", "processed": False,
                 "author_entity_type": "user", "author_entity_id": "user-b"},
            ])


@pytest.mark.parametrize("task_users", [("user-a", "user-b"), ("system",), (None,)])
def test_two_users_extract_only_their_own_session_messages(monkeypatch, task_users):
    backend = MessageBackend()
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
    seen = []
    stored = []
    profiles = []

    def extract(self, messages, **kwargs):
        seen.append([m["id"] for m in messages])
        return [ExtractedFact(content=m["content"], importance=.9,
                              tags=["pets"], source_message_ids=[m["id"]]) for m in messages]

    monkeypatch.setattr(MemoryExtractionService, "extract_from_conversation", extract)
    monkeypatch.setattr("llm_bawt.mcp_server.extraction.extract_profile_attributes_from_fact",
                        lambda fact, user_id, **kwargs: profiles.append((user_id, fact.content)))
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(MEMORY_PROFILE_ATTRIBUTE_ENABLED=True)
    service._default_bot = backend.bot_id
    service._get_background_client = lambda **kw: (object(), "mock-model")
    monkeypatch.setattr("llm_bawt.memory.extraction.storage.ExtractionMemoryStore",
                        lambda backend, bot, user: SimpleNamespace(
                            list_memories=lambda **kw: [],
                            add_memory=lambda **kw: stored.append((user, kw["content"])) or "stored",
                        ))
    try:
        for user in task_users:
            service._process_memory_extraction(SimpleNamespace(bot_id=backend.bot_id, user_id=user, payload={}))
        assert seen == [["message-a"], ["message-b"]]
        assert profiles == [("user-a", "I own a dog named Apple."), ("user-b", "I own a cat named Berry.")]
        assert stored == profiles
    finally:
        backend.engine.dispose()


def test_selection_before_limit_excludes_unowned_deleted_and_other_bot_sessions():
    backend = MessageBackend()
    try:
        with backend.engine.begin() as conn:
            conn.execute(backend.sessions_table.insert(), [
                {"id": "unknown", "bot_id": backend.bot_id, "user_id": None, "status": "active"},
                {"id": "deleted", "bot_id": backend.bot_id, "user_id": "user-a", "status": "deleted"},
                {"id": "foreign", "bot_id": "another-bot", "user_id": "user-a", "status": "active"},
            ])
            for session_id in (None, "unknown", "deleted", "foreign"):
                conn.execute(backend.messages_table.insert(), {
                    "id": str(session_id), "session_id": session_id, "role": "user",
                    "content": "Unproven", "timestamp": 0., "processed": False,
                })
        assert backend.get_unprocessed_message_users() == ["user-a", "user-b"]
        assert [m["id"] for m in backend.get_unprocessed_messages(limit=1, user_id="user-b")] == ["message-b"]
        assert backend.get_unprocessed_messages() == []
        assert backend.get_unprocessed_messages(user_id="system") == []
    finally:
        backend.engine.dispose()


def test_ownerless_system_job_never_falls_back_to_default_user(monkeypatch):
    backend = MessageBackend()
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
    with backend.engine.begin() as conn:
        conn.execute(backend.sessions_table.update().values(user_id=None))
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(DEFAULT_USER="user-a")
    service._get_background_client = lambda **kw: pytest.fail("ownerless job must not extract")
    try:
        result = service._process_memory_extraction(SimpleNamespace(bot_id=backend.bot_id, user_id="system", payload={}))
        assert result["messages_found"] == result["owners_processed"] == 0
        with backend.engine.connect() as conn:
            assert list(conn.execute(select(backend.messages_table.c.processed)).scalars()) == [False, False]
    finally:
        backend.engine.dispose()


def test_system_batch_isolates_failure_and_drops_outside_batch_source(monkeypatch):
    from llm_bawt.service.background_tasks import ExtractionBatchFailed

    backend = MessageBackend()
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
    profiles, writes = [], []
    monkeypatch.setattr("llm_bawt.mcp_server.extraction.extract_profile_attributes_from_fact",
                        lambda **kw: profiles.append(kw))
    monkeypatch.setattr("llm_bawt.memory.extraction.storage.ExtractionMemoryStore", lambda *a: SimpleNamespace(
        list_memories=lambda **kw: [], add_memory=lambda **kw: writes.append(kw),
    ))

    def extract(self, messages, **kwargs):
        if messages[0]["id"] == "message-a":
            raise RuntimeError("owner A extraction failed")
        return [ExtractedFact(content="A different user's fact", importance=.9, tags=["pets"],
                              source_message_ids=["message-a"])]

    monkeypatch.setattr(MemoryExtractionService, "extract_from_conversation", extract)
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(DEFAULT_USER="not-an-owner")
    service._get_background_client = lambda **kw: (object(), "mock-model")
    try:
        with pytest.raises(ExtractionBatchFailed, match="owner A extraction failed"):
            service._process_memory_extraction(SimpleNamespace(bot_id=backend.bot_id, user_id="system", payload={}))
        with backend.engine.connect() as conn:
            processed = dict(conn.execute(select(backend.messages_table.c.id,
                                                 backend.messages_table.c.processed)).all())
        assert processed == {"message-a": False, "message-b": True}
        assert profiles == writes == []
    finally:
        backend.engine.dispose()
