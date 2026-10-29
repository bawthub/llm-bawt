"""TASK-1727: bot-only windows/sessions skip extraction with a recorded reason."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine, insert
from sqlalchemy.pool import StaticPool

from llm_bawt.memory.summary_extraction_policy import summary_has_human_source
from llm_bawt.service.background_tasks import BackgroundTasksMixin

HUMAN = {"id": "h", "role": "user", "content": "I adopted a third dog.",
         "author_entity_type": "user", "author_entity_id": "u"}
DELIVERY = {"id": "d", "role": "user", "content": "status update",
            "author_entity_type": "bot", "author_entity_id": "b"}
REPLY = {"id": "r", "role": "assistant", "content": "ack",
         "author_entity_type": "bot", "author_entity_id": "self"}


class _Backend:
    def __init__(self, rows):
        self.rows, self.processed = rows, []

    def get_unprocessed_messages(self, limit=50):
        return list(self.rows)

    def mark_messages_processed(self, ids):
        self.processed.extend(ids)


def _service(monkeypatch, rows):
    backend = _Backend(rows)
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend",
                        lambda *a, **kw: backend)
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace()
    service._default_bot = "test"
    service.get_memory_client = lambda *a: SimpleNamespace(list_memories=lambda **k: [])
    service._get_background_client = Mock(return_value=(None, None))
    task = SimpleNamespace(bot_id="test", user_id="u", payload={})
    return service, backend, task


def test_bot_only_window_skips_without_client_and_marks_processed(monkeypatch):
    service, backend, task = _service(monkeypatch, [DELIVERY, REPLY])
    result = service._process_memory_extraction(task)
    assert result["skipped"] is True
    assert result["reason"] == "No human-authored messages in window"
    assert backend.processed == ["d", "r"]
    service._get_background_client.assert_not_called()


def test_mixed_window_still_runs_extraction(monkeypatch):
    service, backend, task = _service(monkeypatch, [DELIVERY, HUMAN, REPLY])
    result = service._process_memory_extraction(task)
    assert not result.get("skipped")
    service._get_background_client.assert_called_once()
    assert set(backend.processed) == {"d", "h", "r"}


def _summary_backend(source_rows):
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    meta = MetaData()
    table = Table("messages", meta, Column("id", String, primary_key=True),
                  Column("role", String), Column("author_entity_type", String),
                  Column("author_entity_id", String), Column("summary_metadata", JSON))
    meta.create_all(engine)
    with engine.begin() as conn:
        conn.execute(insert(table), [
            {"id": "summary", "role": "summary",
             "summary_metadata": {"message_ids": [r["id"] for r in source_rows]}}])
        conn.execute(insert(table), [
            {k: r[k] for k in ("id", "role", "author_entity_type", "author_entity_id")}
            for r in source_rows])
    return SimpleNamespace(engine=engine, _messages_table_name="messages")


def test_summary_of_bot_only_session_has_no_human_source():
    assert not summary_has_human_source(_summary_backend([DELIVERY, REPLY]), "summary")
    assert summary_has_human_source(_summary_backend([DELIVERY, HUMAN, REPLY]), "summary")
    assert not summary_has_human_source(_summary_backend([HUMAN]), "missing")


def test_summary_extraction_skips_bot_only_session(monkeypatch):
    from llm_bawt.memory.extraction.service import MemoryExtractionService

    backend = _summary_backend([DELIVERY, REPLY])
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend",
                        lambda *a, **kw: backend)
    monkeypatch.setattr("llm_bawt.memory.summary_extraction_policy.summary_allows_extraction",
                        lambda *a: True)
    extract = Mock(side_effect=AssertionError("bot-only summary reached extraction"))
    monkeypatch.setattr(MemoryExtractionService, "extract_from_summary", extract)
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace()
    service._get_background_client = lambda **_kw: (object(), "mock")
    service.get_memory_client = lambda *a: object()
    service._mark_summary_extracted = Mock()
    result = service._extract_from_summaries(
        [{"created": True, "summary_id": "summary", "summary_text": "bots chatted"}], "test", "u")
    assert result["summaries_skipped_no_human"] == 1
    assert result["summaries_processed"] == 0
    extract.assert_not_called()
    service._mark_summary_extracted.assert_called_once_with("test", "summary")
