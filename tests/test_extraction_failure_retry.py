"""TASK-2214: a failed extraction batch stays unprocessed, with bounded retry."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_bawt.memory.extraction.service import ExtractedFact, ExtractionFailed, MemoryExtractionService
from llm_bawt.service import background_tasks
from llm_bawt.service.background_tasks import (
    MAX_EXTRACTION_ATTEMPTS,
    BackgroundTasksMixin,
    ExtractionBatchFailed,
)

HUMAN = {"id": "h1", "role": "user", "content": "I adopted a third dog named Biscuit.",
         "author_entity_type": "user", "author_entity_id": "u"}
REPLY = {"id": "r1", "role": "assistant", "content": "Congrats on Biscuit!",
         "author_entity_type": "bot", "author_entity_id": "self"}
FACT = ExtractedFact(content="Nick has a dog named Biscuit", tags=["pets"],
                     importance=0.9, source_message_ids=["h1"])


class _Backend:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.processed: list[str] = []

    def get_unprocessed_messages(self, limit: int = 50) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["id"] not in self.processed]

    def mark_messages_processed(self, ids: list[str]) -> None:
        self.processed.extend(ids)


class _Memory:
    def __init__(self, fail_adds: int = 0) -> None:
        self.fail_adds = fail_adds
        self.added: list[str] = []

    def list_memories(self, **_kw: Any) -> list[dict[str, Any]]:
        return []

    def add_memory(self, content: str, **_kw: Any) -> None:
        if self.fail_adds:
            self.fail_adds -= 1
            raise RuntimeError("memory store unavailable")
        self.added.append(content)


@pytest.fixture(autouse=True)
def _fresh_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(background_tasks, "_extraction_attempts", background_tasks._ExtractionAttempts())


def _service(monkeypatch: pytest.MonkeyPatch, memory: _Memory) -> tuple[BackgroundTasksMixin, _Backend, Any]:
    backend = _Backend([HUMAN, REPLY])
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(MEMORY_PROFILE_ATTRIBUTE_ENABLED=False)  # type: ignore[attr-defined]
    service._default_bot = "test"  # type: ignore[attr-defined]
    service.get_memory_client = lambda *a: memory  # type: ignore[attr-defined]
    service._get_background_client = lambda **_kw: (object(), "mock-model")  # type: ignore[attr-defined]
    task = SimpleNamespace(bot_id="test", user_id="u", payload={})
    return service, backend, task


def _extract_script(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> list[bool]:
    """Each call pops an outcome: an exception to raise or a fact list. Records ``strict``."""
    seen: list[bool] = []

    def fake(self: Any, messages: list[dict[str, Any]], use_llm: bool = True, *, strict: bool = False) -> list[ExtractedFact]:
        seen.append(strict)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(MemoryExtractionService, "extract_from_conversation", fake)
    return seen


def test_failed_batch_stays_unprocessed_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    service, backend, task = _service(monkeypatch, memory)
    strict_seen = _extract_script(monkeypatch, [ExtractionFailed("LLM timeout"), [FACT]])

    with pytest.raises(ExtractionBatchFailed, match="attempt 1/3"):
        service._process_memory_extraction(task)
    assert backend.processed == []
    assert memory.added == []

    result = service._process_memory_extraction(task)
    assert result["facts_stored"] == 1
    assert backend.processed == ["h1", "r1"]
    assert memory.added == [FACT.content]
    assert strict_seen == [True, True]


def test_failed_memory_write_leaves_batch_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory(fail_adds=1)
    service, backend, task = _service(monkeypatch, memory)
    _extract_script(monkeypatch, [[FACT], [FACT]])

    with pytest.raises(ExtractionBatchFailed, match="memory write"):
        service._process_memory_extraction(task)
    assert backend.processed == []

    service._process_memory_extraction(task)
    assert backend.processed == ["h1", "r1"]
    assert memory.added == [FACT.content]


def test_poison_batch_is_closed_after_bounded_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real service, LLM times out every time: retried, then closed as a visible failure."""
    memory = _Memory()
    service, backend, task = _service(monkeypatch, memory)
    client = _Client(TimeoutError("synthetic timeout"))
    service._get_background_client = lambda **_kw: (client, "mock-model")  # type: ignore[attr-defined]

    for attempt in range(1, MAX_EXTRACTION_ATTEMPTS):
        with pytest.raises(ExtractionBatchFailed, match=f"attempt {attempt}/"):
            service._process_memory_extraction(task)
        assert backend.processed == []

    with pytest.raises(ExtractionBatchFailed, match="gave up.*heuristics used"):
        service._process_memory_extraction(task)
    assert backend.processed == ["h1", "r1"]


def test_final_attempt_that_recovers_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _Memory()
    service, backend, task = _service(monkeypatch, memory)
    reply = '{"facts": [{"content": "Nick has a dog named Biscuit", "tags": ["pets"], "importance": 0.9, "source_message_ids": ["h1"]}]}'
    replies: list[Any] = [TimeoutError("t1"), TimeoutError("t2"), reply]

    class _Flaky:
        def query(self, **_kw: Any) -> Any:
            r = replies.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r

    service._get_background_client = lambda **_kw: (_Flaky(), "mock-model")  # type: ignore[attr-defined]
    for _ in range(MAX_EXTRACTION_ATTEMPTS - 1):
        with pytest.raises(ExtractionBatchFailed):
            service._process_memory_extraction(task)

    result = service._process_memory_extraction(task)
    assert result["facts_stored"] == 1
    assert backend.processed == ["h1", "r1"]


def test_real_storage_write_failure_leaves_batch_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through MemoryStorage.store_memory and the PostgreSQL writer with a failed transaction.

    Relies on the storage contract from TASK-2246 (store_memory raises when
    the write isn't persisted).
    """
    import asyncio

    import llm_bawt.mcp_server.storage as storage_mod
    from llm_bawt.mcp_server.storage import MemoryStorage
    from llm_bawt.memory.postgresql_memories import PostgreSQLSemanticMemoryMixin

    class _Writer(PostgreSQLSemanticMemoryMixin):
        config = SimpleNamespace()
        embedding_model = "none"
        _memories_table_name = "memories_p_test"

        def ensure_schema(self) -> None:
            pass

        def _memory_write_transaction(self) -> Any:
            raise RuntimeError("synthetic database write outage")

    monkeypatch.setattr(storage_mod, "generate_embedding", lambda *a, **k: [0.1, 0.2])
    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **k: None)
    storage = object.__new__(MemoryStorage)
    storage.config = SimpleNamespace(MEMORY_EMBEDDING_MODEL="none", VERBOSE=False)  # type: ignore[assignment]
    writer = _Writer()
    monkeypatch.setattr(storage, "_get_backend", lambda *a, **k: writer, raising=False)

    def add_memory(content: str, tags: Any = None, importance: float = 0.5, source_message_ids: Any = None) -> str:
        return asyncio.run(storage.store_memory(content=content, bot_id="test", tags=tags,
                                                importance=importance, source_message_ids=source_message_ids)).id

    memory = _Memory()
    memory.add_memory = add_memory  # type: ignore[method-assign,assignment]
    service, backend, task = _service(monkeypatch, memory)
    _extract_script(monkeypatch, [[FACT]])

    with pytest.raises(ExtractionBatchFailed, match="memory write"):
        service._process_memory_extraction(task)
    assert backend.processed == []


def test_profile_write_failure_is_raised_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_bawt.mcp_server import extraction as profile_mod

    def boom() -> Any:
        raise RuntimeError("profile store down")

    monkeypatch.setattr("llm_bawt.mcp_server.server._get_profile_manager", boom)
    fact = ExtractedFact(content="Nick has a dog named Biscuit", tags=["pets"], importance=0.9,
                         source_message_ids=["h1"], profile_attribute={"category": "fact", "key": "pets"})
    config = SimpleNamespace(MEMORY_PROFILE_ATTRIBUTE_ENABLED=True)
    assert profile_mod.extract_profile_attributes_from_fact(fact, "u", config) is False  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="profile store down"):
        profile_mod.extract_profile_attributes_from_fact(fact, "u", config, raise_errors=True)  # type: ignore[arg-type]


def test_failure_marks_run_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_bawt.service.tasks import TaskStatus, TaskType

    service, backend, _ = _service(monkeypatch, _Memory())
    _extract_script(monkeypatch, [ExtractionFailed("LLM timeout")])
    task = SimpleNamespace(bot_id="test", user_id="u", payload={}, task_id="t-2214",
                           task_type=TaskType.MEMORY_EXTRACTION)

    result = service._run_task(task)  # type: ignore[arg-type]

    assert result.status is TaskStatus.FAILED
    assert "left unprocessed" in (result.error or "")
    assert backend.processed == []


class _Client:
    def __init__(self, reply: Any) -> None:
        self.reply = reply

    def query(self, **_kw: Any) -> Any:
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


@pytest.mark.parametrize("reply", [TimeoutError("timed out"), "no json here", "{not: valid json,,"],
                         ids=["timeout", "no-json", "bad-json"])
def test_strict_extraction_raises_instead_of_degrading(reply: Any) -> None:
    service = MemoryExtractionService(llm_client=_Client(reply))
    with pytest.raises(ExtractionFailed):
        service.extract_from_conversation([HUMAN, REPLY], strict=True)
    # The default stays lenient for interactive callers.
    assert isinstance(service.extract_from_conversation([HUMAN, REPLY]), list)
