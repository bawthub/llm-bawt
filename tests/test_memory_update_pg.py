"""Private PostgreSQL regression for MCP memory updates (TASK-1854)."""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event, local

import pytest
from sqlalchemy import event, text

from llm_bawt.memory import embeddings
from llm_bawt.memory.errors import MemoryUpdateConflictError, MemoryUpdateDuplicateError
from llm_bawt.mcp_server import storage as storage_module
from llm_bawt.mcp_server.storage import MemoryStorage
from tests.pg_lock_wait import wait_blocked_by
from tests.private_postgres import private_app_pg

pytestmark = pytest.mark.integration


@pytest.fixture
def memory_rig(private_app_pg, monkeypatch):
    vectors: dict[str, int] = {}

    def embed(content, *_args, **_kwargs):
        index = vectors.setdefault(content, len(vectors))
        vector = [0.0] * embeddings.EMBED_DIM
        vector[index] = 1.0
        return vector

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    monkeypatch.setattr(storage_module, "generate_embedding", embed)
    storage = MemoryStorage(private_app_pg)
    bot_id = f"memory-update-{uuid.uuid4().hex[:12]}"
    backend = storage._get_backend(bot_id, provision_schema=True)
    return storage, backend, bot_id


def row(backend, memory_id):
    with backend.engine.connect() as conn:
        return conn.execute(text(f"""
            SELECT content, tags, importance, source_message_ids, intent, stakes,
                   emotional_charge, recurrence_keywords, embedding,
                   meaning_embedding, meaning_updated_at, created_at
            FROM {backend._memories_table_name} WHERE id=:id
        """), {"id": memory_id}).mappings().one()


def test_same_content_update_persists_sources_and_preserves_other_fields(memory_rig):
    storage, backend, bot_id = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "A stable fact", tags=["existing"], importance=0.3,
                       source_message_ids=["original"], intent="remember",
                       stakes="important", emotional_charge=0.7,
                       recurrence_keywords=["daily"])
    before = row(backend, memory_id)

    result = asyncio.run(storage.update_memory(
        memory_id, bot_id=bot_id, content="A stable fact", importance=0.8,
        source_message_ids=["new-human"],
    ))

    after = row(backend, memory_id)
    assert result is not None
    assert result.source_message_ids == ["new-human"]
    assert result.importance == after["importance"] == 0.8
    assert after["source_message_ids"] == ["new-human"]
    assert after["tags"] == ["existing"]
    assert after["intent"] == "remember"
    assert after["stakes"] == "important"
    assert after["emotional_charge"] == 0.7
    assert after["recurrence_keywords"] == ["daily"]
    assert after["embedding"] == before["embedding"]
    assert after["meaning_embedding"] == before["meaning_embedding"]
    assert after["meaning_updated_at"] == before["meaning_updated_at"]
    assert after["created_at"] == before["created_at"]

    normalized = asyncio.run(storage.update_memory(
        memory_id, bot_id=bot_id, tags=["  Priority ", "  "],
    ))
    assert normalized is not None and normalized.tags == ["priority"]
    assert row(backend, memory_id)["source_message_ids"] == ["new-human"]
    cleared = asyncio.run(storage.update_memory(
        memory_id, bot_id=bot_id, source_message_ids=[],
    ))
    assert cleared is not None and cleared.source_message_ids == []
    assert row(backend, memory_id)["source_message_ids"] == []
    assert row(backend, memory_id)["intent"] == "remember"


def test_changed_content_updates_but_other_memory_duplicate_is_refused(memory_rig):
    storage, backend, bot_id = memory_rig
    memory_id, other_id = str(uuid.uuid4()), str(uuid.uuid4())
    backend.add_memory(memory_id, "First fact", source_message_ids=["source"])
    backend.add_memory(other_id, "Other fact", source_message_ids=["other-source"])

    changed = asyncio.run(storage.update_memory(
        memory_id, bot_id=bot_id, content="Updated fact",
    ))
    assert changed is not None and changed.content == "Updated fact"
    assert row(backend, memory_id)["source_message_ids"] == ["source"]
    assert row(backend, memory_id)["embedding"] is not None
    assert row(backend, memory_id)["meaning_embedding"] is not None

    with pytest.raises(MemoryUpdateDuplicateError, match="duplicate"):
        asyncio.run(storage.update_memory(
            memory_id, bot_id=bot_id, content="Other fact",
        ))
    assert row(backend, memory_id)["content"] == "Updated fact"
    assert row(backend, other_id)["content"] == "Other fact"


def test_case_only_change_does_not_match_its_own_text(memory_rig):
    storage, backend, bot_id = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "A stable fact")

    updated = asyncio.run(storage.update_memory(
        memory_id, bot_id=bot_id, content="a stable fact",
    ))
    assert updated is not None and updated.content == "a stable fact"
    assert row(backend, memory_id)["content"] == "a stable fact"


def test_near_identical_update_does_not_match_its_own_vector(memory_rig, monkeypatch):
    storage, backend, bot_id = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "A stable fact")
    original_embed = embeddings.generate_embedding

    def similar_embed(content, *args, **kwargs):
        return original_embed("A stable fact" if content == "A stable true fact" else content,
                              *args, **kwargs)

    monkeypatch.setattr(embeddings, "generate_embedding", similar_embed)
    updated = asyncio.run(storage.update_memory(
        memory_id, bot_id=bot_id, content="A stable true fact",
    ))
    assert updated is not None and updated.content == "A stable true fact"
    assert row(backend, memory_id)["content"] == "A stable true fact"


def test_competing_vector_rejects_update_even_when_text_differs(memory_rig, monkeypatch):
    storage, backend, bot_id = memory_rig
    memory_id, competitor_id = str(uuid.uuid4()), str(uuid.uuid4())
    backend.add_memory(memory_id, "A stable fact")
    backend.add_memory(competitor_id, "Competitor statement")
    original_embed = embeddings.generate_embedding

    def competitor_embed(content, *args, **kwargs):
        return original_embed("Competitor statement" if content == "Rephrased competitor" else content,
                              *args, **kwargs)

    monkeypatch.setattr(embeddings, "generate_embedding", competitor_embed)
    with pytest.raises(MemoryUpdateDuplicateError, match="duplicate"):
        asyncio.run(storage.update_memory(
            memory_id, bot_id=bot_id, content="Rephrased competitor",
        ))
    assert row(backend, memory_id)["content"] == "A stable fact"


def test_unknown_id_returns_none_and_backend_error_is_not_hidden(memory_rig, monkeypatch):
    storage, backend, bot_id = memory_rig
    assert asyncio.run(storage.update_memory(str(uuid.uuid4()), bot_id=bot_id,
                                             content="Unknown")) is None

    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "Existing")

    def fail(_memory_id, **_kwargs):
        raise RuntimeError("database update failed")

    monkeypatch.setattr(backend, "update_memory", fail, raising=False)
    with pytest.raises(RuntimeError, match="database update failed"):
        asyncio.run(storage.update_memory(memory_id, bot_id=bot_id,
                                          content="Different"))
    assert row(backend, memory_id)["content"] == "Existing"


def test_changed_content_during_embed_outage_clears_stale_vectors(memory_rig, monkeypatch):
    storage, backend, bot_id = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "Original fact", intent="remember")
    before = row(backend, memory_id)
    assert before["embedding"] is not None
    assert before["meaning_embedding"] is not None

    monkeypatch.setattr(embeddings, "generate_embedding", lambda *_a, **_k: None)
    monkeypatch.setattr(storage_module, "generate_embedding", lambda *_a, **_k: None)
    changed = asyncio.run(storage.update_memory(
        memory_id, bot_id=bot_id, content="Fact changed while embeddings are down",
    ))
    after = row(backend, memory_id)
    assert changed is not None and changed.content == after["content"]
    assert after["intent"] == "remember"
    assert after["embedding"] is None
    assert after["meaning_embedding"] is None
    assert after["meaning_updated_at"] is None


def test_update_rejects_exact_duplicate_with_other_row_missing_vector(memory_rig, monkeypatch):
    storage, backend, bot_id = memory_rig
    memory_id, other_id = str(uuid.uuid4()), str(uuid.uuid4())
    backend.add_memory(memory_id, "Original fact")
    embed = embeddings.generate_embedding
    monkeypatch.setattr(embeddings, "generate_embedding", lambda *_a, **_k: None)
    backend.add_memory(other_id, "Fact from an outage")
    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    assert row(backend, other_id)["embedding"] is None

    with pytest.raises(MemoryUpdateDuplicateError, match="duplicate"):
        asyncio.run(storage.update_memory(
            memory_id, bot_id=bot_id, content="Fact from an outage",
        ))
    assert row(backend, memory_id)["content"] == "Original fact"


def test_concurrent_change_refuses_stale_update(memory_rig, monkeypatch):
    storage, backend, bot_id = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "Original fact", tags=["initial"])
    embed = embeddings.generate_embedding

    def change_after_read(content, *args, **kwargs):
        if content == "Attempted overwrite":
            with backend.engine.begin() as conn:
                conn.execute(text(f"""
                    UPDATE {backend._memories_table_name}
                    SET tags='["newer"]'::jsonb,
                        updated_at=updated_at + interval '1 second'
                    WHERE id=:id
                """), {"id": memory_id})
        return embed(content, *args, **kwargs)

    monkeypatch.setattr(embeddings, "generate_embedding", change_after_read)
    with pytest.raises(MemoryUpdateConflictError, match="changed concurrently"):
        asyncio.run(storage.update_memory(
            memory_id, bot_id=bot_id, content="Attempted overwrite",
        ))
    assert row(backend, memory_id)["content"] == "Original fact"
    assert row(backend, memory_id)["tags"] == ["newer"]


@pytest.mark.parametrize("writers", [("add", "add"), ("update", "update"),
                                    ("add", "update"), ("update", "add")])
@pytest.mark.parametrize("vectors", [False, True], ids=["exact", "semantic"])
def test_concurrent_duplicate_writers_serialize(memory_rig, monkeypatch, writers, vectors):
    """Pause writer one after dedup, then force writer two into that window."""
    _storage, backend, _bot_id = memory_rig
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    contents = ["Shared fact", "Paraphrased fact" if vectors else "SHARED FACT"]
    for index, operation in enumerate(writers):
        if operation == "update":
            backend.add_memory(ids[index], f"Original fact {index}")
    vector = None
    # Seed rows must differ from the shared vector.
    if vectors:
        vector = [0.0] * (embeddings.EMBED_DIM - 1) + [1.0]
    monkeypatch.setattr(embeddings, "generate_embedding", lambda *_a, **_k: vector)
    state = local()
    first_at_write, release_first = Event(), Event()
    holder = {}

    def pause_first(conn, _cursor, statement, _params, _context, _many):
        sql = " ".join(statement.split())
        is_write = (sql.startswith(f"INSERT INTO {backend._memories_table_name} ")
                    or sql.startswith(f"UPDATE {backend._memories_table_name} "))
        if is_write and getattr(state, "index", None) == 0:
            holder["pid"] = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
            first_at_write.set()
            assert release_first.wait(15), "first writer was not released"

    def write(index):
        state.index = index
        try:
            if writers[index] == "add":
                backend.add_memory(ids[index], contents[index])
            else:
                backend.update_memory(ids[index], content=contents[index])
            return "written"
        except MemoryUpdateDuplicateError:
            return "duplicate"

    event.listen(backend.engine, "before_cursor_execute", pause_first)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(write, 0)
            try:
                assert first_at_write.wait(15), "first writer did not reach its write"
                second = pool.submit(write, 1)
                wait_error = None
                try:
                    wait_blocked_by(backend.engine, holder["pid"], second)
                except pytest.fail.Exception as exc:
                    # On unfixed code writer two commits before writer one:
                    # let both finish so the row-count assertion proves the race.
                    wait_error = exc
                    if not second.done():
                        raise
            finally:
                release_first.set()
            assert first.result(timeout=15) == "written"
            second_result = second.result(timeout=15)
        with backend.engine.connect() as conn:
            persisted = conn.execute(text(f"""
                SELECT id FROM {backend._memories_table_name}
                WHERE content IN (:first, :second)
            """), {"first": contents[0], "second": contents[1]}).scalars().all()
        assert persisted == [ids[0]], f"concurrent duplicate rows persisted: {persisted}"
        assert wait_error is None, str(wait_error)
        assert second_result == ("duplicate" if writers[1] == "update" else "written")
    finally:
        release_first.set()
        event.remove(backend.engine, "before_cursor_execute", pause_first)


@pytest.mark.parametrize("operation", ["add", "update"])
def test_memory_embeddings_run_outside_write_lock(memory_rig, monkeypatch, operation):
    _storage, backend, _bot_id = memory_rig
    memory_id = str(uuid.uuid4())
    if operation == "update":
        backend.add_memory(memory_id, "Original fact")
    probes = []
    embed = embeddings.generate_embedding

    def probe_lock(content, *args, **kwargs):
        # Another connection must be able to take the writer's lock during
        # both content and meaning embedding calls.
        with backend.engine.begin() as conn:
            probes.append(conn.execute(text("""
                SELECT pg_try_advisory_xact_lock(
                    hashtext('memory-dedup'), hashtext(:partition))
            """), {"partition": backend._memories_table_name}).scalar_one())
        return embed(content, *args, **kwargs)

    monkeypatch.setattr(embeddings, "generate_embedding", probe_lock)
    if operation == "add":
        backend.add_memory(memory_id, "New fact")
    else:
        backend.update_memory(memory_id, content="New fact")
    assert probes == [True, True]
    assert row(backend, memory_id)["content"] == "New fact"


def test_memory_write_lock_is_partition_scoped_and_transaction_released(memory_rig):
    storage, backend, _bot_id = memory_rig
    other = storage._get_backend(f"memory-other-{uuid.uuid4().hex[:12]}",
                                 provision_schema=True)
    other_id = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="rollback"):
            with backend._memory_write_transaction() as conn:
                # A different bot must write even while this partition is locked.
                pool.submit(other.add_memory, other_id, "Other bot fact").result(timeout=15)
                assert row(other, other_id)["content"] == "Other bot fact"
                conn.execute(text(f"""
                    INSERT INTO {backend._memories_table_name} (id, content)
                    VALUES (:id, 'Rolled back fact')
                """), {"id": str(uuid.uuid4())})
                raise RuntimeError("rollback")
        # Failure must release the advisory lock, not leak it into the pool.
        same_id = str(uuid.uuid4())
        pool.submit(backend.add_memory, same_id, "After rollback").result(timeout=15)
    assert row(backend, same_id)["content"] == "After rollback"
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"""
            SELECT count(*) FROM {backend._memories_table_name}
            WHERE content = 'Rolled back fact'
        """)).scalar_one() == 0
