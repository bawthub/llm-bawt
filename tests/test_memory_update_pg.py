"""Private PostgreSQL regression for MCP memory updates (TASK-1854)."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.memory import embeddings
from llm_bawt.memory.errors import MemoryUpdateConflictError, MemoryUpdateDuplicateError
from llm_bawt.mcp_server import storage as storage_module
from llm_bawt.mcp_server.storage import MemoryStorage
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
