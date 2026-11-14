"""Private PostgreSQL regressions: consolidation never loses memories (TASK-2246).

Originals are deleted only after a verified surviving row holds the merge. A
deduplicated merge keeps the row it folded into, a failed write keeps every
original, and a concurrent edit keeps the edited original. Also covers the
memory-store receipt for the same silent-write contract (TASK-2238).
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from sqlalchemy import event, text

from llm_bawt.memory import embeddings
from llm_bawt.memory.consolidation import MemoryConsolidator
from llm_bawt.memory.errors import MemoryWriteOutcome
from llm_bawt.mcp_server import storage as storage_module
from llm_bawt.mcp_server.storage import MemoryStorage

pytestmark = pytest.mark.integration

ORIGINALS = (
    "User's favorite color is blue and they like the ocean",
    "User likes the color blue",
)
MERGED = "User's favorite color is blue; they also love the ocean"


def _vector(index: int) -> list[float]:
    vector = [0.0] * embeddings.EMBED_DIM
    vector[index] = 1.0
    return vector


def _literal(vector: list[float]) -> str:
    return f"[{','.join(str(x) for x in vector)}]"


@pytest.fixture
def rig(private_app_pg, monkeypatch):
    # Originals and anything not named share one direction (they cluster and
    # semantically dedup); MERGED gets its own direction so it inserts.
    state = {"available": True}

    def embed(content, *_args, **_kwargs):
        if not state["available"]:
            return None
        return _vector(1 if content == MERGED else 0)

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    monkeypatch.setattr(storage_module, "generate_embedding", embed)
    storage = MemoryStorage(private_app_pg)
    bot_id = f"consolidate-{uuid.uuid4().hex[:12]}"
    backend = storage._get_backend(bot_id, provision_schema=True)
    ids = []
    with backend.engine.begin() as conn:
        for index, content in enumerate(ORIGINALS):
            memory_id = str(uuid.uuid4())
            ids.append(memory_id)
            conn.execute(text(f"""
                INSERT INTO {backend._memories_table_name}
                (id, content, tags, importance, source_message_ids, embedding)
                VALUES (:id, :content, CAST(:tags AS jsonb), :importance, :src, :emb)
            """), {
                "id": memory_id, "content": content, "tags": json.dumps(["preference"]),
                "importance": 0.9 - index * 0.1, "src": json.dumps([f"m{index}"]),
                "emb": _literal(_vector(0)),
            })
    return storage, backend, bot_id, ids, state


def _rows(backend) -> dict[str, str]:
    with backend.engine.connect() as conn:
        return {
            str(r.id): r.content
            for r in conn.execute(text(f"SELECT id, content FROM {backend._memories_table_name}"))
        }


class _Llm:
    def __init__(self, reply: str):
        self.reply = reply

    def stream_raw(self, _messages):
        yield self.reply


def test_semantic_duplicate_merge_keeps_every_original(rig):
    _storage, backend, _bot, ids, _state = rig
    # Give the longer representative a slightly different stored direction.
    # Its generated [1, 0, ...] vector now deterministically dedups onto the
    # shorter/different text, instead of relying on an equal-vector SQL tie.
    vector = _vector(0)
    vector[1] = 0.01
    with backend.engine.begin() as conn:
        conn.execute(text(f"UPDATE {backend._memories_table_name} SET embedding=:vector WHERE id=:id"),
                     {"vector": _literal(vector), "id": ids[0]})

    result = MemoryConsolidator(backend).consolidate()

    # The merge folded into a row with different text, so no information may
    # be dropped: every original survives and an error is reported.
    assert _rows(backend) == dict(zip(ids, ORIGINALS))
    assert result.new_memories_created == 0
    assert result.memories_consolidated == 0
    assert any("merged content" in e for e in result.errors)


def test_exact_duplicate_merge_without_embeddings_keeps_the_original(rig):
    _storage, backend, _bot, ids, state = rig
    state["available"] = False  # forces add_memory's exact-content path

    result = MemoryConsolidator(backend).consolidate()

    assert _rows(backend) == {ids[0]: ORIGINALS[0]}
    assert result.new_memories_created == 0
    assert result.errors == []


def _metadata(backend):
    with backend.engine.connect() as conn:
        return {row["id"]: dict(row) for row in conn.execute(text(f"""
            SELECT id, tags, source_message_ids, importance, access_count, updated_at
            FROM {backend._memories_table_name}
        """)).mappings()}


def _metadata_cluster(rig):
    _storage, backend, _bot, ids, state = rig
    state["available"] = False
    with backend.engine.begin() as conn:
        for index, memory_id in enumerate(ids):
            conn.execute(text(f"""
                UPDATE {backend._memories_table_name}
                SET tags=CAST(:tags AS jsonb), importance=:importance, access_count=:count
                WHERE id=:id
            """), {"id": memory_id, "tags": json.dumps(["first" if index == 0 else "second"]),
                   "importance": 0.5 if index == 0 else 1.0, "count": 4 if index == 0 else 7})
    consolidator = MemoryConsolidator(backend)
    cluster = consolidator.find_clusters(consolidator.get_all_active_memories_with_embeddings())[0]
    return consolidator, cluster


@pytest.mark.parametrize("kind", ["original", "inserted", "external"])
def test_replacement_preserves_union_metadata_and_access_counts(rig, kind):
    _storage, backend, _bot, ids, _state = rig
    consolidator, cluster = _metadata_cluster(rig)
    content = ORIGINALS[0] if kind == "original" else MERGED
    external = None
    if kind == "external":
        external = str(uuid.uuid4())
        assert backend.add_memory(external, MERGED, tags=["external"],
                                  source_message_ids=["outside"], importance=0.3).persisted
        with backend.engine.begin() as conn:
            conn.execute(text(f"UPDATE {backend._memories_table_name} SET access_count=20 WHERE id=:id"),
                         {"id": external})
    outcome = consolidator.create_merged_memory(content, cluster)
    if external:
        assert outcome.memory_id == external
    deleted, kept = consolidator.replace_originals(outcome.memory_id, cluster, content)
    assert kept == [] and deleted == (1 if kind == "original" else 2)
    rows = _metadata(backend)
    assert set(rows) == {outcome.memory_id}
    survivor = rows[outcome.memory_id]
    assert set(survivor["tags"]) == {"first", "second"} | ({"external"} if external else set())
    assert set(survivor["source_message_ids"]) == {"m0", "m1"} | ({"outside"} if external else set())
    assert survivor["importance"] == 1.0
    assert survivor["access_count"] == (20 if external else 11)
    assert outcome.memory_id == ids[0] if kind == "original" else outcome.memory_id not in ids


def test_changed_original_survivor_preserves_every_row_and_current_metadata(rig):
    _storage, backend, _bot, ids, _state = rig
    consolidator, cluster = _metadata_cluster(rig)
    outcome = consolidator.create_merged_memory(ORIGINALS[0], cluster)
    with backend.engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {backend._memories_table_name}
            SET tags='["later"]'::jsonb, access_count=19, updated_at=updated_at+interval '1 second'
            WHERE id=:id
        """), {"id": outcome.memory_id})
    before = _metadata(backend)
    with pytest.raises(RuntimeError, match="original survivor changed"):
        consolidator.replace_originals(outcome.memory_id, cluster, ORIGINALS[0])
    assert _metadata(backend) == before and set(before) == set(ids)


def test_changed_other_original_is_kept_without_importing_its_metadata(rig):
    _storage, backend, _bot, ids, _state = rig
    consolidator, cluster = _metadata_cluster(rig)
    outcome = consolidator.create_merged_memory(ORIGINALS[0], cluster)
    with backend.engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {backend._memories_table_name}
            SET tags='["later"]'::jsonb, updated_at=updated_at+interval '1 second' WHERE id=:id
        """), {"id": ids[1]})
    assert consolidator.replace_originals(outcome.memory_id, cluster, ORIGINALS[0]) == (0, [ids[1]])
    survivor = _metadata(backend)[outcome.memory_id]
    assert survivor["tags"] == ["first"]
    assert survivor["source_message_ids"] == ["m0"]
    assert survivor["access_count"] == 4


@pytest.mark.parametrize("stage", ["metadata", "delete"])
def test_replacement_failure_rolls_back_metadata_and_keeps_originals(rig, stage):
    _storage, backend, _bot, ids, _state = rig
    consolidator, cluster = _metadata_cluster(rig)
    outcome = consolidator.create_merged_memory(ORIGINALS[0], cluster)
    before = _metadata(backend)

    def fail(_conn, _cursor, statement, _parameters, _context, _executemany):
        if ((stage == "metadata" and "SET tags=CAST" in statement)
                or (stage == "delete" and statement.lstrip().startswith("DELETE FROM"))):
            raise RuntimeError("synthetic replacement failure")

    event.listen(backend.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="synthetic replacement failure"):
            consolidator.replace_originals(outcome.memory_id, cluster, ORIGINALS[0])
    finally:
        event.remove(backend.engine, "before_cursor_execute", fail)
    assert _metadata(backend) == before and set(before) == set(ids)


def test_failed_write_keeps_every_original_and_reports_an_error(rig, monkeypatch):
    _storage, backend, _bot, ids, _state = rig

    def broken_transaction():
        raise RuntimeError("insert failed")

    monkeypatch.setattr(backend, "_memory_write_transaction", broken_transaction)

    result = MemoryConsolidator(backend).consolidate()

    assert _rows(backend) == dict(zip(ids, ORIGINALS))
    assert result.new_memories_created == 0
    assert result.clusters_merged == 0
    assert len(result.errors) == 1 and "not persisted" in result.errors[0]


def test_valid_llm_merge_replaces_originals_with_one_new_row(rig):
    _storage, backend, _bot, ids, _state = rig

    result = MemoryConsolidator(backend, llm_client=_Llm(MERGED)).consolidate()

    rows = _rows(backend)
    assert list(rows.values()) == [MERGED]
    assert not set(ids) & set(rows)
    assert result.new_memories_created == 1
    assert result.memories_consolidated == 2
    assert result.errors == []


def test_concurrently_edited_original_is_kept(rig, monkeypatch):
    _storage, backend, _bot, ids, _state = rig
    consolidator = MemoryConsolidator(backend, llm_client=_Llm(MERGED))
    write = consolidator.create_merged_memory

    def write_then_edit(content, cluster):
        outcome = write(content, cluster)
        with backend.engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE {backend._memories_table_name}
                SET content = 'edited meanwhile', updated_at = updated_at + interval '1 second'
                WHERE id = :id
            """), {"id": ids[1]})
        return outcome

    monkeypatch.setattr(consolidator, "create_merged_memory", write_then_edit)

    result = consolidator.consolidate()

    rows = _rows(backend)
    assert rows.get(ids[1]) == "edited meanwhile"
    assert ids[0] not in rows
    assert MERGED in rows.values()
    assert result.memories_consolidated == 1
    assert any(ids[1] in e for e in result.errors)


def test_missing_survivor_deletes_nothing(rig):
    _storage, backend, _bot, ids, _state = rig
    consolidator = MemoryConsolidator(backend)
    cluster = consolidator.find_clusters(consolidator.get_all_active_memories_with_embeddings())[0]

    with pytest.raises(RuntimeError, match="not present"):
        consolidator.replace_originals(str(uuid.uuid4()), cluster, MERGED)

    assert _rows(backend) == dict(zip(ids, ORIGINALS))


def test_store_memory_returns_existing_id_for_duplicate_and_raises_on_failure(rig, monkeypatch):
    storage, backend, bot_id, ids, _state = rig

    memory = asyncio.run(storage.store_memory(ORIGINALS[1] + " again", bot_id=bot_id))
    assert memory.id in ids

    monkeypatch.setattr(
        type(backend), "add_memory",
        lambda *_a, **_k: MemoryWriteOutcome("failed", error="boom"),
    )
    with pytest.raises(RuntimeError, match="not stored"):
        asyncio.run(storage.store_memory("brand new fact", bot_id=bot_id))


def test_survivor_edited_before_replacement_keeps_every_original(rig, monkeypatch):
    _storage, backend, _bot, ids, _state = rig
    consolidator = MemoryConsolidator(backend, llm_client=_Llm(MERGED))
    write = consolidator.create_merged_memory

    def write_then_edit_survivor(content, cluster):
        outcome = write(content, cluster)
        with backend.engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE {backend._memories_table_name}
                SET content = 'User now likes red', updated_at = updated_at + interval '1 second'
                WHERE id = :id
            """), {"id": str(outcome.memory_id)})
        return outcome

    monkeypatch.setattr(consolidator, "create_merged_memory", write_then_edit_survivor)

    result = consolidator.consolidate()

    rows = _rows(backend)
    assert {i: rows.get(i) for i in ids} == dict(zip(ids, ORIGINALS))
    assert result.memories_consolidated == 0
    assert any("merged content" in e for e in result.errors)
