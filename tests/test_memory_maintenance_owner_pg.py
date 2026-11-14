"""Synthetic ownership boundaries for meaning, maintenance and consolidation."""
from uuid import uuid4

import pytest
from sqlalchemy import text

from llm_bawt.memory.consolidation import MemoryConsolidator
from llm_bawt.memory.maintenance import MemoryMaintenance, update_memory_meaning
from llm_bawt.memory.ownership import MemoryAuthority, OPERATOR_MEMORY_AUTHORITY
from tests.private_postgres import private_app_pg  # noqa: F401
from tests.test_memory_update_pg import memory_rig  # noqa: F401
from tests.test_memory_crud_owner_pg import owned_clients as _owned_clients

owned_clients = _owned_clients
pytestmark = pytest.mark.integration


def _seed(backend, owner, content, *, sources=None):
    """Seed duplicates deliberately; normal add would correctly deduplicate."""
    from llm_bawt.memory.embeddings import EMBED_DIM
    memory_id = str(uuid4())
    import json
    with backend.engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {backend._memories_table_name}
                (id, content, owner_user_id, embedding, tags, importance,
                 source_message_ids, recurrence_keywords, created_at)
            VALUES (:id, :content, :owner, :embedding, '["fact"]', 0.01,
                    :sources, '["daily"]', now()-interval '10 years')
        """), {"id": memory_id, "content": content, "owner": owner,
               "embedding": str([1.0] + [0.0] * (EMBED_DIM - 1)),
               "sources": json.dumps(sources or [])})
    return memory_id


def _rows(backend):
    with backend.engine.connect() as conn:
        return {r["id"]: dict(r) for r in conn.execute(text(f"""
            SELECT id, content, owner_user_id, intent, tags, source_message_ids
            FROM {backend._memories_table_name}
        """)).mappings()}


def test_meaning_requires_owner_even_for_tags_only(owned_clients):
    backend, a, b, _sources = owned_clients
    target = b.store_memory("B's meaning metadata")
    assert not a.update_memory_meaning(target.id, intent="foreign inference")
    assert not a.update_memory_meaning(target.id, updated_tags=["foreign"])
    assert not update_memory_meaning(backend, target.id, intent="anonymous")
    assert b.update_memory_meaning(target.id, intent="own inference", updated_tags=["own"])
    row = _rows(backend)[target.id]
    assert row["intent"] == "own inference"
    assert row["tags"] == ["own"]
    assert row["owner_user_id"] == "owner-b"


def test_meaning_owner_change_during_embedding_refuses_stale_write(owned_clients, monkeypatch):
    backend, a, _b, _sources = owned_clients
    target = a.store_memory("Meaning race")

    def transfer(*_args, **_kwargs):
        with backend.engine.begin() as conn:
            conn.execute(text(f"UPDATE {backend._memories_table_name} SET owner_user_id='owner-b' WHERE id=:id"),
                         {"id": target.id})
        return None

    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", transfer)
    assert not a.update_memory_meaning(target.id, intent="stale owner's intent")
    assert _rows(backend)[target.id]["intent"] is None


def test_personal_maintenance_never_prunes_foreign_or_ambiguous_rows(owned_clients):
    backend, a, _b, _sources = owned_clients
    own = _seed(backend, "owner-a", "own stale row")
    foreign = _seed(backend, "owner-b", "foreign stale row")
    ambiguous = _seed(backend, None, "legacy stale row")
    denied = MemoryMaintenance(backend).run(run_consolidation=False, run_decay_pruning=True)
    assert denied.decay_memories_archived == 0
    result = a.run_maintenance(run_consolidation=False, run_decay_pruning=True)
    assert result["errors"] == []
    assert result["decay_memories_archived"] == 1
    assert own not in _rows(backend)
    assert set(_rows(backend)) == {foreign, ambiguous}


def test_operator_recurrence_groups_owners_and_null(owned_clients):
    backend, _a, _b, _sources = owned_clients
    for owner in ("owner-a", "owner-b", None):
        _seed(backend, owner, f"{owner} repeats daily")
    maint = MemoryMaintenance(backend, authority=OPERATOR_MEMORY_AUTHORITY)
    result = maint.run(run_consolidation=False)
    assert result.recurrence_themes_detected == 0
    assert all("recurring" not in r["tags"] for r in _rows(backend).values())
    _seed(backend, "owner-a", "second daily A")
    _seed(backend, "owner-a", "third daily A")
    result = maint.run(run_consolidation=False)
    assert result.recurrence_themes_detected == 1
    assert result.recurrence_memories_tagged == 3
    assert all(("recurring" in r["tags"]) == (r["owner_user_id"] == "owner-a")
               for r in _rows(backend).values())


def test_consolidation_groups_owners_and_never_claims_null(owned_clients, monkeypatch):
    backend, a, _b, sources = owned_clients
    # The fixture intentionally has two similar memories in each owner group.
    for owner in ("owner-a", "owner-b", None):
        for index in range(2):
            _seed(backend, owner, f"{owner} fact {index}",
                  sources=[sources["owner-a"]] if owner is None else [])
    personal = MemoryConsolidator(backend, authority=MemoryAuthority(user_id="owner-a"))
    eligible = personal.get_all_active_memories_with_embeddings()
    assert {r["owner_user_id"] for r in eligible} == {"owner-a", None}
    assert len(personal.find_clusters(eligible)) == 2
    assert MemoryConsolidator(backend).get_all_active_memories_with_embeddings() == []

    # Force exact dedup so heuristic survivor text can be verified.
    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **k: None)
    monkeypatch.setattr("llm_bawt.memory.consolidation.get_local_llm_client", lambda _config: None)
    result = a.consolidate_memories(dry_run=False)
    assert result["errors"] == []
    assert result["memories_consolidated"] == 2
    rows = list(_rows(backend).values())
    assert [r["owner_user_id"] for r in rows].count(None) == 1
    assert [r["owner_user_id"] for r in rows].count("owner-a") == 1
    assert [r["owner_user_id"] for r in rows].count("owner-b") == 2

    operator = MemoryConsolidator(backend, authority=OPERATOR_MEMORY_AUTHORITY)
    result = operator.consolidate()
    assert result.errors == []
    assert result.memories_consolidated == 1
    assert len(_rows(backend)) == 3


@pytest.mark.parametrize("changed", ["survivor", "original"])
def test_replacement_rechecks_owner_even_without_timestamp_change(owned_clients, monkeypatch, changed):
    backend, _a, _b, _sources = owned_clients
    originals = [_seed(backend, "owner-a", f"A fact {i}") for i in range(2)]
    c = MemoryConsolidator(backend, authority=MemoryAuthority(user_id="owner-a"))
    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **k: None)
    cluster = c.find_clusters(c.get_all_active_memories_with_embeddings())[0]
    outcome = c.create_merged_memory("new combined memory", cluster)
    assert outcome.status == "inserted"
    target = outcome.memory_id if changed == "survivor" else originals[0]
    with backend.engine.begin() as conn:
        conn.execute(text(f"UPDATE {backend._memories_table_name} SET owner_user_id='owner-b' WHERE id=:id"),
                     {"id": target})
    if changed == "survivor":
        with pytest.raises(RuntimeError, match="not present"):
            c.replace_originals(outcome.memory_id, cluster, "new combined memory")
        assert set(originals) <= set(_rows(backend))
    else:
        deleted, kept = c.replace_originals(outcome.memory_id, cluster, "new combined memory")
        assert deleted == 1
        assert kept == [target]
        assert target in _rows(backend)


def test_operator_merge_insert_retains_explicit_owner(owned_clients, monkeypatch):
    backend, _a, _b, _sources = owned_clients
    for owner in ("owner-a", "owner-b", None):
        for i in range(2):
            _seed(backend, owner, f"{owner} original {i}")
    c = MemoryConsolidator(backend, authority=OPERATOR_MEMORY_AUTHORITY)
    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **k: None)
    clusters = c.find_clusters(c.get_all_active_memories_with_embeddings())
    for cluster in clusters:
        # Same merged text must not dedup onto another owner (including NULL).
        outcome = c.create_merged_memory("identical merge", cluster)
        assert outcome.status == "inserted"
        assert _rows(backend)[outcome.memory_id]["owner_user_id"] == cluster.memories[0]["owner_user_id"]
        assert c.replace_originals(outcome.memory_id, cluster, "identical merge") == (2, [])
    assert len(_rows(backend)) == 3


def test_recall_source_preserves_provenance_and_excludes_foreign_explicit_owner(owned_clients):
    import asyncio
    from llm_bawt.recall.contracts import RecallScope
    from llm_bawt.recall.postgres import MemorySource, RecallDatabase

    backend, _a, _b, sources = owned_clients
    foreign = _seed(backend, "owner-b", "synthetic recall fact", sources=[sources["owner-a"]])
    legacy = _seed(backend, None, "synthetic recall fact", sources=[sources["owner-a"]])
    own = _seed(backend, "owner-a", "synthetic recall fact", sources=[sources["owner-a"]])
    _seed(backend, "owner-a", "synthetic recall manual fact")
    source = MemorySource(RecallDatabase(backend.engine))
    scope = RecallScope(backend.bot_id, "owner-a", None, frozenset(), frozenset({backend.bot_id}))
    rows = asyncio.run(source.search("synthetic recall", scope, ()))
    assert {r.id for r in rows} == {legacy, own}
    assert foreign not in {r.id for r in rows}


def test_changed_original_during_merge_embedding_refuses_insert(owned_clients, monkeypatch):
    backend, _a, _b, _sources = owned_clients
    originals = [_seed(backend, "owner-a", f"Original {i}") for i in range(2)]
    c = MemoryConsolidator(backend, authority=MemoryAuthority(user_id="owner-a"))
    cluster = c.find_clusters(c.get_all_active_memories_with_embeddings())[0]

    def transfer(*_args, **_kwargs):
        with backend.engine.begin() as conn:
            conn.execute(text(f"UPDATE {backend._memories_table_name} SET owner_user_id='owner-b' WHERE id=:id"),
                         {"id": originals[0]})
        return None

    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", transfer)
    with pytest.raises(RuntimeError, match="not persisted"):
        c.create_merged_memory("not authorized to merge", cluster)
    assert set(_rows(backend)) == set(originals)


def test_original_survivor_keeps_core_importance_and_forget_provenance(owned_clients, monkeypatch):
    backend, _a, _b, sources = owned_clients
    second_source = str(uuid4())
    with backend.engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {backend._messages_table_name}
                (id, role, content, timestamp, session_id, author_entity_type, author_entity_id)
            SELECT :id, role, 'Second owned source', timestamp+1, session_id,
                   author_entity_type, author_entity_id
            FROM {backend._messages_table_name} WHERE id=:source
        """), {"id": second_source, "source": sources["owner-a"]})
    survivor = _seed(backend, "owner-a", "Detailed preference " * 8, sources=[sources["owner-a"]])
    other = _seed(backend, "owner-a", "Name is Alex", sources=[second_source])
    with backend.engine.begin() as conn:
        for memory_id, tag, importance, count in (
            (survivor, "preference", 0.5, 4), (other, "identity", 1.0, 7),
        ):
            conn.execute(text(f"""
                UPDATE {backend._memories_table_name}
                SET tags=jsonb_build_array(CAST(:tag AS text)), importance=:importance, access_count=:count
                WHERE id=:id
            """), {"id": memory_id, "tag": tag, "importance": importance, "count": count})
    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **k: None)
    authority = MemoryAuthority(user_id="owner-a")
    result = MemoryConsolidator(backend, authority=authority).consolidate()
    assert result.errors == [] and result.memories_consolidated == 1
    with backend.engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT id, importance, tags, source_message_ids, access_count
            FROM {backend._memories_table_name}
        """)).mappings().one()
    assert row["id"] == survivor and row["importance"] == 1.0
    assert set(row["tags"]) == {"preference", "identity"}
    assert set(row["source_message_ids"]) == {sources["owner-a"], second_source}
    assert row["access_count"] == 11
    assert backend.delete_memories_by_source_message_ids([second_source], authority=authority) == 1
    assert _rows(backend) == {}
