"""Private PostgreSQL: extraction dedup and mutations retain source ownership."""
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text

from llm_bawt.memory.extraction.storage import ExtractionMemoryStore
from llm_bawt.profiles import EntityType, ProfileManager
from tests.private_postgres import private_app_pg  # noqa: F401
from tests.test_memory_update_pg import memory_rig  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def owned_memory(memory_rig, monkeypatch):  # noqa: F811 - imported pytest fixture
    _, backend, bot = memory_rig
    policy = SimpleNamespace(partitions={bot: backend.bot_id_sanitized},
                             readable_bots=frozenset([bot]), assert_current=lambda: None)
    monkeypatch.setattr("llm_bawt.memory.extraction.storage.BotPolicyResolver",
                        lambda engine: SimpleNamespace(resolve_sync=lambda: policy))
    # A session user string alone is not proof of a known non-service human.
    profiles = ProfileManager(backend.config)
    for owner in ("owner-a", "owner-b"):
        profiles.get_or_create_profile(EntityType.USER, owner)
    sources = {}
    with backend.engine.begin() as conn:
        for owner in ("owner-a", "owner-b"):
            session, message = str(uuid4()), str(uuid4())
            conn.execute(text("""
                INSERT INTO sessions (id, bot_id, user_id, started_at, status)
                VALUES (:id, :bot, :user, now(), 'active')
            """), {"id": session, "bot": bot, "user": owner})
            conn.execute(text(f"""
                INSERT INTO {backend._messages_table_name}
                    (id, role, content, timestamp, session_id, processed, author_entity_type, author_entity_id)
                VALUES (:id, 'user', 'A synthetic fact', 1, :session, false, 'user', :user)
            """), {"id": message, "session": session, "user": owner})
            sources[owner] = message
    return backend, ExtractionMemoryStore(backend, bot, "owner-a"), ExtractionMemoryStore(backend, bot, "owner-b"), sources


@pytest.mark.parametrize("vector", [True, False])
def test_same_fact_deduplicates_only_within_owner(owned_memory, monkeypatch, vector):
    backend, a, b, sources = owned_memory
    if not vector:
        monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **kw: None)
    manual = str(uuid4())
    backend.add_memory(manual, "Same fact", source_message_ids=[])
    first = a.add_memory(content="Same fact", source_message_ids=[sources["owner-a"]])
    second = b.add_memory(content="Same fact", source_message_ids=[sources["owner-b"]])
    assert len({manual, first, second}) == 3
    assert a.add_memory(content="Same fact", source_message_ids=[sources["owner-a"]]) == first
    assert [m["id"] for m in a.list_memories(limit=1)] == [first]
    assert [m["id"] for m in b.list_memories(limit=1)] == [second]
    assert a.update_memory(second, content="Wrong attribution", source_message_ids=[sources["owner-a"]]) is None
    assert a.delete_memory(second) is False
    assert a.delete_memory(manual) is False
    assert b.list_memories()[0]["content"] == "Same fact"
    assert a.update_memory(first, content="A corrected fact", source_message_ids=[sources["owner-a"]])["content"] == "A corrected fact"
    # An update is not swallowed by another owner's identical fact, either.
    assert a.update_memory(first, content="Same fact")["content"] == "Same fact"


def test_unproven_or_mixed_sources_cannot_be_inserted_or_replace_owned_sources(owned_memory):
    backend, a, b, sources = owned_memory
    for ids in ([], [str(uuid4())], [sources["owner-b"]], list(sources.values())):
        with pytest.raises(RuntimeError, match="sources do not belong"):
            a.add_memory(content="Unproven fact", source_message_ids=ids)
    first = a.add_memory(content="A fact", source_message_ids=[sources["owner-a"]])
    with pytest.raises(ValueError, match="sources do not belong"):
        a.update_memory(first, source_message_ids=[sources["owner-b"]])
    assert a.list_memories()[0]["source_message_ids"] == [sources["owner-a"]]
    assert b.list_memories() == []


@pytest.mark.parametrize("action", ["UPDATE", "DELETE"])
def test_cross_owner_model_action_is_skipped_without_retrying_batch(owned_memory, monkeypatch, action):
    from llm_bawt.memory.extraction.service import ExtractedFact, MemoryAction, MemoryExtractionService
    from llm_bawt.service.background_tasks import BackgroundTasksMixin

    backend, a, b, sources = owned_memory
    a.add_memory(content="Existing A fact", source_message_ids=[sources["owner-a"]])
    target = b.add_memory(content="Existing B fact", source_message_ids=[sources["owner-b"]])
    fact = ExtractedFact(content="A new fact", importance=.9, tags=["test"],
                         source_message_ids=[sources["owner-a"]])
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
    monkeypatch.setattr(MemoryExtractionService, "extract_from_conversation", lambda *a, **kw: [fact])
    monkeypatch.setattr(MemoryExtractionService, "determine_memory_actions",
                        lambda *a, **kw: [MemoryAction(action=action, fact=fact, target_memory_id=target)])
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(MEMORY_PROFILE_ATTRIBUTE_ENABLED=False)
    service._get_background_client = lambda **kw: (object(), "mock-model")
    result = service._process_memory_extraction(SimpleNamespace(bot_id=backend.bot_id, user_id="owner-a", payload={}))
    assert result["facts_stored"] == 0
    assert b.list_memories()[0]["content"] == "Existing B fact"
    assert backend.get_unprocessed_messages(user_id="owner-a") == []
    assert [m["id"] for m in backend.get_unprocessed_messages(user_id="owner-b")] == [sources["owner-b"]]


def test_source_ownership_change_during_embedding_refuses_update(owned_memory, monkeypatch):
    backend, a, b, sources = owned_memory
    memory = a.add_memory(content="Original", source_message_ids=[sources["owner-a"]])
    changed = False

    def embed(*args, **kwargs):
        nonlocal changed
        if not changed:
            with backend.engine.begin() as conn:
                conn.execute(text(f"""
                    UPDATE sessions SET user_id='owner-b', status='archived' WHERE id=(
                        SELECT session_id FROM {backend._messages_table_name} WHERE id=:id)
                """), {"id": sources["owner-a"]})
            changed = True
        return None

    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", embed)
    assert a.update_memory(memory, content="Must not overwrite") is None
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT content FROM {backend._memories_table_name} WHERE id=:id"),
                            {"id": memory}).scalar_one() == "Original"
