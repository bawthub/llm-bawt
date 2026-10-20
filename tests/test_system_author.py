"""System attribution reuses a typed bot identity, never a runnable bot."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_identities import seed_system_author
from llm_bawt.entity_presentation import EntityPresentationResolver
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.runtime_settings import BotProfileStore
from tests.test_entity_presentation import Engine


def test_seed_is_repeatable_and_keeps_user_identity_separate():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql("""CREATE TABLE entity_profiles (
            entity_type TEXT, entity_id TEXT, display_name TEXT, description TEXT,
            created_at TIMESTAMP, updated_at TIMESTAMP,
            UNIQUE(entity_type, entity_id))
        """)
        conn.exec_driver_sql("""INSERT INTO entity_profiles
            (entity_type, entity_id, display_name) VALUES ('USER', 'system', 'Existing User')
        """)
        seed_system_author(conn)
        seed_system_author(conn)
        assert conn.execute(text("""SELECT entity_type, entity_id, display_name
            FROM entity_profiles ORDER BY entity_type""")).all() == [
                ("BOT", "system", "System"), ("USER", "system", "Existing User"),
            ]
        conn.exec_driver_sql("UPDATE entity_profiles SET display_name='Automation' WHERE entity_type='BOT'")
        seed_system_author(conn)
        assert conn.execute(text("SELECT display_name FROM entity_profiles WHERE entity_type='BOT'")).scalar_one() == "Automation"
    engine.dispose()


def test_presentation_uses_typed_profile_without_runnable_bot():
    engine = Engine()
    engine.bot_rows = []
    engine.profile_rows = [
        {"entity_type": "BOT", "entity_id": "system", "display_name": "System"},
        {"entity_type": "USER", "entity_id": "system", "display_name": "Existing User"},
    ]
    resolved = EntityPresentationResolver(engine).resolve_many([
        AuthorReference.bot("system"), AuthorReference.user("system"),
    ])
    assert resolved[("bot", "system")].display_name == "System"
    assert resolved[("bot", "system")].status == "canonical"
    assert resolved[("user", "system")].display_name == "Existing User"


@pytest.mark.parametrize("slug", ["system", " System ", "*"])
def test_reserved_author_cannot_become_runnable_bot(slug):
    store = object.__new__(BotProfileStore)
    store.engine = object()  # Validation must fail before opening a connection.
    with pytest.raises(ValueError, match="reserved for system actions"):
        store.upsert({"slug": slug, "name": "No inference identity"})
