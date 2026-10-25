"""Tag rename and tag instructions against an isolated PostgreSQL DB (TASK-1249).

A role tag's instructions are a versioned global prompt-registry row reached
through the tag's opaque ``prompt_key``. The key is allocated on the first save
and never re-derived from the slug: a rename keeps it, clearing detaches it and
deleting the tag leaves the prompt and its history. The tag-context facts are
read from the same DB. Fixture bots and tags only.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlmodel import SQLModel

from llm_bawt.bot_tags.store import BotTagStore
from llm_bawt.bot_tags.errors import TagConflict
from llm_bawt.prompt_registry import PromptTemplate, PromptTemplateVersion
from tests.test_bot_tag_api_pg import api  # noqa: F401
from tests.test_bot_tags_pg import rig  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def tags(api):  # noqa: F811
    SQLModel.metadata.create_all(
        api.rig.engine, tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__])
    api.client.post("/v1/bot-tags", json={
        "slug": "fixture_role", "name": "Fixture Role", "capabilities": ["review"]})
    api.patch("fixture-bot", tags=["fixture_role"])
    api.published.clear()
    api.reloads.clear()
    def prompts():
        with api.rig.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(
                "SELECT t.key, t.body, t.category, count(v.id) FROM prompt_templates t"
                " LEFT JOIN prompt_template_versions v"
                "   ON (v.key, v.scope_type, v.scope_id) = (t.key, t.scope_type, t.scope_id)"
                " GROUP BY t.id ORDER BY t.key"))]

    api.prompts = prompts
    api.instructions = lambda slug, body: api.client.put(
        f"/v1/bot-tags/{slug}/instructions", json={"body": body, "updated_by": "fixture-bot"})
    return api


def _tag(api, slug):
    return next((t for t in api.client.get("/v1/bot-tags").json()["data"] if t["slug"] == slug),
                None)


# --- rename ---------------------------------------------------------------------


def test_a_rename_moves_assignments_key_and_capabilities_and_announces_holders(tags):
    tags.instructions("fixture_role", "Review carefully.")

    renamed = tags.client.patch("/v1/bot-tags/fixture_role",
                                json={"slug": "fixture_renamed", "name": "Renamed"})

    assert renamed.status_code == 200, renamed.text
    body = renamed.json()
    assert (body["slug"], body["name"], body["prompt_key"], body["capabilities"]) == (
        "fixture_renamed", "Renamed", "role.fixture_role", ["review"])
    assert body["used_by"] == [{"bot": "fixture-bot", "has_profile": True}]
    assert tags.rig.tags("fixture-bot") == ["fixture_renamed"]
    assert _tag(tags, "fixture_role") is None
    assert tags.published == ["fixture-bot"] and tags.reloads == [True]


def test_a_rename_onto_an_existing_slug_is_a_409_that_writes_nothing(tags):
    tags.client.post("/v1/bot-tags", json={"slug": "fixture_other", "name": "Other"})
    before = tags.client.get("/v1/bot-tags").json()

    for target in ("fixture_other", "private"):
        response = tags.client.patch("/v1/bot-tags/fixture_role", json={"slug": target})
        assert response.status_code == 409, response.text

    assert tags.client.get("/v1/bot-tags").json() == before
    assert tags.rig.tags("fixture-bot") == ["fixture_role"]
    assert tags.published == []


@pytest.mark.parametrize("path, target, status", [
    ("/v1/bot-tags/fixture_role", "Fixture_Bad", 422),
    ("/v1/bot-tags/private", "fixture_shadow", 409),
    ("/v1/bot-tags/fixture_missing", "fixture_shadow", 404),
])
def test_a_bad_rename_is_refused(tags, path, target, status):
    assert tags.client.patch(path, json={"slug": target}).status_code == status
    assert _tag(tags, "fixture_shadow") is None


def test_a_same_slug_patch_is_not_a_rename(tags):
    response = tags.client.patch("/v1/bot-tags/fixture_role", json={"slug": "fixture_role"})

    assert response.status_code == 200, response.text
    assert tags.published == [] and tags.rig.tags("fixture-bot") == ["fixture_role"]


def test_capabilities_follow_the_renamed_tag(tags):
    tags.client.patch("/v1/bot-tags/fixture_role", json={"slug": "fixture_renamed"})

    assert BotTagStore(tags.rig.engine).capabilities_by_tag() == {"fixture_renamed": ("review",)}


# --- instructions ---------------------------------------------------------------


def test_the_first_save_allocates_a_key_and_later_saves_add_versions(tags):
    first = tags.instructions("fixture_role", "Review carefully.")
    assert first.status_code == 200, first.text
    assert first.json()["prompt_key"] == "role.fixture_role"

    second = tags.instructions("fixture_role", "Review very carefully.")
    assert second.json()["prompt_key"] == "role.fixture_role"
    assert tags.prompts() == [("role.fixture_role", "Review very carefully.", "bot_tag", 2)]


def test_a_taken_key_gets_the_first_free_suffix(tags):
    with tags.rig.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO bot_tag_catalog (slug, name, kind, prompt_key, system)"
            " VALUES ('fixture_holder', 'Holder', 'role', 'role.fixture_role', FALSE)"))
    tags.client.post("/v1/bot-tags", json={"slug": "fixture_x", "name": "X"})
    tags.instructions("fixture_x", "Orphan prompt.")
    tags.client.patch("/v1/bot-tags/fixture_x", json={"slug": "fixture_y"})
    tags.client.post("/v1/bot-tags", json={"slug": "fixture_x", "name": "X again"})

    held = tags.instructions("fixture_role", "Mine.").json()["prompt_key"]
    fresh = tags.instructions("fixture_x", "New one.").json()["prompt_key"]

    assert held == "role.fixture_role_2"
    # role.fixture_x still belongs to the renamed fixture_y tag.
    assert fresh == "role.fixture_x_2"
    assert _tag(tags, "fixture_y")["prompt_key"] == "role.fixture_x"


def test_clearing_detaches_the_key_and_keeps_the_prompt(tags):
    tags.instructions("fixture_role", "Review carefully.")

    cleared = tags.client.delete("/v1/bot-tags/fixture_role/instructions")

    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["prompt_key"] is None
    assert tags.prompts() == [("role.fixture_role", "Review carefully.", "bot_tag", 1)]
    # The kept row is never silently re-attached: the next save takes a fresh key.
    assert tags.instructions("fixture_role", "Again.").json()["prompt_key"] == "role.fixture_role_2"


def test_deleting_a_tag_keeps_its_prompt_and_history(tags):
    tags.client.post("/v1/bot-tags", json={"slug": "fixture_free", "name": "Free"})
    tags.instructions("fixture_free", "Body.")

    assert tags.client.delete("/v1/bot-tags/fixture_free").status_code == 200
    assert ("role.fixture_free", "Body.", "bot_tag", 1) in tags.prompts()


@pytest.mark.parametrize("slug, body, status", [
    ("fixture_role", "   ", 422),
    ("fixture_role", "x" * 20001, 422),
    ("private", "Behaviour tags are code.", 409),
    ("fixture_missing", "Body.", 404),
])
def test_a_refused_save_writes_nothing(tags, slug, body, status):
    before = tags.client.get("/v1/bot-tags").json()

    response = tags.instructions(slug, body)

    assert response.status_code == status, response.text
    assert tags.client.get("/v1/bot-tags").json() == before
    assert tags.prompts() == []


def test_a_key_taken_after_allocation_is_a_conflict_and_never_overwritten(tags, monkeypatch):
    """Another writer creates the chosen key between allocation and the write."""
    store = BotTagStore(tags.rig.engine)
    allocate = BotTagStore._allocate_prompt_key

    def racing_allocate(conn, slug):
        key = allocate(conn, slug)
        with tags.rig.engine.begin() as other:
            other.execute(text(
                "INSERT INTO prompt_templates (key, scope_type, scope_id, format, body,"
                " created_at, updated_at) VALUES (:k, 'global', '*', 'plain_text', 'Theirs.',"
                " now(), now())"), {"k": key})
        return key

    monkeypatch.setattr(BotTagStore, "_allocate_prompt_key", staticmethod(racing_allocate))

    with pytest.raises(TagConflict):
        store.save_instructions("fixture_role", "Mine.", None)
    assert _tag(tags, "fixture_role")["prompt_key"] is None
    assert tags.prompts() == [("role.fixture_role", "Theirs.", None, 0)]


def test_a_key_held_by_another_tag_is_refused_on_create_and_patch(tags):
    """Rook r1 Blocker: a shared key let one tag's save overwrite another's prompt."""
    tags.instructions("fixture_role", "Mine.")
    tags.client.post("/v1/bot-tags", json={"slug": "fixture_other", "name": "Other"})

    created = tags.client.post("/v1/bot-tags", json={
        "slug": "fixture_thief", "name": "Thief", "prompt_key": "role.fixture_role"})
    patched = tags.client.patch("/v1/bot-tags/fixture_other",
                                json={"prompt_key": "role.fixture_role"})

    assert (created.status_code, patched.status_code) == (409, 409), (created.text, patched.text)
    assert _tag(tags, "fixture_thief") is None
    assert _tag(tags, "fixture_other")["prompt_key"] is None
    assert tags.instructions("fixture_other", "Theirs.").json()["prompt_key"] == "role.fixture_other"
    assert ("role.fixture_role", "Mine.", "bot_tag", 1) in tags.prompts()


def test_a_tag_may_keep_or_reset_its_own_key(tags):
    tags.instructions("fixture_role", "Mine.")

    same = tags.client.patch("/v1/bot-tags/fixture_role", json={"prompt_key": "role.fixture_role"})

    assert same.status_code == 200, same.text


def test_the_database_holds_each_key_to_one_tag(tags):
    from sqlalchemy.exc import IntegrityError

    tags.instructions("fixture_role", "Mine.")

    with pytest.raises(IntegrityError, match="bot_tag_catalog_prompt_key_key"):
        with tags.rig.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO bot_tag_catalog (slug, name, kind, prompt_key, system)"
                " VALUES ('fixture_dup', 'Dup', 'role', 'role.fixture_role', FALSE)"))


def test_a_key_shared_before_the_index_blocks_saves_and_startup_survives(tags, caplog):
    from llm_bawt.bot_tags import catalog

    tags.instructions("fixture_role", "Mine.")
    with tags.rig.engine.begin() as conn:
        conn.execute(text(f"DROP INDEX {catalog.PROMPT_KEY_INDEX}"))
        conn.execute(text(
            "INSERT INTO bot_tag_catalog (slug, name, kind, prompt_key, system)"
            " VALUES ('fixture_dup', 'Dup', 'role', 'role.fixture_role', FALSE)"))

    with tags.rig.engine.begin() as conn:
        catalog.ensure_schema(conn)
        assert not catalog.owned(conn, catalog.PROMPT_KEY_INDEX, "i")
    assert "role.fixture_role" in caplog.text

    response = tags.instructions("fixture_dup", "Overwrite?")

    assert response.status_code == 409, response.text
    assert ("role.fixture_role", "Mine.", "bot_tag", 1) in tags.prompts()


# --- the tag-context facts --------------------------------------------------------


def test_tag_context_facts_list_visible_agent_bots_and_this_bots_instructions(tags):
    from types import SimpleNamespace

    from llm_bawt.bot_tags.context import load_tag_context
    from tests.test_bot_tags_pg import upsert

    for slug, extra in (("fixture-chat", {"bot_type": "chat", "agent_backend": None}),
                        ("fixture-hidden", {}), ("fixture-off", {}), ("fixture-plain", {})):
        upsert(tags.store, slug, **({"bot_type": "agent", "agent_backend": "claude-code"} | extra))
    upsert(tags.store, "fixture-bot", bot_type="agent", agent_backend="claude-code")
    tags.patch("fixture-hidden", tags=["private"])
    tags.patch("fixture-off", tags=["inactive", "fixture_role"])
    tags.patch("fixture-bot", tags=["fixture_role"])
    tags.instructions("fixture_role", "Review carefully.")
    bodies = {"role.fixture_role": "Review carefully."}

    context = load_tag_context(tags.rig.engine, SimpleNamespace(ensure_model_catalog=lambda: None),
                               "fixture-bot", bodies.get, profiles=tags.store.list_all())

    assert context.bot_roster == (
        "- fixture-bot: tags fixture_role; endpoint unknown; provider unknown\n"
        "- fixture-plain: tags none; endpoint unknown; provider unknown")
    assert context.own_tag_instructions == "### Fixture Role (`fixture_role`)\nReview carefully."
    assert context.tag_catalog == "- `fixture_role` (Fixture Role): capabilities review"
    assert context.flags == ()
