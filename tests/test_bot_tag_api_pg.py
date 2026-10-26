"""The bot tag API against an isolated, networkless PostgreSQL DB (TASK-1329 S5).

BOT_TAGS.md §8: ``tags`` on the profile responses and the profile PATCH
(full-set replace), the ``/v1/bot-tags`` catalog and assignment routes, the
migration's WARNING receipt and ``Bot.tags`` on ``/v1/bots``. The real routes
run over HTTP on the real store; stubbed are payload validation, the route and
switch gates (``test_profile_route_gate_pg.py``) and the post-commit side
effects, which are recorded. Fixture bots only, one private DB per test.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from llm_bawt import bots as bots_module
from llm_bawt.bot_tags import BehaviorTag, catalog
from llm_bawt.bot_tags.store import BotTagStore
from llm_bawt.bot_tags.vocabulary import API_ASSIGNER, MIGRATION_ASSIGNER
from llm_bawt.service.routes import bot_tags as bot_tag_routes
from llm_bawt.service.routes import models as model_routes
from llm_bawt.service.routes import settings
from tests.bot_tag_fixtures import tag
from tests.pg_lock_wait import wait_blocked_by
from tests.test_bot_tags_pg import LEGACY, legacy_profiles, rig, upsert  # noqa: F401


def _backend_pid(conn) -> int:
    return conn.execute(text("SELECT pg_backend_pid()")).scalar_one()

pytestmark = pytest.mark.integration


@pytest.fixture
def api(rig, monkeypatch):  # noqa: F811
    store = rig.store()
    service = SimpleNamespace(config=None)
    ctx = SimpleNamespace(rig=rig, store=store, published=[], reloads=[])

    async def publish(_service, slug):
        ctx.published.append(slug)

    for module in (settings, bot_tag_routes):
        monkeypatch.setattr(module, "get_service", lambda: service)
        monkeypatch.setattr(module, "get_bot_profile_store", lambda _config: store)
    monkeypatch.setattr(settings, "_validate_profile_payload", lambda _payload: None)
    monkeypatch.setattr(settings.ProfileRouteGate, "for_payload",
                        classmethod(lambda cls, *_args: cls({})))
    monkeypatch.setattr(settings, "ProfileSaveGuard", lambda *_gates: None)
    monkeypatch.setattr(settings, "_reload_bot_registry", lambda: ctx.reloads.append(True))
    monkeypatch.setattr(settings, "_reload_service_model_catalog", lambda _service: None)
    monkeypatch.setattr(settings, "_invalidate_bot_instance_cache", lambda *_args: None)
    monkeypatch.setattr(settings, "_clear_session_model_overrides", lambda *_a, **_k: None)
    monkeypatch.setattr(settings, "_resolve_effective_bot_prompt", lambda *_args: "")
    monkeypatch.setattr(settings, "_effective_bot_settings", lambda *_args: {})
    monkeypatch.setattr(settings, "_publish_bot_profile_changed", publish)

    app = FastAPI()
    app.include_router(settings.router)
    app.include_router(bot_tag_routes.router)
    ctx.client = TestClient(app)
    ctx.patch = lambda slug, **body: ctx.client.patch(f"/v1/bots/{slug}/profile", json=body)
    ctx.column = lambda slug: store.get(slug).include_in_global_search
    def assigned(slug):
        with rig.engine.connect() as conn:
            return {row["tag_slug"]: (row["assigned_by"], row["assigned_at"])
                    for row in conn.execute(text(
                        "SELECT tag_slug, assigned_by, assigned_at FROM bot_profile_tags"
                        " WHERE bot_slug=:s"), {"s": slug}).mappings()}

    ctx.assigned = assigned
    upsert(store, "fixture-bot")
    return ctx


# --- the migration receipt --------------------------------------------------


def test_a_fresh_migration_logs_a_warning_and_a_second_bootstrap_logs_nothing(rig, caplog):  # noqa: F811
    legacy_profiles(rig, fixture_private=False, fixture_public=True)
    with caplog.at_level(logging.INFO, logger="llm_bawt.runtime_settings"):
        rig.store()
        first = [r for r in caplog.records if "legacy private assignments" in r.getMessage()]
        caplog.clear()
        rig.store()
        second = [r for r in caplog.records if "legacy private assignments" in r.getMessage()]

    assert [(r.levelno, r.getMessage()) for r in first] == [
        (logging.WARNING, f"Bot tags: migrated {len(LEGACY)} legacy private assignments")]
    assert second == []
    # The rows are the primary receipt; the log line only points at them.
    assert rig.scalar("SELECT count(*) FROM bot_profile_tags WHERE assigned_by=:by",
                      by=MIGRATION_ASSIGNER) == len(LEGACY)


# --- PATCH /v1/bots/{slug}/profile tags -------------------------------------


def test_patch_tags_is_a_full_set_replace_that_mirrors_the_column(api):
    made = api.patch("fixture-bot", tags=LEGACY)
    assert made.status_code == 200, made.text
    assert made.json()["tags"] == LEGACY and made.json()["include_in_global_search"] is False
    private_at = api.assigned("fixture-bot")["private"][1]

    kept = api.patch("fixture-bot", tags=["private"])
    assert kept.json()["tags"] == ["private"] and api.column("fixture-bot") is False
    assert api.assigned("fixture-bot")["private"][1] == private_at  # kept, not re-inserted

    cleared = api.patch("fixture-bot", tags=[])
    assert cleared.json()["tags"] == [] and api.column("fixture-bot") is True
    assert api.client.get("/v1/bots/fixture-bot/profile").json()["tags"] == []
    assert api.published == ["fixture-bot"] * 3 and len(api.reloads) == 3


def test_a_patch_without_tags_keeps_them_and_reports_them(api):
    api.patch("fixture-bot", tags=["no_global_search"])
    response = api.patch("fixture-bot", description="unrelated edit")

    assert response.json()["tags"] == ["no_global_search"]
    assert api.client.get("/v1/bots/profiles").json()["profiles"][0]["tags"] == ["no_global_search"]


@pytest.mark.parametrize("body", [
    {"tags": ["fixture_unknown"]},
    {"tags": [" Private"]},
    {"tags": ["Private"]},
    {"tags": None},
    {"tags": "private"},
    {"tags": ["private"], "include_in_global_search": True},
    {"tags": [], "include_in_global_search": False},
    {"tags": ["private"], "updated_by": ""},
])
def test_a_refused_tag_patch_writes_nothing(api, body):
    api.patch("fixture-bot", tags=["no_global_search"])
    before = api.assigned("fixture-bot")

    response = api.patch("fixture-bot", description="must not land", **body)

    assert response.status_code == 422, response.text
    assert api.assigned("fixture-bot") == before
    assert api.column("fixture-bot") is True
    assert api.store.get("fixture-bot").description == ""


def test_a_flag_that_agrees_with_the_tags_is_accepted(api):
    response = api.patch("fixture-bot", tags=["private"], include_in_global_search=False)

    assert response.status_code == 200, response.text
    assert response.json()["tags"] == ["private"] and api.column("fixture-bot") is False


def test_updated_by_is_the_assigner_and_api_is_the_default(api):
    api.patch("fixture-bot", tags=["private"], updated_by="fixture-operator")
    api.patch("fixture-bot", tags=["private", "no_global_search"])

    by = {slug: who for slug, (who, _at) in api.assigned("fixture-bot").items()}
    assert by == {"private": "fixture-operator", "no_global_search": API_ASSIGNER}


def test_concurrent_full_set_replaces_serialize_instead_of_committing_a_union(api):
    # The holder closes (and so releases the lock) before the pool joins the racer.
    with ThreadPoolExecutor(max_workers=1) as pool, api.rig.engine.connect() as holder:
        holder.begin()
        catalog.lock_bot(holder, "fixture-bot")
        catalog.replace_assignments(holder, "fixture-bot", ["no_global_search"], "fixture-holder")
        racer = pool.submit(upsert, api.store, "fixture-bot", tags=["private"])
        wait_blocked_by(api.rig.engine, _backend_pid(holder), racer, timeout=10)
        holder.commit()
        racer.result(timeout=10)

    assert api.rig.tags("fixture-bot") == ["private"]
    assert api.column("fixture-bot") is False


# --- expected_tags compare-and-set (TASK-1398) ---------------------------------


def test_a_stale_expected_set_is_a_409_with_the_current_tags_and_writes_nothing(api):
    api.patch("fixture-bot", tags=["no_global_search"])
    before = api.assigned("fixture-bot")

    response = api.patch("fixture-bot", description="must not land",
                         tags=["inactive"], expected_tags=[])

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "tags_changed", "current_tags": ["no_global_search"],
        "message": "fixture-bot's tags changed since they were read; nothing was saved"}
    assert api.assigned("fixture-bot") == before
    assert api.store.get("fixture-bot").description == ""


def test_a_matching_expected_set_replaces_regardless_of_order_or_duplicates(api):
    api.patch("fixture-bot", tags=["no_global_search", "inactive"])

    response = api.patch("fixture-bot", tags=["private"],
                         expected_tags=["inactive", "no_global_search", "inactive"])

    assert response.status_code == 200, response.text
    assert response.json()["tags"] == ["private"] and api.column("fixture-bot") is False


@pytest.mark.parametrize("body", [
    # expected_tags without tags is a valid precondition since TASK-1521
    # (test_bot_profile_patch_pg.py).
    {"expected_tags": None, "tags": []},
    {"expected_tags": "private", "tags": []},
])
def test_a_malformed_expected_set_is_refused(api, body):
    response = api.patch("fixture-bot", description="must not land", **body)

    assert response.status_code == 422, response.text
    assert api.store.get("fixture-bot").description == ""


def test_two_adds_from_one_read_keep_both_tags_instead_of_dropping_private(api):
    """Both writers read [inactive]; one adds private, the other no_global_search.

    The holder commits private under the bot lock while the racing PATCH waits.
    The racer's expected set is stale, so it gets a 409 carrying the current
    set; recomputing its one-tag edit on that set keeps private.
    """
    api.patch("fixture-bot", tags=["inactive"])
    with ThreadPoolExecutor(max_workers=1) as pool, api.rig.engine.connect() as holder:
        holder.begin()
        catalog.lock_bot(holder, "fixture-bot")
        catalog.check_expected_assignments(holder, "fixture-bot", ["inactive"])
        catalog.replace_assignments(holder, "fixture-bot", ["inactive", "private"], "fixture-holder")
        racer = pool.submit(api.patch, "fixture-bot",
                            tags=["inactive", "no_global_search"], expected_tags=["inactive"])
        wait_blocked_by(api.rig.engine, _backend_pid(holder), racer, timeout=10)
        holder.commit()
        stale = racer.result(timeout=10)

    assert stale.status_code == 409, stale.text
    current = stale.json()["detail"]["current_tags"]
    assert current == ["inactive", "private"]
    assert api.rig.tags("fixture-bot") == ["inactive", "private"]

    retried = api.patch("fixture-bot", tags=sorted({*current, "no_global_search"}),
                        expected_tags=current)
    assert retried.status_code == 200, retried.text
    assert api.rig.tags("fixture-bot") == ["inactive", "no_global_search", "private"]
    assert api.column("fixture-bot") is False


# --- /v1/bot-tags catalog -----------------------------------------------------


def test_a_role_tag_is_created_listed_edited_and_deleted(api):
    created = api.client.post("/v1/bot-tags", json={
        "slug": "fixture_role", "name": " Fixture Role ", "prompt_key": "role.fixture",
        "capabilities": ["read"]})
    assert created.status_code == 201, created.text
    assert created.json() | {"created_at": None, "updated_at": None} == {
        "slug": "fixture_role", "name": "Fixture Role", "description": "", "kind": "role",
        "prompt_key": "role.fixture", "capabilities": ["read"], "system": False, "used_by": [],
        "created_at": None, "updated_at": None}

    edited = api.client.patch("/v1/bot-tags/fixture_role",
                              json={"prompt_key": None, "capabilities": ["read", "write"]})
    assert edited.status_code == 200, edited.text
    assert (edited.json()["prompt_key"], edited.json()["capabilities"]) == (None, ["read", "write"])
    assert edited.json()["name"] == "Fixture Role"

    listed = {t["slug"]: t for t in api.client.get("/v1/bot-tags").json()["data"]}
    assert listed["fixture_role"]["capabilities"] == ["read", "write"]
    assert listed["private"]["system"] is True and listed["private"]["kind"] == "behavior"

    assert api.client.delete("/v1/bot-tags/fixture_role").json() == {"deleted": "fixture_role"}
    assert api.client.patch("/v1/bot-tags/fixture_role", json={"name": "x"}).status_code == 404


@pytest.mark.parametrize("body, status", [
    ({"slug": "private", "name": "Shadow"}, 409),
    ({"slug": "fixture_dupe", "name": "Second"}, 409),
    ({"slug": "Fixture_Bad", "name": "Bad"}, 422),
    ({"slug": "fixture_bad", "name": "   "}, 422),
    ({"slug": "fixture_bad", "name": "Bad", "prompt_key": "persona.bad"}, 422),
    ({"slug": "fixture_bad", "name": "Bad", "prompt_key": "role.has space"}, 422),
    ({"slug": "fixture_bad", "name": "Bad", "capabilities": ["read", "read"]}, 422),
    ({"slug": "fixture_bad", "name": "Bad", "capabilities": [" read"]}, 422),
    ({"slug": "fixture_bad", "name": "Bad", "kind": "behavior"}, 422),
])
def test_a_refused_role_create_writes_nothing(api, body, status):
    api.client.post("/v1/bot-tags", json={"slug": "fixture_dupe", "name": "First"})
    before = api.client.get("/v1/bot-tags").json()

    response = api.client.post("/v1/bot-tags", json=body)

    assert response.status_code == status, response.text
    assert api.client.get("/v1/bot-tags").json() == before


@pytest.mark.parametrize("method, path, body", [
    ("patch", "/v1/bot-tags/private", {"name": "Renamed"}),
    ("delete", "/v1/bot-tags/private", None),
    ("delete", "/v1/bot-tags/fixture_role", None),
])
def test_system_and_assigned_tags_refuse_edits_and_deletes(api, method, path, body):
    api.client.post("/v1/bot-tags", json={"slug": "fixture_role", "name": "Fixture"})
    api.patch("fixture-bot", tags=["fixture_role"])
    before = api.client.get("/v1/bot-tags").json()

    response = api.client.request(method.upper(), path, json=body)

    assert response.status_code == 409, response.text
    assert api.client.get("/v1/bot-tags").json() == before


def test_used_by_lists_bots_with_and_without_a_profile(api):
    api.patch("fixture-bot", tags=["private"])
    with api.rig.engine.begin() as conn:
        tag(conn, "fixture-gone", BehaviorTag.PRIVATE)

    private = next(t for t in api.client.get("/v1/bot-tags").json()["data"] if t["slug"] == "private")
    assert private["used_by"] == [{"bot": "fixture-bot", "has_profile": True},
                                  {"bot": "fixture-gone", "has_profile": False}]


# --- DELETE /v1/bot-tags/{slug}/assignments/{bot} ---------------------------


def test_removing_a_profileless_assignment_works_and_reloads(api):
    with api.rig.engine.begin() as conn:
        tag(conn, "fixture-gone", BehaviorTag.PRIVATE, BehaviorTag.NO_GLOBAL_SEARCH)

    response = api.client.delete("/v1/bot-tags/private/assignments/fixture-gone")

    assert response.status_code == 200, response.text
    assert response.json() == {"tag": "private", "bot": "fixture-gone", "has_profile": False}
    assert api.rig.tags("fixture-gone") == ["no_global_search"]
    assert api.published == ["fixture-gone"] and api.reloads == [True]


def test_removing_private_sets_the_column_and_other_untags_leave_it(api):
    api.patch("fixture-bot", tags=LEGACY)
    api.client.delete("/v1/bot-tags/private/assignments/fixture-bot")
    assert api.rig.tags("fixture-bot") == ["no_global_search"] and api.column("fixture-bot") is True

    # A bypassing writer left the column false with no private tag: an
    # unrelated untag must not loosen it.
    with api.rig.engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE bot_profiles SET include_in_global_search=false WHERE slug='fixture-bot'")
    api.client.delete("/v1/bot-tags/no_global_search/assignments/fixture-bot")
    assert api.rig.tags("fixture-bot") == [] and api.column("fixture-bot") is False


@pytest.mark.parametrize("path, status, detail", [
    ("/v1/bot-tags/fixture_unknown/assignments/fixture-bot", 404, "not found"),
    ("/v1/bot-tags/inactive/assignments/fixture-bot", 404, "has no 'inactive' tag"),
    ("/v1/bot-tags/private/assignments/Fixture-Bot", 422, "canonical bot slug"),
    ("/v1/bot-tags/Private/assignments/fixture-bot", 422, "no normalization"),
])
def test_a_refused_assignment_delete_writes_nothing(api, path, status, detail):
    api.patch("fixture-bot", tags=["private"])
    api.published.clear()

    response = api.client.delete(path)

    assert response.status_code == status and detail in response.json()["detail"]
    assert api.rig.tags("fixture-bot") == ["private"] and api.column("fixture-bot") is False
    assert api.published == []


# --- Bot.tags and /v1/bots -----------------------------------------------------


def test_bot_tags_load_for_every_assigned_bot_or_report_unavailable(rig, monkeypatch):  # noqa: F811
    store = rig.store()
    upsert(store, "fixture-private", tags=["private"])
    with rig.engine.begin() as conn:
        tag(conn, "fixture-gone", BehaviorTag.PRIVATE)

    assert bots_module._load_bot_tags(rig.engine) == {
        "fixture-gone": ["private"], "fixture-private": ["private"]}

    def broken(_self):
        raise RuntimeError("fixture tag read failure")

    monkeypatch.setattr(BotTagStore, "tags_by_bot", broken)
    assert bots_module._load_bot_tags(rig.engine) is None


@pytest.mark.parametrize("tags, expected", [({"fixture-private": ["private"]}, ["private"]),
                                            ({}, []), (None, None)])
def test_db_profiles_carry_tags_and_a_tag_failure_never_drops_a_bot(rig, monkeypatch, tags,  # noqa: F811
                                                                    expected):
    store = rig.store()
    upsert(store, "fixture-private")
    monkeypatch.setattr("llm_bawt.utils.config.has_database_credentials", lambda _config: True)
    monkeypatch.setattr("llm_bawt.runtime_settings.BotProfileStore", lambda _config: store)
    monkeypatch.setattr(bots_module, "_load_bot_tags", lambda _engine: tags)

    profiles = bots_module._load_db_bot_profiles()

    assert profiles["fixture-private"]["tags"] == expected
    monkeypatch.setattr(bots_module, "_load_db_bot_profiles", lambda: profiles)
    bots_module._load_bots_config()
    assert bots_module.get_bot("fixture-private").tags == expected


@pytest.mark.parametrize("tags", [["private"], None])
def test_v1_bots_reports_each_bots_tags(monkeypatch, tags):
    bot = bots_module.Bot(slug="fixture-bot", name="Fixture", description="",
                          system_prompt="fixture", tags=tags)
    monkeypatch.setattr(model_routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(model_routes, "BotManager",
                        lambda _config: SimpleNamespace(list_bots=lambda: [bot]))
    app = FastAPI()
    app.include_router(model_routes.router)

    assert TestClient(app).get("/v1/bots").json()["data"][0]["tags"] == tags
