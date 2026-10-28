"""Skill assignments against an isolated, networkless PostgreSQL DB (TASK-1309).

The real store, service and routes on a private DB with the real tag tables;
stubbed are the bot list (fixture bots), the caps and the skill tree (a temp
directory). Fixture bots and fixture skills only, one private DB per test.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from agent_bridge.skill_sources import AmbientSource
from llm_bawt.bot_tags.facade import TagSnapshot
from llm_bawt.service.routes import skills as skill_routes
from llm_bawt.skills.resolver import Caps
from llm_bawt.skills.service import SkillService, SkillServiceError
from tests.bot_tag_fixtures import make_private, tag
from tests.test_bot_tags_pg import rig, upsert  # noqa: F401

pytestmark = pytest.mark.integration

ROLE = "fixture_role"
_BOTS = {
    "fixture-one": ("agent", "claude-code", {}),
    "fixture-two": ("agent", "codex", {}),
    "fixture-local": ("agent", "local", {}),
    "fixture-chat": ("chat", None, {}),
    "fixture-bundle": ("agent", "claude-code", {"skill_bundle": "fixture-pkg"}),
    "fixture-private": ("agent", "claude-code", {}),
}


def _bot(slug):
    bot_type, backend, config = _BOTS[slug]
    return SimpleNamespace(slug=slug, bot_type=bot_type, agent_backend=backend,
                           agent_backend_config=config)


@pytest.fixture
def world(rig, tmp_path):  # noqa: F811
    store = rig.store()
    for slug in _BOTS:
        upsert(store, slug)
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO bot_tag_catalog (slug, name, description, kind) "
                             f"VALUES ('{ROLE}', 'Fixture role', '', 'role')")
        tag(conn, "fixture-one", ROLE)
        tag(conn, "fixture-two", ROLE)
        make_private(conn, "fixture-private", "tag")

    root = tmp_path / "skills"
    for name in ("alpha", "beta", "gamma"):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: the {name} skill\n---\n"
            f"# {name}\n" + f"Instructions for {name}. " * 40)
    source = AmbientSource((root,))
    sizes = {ref: source.read_body(ref).tokens for ref in ("alpha", "beta", "gamma")}
    # alpha and beta fit together; any third injected skill does not.
    caps = {"value": Caps(per_skill=10_000, total=sizes["alpha"] + sizes["beta"])}
    service = SkillService(rig.engine, list_bots=lambda: [_bot(s) for s in _BOTS],
                           caps=lambda: caps["value"], source=source)
    return SimpleNamespace(rig=rig, service=service, caps=caps, sizes=sizes, root=root)


def _rows(world, slug=None):
    return {(r["scope_type"], r["scope_id"], r["skill_ref"]): r["mode"]
            for r in world.service.assignments()
            if slug is None or r["scope_id"] == slug}


def _status(call, *args):
    with pytest.raises(SkillServiceError) as caught:
        call(*args)
    return caught.value.status, caught.value.detail


def _entries(world, slug):
    return {e["ref"]: e for e in world.service.effective(slug)["entries"]}


def test_a_bot_without_rows_keeps_discovery_and_one_disable_makes_an_allow_list(world):
    assert world.service.resolve_bot("fixture-one").manifest() is None

    world.service.put("bot", "fixture-one", "beta", "disable", 0, "fixture-actor")

    manifest = world.service.resolve_bot("fixture-one").manifest()
    assert [e["ref"] for e in manifest["entries"]] == ["alpha", "gamma"]
    # A skill added to the tree later is allowed with no row written.
    (world.root / "delta").mkdir()
    (world.root / "delta" / "SKILL.md").write_text("---\nname: delta\ndescription: d\n---\n# d\n")
    fresh = SkillService(world.rig.engine, list_bots=world.service._list_bots,
                         caps=world.service._caps, source=AmbientSource((world.root,)))
    assert [e["ref"] for e in fresh.resolve_bot("fixture-one").manifest()["entries"]] == [
        "alpha", "delta", "gamma"]
    world.service.remove("bot", "fixture-one", "beta")
    assert world.service.resolve_bot("fixture-one").manifest() is None


def test_bot_rows_need_a_writable_bot_and_valid_values(world):
    put = world.service.put
    assert _status(put, "bot", "fixture-private", "alpha", "enable", 0, None)[0] == 403
    assert _status(put, "bot", "fixture-bundle", "alpha", "enable", 0, None)[0] == 409
    assert _status(put, "bot", "fixture-unknown", "alpha", "enable", 0, None)[0] == 404
    for args in [("bot", "fixture-one", "alpha", "loud", 0, None),
                 ("bot", "fixture-one", "alpha", "inject", -1, None),
                 ("bot", "fixture-one", "alpha", "inject", 10_001, None),
                 ("bot", "fixture-one", "alpha", "inject", True, None),
                 ("bot", "fixture-one", "missing", "inject", 0, None),
                 ("team", "fixture-one", "alpha", "inject", 0, None)]:
        assert _status(put, *args)[0] == 422, args
    put("bot", "fixture-one", "alpha", "disable", 0, None)
    assert _entries(world, "fixture-one")["alpha"]["applied"] is True
    assert world.service.effective("fixture-one")["manifest"] is not None


def test_a_legacy_mode_name_is_stored_under_its_new_name(world):
    assert world.service.put("bot", "fixture-one", "alpha", "injected", 0, None)["mode"] == "inject"
    assert _rows(world)[("bot", "fixture-one", "alpha")] == "inject"


def test_an_over_cap_write_is_refused_and_rolled_back(world):
    world.service.put("bot", "fixture-one", "alpha", "inject", 0, None)
    world.service.put("bot", "fixture-one", "beta", "inject", 1, None)

    status, detail = _status(world.service.put, "bot", "fixture-one", "gamma", "inject", 2, None)

    assert status == 409 and detail["code"] == "over_cap"
    assert detail["bots"] == [{
        "bot": "fixture-one", "refs": ["gamma"],
        "injected_tokens": world.sizes["alpha"] + world.sizes["beta"],
        "per_skill_cap": 10_000, "total_cap": world.caps["value"].total}]
    assert ("bot", "fixture-one", "gamma") not in _rows(world)
    resolution = world.service.resolve_bot("fixture-one")
    assert [e.ref for e in resolution.injected] == ["alpha", "beta"]


def test_a_lowered_cap_does_not_block_unrelated_writes(world):
    world.service.put("bot", "fixture-one", "alpha", "inject", 0, None)
    world.service.put("bot", "fixture-one", "beta", "inject", 1, None)
    world.caps["value"] = Caps(per_skill=10_000, total=world.sizes["alpha"])

    world.service.put("bot", "fixture-one", "gamma", "disable", 2, None)

    assert world.service.resolve_bot("fixture-one").over_cap == {"beta"}


def test_tag_rows_reach_every_tagged_bot_and_a_bot_row_overrides(world):
    world.service.put("tag", ROLE, "gamma", "disable", 0, "fixture-actor")

    for slug in ("fixture-one", "fixture-two"):
        gamma = _entries(world, slug)["gamma"]
        assert (gamma["mode"], gamma["origin"], gamma["applied"]) == ("disable", f"tag:{ROLE}", True)
    world.service.put("bot", "fixture-one", "gamma", "enable", 0, None)
    gamma = _entries(world, "fixture-one")["gamma"]
    assert (gamma["mode"], gamma["origin"], gamma["reason"]) == ("enable", "bot", None)
    assert world.service.resolve_bot("fixture-one").manifest() is None
    assert "gamma" not in [e["ref"] for e in
                           world.service.resolve_bot("fixture-two").manifest()["entries"]]


def test_a_tag_write_that_pushes_a_tagged_bot_over_the_cap_rolls_back(world):
    world.service.put("bot", "fixture-one", "alpha", "inject", 0, None)
    world.service.put("bot", "fixture-one", "beta", "inject", 1, None)

    status, detail = _status(world.service.put, "tag", ROLE, "gamma", "inject", 5, None)

    assert status == 409 and [b["bot"] for b in detail["bots"]] == ["fixture-one"]
    assert ("tag", ROLE, "gamma") not in _rows(world)


def test_tag_scope_refuses_unknown_tags(world):
    assert _status(world.service.put, "tag", "fixture_unknown_tag", "alpha", "disable", 0,
                   None)[0] == 422


def test_the_schema_refuses_an_unknown_mode(world):
    world.service.assignments()  # bootstraps the tables
    with pytest.raises(IntegrityError):
        with world.rig.engine.begin() as conn:
            conn.execute(text("INSERT INTO skill_assignments (scope_type, scope_id, skill_ref, mode)"
                              f" VALUES ('tag', '{ROLE}', 'alpha', 'loud')"))


def test_the_migration_turns_an_adopt_era_table_into_exceptions(world):
    from llm_bawt.skills import store as store_module

    with world.rig.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE skill_assignments (
                scope_type TEXT NOT NULL CHECK (scope_type IN ('bot', 'tag')),
                scope_id TEXT NOT NULL, skill_ref TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('off', 'available', 'injected')),
                position INTEGER NOT NULL DEFAULT 0 CHECK (position >= 0),
                updated_by TEXT, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (scope_type, scope_id, skill_ref),
                CONSTRAINT skill_assignments_off_is_bot_only
                    CHECK (mode <> 'off' OR scope_type = 'bot'))"""))
        conn.execute(text("CREATE TABLE skill_managed_bots (bot_slug TEXT PRIMARY KEY,"
                          " adopted_by TEXT, adopted_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        conn.execute(text("INSERT INTO skill_managed_bots (bot_slug) VALUES ('fixture-one')"))
        conn.execute(text(f"""
            INSERT INTO skill_assignments (scope_type, scope_id, skill_ref, mode) VALUES
            ('bot', 'fixture-one', 'alpha', 'available'), ('bot', 'fixture-one', 'beta', 'off'),
            ('bot', 'fixture-one', 'gamma', 'injected'), ('tag', '{ROLE}', 'alpha', 'available')"""))

    for _ in range(2):  # idempotent: a second process start changes nothing
        with world.rig.engine.begin() as conn:
            store_module._bootstrap(conn)

    assert _rows(world) == {("bot", "fixture-one", "beta"): "disable",
                            ("bot", "fixture-one", "gamma"): "inject",
                            ("tag", ROLE, "alpha"): "enable"}
    with world.rig.engine.begin() as conn:
        assert conn.execute(text("SELECT count(*) FROM skill_managed_bots")).scalar_one() == 0
        # Both vocabularies are writable for one release (a rolling restart's old writer).
        conn.execute(text(f"INSERT INTO skill_assignments (scope_type, scope_id, skill_ref, mode)"
                          f" VALUES ('tag', '{ROLE}', 'beta', 'off')"))
        names = set(conn.execute(text("SELECT conname FROM pg_constraint WHERE conrelid ="
                                      " 'skill_assignments'::regclass")).scalars())
    assert "skill_assignments_off_is_bot_only" not in names
    assert "skill_assignments_mode_check" not in names


def test_removing_a_missing_row_is_404_and_changes_nothing(world):
    world.service.put("bot", "fixture-one", "alpha", "disable", 0, None)
    before = _rows(world)
    assert _status(world.service.remove, "tag", ROLE, "alpha")[0] == 404
    assert _rows(world) == before


def test_writes_fail_closed_when_tags_are_unavailable(world):
    blind = SkillService(world.rig.engine, list_bots=world.service._list_bots,
                         caps=world.service._caps, source=world.service.source,
                         tags=SimpleNamespace(snapshot=lambda _conn: TagSnapshot.unavailable()))
    assert _status(blind.put, "bot", "fixture-one", "alpha", "disable", 0, None)[0] == 503
    assert _rows(world) == {}


def test_private_bots_show_only_a_lock(world):
    world.service.assignments()  # bootstraps the tables
    with world.rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO skill_assignments (scope_type, scope_id, skill_ref, mode)"
                          " VALUES ('bot', 'fixture-private', 'alpha', 'inject')"))

    summary = {b["slug"]: b for b in world.service.bots_summary()["bots"]}
    view = world.service.effective("fixture-private")

    assert summary["fixture-private"] == {"slug": "fixture-private", "locked": True}
    assert (view["locked"], view["entries"], view["manifest"]) == (True, [], None)
    assert world.service.resolve_bot("fixture-private").injected == ()


def test_an_unmounted_source_applies_no_disable_and_says_so(world, monkeypatch, tmp_path):
    world.service.put("bot", "fixture-one", "alpha", "inject", 0, None)
    world.service.put("bot", "fixture-one", "beta", "disable", 1, None)
    unmounted = SkillService(world.rig.engine, list_bots=world.service._list_bots,
                             caps=world.service._caps,
                             source=AmbientSource((tmp_path / "not-mounted",)))
    monkeypatch.setattr(skill_routes, "skill_service", lambda: unmounted)
    app = FastAPI()
    app.include_router(skill_routes.router)
    client = TestClient(app)

    assert client.get("/v1/skills").json() == {"data": []}
    view = client.get("/v1/skills/bots/fixture-one").json()
    assert view["injected_tokens"] == 0 and view["catalog_unreadable"] is True
    assert view["manifest"] is None  # never an empty allow-list that hides every skill
    summary = {b["slug"]: b for b in client.get("/v1/skills/bots").json()["bots"]}
    assert summary["fixture-one"]["catalog_unreadable"] is True


def test_routes_serve_the_service_over_http(world, monkeypatch):
    monkeypatch.setattr(skill_routes, "skill_service", lambda: world.service)
    app = FastAPI()
    app.include_router(skill_routes.router)
    client = TestClient(app)

    catalog = {s["ref"]: s for s in client.get("/v1/skills").json()["data"]}
    assert sorted(catalog) == ["alpha", "beta", "gamma"]
    assert catalog["alpha"]["file_count"] == 1
    assert catalog["alpha"]["token_estimate"] == world.sizes["alpha"]
    assert {h["harness"] for h in client.get("/v1/skills/harnesses").json()["data"]} == {
        "claude-code", "codex"}
    tree = client.get("/v1/skills/catalog/alpha/tree").json()
    assert [f["path"] for f in tree["files"]] == ["SKILL.md"]
    assert client.get("/v1/skills/catalog/alpha/file", params={"path": "SKILL.md"}).json()[
        "text"].startswith("---\nname: alpha")
    assert client.get("/v1/skills/catalog/alpha/file",
                      params={"path": "../beta/SKILL.md"}).status_code == 400

    put = client.put("/v1/skills/assignments", json={
        "scope_type": "bot", "scope_id": "fixture-one", "skill_ref": "alpha",
        "mode": "inject", "position": 0, "updated_by": "fixture-actor"})
    assert put.status_code == 200 and put.json()["mode"] == "inject"
    over = [client.put("/v1/skills/assignments", json={
        "scope_type": "bot", "scope_id": "fixture-one", "skill_ref": ref,
        "mode": "inject", "position": i + 1}) for i, ref in enumerate(("beta", "gamma"))]
    assert [r.status_code for r in over] == [200, 409]
    assert over[1].json()["detail"]["code"] == "over_cap"
    assert client.get("/v1/skills/bots/fixture-one").json()["injected_tokens"] == (
        world.sizes["alpha"] + world.sizes["beta"])
    assert client.delete("/v1/skills/assignments", params={
        "scope_type": "bot", "scope_id": "fixture-one", "skill_ref": "beta"}).status_code == 200
    assert client.get("/v1/skills/assignments",
                      params={"scope_type": "bot", "scope_id": "fixture-one"}).status_code == 200
    assert client.get("/v1/skills/bots/fixture-private").json()["locked"] is True
    # The adopt-era batch routes are gone.
    assert client.post("/v1/skills/bots/fixture-one/adopt", json={}).status_code in (404, 405)
    assert client.delete("/v1/skills/bots/fixture-one/managed").status_code in (404, 405)


def test_routes_report_a_missing_database_as_503(monkeypatch):
    monkeypatch.setattr(skill_routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(skill_routes, "get_bot_profile_store",
                        lambda _config: SimpleNamespace(engine=None))
    app = FastAPI()
    app.include_router(skill_routes.router)
    assert TestClient(app).get("/v1/skills/bots").status_code == 503
