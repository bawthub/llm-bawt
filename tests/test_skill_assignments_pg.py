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
    "fixture-claw": ("agent", "openclaw", {}),
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


def test_adopt_seeds_every_skill_as_available_and_release_drops_it(world):
    assert world.service.resolve_bot("fixture-one").manifest() is None

    view = world.service.adopt("fixture-one", "fixture-actor")

    assert view["managed"] is True
    assert {ref: (e["mode"], e["origin"], e["applied"]) for ref, e in
            {e["ref"]: e for e in view["entries"]}.items()} == {
        ref: ("available", "bot", True) for ref in ("alpha", "beta", "gamma")}
    assert world.service.resolve_bot("fixture-one").manifest()["injected"] == []
    with world.rig.engine.connect() as conn:
        assert conn.execute(text("SELECT adopted_by FROM skill_managed_bots WHERE bot_slug="
                                 "'fixture-one'")).scalar_one() == "fixture-actor"

    world.service.put("bot", "fixture-one", "alpha", "injected", 0, "fixture-actor")
    world.service.adopt("fixture-one", "fixture-actor")  # re-adopt keeps existing rows
    assert _rows(world)[("bot", "fixture-one", "alpha")] == "injected"

    world.service.release("fixture-one")
    assert _rows(world, "fixture-one") == {}
    assert world.service.resolve_bot("fixture-one").manifest() is None
    assert _status(world.service.release, "fixture-one")[0] == 404


@pytest.mark.parametrize(("slug", "status"), [
    ("fixture-private", 403), ("fixture-bundle", 409), ("fixture-claw", 422),
    ("fixture-chat", 422), ("fixture-unknown", 404)])
def test_adopt_refuses_bots_that_cannot_take_managed_skills(world, slug, status):
    assert _status(world.service.adopt, slug, None)[0] == status
    with world.rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM skill_managed_bots")).scalar_one() == 0


def test_bot_rows_need_an_adopted_writable_bot_and_valid_values(world):
    put = world.service.put
    assert _status(put, "bot", "fixture-one", "alpha", "injected", 0, None)[0] == 409
    assert _status(put, "bot", "fixture-private", "alpha", "available", 0, None)[0] == 403
    world.service.adopt("fixture-one", None)
    for args in [("bot", "fixture-one", "alpha", "loud", 0, None),
                 ("bot", "fixture-one", "alpha", "injected", -1, None),
                 ("bot", "fixture-one", "alpha", "injected", 10_001, None),
                 ("bot", "fixture-one", "alpha", "injected", True, None),
                 ("bot", "fixture-one", "missing", "injected", 0, None),
                 ("team", "fixture-one", "alpha", "injected", 0, None)]:
        assert _status(put, *args)[0] == 422, args
    put("bot", "fixture-one", "alpha", "off", 0, None)
    assert _entries(world, "fixture-one")["alpha"]["applied"] is False


def test_an_over_cap_write_is_refused_and_rolled_back(world):
    world.service.adopt("fixture-one", None)
    world.service.put("bot", "fixture-one", "alpha", "injected", 0, None)
    world.service.put("bot", "fixture-one", "beta", "injected", 1, None)

    status, detail = _status(world.service.put, "bot", "fixture-one", "gamma", "injected", 2, None)

    assert status == 409 and detail["code"] == "over_cap"
    assert detail["bots"] == [{
        "bot": "fixture-one", "refs": ["gamma"],
        "injected_tokens": world.sizes["alpha"] + world.sizes["beta"],
        "per_skill_cap": 10_000, "total_cap": world.caps["value"].total}]
    assert _rows(world)[("bot", "fixture-one", "gamma")] == "available"
    resolution = world.service.resolve_bot("fixture-one")
    assert [e.ref for e in resolution.injected] == ["alpha", "beta"]


def test_a_lowered_cap_does_not_block_unrelated_writes(world):
    world.service.adopt("fixture-one", None)
    world.service.put("bot", "fixture-one", "alpha", "injected", 0, None)
    world.service.put("bot", "fixture-one", "beta", "injected", 1, None)
    world.caps["value"] = Caps(per_skill=10_000, total=world.sizes["alpha"])

    world.service.put("bot", "fixture-one", "gamma", "off", 2, None)

    assert world.service.resolve_bot("fixture-one").over_cap == {"beta"}


def test_tag_rows_reach_tagged_managed_bots_without_a_bot_row(world):
    world.service.adopt("fixture-one", None)
    world.service.remove("bot", "fixture-one", "gamma")

    world.service.put("tag", ROLE, "gamma", "injected", 0, "fixture-actor")

    gamma = _entries(world, "fixture-one")["gamma"]
    assert (gamma["mode"], gamma["origin"], gamma["applied"]) == ("injected", f"tag:{ROLE}", True)
    # A bot row shadows the tag rule: alpha stays as adoption seeded it.
    world.service.put("tag", ROLE, "alpha", "injected", 0, None)
    assert _entries(world, "fixture-one")["alpha"]["mode"] == "available"
    # The unmanaged tagged bot is untouched.
    assert world.service.resolve_bot("fixture-two").entries == ()


def test_a_tag_write_that_pushes_a_tagged_bot_over_the_cap_rolls_back(world):
    world.service.adopt("fixture-one", None)
    world.service.put("bot", "fixture-one", "alpha", "injected", 0, None)
    world.service.put("bot", "fixture-one", "beta", "injected", 1, None)
    world.service.remove("bot", "fixture-one", "gamma")

    status, detail = _status(world.service.put, "tag", ROLE, "gamma", "injected", 5, None)

    assert status == 409 and [b["bot"] for b in detail["bots"]] == ["fixture-one"]
    assert ("tag", ROLE, "gamma") not in _rows(world)


@pytest.mark.parametrize(("args", "status"), [
    (("tag", ROLE, "alpha", "off", 0, None), 422),
    (("tag", "fixture_unknown_tag", "alpha", "available", 0, None), 422)])
def test_tag_scope_refuses_off_and_unknown_tags(world, args, status):
    assert _status(world.service.put, *args)[0] == status


def test_the_schema_itself_refuses_off_at_tag_scope(world):
    world.service.assignments()  # bootstraps the tables
    with pytest.raises(IntegrityError):
        with world.rig.engine.begin() as conn:
            conn.execute(text("INSERT INTO skill_assignments (scope_type, scope_id, skill_ref, mode)"
                              f" VALUES ('tag', '{ROLE}', 'alpha', 'off')"))


def test_removing_a_missing_row_is_404_and_changes_nothing(world):
    world.service.adopt("fixture-one", None)
    before = _rows(world)
    assert _status(world.service.remove, "tag", ROLE, "alpha")[0] == 404
    assert _rows(world) == before


def test_private_bots_show_only_a_lock(world):
    world.service.assignments()  # bootstraps the tables
    with world.rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO skill_managed_bots (bot_slug) VALUES ('fixture-private')"))
        conn.execute(text("INSERT INTO skill_assignments (scope_type, scope_id, skill_ref, mode)"
                          " VALUES ('bot', 'fixture-private', 'alpha', 'injected')"))

    summary = {b["slug"]: b for b in world.service.bots_summary()["bots"]}
    view = world.service.effective("fixture-private")

    assert summary["fixture-private"] == {"slug": "fixture-private", "locked": True, "managed": False}
    assert (view["locked"], view["managed"], view["entries"], view["manifest"]) == (
        True, False, [], None)
    assert world.service.resolve_bot("fixture-private").injected == ()


def _plant_managed(world, slug):
    world.service.assignments()  # bootstraps the tables
    with world.rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO skill_managed_bots (bot_slug) VALUES (:s)"), {"s": slug})
        conn.execute(text("INSERT INTO skill_assignments (scope_type, scope_id, skill_ref, mode)"
                          " VALUES ('bot', :s, 'alpha', 'injected')"), {"s": slug})


def _managed_rows(world):
    with world.rig.engine.connect() as conn:
        return (sorted(conn.execute(text("SELECT bot_slug FROM skill_managed_bots")).scalars()),
                _rows(world))


@pytest.mark.parametrize(("slug", "status"), [
    ("fixture-private", 403), ("fixture-unknown", 404)])
def test_release_refuses_protected_and_unknown_bots_and_keeps_their_rows(world, slug, status):
    _plant_managed(world, slug)
    before = _managed_rows(world)
    assert _status(world.service.release, slug)[0] == status
    assert _managed_rows(world) == before


def test_release_fails_closed_when_tags_are_unavailable(world):
    world.service.adopt("fixture-one", None)
    before = _managed_rows(world)
    blind = SkillService(world.rig.engine, list_bots=world.service._list_bots,
                         caps=world.service._caps, source=world.service.source,
                         tags=SimpleNamespace(snapshot=lambda _conn: TagSnapshot.unavailable()))
    assert _status(blind.release, "fixture-one")[0] == 503
    assert _managed_rows(world) == before


def test_a_bundle_bot_can_still_be_released(world):
    _plant_managed(world, "fixture-bundle")
    world.service.release("fixture-bundle")
    assert _managed_rows(world) == ([], {})


def test_an_unmounted_source_serves_an_empty_catalog_and_injects_nothing(world, monkeypatch, tmp_path):
    world.service.adopt("fixture-one", None)
    world.service.put("bot", "fixture-one", "alpha", "injected", 0, None)
    unmounted = SkillService(world.rig.engine, list_bots=world.service._list_bots,
                             caps=world.service._caps,
                             source=AmbientSource((tmp_path / "not-mounted",)))
    monkeypatch.setattr(skill_routes, "skill_service", lambda: unmounted)
    app = FastAPI()
    app.include_router(skill_routes.router)
    client = TestClient(app)

    assert client.get("/v1/skills").json() == {"data": []}
    view = client.get("/v1/skills/bots/fixture-one")
    assert view.status_code == 200 and view.json()["injected_tokens"] == 0
    resolution = unmounted.resolve_bot("fixture-one")
    assert resolution.injected == () and resolution.manifest()["entries"] == []


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
        "claude-code", "codex", "openclaw"}
    tree = client.get("/v1/skills/catalog/alpha/tree").json()
    assert [f["path"] for f in tree["files"]] == ["SKILL.md"]
    assert client.get("/v1/skills/catalog/alpha/file", params={"path": "SKILL.md"}).json()[
        "text"].startswith("---\nname: alpha")
    assert client.get("/v1/skills/catalog/alpha/file",
                      params={"path": "../beta/SKILL.md"}).status_code == 400

    assert client.post("/v1/skills/bots/fixture-one/adopt", json={}).status_code == 200
    put = client.put("/v1/skills/assignments", json={
        "scope_type": "bot", "scope_id": "fixture-one", "skill_ref": "alpha",
        "mode": "injected", "position": 0, "updated_by": "fixture-actor"})
    assert put.status_code == 200 and put.json()["mode"] == "injected"
    over = [client.put("/v1/skills/assignments", json={
        "scope_type": "bot", "scope_id": "fixture-one", "skill_ref": ref,
        "mode": "injected", "position": i + 1}) for i, ref in enumerate(("beta", "gamma"))]
    assert [r.status_code for r in over] == [200, 409]
    assert over[1].json()["detail"]["code"] == "over_cap"
    assert client.get("/v1/skills/bots/fixture-one").json()["injected_tokens"] == (
        world.sizes["alpha"] + world.sizes["beta"])
    assert client.delete("/v1/skills/assignments", params={
        "scope_type": "bot", "scope_id": "fixture-one", "skill_ref": "beta"}).status_code == 200
    assert client.get("/v1/skills/assignments",
                      params={"scope_type": "bot", "scope_id": "fixture-one"}).status_code == 200
    assert client.get("/v1/skills/bots/fixture-private").json()["locked"] is True
    assert client.delete("/v1/skills/bots/fixture-one/managed").status_code == 200
    assert client.delete("/v1/skills/bots/fixture-one/managed").status_code == 404


def test_routes_report_a_missing_database_as_503(monkeypatch):
    monkeypatch.setattr(skill_routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(skill_routes, "get_bot_profile_store",
                        lambda _config: SimpleNamespace(engine=None))
    app = FastAPI()
    app.include_router(skill_routes.router)
    assert TestClient(app).get("/v1/skills/bots").status_code == 503
