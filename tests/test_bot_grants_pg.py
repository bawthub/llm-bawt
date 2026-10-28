"""Grant tags against an isolated PostgreSQL DB (TASK-1609).

``approve_ops`` is written only by ``PUT /v1/bot-grants/{tag}/bots/{slug}``:
a person through the service bearer, or a grant-manager bot through its signed
turn. The profile PATCH and the single-assignment DELETE refuse to move it.
The real routes run over HTTP on the real store; the bearer check, the turn
capability and the runtime settings store are stand-ins. Fixture bots only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from llm_bawt.bot_tags import BehaviorTag, grants
from llm_bawt.service.routes import bot_grants as grant_routes
from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER, TaskTurnContextError
from tests.bot_tag_fixtures import tag
from tests.test_bot_tag_api_pg import api  # noqa: F401
from tests.test_bot_tags_pg import rig, upsert  # noqa: F401

pytestmark = pytest.mark.integration

GRANT = BehaviorTag.APPROVE_OPS
CAPABILITY = "fixture-manage"
ROLE = "fixture_manager"
MANAGER = "fixture-manager"
BEARER = "Bearer fixture-token"
PERSON = "fixture-person"


@pytest.fixture
def grant_api(api, monkeypatch):  # noqa: F811
    settings = {grants.MANAGER_CAPABILITY_SETTING: CAPABILITY}

    def bearer(authorization):
        if authorization != BEARER:
            raise HTTPException(401, "unauthorized")

    def open_turn(token):
        if not token.startswith("turn:"):
            raise TaskTurnContextError("bad turn capability")
        return SimpleNamespace(bot_id=token.removeprefix("turn:"))

    monkeypatch.setattr(grant_routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(grant_routes, "require_service_token", bearer)
    monkeypatch.setattr(grant_routes, "open_task_turn_context", open_turn)
    monkeypatch.setattr(grant_routes, "get_runtime_settings_store", lambda _config: SimpleNamespace(
        get_scope_settings=lambda scope, scope_id: dict(settings)
        if (scope, scope_id) == ("global", "*") else {}))
    api.client.app.include_router(grant_routes.router)
    api.settings = settings

    created = api.client.post("/v1/bot-tags", json={
        "slug": ROLE, "name": "Fixture Manager", "capabilities": [CAPABILITY]})
    assert created.status_code == 201, created.text
    for slug in (MANAGER, "fixture-target", "fixture-peer"):
        upsert(api.store, slug)
    assert api.patch(MANAGER, tags=[ROLE]).status_code == 200
    api.published.clear()
    api.reloads.clear()

    def as_person(slug, *, granted=True, reason="fixture reason", actor=PERSON, auth=BEARER):
        body = {"granted": granted, "reason": reason}
        if actor is not None:
            body["actor"] = actor
        return api.client.put(f"/v1/bot-grants/{GRANT}/bots/{slug}", json=body,
                              headers={"Authorization": auth} if auth else {})

    def as_bot(caller, slug, *, granted=True, tag_slug=str(GRANT), **extra):
        return api.client.put(f"/v1/bot-grants/{tag_slug}/bots/{slug}",
                              json={"granted": granted, "reason": "fixture reason", **extra},
                              headers={TASK_TURN_CONTEXT_HEADER: f"turn:{caller}"})

    # The fixture's own profile creation and role patch write audit rows too.
    baseline = api.rig.scalar("SELECT coalesce(max(id), 0) FROM bot_profile_changes")

    def changes(slug):
        with api.rig.engine.connect() as conn:
            return [dict(r) for r in conn.execute(text(
                "SELECT changed_by, change_note, changes FROM bot_profile_changes"
                " WHERE bot_slug = :s AND id > :baseline ORDER BY id"),
                {"s": slug, "baseline": baseline}).mappings()]

    api.as_person, api.as_bot, api.changes = as_person, as_bot, changes
    return api


def _code(response) -> str:
    return response.json()["detail"]["code"]


# --- the grant route ----------------------------------------------------------


def test_a_person_grants_with_provenance_and_an_audit_row(grant_api):
    response = grant_api.as_person("fixture-target")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["granted"] is True and body["changed"] is True
    assert body["granted_by"] == PERSON and body["granted_at"] is not None
    assert grant_api.rig.tags("fixture-target") == [str(GRANT)]
    assert grant_api.changes("fixture-target") == [{
        "changed_by": PERSON, "change_note": "fixture reason",
        "changes": {"tags": {"before": [], "after": [str(GRANT)]}}}]
    assert grant_api.published == ["fixture-target"] and grant_api.reloads == [True]

    listed = grant_api.client.get("/v1/bot-grants").json()["data"]
    assert [(g["tag"], g["bot"], g["granted_by"]) for g in listed] == [
        (str(GRANT), "fixture-target", PERSON)]


def test_a_repeated_grant_changes_nothing_and_a_revoke_is_audited(grant_api):
    first = grant_api.as_person("fixture-target").json()
    again = grant_api.as_person("fixture-target", actor="fixture-other").json()

    assert again["changed"] is False
    assert (again["granted_by"], again["granted_at"]) == (first["granted_by"], first["granted_at"])
    assert len(grant_api.changes("fixture-target")) == 1

    revoked = grant_api.as_person("fixture-target", granted=False).json()
    assert revoked["granted"] is False and revoked["changed"] is True
    assert grant_api.rig.tags("fixture-target") == []
    assert grant_api.changes("fixture-target")[-1]["changes"] == {
        "tags": {"before": [str(GRANT)], "after": []}}
    assert grant_api.client.get("/v1/bot-grants").json() == {"data": []}


@pytest.mark.parametrize("kwargs, status", [
    ({"auth": None}, 401),
    ({"auth": "Bearer wrong"}, 401),
    ({"actor": None}, 422),
    ({"actor": f"{grants.BOT_ACTOR_PREFIX}{MANAGER}"}, 422),
    ({"actor": " "}, 422),
    ({"reason": ""}, 422),
])
def test_a_refused_person_write_writes_nothing(grant_api, kwargs, status):
    response = grant_api.as_person("fixture-target", **kwargs)

    assert response.status_code == status, response.text
    assert grant_api.rig.tags("fixture-target") == []
    assert grant_api.changes("fixture-target") == []


def test_a_grant_manager_bot_grants_as_itself(grant_api):
    response = grant_api.as_bot(MANAGER, "fixture-target")

    assert response.status_code == 200, response.text
    assert response.json()["granted_by"] == f"bot:{MANAGER}"
    assert grant_api.changes("fixture-target")[0]["changed_by"] == f"bot:{MANAGER}"


def test_a_bot_cannot_grant_itself_even_as_a_manager(grant_api):
    response = grant_api.as_bot(MANAGER, MANAGER)

    assert response.status_code == 403 and _code(response) == "self_grant_refused"
    assert grant_api.rig.tags(MANAGER) == [ROLE]


def test_a_bot_without_the_manager_role_is_refused(grant_api):
    response = grant_api.as_bot("fixture-peer", "fixture-target")

    assert response.status_code == 403 and _code(response) == "not_grant_manager"
    assert grant_api.rig.tags("fixture-target") == []


def test_no_bot_manages_grants_while_the_setting_is_unset(grant_api):
    grant_api.settings.clear()

    response = grant_api.as_bot(MANAGER, "fixture-target")

    assert response.status_code == 403 and _code(response) == "not_grant_manager"
    # A person still can.
    assert grant_api.as_person("fixture-target").status_code == 200


def test_a_private_manager_is_not_a_manager(grant_api):
    assert grant_api.patch(MANAGER, tags=[ROLE, "private"]).status_code == 200

    response = grant_api.as_bot(MANAGER, "fixture-target")

    assert response.status_code == 403 and _code(response) == "not_grant_manager"


@pytest.mark.parametrize("target", ["fixture-hidden", "fixture-nobody"])
def test_a_private_and_an_unknown_target_get_the_same_refusal(grant_api, target):
    upsert(grant_api.store, "fixture-hidden")
    assert grant_api.patch("fixture-hidden", tags=["private"]).status_code == 200

    response = grant_api.as_bot(MANAGER, target)

    assert response.status_code == 403
    assert response.json()["detail"] == {"code": "access_denied", "message": "no such grantable bot"}


def test_a_bad_turn_capability_and_a_body_actor_are_refused(grant_api):
    bad = grant_api.client.put(f"/v1/bot-grants/{GRANT}/bots/fixture-target",
                               json={"granted": True, "reason": "x"},
                               headers={TASK_TURN_CONTEXT_HEADER: "forged"})
    spoofed = grant_api.as_bot(MANAGER, "fixture-target", actor=PERSON)

    assert bad.status_code == 403
    assert spoofed.status_code == 422
    assert grant_api.rig.tags("fixture-target") == []


@pytest.mark.parametrize("tag_slug", ["inactive", ROLE, "fixture_unknown"])
def test_only_a_grant_tag_can_be_set_here(grant_api, tag_slug):
    response = grant_api.as_bot(MANAGER, "fixture-target", tag_slug=tag_slug)

    assert response.status_code == 404
    assert grant_api.rig.tags("fixture-target") == []


def test_a_grant_needs_a_profile(grant_api):
    response = grant_api.as_person("fixture-gone")

    assert response.status_code == 404
    assert grant_api.rig.tags("fixture-gone") == []


# --- every other tag writer refuses to move a grant tag ------------------------


@pytest.mark.parametrize("wanted", [["inactive", str(GRANT)], [str(GRANT)]])
def test_the_profile_patch_cannot_add_a_grant_tag(grant_api, wanted):
    response = grant_api.patch("fixture-target", tags=wanted)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "grant_locked"
    assert response.json()["detail"]["tags"] == [str(GRANT)]
    assert grant_api.rig.tags("fixture-target") == []


def test_the_profile_patch_cannot_remove_a_grant_tag_but_may_keep_it(grant_api):
    grant_api.as_person("fixture-target")

    removed = grant_api.patch("fixture-target", tags=["inactive"])
    kept = grant_api.patch("fixture-target", tags=[str(GRANT), "inactive"])

    assert removed.status_code == 409 and removed.json()["detail"]["code"] == "grant_locked"
    assert kept.status_code == 200, kept.text
    assert grant_api.rig.tags("fixture-target") == sorted([str(GRANT), "inactive"])


def test_the_assignment_delete_cannot_revoke_a_grant_tag(grant_api):
    grant_api.as_person("fixture-target")

    response = grant_api.client.delete(f"/v1/bot-tags/{GRANT}/assignments/fixture-target")

    assert response.status_code == 409
    assert grant_api.rig.tags("fixture-target") == [str(GRANT)]


# --- holds_grant ---------------------------------------------------------------


def test_holds_grant_is_live_and_fails_closed_for_private_and_unknown_bots(grant_api):
    engine = grant_api.rig.engine

    def holds(slug):
        with engine.connect() as conn:
            return grants.holds_grant(conn, slug, GRANT)

    assert holds("fixture-target") is False
    grant_api.as_person("fixture-target")
    assert holds("fixture-target") is True
    assert holds("fixture-nobody") is False

    with engine.begin() as conn:
        tag(conn, "fixture-target", BehaviorTag.PRIVATE)
    assert holds("fixture-target") is False
