"""TASK-2543: manual, subject-specific summary saves, not fact curation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from llm_bawt.profiles import STATUS_CANDIDATE, EntityProfile, EntityType, ProfileAttribute, ProfileManager
from llm_bawt.service import profile_summary
from llm_bawt.service.routes import profiles, service_auth

U = EntityType.USER
PATH = "/v1/internal/profile-summary/regenerate"


@pytest.fixture
def setup(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine, tables=[EntityProfile.__table__, ProfileAttribute.__table__])
    manager = ProfileManager.__new__(ProfileManager)
    manager.engine = engine
    for entity in ("fixture", "other"):
        manager.get_or_create_profile(U, entity)
        manager.update_profile(U, entity, display_name="Avery", summary="Previous summary")
    manager.set_attribute(U, "fixture", "interest", "hobby", "Garden walks", source="explicit")
    manager.set_attribute(U, "other", "interest", "hobby", "Other profile detail", source="explicit")
    model = Mock()
    model.query_bounded.return_value = '```json\n{"interests": "Enjoys garden walks."}\n```'
    service = SimpleNamespace(config=object(), _get_background_client=lambda **_: (model, "configured"))
    monkeypatch.setattr(profiles, "get_service", lambda: service)
    monkeypatch.setattr(profiles, "get_profile_manager", lambda _: manager)
    monkeypatch.setattr(profile_summary, "get_profile_manager", lambda _: manager)
    monkeypatch.setattr(profile_summary, "resolve_job_model", lambda *_: "configured")
    mark = Mock()
    monkeypatch.setattr(profile_summary, "mark", mark)
    monkeypatch.setattr(service_auth, "service_token_source", lambda: SimpleNamespace(check=lambda header:
        service_auth.TokenCheck.OK if header == "Bearer fixture-only" else service_auth.TokenCheck.INVALID))
    app = FastAPI()
    app.include_router(profiles.router)
    with TestClient(app) as client:
        yield client, manager, model, mark
    engine.dispose()


def regenerate(client, entity="fixture"):
    return client.post(PATH, headers={"Authorization": "Bearer fixture-only"}, json={"entity_id": entity})


def test_zero_candidates_regenerates_and_reloads_only_the_target_summary(setup):
    client, manager, model, mark = setup
    before = manager.get_profile(U, "fixture")
    assert manager.attributes_with_status(U, "fixture", (STATUS_CANDIDATE,)) == []
    response = regenerate(client)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    saved = response.json()
    assert saved["summary"] == "Interests: Enjoys garden walks."
    reloaded = client.get("/v1/profiles/user/fixture").json()
    assert reloaded["summary"] == saved["summary"]
    assert reloaded["summary_updated_at"] == saved["summary_updated_at"].replace("Z", "+00:00")
    assert manager.get_profile(U, "fixture").summary_updated_at > before.summary_updated_at
    assert manager.get_profile(U, "other").summary == "Previous summary"
    assert reloaded["display_name"] == "Avery"
    assert [attribute["value"] for attribute in reloaded["attributes"]] == ["Garden walks"]
    assert manager.attributes_with_status(U, "fixture", (STATUS_CANDIDATE,)) == []
    prompt = model.query_bounded.call_args.args[0][0].content
    assert "Garden walks" in prompt and "Other profile detail" not in prompt and "Previous summary" not in prompt
    mark.assert_called_once()


@pytest.mark.parametrize("raw", ["failed", "{}", '{"name": []}'])
def test_invalid_provider_result_keeps_last_good_summary_and_timestamp(setup, raw):
    client, manager, model, mark = setup
    old = manager.get_profile(U, "fixture")
    model.query_bounded.return_value = raw
    assert regenerate(client).status_code == 502
    assert manager.get_profile(U, "fixture").summary == old.summary
    assert manager.get_profile(U, "fixture").summary_updated_at == old.summary_updated_at
    mark.assert_not_called()


def test_concurrent_edit_wins(setup):
    client, manager, model, mark = setup

    def complete_after_edit(*_, **__):
        manager.update_profile(U, "fixture", summary="Newer manual edit")
        return '{"interests": "Generated from the previous revision."}'

    model.query_bounded.side_effect = complete_after_edit
    assert regenerate(client).status_code == 409
    assert manager.get_profile(U, "fixture").summary == "Newer manual edit"
    mark.assert_not_called()


def test_missing_profile_and_untrusted_call_do_not_run_model(setup):
    client, manager, model, mark = setup
    assert client.post(PATH, json={"entity_id": "fixture"}).status_code == 401
    assert regenerate(client, "not-created").status_code == 404
    assert manager.get_profile(U, "not-created") is None
    model.query_bounded.assert_not_called()
    mark.assert_not_called()
