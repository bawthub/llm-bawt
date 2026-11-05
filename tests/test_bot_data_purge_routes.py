"""Both destructive purge routes go through the guarded executor (TASK-2045)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt import bot_purge_execute as ex
from llm_bawt.service.routes import settings

DIGEST = "a" * 64


@pytest.fixture
def env(monkeypatch):
    service = SimpleNamespace(config=object())
    engine = object()
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", lambda config: engine)
    store = SimpleNamespace(engine=MagicMock(), get=Mock(return_value=object()), delete=Mock(return_value=True))
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda config: store)
    monkeypatch.setattr("llm_bawt.bot_slug_aliases.aliases_for", lambda conn, slug: [])
    effects = {}
    for name in ("_reload_bot_registry", "_reload_service_model_catalog",
                 "_invalidate_bot_instance_cache", "_clear_session_model_overrides"):
        effects[name] = Mock()
        monkeypatch.setattr(settings, name, effects[name])
    published = AsyncMock()
    monkeypatch.setattr(settings, "_publish_bot_profile_changed", published)
    execute = Mock(return_value={"slug": "source", "deleted_rows": {"sessions": 3}, "profile_deleted": True})
    monkeypatch.setattr(ex, "execute_bot_purge", execute)
    app = FastAPI()
    app.include_router(settings.router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, effects=effects, published=published,
                              execute=execute, engine=engine)


def test_combined_delete_runs_one_guarded_transaction_then_refreshes(env):
    response = env.client.delete(f"/v1/bots/SOURCE/profile?purge=true&manifest_digest={DIGEST}")
    assert response.status_code == 200
    assert response.json() == {"success": True, "slug": "source", "purge": env.execute.return_value}
    env.execute.assert_called_once_with(env.engine, "source", manifest_digest=DIGEST, delete_profile=True)
    env.store.delete.assert_not_called()  # the executor deletes the profile in its transaction
    for effect in env.effects.values():
        effect.assert_called_once()
    env.published.assert_awaited_once()


def test_standalone_purge_requires_digest_and_keeps_profile(env):
    response = env.client.post(f"/v1/bots/source/purge-data?manifest_digest={DIGEST}")
    assert response.status_code == 200
    env.execute.assert_called_once_with(env.engine, "source", manifest_digest=DIGEST, delete_profile=False)
    env.store.delete.assert_not_called()


@pytest.mark.parametrize("error,status,code", [
    (ex.PurgeStale("A manifest_digest from a fresh purge preview is required", code="preview_required"), 409, "preview_required"),
    (ex.PurgeStale("changed"), 409, "stale_preview"),
    (ex.PurgeLocked("running", code="active"), 423, "active"),
    (ex.PurgeLocked("reserved", code="protected"), 423, "protected"),
    (ex.PurgeRefused("missing", code="not_found"), 404, "not_found"),
    (ex.PurgeRefused("aliases", code="aliases"), 409, "aliases"),
    (ex.PurgeUnavailable("PostgreSQL is unavailable"), 503, "unavailable"),
])
@pytest.mark.parametrize("path,method", [
    ("/v1/bots/source/profile?purge=true", "delete"),
    ("/v1/bots/source/purge-data", "post"),
])
def test_refusals_map_to_contract_and_leave_caches_alone(env, error, status, code, path, method):
    env.execute.side_effect = error
    response = getattr(env.client, method)(path)
    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["code"] == code and "Nothing was deleted" in detail["message"]
    env.store.delete.assert_not_called()
    for effect in env.effects.values():
        effect.assert_not_called()
    env.published.assert_not_awaited()


def test_legacy_purge_without_digest_reaches_the_guard(env):
    env.execute.side_effect = ex.PurgeStale("required", code="preview_required")
    assert env.client.delete("/v1/bots/source/profile?purge=true").status_code == 409
    assert env.client.post("/v1/bots/source/purge-data").status_code == 409
    assert all(call.kwargs["manifest_digest"] is None for call in env.execute.call_args_list)


def test_profile_only_delete_is_unchanged(env):
    response = env.client.delete("/v1/bots/source/profile")
    assert response.status_code == 200
    assert response.json() == {"success": True, "slug": "source"}
    env.execute.assert_not_called()
    # TASK-2044: without a purge the data is retained, so privacy tags stay.
    env.store.delete.assert_called_once_with("source", drop_privacy_tags=False)
