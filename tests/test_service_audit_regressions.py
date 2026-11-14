"""TASK-2250: offline cache, queue, continuity and fallback regressions."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.health_checks import DEGRADED, DOWN, check_ops_queue
from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin
from llm_bawt.service.session_lifecycle.execution import NewSessionExecution
from tests.test_background_job_model_resolution import _make_service as background_service
from tests.test_health_checks import NOW, _job
from tests.test_instance_cache_local_mode import _fake_instance, _make_service
from tests.test_inter_bot_steer_generation import (
    BASE_ID, TARGET, _Bridge, _dispatcher, _steer_until_settled,
)
from tests.test_maintenance_fallback import PRIMARY_PROVIDER, Rig, _spend, _StatusError
from tests.test_reset_continuity import counting_llm, resolver, service_with


def test_catalog_reload_retires_all_pools_without_closing_borrowed_clients():
    svc = _make_service()
    old = _fake_instance(local_mode=False)
    svc._llm_bawt_cache[("test-model", "fixture", "user", False)] = old
    svc._client_cache["test-model"] = old.client
    svc._bg_client_cache["test-model"] = Mock()
    assert svc.invalidate_model_catalog() == 1
    assert svc._llm_bawt_cache == svc._client_cache == svc._bg_client_cache == {}
    old.client.close.assert_not_called()
    with patch("llm_bawt.service.core.ServiceLLMBawt", return_value=_fake_instance(False)) as build:
        svc._get_llm_bawt("test-model", "fixture", "user")
    assert build.call_args.kwargs["existing_client"] is None


def test_catalog_reload_during_chat_build_cannot_repopulate_new_generation():
    svc = _make_service()
    built = _fake_instance(False)

    def build(**_kwargs):
        svc.invalidate_model_catalog()
        return built

    with patch("llm_bawt.service.core.ServiceLLMBawt", side_effect=build):
        assert svc._get_llm_bawt("test-model", "fixture", "user") is built
    assert svc._llm_bawt_cache == svc._client_cache == {}


@pytest.mark.parametrize("reload_during_build", [False, True])
def test_catalog_reload_rebuilds_same_background_endpoint(monkeypatch, reload_during_build):
    svc = background_service({"type": "openai", "model_key": "model", "model_id": "old"})
    svc._llm_bawt_cache = {}
    svc._client_cache = {}
    clients = [object(), object()]
    builds = []

    def build(_config, definition, _alias):
        builds.append(definition["model_id"])
        if reload_during_build and len(builds) == 1:
            svc.invalidate_model_catalog()
        return clients[len(builds) - 1], "test"

    monkeypatch.setattr("llm_bawt.service.background_clients.build_background_client", build)
    assert svc._resolve_background_endpoint("model")[0] is clients[0]
    if reload_during_build:
        assert svc._bg_client_cache == {}
    else:
        assert svc._resolve_background_endpoint("model")[0] is clients[0]
        svc.invalidate_model_catalog()
    svc.config.resolve_model.return_value = {"type": "openai", "model_key": "model", "model_id": "new"}
    assert svc._resolve_background_endpoint("model")[0] is clients[1]
    assert builds == ["old", "new"]


def test_reload_route_uses_catalog_invalidation(monkeypatch):
    from llm_bawt.service.routes import models
    svc = _make_service()
    svc.invalidate_model_catalog = Mock(return_value=3)
    svc._load_available_models = Mock()
    monkeypatch.setattr(models, "get_service", lambda: svc)
    monkeypatch.setattr("llm_bawt.service.dependencies.get_model_catalog_engine", lambda _: object())
    monkeypatch.setattr("llm_bawt.memory.model_catalog_migration.migrate_model_catalog", lambda _: None)
    monkeypatch.setattr("llm_bawt.model_catalog.ModelCatalogStore", lambda _: SimpleNamespace(load=lambda: []))
    assert models.reload_models_catalog()["cleared_instances"] == 3
    svc.invalidate_model_catalog.assert_called_once_with()


def test_explicit_single_endpoint_ref_is_not_rejected_as_missing(monkeypatch):
    from llm_bawt.model_catalog import ModelNotFoundError
    from tests.test_model_consolidation import _make_manager, _patch_bot_manager
    catalog = ModelCatalog([ModelEndpoint(
        1, ModelIdentity(1, "chosen", "openai", "Chosen"),
        AccessPath(1, "fixture-path", "openai", "chat-completions", "http://fixture", "api-key"),
        "upstream",
    )])
    bot = SimpleNamespace(slug="fixture", agent_backend=None, harness="chat", default_model="chosen", endpoint_id=None)
    _patch_bot_manager(monkeypatch, bot)
    svc = _make_manager(defined_models=catalog.compatibility_mapping(), available=["chosen"], agent_backend_models={})

    def resolve(ref, harness=None, default=None):
        try:
            return catalog.resolve(ref, harness=harness)
        except ModelNotFoundError:
            return default

    svc.config.resolve_model = resolve
    assert "chosen@fixture-path" not in svc._available_models
    assert svc._resolve_request_model("chosen@fixture-path", "fixture", False) == ("chosen@fixture-path", [])
    assert svc._resolve_request_model("missing", "fixture", False)[1]


@pytest.mark.parametrize("state,old,expected", [
    ("QUEUED", _job("old", error="blocked", submitted=1000), DOWN),
    ("QUEUED", _job("old", submitted=1000), DEGRADED),
    ("RUNNING", _job("old", submitted=1000, reconciled=1000), DEGRADED),
    ("ACCEPTED", _job("old", submitted=1000), DEGRADED),
    ("DISPATCHING", _job("old", submitted=1000), DEGRADED),
])
def test_queue_health_includes_jobs_older_than_display_limit(state, old, expected):
    jobs = [_job("fresh", submitted=1) for _ in range(50)] + [old]
    target = state.lower()
    store = SimpleNamespace(engine=object())
    store.list_jobs = lambda *, state, limit: jobs[:limit] if state == target else []
    result = check_ops_queue(store, NOW)
    assert result["status"] == expected
    assert "old" in result["reason"]


def test_accepted_unpersisted_steer_stops_at_budget_without_new_generation(monkeypatch):
    bridge = _Bridge(persisted=[False] * 20)
    dispatcher = _dispatcher(monkeypatch, bridge)
    result = _steer_until_settled(dispatcher)
    assert result.status == "FAILED"
    assert result.attempt_count == result.max_attempts == 5
    assert bridge.calls == [(TARGET, BASE_ID)] * 5
    assert bridge.fresh == 1
    assert dispatcher.store.delays == [1, 2, 4, 8]
    assert result.transport_accepted_at is not None
    assert result.delivery_mode == "steer"
    assert "steered_persistence_failed" in result.last_error
    assert "do not resend" in result.last_error


@pytest.mark.parametrize("summarized,expected", [(False, "inline+summaries"), (True, "summaries")])
def test_retained_reset_carries_timeout_fallback_into_seed(monkeypatch, summarized, expected):
    llm, reads = counting_llm("summaries")
    service = service_with({})
    service._maybe_summarize_on_new = Mock(return_value=summarized)
    seed = Mock(return_value={"messages": []})
    monkeypatch.setattr("llm_bawt.service.routes.history.build_context_seed", seed)
    result = NewSessionExecution.prepare_agent(
        service, llm, SimpleNamespace(inter_bot_session_policy="reset_retain_history", inter_bot_seed_session_id="old"),
        bot_id="fixture", user_prompt="task", model_alias="m", thread_binding={"thread_session_id": "new"},
        delivery_id=None, backend_name="claude-code", is_agent=False,
    )
    assert seed.call_args.kwargs["history_scope"] == result.history_scope == expected
    assert service._maybe_summarize_on_new.call_args.kwargs["history_scope"] == "summaries"
    assert reads.count("history_scope") == 1


@pytest.mark.parametrize("result,expected", [
    ({"errors": ["provider unavailable"], "summaries_created": 0}, False),
    (RuntimeError("provider unavailable"), False),
    ({"errors": [], "summaries_created": 0}, True),
    ({"errors": [], "summaries_created": 1}, True),
])
def test_finished_summary_worker_reports_success_not_just_completion(monkeypatch, result, expected):
    worker = Mock()
    if isinstance(result, Exception):
        worker.summarize_thread.side_effect = result
    else:
        worker.summarize_thread.return_value = result
    monkeypatch.setattr("llm_bawt.memory.summarization.HistorySummarizer", lambda *a, **k: worker)
    monkeypatch.setattr("llm_bawt.memory.summarization_limits.resolve_summarization_limits", lambda *a: (1000, 100))
    llm = SimpleNamespace(config_resolver=resolver("summaries"), history_manager=SimpleNamespace(_db_backend=object()))
    svc = SimpleNamespace(config=None, _get_background_client=lambda **k: (None, None))
    assert BridgeSessionResetMixin._maybe_summarize_on_new(
        svc, llm, "fixture", "/new", thread_binding={"thread_session_id": "old"}, history_scope="summaries",
    ) is expected


@pytest.mark.parametrize("failure", ["ERROR: 401 Unauthorized", "", "  ", _StatusError(401)])
def test_failed_maintenance_probe_preserves_exhaustion_and_releases_claim(failure):
    rig = Rig()
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "backup"
    client = rig.client()
    assert client.query() == "backup"
    previous = rig.store.get(PRIMARY_PROVIDER)
    rig.clock.now += 601
    rig.primary_inner.query.side_effect = failure if isinstance(failure, Exception) else None
    rig.primary_inner.query.return_value = failure
    if isinstance(failure, Exception):
        with pytest.raises(type(failure)):
            client.query()
    else:
        assert client.query() == failure
    retained = rig.store.get(PRIMARY_PROVIDER)
    assert retained is not None and retained.since == previous.since
    assert retained.until == rig.clock.now + rig.cooldown
    assert rig.store.episodes() == []
    assert client.query() == "backup"
    rig.clock.now += 601
    rig.primary_inner.query.side_effect = None
    rig.primary_inner.query.return_value = "recovered"
    assert client.query() == "recovered"
    assert rig.store.get(PRIMARY_PROVIDER) is None
    assert len(rig.store.episodes()) == 1
