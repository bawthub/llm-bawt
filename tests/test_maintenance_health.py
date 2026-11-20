"""TASK-1704 slice A: maintenance/utility call failures are recorded loudly."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_bawt.service import maintenance_health as mh
from llm_bawt.service.maintenance_health import MaintenanceHealth, MonitoredClient, classify_failure


class _StatusError(Exception):
    def __init__(self, status: int, text: str = ""):
        super().__init__(text or f"Error code: {status}")
        self.status_code = status


@pytest.mark.parametrize("error, expected", [
    (_StatusError(403, "Error code: 403 - {'code': 'personal-team-blocked:spending-limit'}"), mh.SPENDING_LIMIT),
    (_StatusError(402), mh.SPENDING_LIMIT),
    (_StatusError(429, "rate limited"), mh.RATE_LIMIT),
    (_StatusError(403, "forbidden"), mh.AUTH),
    (_StatusError(401), mh.AUTH),
    (_StatusError(503), mh.UNAVAILABLE),
    (TimeoutError("read timed out"), mh.TIMEOUT),
    (ValueError("weird"), mh.UNKNOWN),
    ("ERROR: OpenAI API Error - Error code: 403 - {'code': 'personal-team-blocked:spending-limit'}", mh.SPENDING_LIMIT),
    ("ERROR: OpenAI API Error - Error code: 429 - slow down", mh.RATE_LIMIT),
])
def test_classifier_reads_code_before_status(error, expected):
    assert classify_failure(error) == expected


def _client(health, inner):
    return MonitoredClient(inner, job="history_summarization", model="grok-4.3",
                           provider="xai", credential="Default", health=health)


def _only(health):
    return health.checks_payload()["checks"]


def test_repeated_failures_collapse_into_one_row_and_success_clears_it():
    health = MaintenanceHealth()
    inner = MagicMock()
    inner.query.side_effect = _StatusError(403, "personal-team-blocked:spending-limit")
    client = _client(health, inner)
    for _ in range(3):
        with pytest.raises(_StatusError):
            client.query(messages=[])
    rows = _only(health)
    assert len(rows) == 1 and rows[0]["status"] == "down"
    reason = rows[0]["reason"]
    assert rows[0]["name"] == "history_summarization"
    for part in ("spending_limit", "xai/Default", "grok-4.3", "3x since"):
        assert part in reason

    inner.query.side_effect = None
    inner.query.return_value = "a summary"
    assert client.query(messages=[]) == "a summary"
    assert _only(health) == [{"name": "history_summarization", "status": "ok", "reason": None}]


def test_error_string_and_empty_replies_count_as_failures():
    health = MaintenanceHealth()
    inner = MagicMock()
    inner.query_bounded.return_value = "ERROR: OpenAI API Error - Error code: 429 - slow"
    assert _client(health, inner).query_bounded(messages=[]).startswith("ERROR:")
    assert "rate_limit" in _only(health)[0]["reason"]
    inner.query_bounded.return_value = "   "
    _client(health, inner).query_bounded(messages=[])
    assert "empty_response" in _only(health)[0]["reason"]


def test_proxy_delegates_other_attributes():
    inner = SimpleNamespace(model="m", query=lambda **_: "x")
    client = MonitoredClient(inner, job="j", model=None, provider=None, credential=None,
                             health=MaintenanceHealth())
    assert client.model == "m" and client.wrapped is inner


def test_background_client_without_usable_client_is_a_recorded_failure(monkeypatch):
    from llm_bawt.service.instance_manager import InstanceManagerMixin

    health = MaintenanceHealth()
    monkeypatch.setattr(mh, "_registry", health)
    svc = InstanceManagerMixin.__new__(InstanceManagerMixin)
    svc.config = SimpleNamespace()
    with patch.object(InstanceManagerMixin, "_resolve_background_endpoint", return_value=(None, None, None)), \
         patch.object(InstanceManagerMixin, "_background_credential_labels", return_value=(None, None)):
        assert svc._get_background_client("grok-4.3", job="memory_extraction") == (None, None)
    row = _only(health)[0]
    assert row["name"] == "memory_extraction" and row["status"] == "down"
    assert "unavailable" in row["reason"] and "grok-4.3" in row["reason"]


def test_background_client_is_wrapped_with_job_labels(monkeypatch):
    from llm_bawt.service.instance_manager import InstanceManagerMixin

    health = MaintenanceHealth()
    monkeypatch.setattr(mh, "_registry", health)
    svc = InstanceManagerMixin.__new__(InstanceManagerMixin)
    inner = MagicMock()
    inner.query.return_value = "ok"
    with patch.object(InstanceManagerMixin, "_resolve_background_endpoint",
                      return_value=(inner, "grok-4.3", {})), \
         patch.object(InstanceManagerMixin, "_background_credential_labels", return_value=("xai", "Default")):
        client, alias = svc._get_background_client("grok-4.3", job="profile_maintenance")
    assert alias == "grok-4.3" and client.wrapped is inner
    client.query(messages=[])
    assert _only(health) == [{"name": "profile_maintenance", "status": "ok", "reason": None}]


def test_429_is_rate_limit_even_when_text_mentions_spending():
    err = _StatusError(429, "Error code: 429 - {'code': 'team:spending-limit'} credit balance low")
    assert classify_failure(err) == mh.RATE_LIMIT
    assert classify_failure(_StatusError(400, "your credit balance is low")) == mh.UNKNOWN


def test_raw_error_text_never_reaches_reason_or_log(caplog):
    secret = "xai-SUPERSECRETKEY123"
    health = MaintenanceHealth()
    inner = MagicMock()
    inner.query.side_effect = _StatusError(
        401, f"Error code: 401 - {{'code': 'invalid_api_key', 'error': 'bad key {secret}'}}")
    with pytest.raises(_StatusError):
        _client(health, inner).query(messages=[])
    reason = _only(health)[0]["reason"]
    assert "HTTP 401" in reason and "invalid_api_key" not in reason
    inner.query.side_effect = None
    inner.query.return_value = f"ERROR: request echoed {secret}"
    _client(health, inner).query(messages=[])
    inner.query.side_effect = ValueError(f"request echoed {{'code':'{secret}'}}")
    with pytest.raises(ValueError):
        _client(health, inner).query(messages=[])
    assert secret not in str(health.checks_payload()) and secret not in caplog.text


def test_unset_utility_model_is_shown_but_not_a_failure(monkeypatch):
    from fastapi import HTTPException
    from llm_bawt.service import utility_completion as uc

    health = MaintenanceHealth()
    monkeypatch.setattr(mh, "_registry", health)
    monkeypatch.setattr(uc, "get_service", lambda: SimpleNamespace(config=SimpleNamespace()))
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda *_a, **_k: None)
    with pytest.raises(HTTPException):
        uc.complete_utility(SimpleNamespace(model=None))
    assert _only(health) == [{
        "name": "utility_completion", "status": "ok",
        "reason": "not configured (maintenance_model is not set); off until a model is set",
    }]


@pytest.fixture
def unconfigured_jobs(monkeypatch):
    from llm_bawt.service.background_tasks import BackgroundTasksMixin
    from llm_bawt.service.instance_manager import InstanceManagerMixin

    class Service(BackgroundTasksMixin, InstanceManagerMixin):
        pass

    health = MaintenanceHealth()
    monkeypatch.setattr(mh, "_registry", health)
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda *_a, **_k: None)
    svc = Service.__new__(Service)
    svc.config = SimpleNamespace(resolve_summarization_job=dict,
                                 resolve_context_budget=lambda _alias: (0, 0, 1000))
    svc._default_bot = "test-bot"
    svc._llm_bawt_cache = {}
    return svc, health


def _assert_unconfigured(health, job, fallback):
    row = next(row for row in _only(health) if row["name"] == job)
    effect = "using the built-in fallback" if fallback else "off until a model is set"
    assert row == {"name": job, "status": "ok",
                   "reason": f"not configured (maintenance_model is not set); {effect}"}


@pytest.mark.parametrize("fallback", [False, True])
def test_scheduled_history_respects_heuristics_and_summary_extraction_skips(
    monkeypatch, unconfigured_jobs, fallback,
):
    svc, health = unconfigured_jobs
    summarizer = MagicMock()
    summarizer.return_value.summarize_eligible_sessions.return_value = {
        "results": [{"created": True, "summary": "A completed summary"}],
    }
    monkeypatch.setattr("llm_bawt.memory.summarization.HistorySummarizer", summarizer)
    monkeypatch.setattr("llm_bawt.bots.BotManager", MagicMock())
    monkeypatch.setattr("llm_bawt.runtime_settings.RuntimeSettingsResolver", MagicMock())
    task = SimpleNamespace(bot_id="test-bot", user_id="system", payload={"use_heuristic_fallback": fallback})
    result = svc._process_history_summarization(task)
    assert result["extraction"]["extraction_method"] == "skipped"
    assert summarizer.return_value.summarize_eligible_sessions.call_args.kwargs["use_heuristic_fallback"] is fallback
    _assert_unconfigured(health, "history_summarization", fallback)
    _assert_unconfigured(health, "summary_fact_extraction", False)


@pytest.mark.parametrize("route", ["summarize_history", "rebuild_history_summaries"])
@pytest.mark.parametrize("fallback", [False, True])
def test_manual_history_respects_heuristic_opt_in(monkeypatch, unconfigured_jobs, route, fallback):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llm_bawt.service.routes import history_summaries

    svc, health = unconfigured_jobs
    summarizer = MagicMock()
    summarizer.summarize_eligible_sessions.return_value = {}
    summarizer.rebuild_recent_sessions.return_value = {}
    monkeypatch.setattr("llm_bawt.memory.summarization.HistorySummarizer", lambda *_a, **_k: summarizer)
    # Prompt construction is irrelevant when no client can render or send one.
    monkeypatch.setattr("llm_bawt.prompt_registry.PromptResolver", MagicMock())
    monkeypatch.setattr(history_summaries, "get_service", lambda: svc)
    app = FastAPI()
    app.post("/summary")(getattr(history_summaries, route))
    with TestClient(app) as client:
        # Exercise the actual FastAPI default as well as explicit opt-in.
        response = client.post("/summary", params={"use_heuristic": "true"} if fallback else {})
    assert response.status_code == 200
    summarize = (summarizer.summarize_eligible_sessions if route == "summarize_history"
                 else summarizer.rebuild_recent_sessions)
    assert summarize.call_args.kwargs["use_heuristic_fallback"] is fallback
    _assert_unconfigured(health, "history_summary_rebuild", fallback)


def test_failure_is_attributed_to_model_attempted_even_if_settings_change(monkeypatch, unconfigured_jobs):
    svc, health = unconfigured_jobs
    settings = {"extraction_model": "oauth-only-model"}
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda _config, key: settings.get(key))

    def lookup(_config, alias):
        assert alias == "oauth-only-model"
        settings.clear()
        return None

    monkeypatch.setattr("llm_bawt.service.background_clients.resolve_background_model", lookup)
    assert svc._get_background_client(job="summary_fact_extraction") == (None, None)
    row = _only(health)[0]
    assert row["status"] == "down" and "oauth-only-model" in row["reason"]


def test_success_after_unconfigured_clears_the_note():
    health = MaintenanceHealth()
    health.record_unconfigured("history_summarization", "not configured")
    inner = MagicMock()
    inner.query.return_value = "a summary"
    _client(health, inner).query(messages=[])
    assert _only(health) == [{"name": "history_summarization", "status": "ok", "reason": None}]


def test_route_serves_aggregator_shape(monkeypatch):
    from llm_bawt.service.routes.health import get_maintenance_health

    health = MaintenanceHealth()
    health.record_failure("j", mh.AUTH, "401", model="m", provider="xai", credential="Default")
    monkeypatch.setattr(mh, "_registry", health)
    from llm_bawt.service import maintenance_fallback as mf
    from llm_bawt.service.routes import health as health_route
    monkeypatch.setattr(mf, "default_store", lambda _config: mf.MemoryExhaustionStore())
    monkeypatch.setattr(health_route, "get_service", lambda: SimpleNamespace(config=None))
    payload = get_maintenance_health()
    assert payload["checks"][0]["status"] == "down"
