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
    with patch.object(InstanceManagerMixin, "_resolve_background_client", return_value=(None, None)), \
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
    with patch.object(InstanceManagerMixin, "_resolve_background_client", return_value=(inner, "grok-4.3")), \
         patch.object(InstanceManagerMixin, "_background_credential_labels", return_value=("xai", "Default")):
        client, alias = svc._get_background_client(job="profile_maintenance")
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
    assert "HTTP 401" in reason and "code invalid_api_key" in reason
    inner.query.side_effect = None
    inner.query.return_value = f"ERROR: request echoed {secret}"
    _client(health, inner).query(messages=[])
    assert secret not in str(health.checks_payload()) and secret not in caplog.text


def test_unset_model_is_a_recorded_failure(monkeypatch):
    from fastapi import HTTPException
    from llm_bawt.service import utility_completion as uc

    health = MaintenanceHealth()
    monkeypatch.setattr(mh, "_registry", health)
    monkeypatch.setattr(uc, "get_service", lambda: SimpleNamespace(config=SimpleNamespace()))
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda *_a, **_k: None)
    with pytest.raises(HTTPException):
        uc.complete_utility(SimpleNamespace(model=None))
    row = _only(health)[0]
    assert row["name"] == "utility_completion" and "unconfigured" in row["reason"]
    assert "maintenance_model is not set" in row["reason"]


def test_route_serves_aggregator_shape(monkeypatch):
    from llm_bawt.service.routes.health import get_maintenance_health

    health = MaintenanceHealth()
    health.record_failure("j", mh.AUTH, "401", model="m", provider="xai", credential="Default")
    monkeypatch.setattr(mh, "_registry", health)
    payload = get_maintenance_health()
    assert payload["checks"][0]["status"] == "down"
