"""Accounting tests: unknown is not zero; context isn't cumulative spend."""
import pytest

from llm_bawt.service.task_retro_accounting import estimated_quota_share, usage_facts, usage_totals


def test_cumulative_requests_not_resident_context_and_no_session_cost():
    facts = usage_facts({
        "input_tokens": 2, "cache_read_tokens": 199000,
        "session_cost_usd": 99, "output_tokens": 456,
        "turn_request_totals": {"input_tokens": 3000000, "cache_read_tokens": 2000000,
                                "cache_creation_tokens": 500000, "requests": 20},
        "access_path": {"vendor": "anthropic", "billing_mode": "subscription"},
    })
    assert facts["cost_usd"] is None
    assert facts["fresh_input_tokens"] == 1000000
    assert facts["cache_read_tokens"] == 2000000
    assert facts["cache_write_tokens"] == 500000
    assert facts["model_requests"] == 20
    assert facts["output_tokens"] == 456
    assert facts["billing_mode"] == "subscription"


@pytest.mark.parametrize("raw", [None, "", "{", "null", "[]", [],
                                {"input_tokens": 199000, "cache_read_tokens": 190000}])
def test_old_or_missing_request_totals_remain_unmeasured(raw):
    facts = usage_facts(raw)
    assert facts["fresh_input_tokens"] is None
    assert facts["cache_read_tokens"] is None
    assert facts["model_requests"] is None


@pytest.mark.parametrize("invalid", [None, True, -1, float("nan"), float("inf"), "oops"])
def test_bad_cost_is_not_zero(invalid):
    assert usage_facts({"total_cost_usd": invalid})["cost_usd"] is None


def test_partial_sums_and_separate_coverage():
    totals = usage_totals([usage_facts({"total_cost_usd": "0.10", "output_tokens": 0}),
                           usage_facts({"total_cost_usd": "0.20"}), usage_facts(None)])
    assert totals["cost_usd"] == "0.30"
    assert totals["output_tokens"] == 0
    assert totals["fresh_input_tokens"] is None
    assert totals["coverage_pct"]["cost_usd"] == 66.7
    assert totals["coverage_pct"]["output_tokens"] == 33.3
    assert totals["coverage_pct"]["fresh_input_tokens"] == 0
    assert usage_totals([])["coverage_pct"]["cost_usd"] is None


def test_impossible_request_counters_not_counted():
    facts = usage_facts({"turn_request_totals": {
        "input_tokens": 100, "cache_read_tokens": 90, "cache_creation_tokens": 20}})
    assert facts["fresh_input_tokens"] is None
    assert facts["cache_read_tokens"] is None


def test_quota_share_estimate_not_fraction_of_bill_or_current_utc_week():
    assert estimated_quota_share("12", "120", "65") == 6.5
    assert estimated_quota_share("0", "120", "65") == 0
    for args in [(None, 120, 65), (12, 0, 65), (121, 120, 65), (12, 120, None), (12, 120, 101)]:
        assert estimated_quota_share(*args) is None


@pytest.mark.parametrize("extra", [
    {}, {"start": "2026-11-01T00:00:00Z"},
    {"start": "2026-11-01T00:00:00Z", "end": "2026-11-09T00:00:00Z"},
    {"start": "2026-11-01T00:00:00", "end": "2026-11-02T00:00:00"},
    {"task_refs": []}, {"task_refs": ["TASK-1"], "links": [{}]},
])
def test_read_route_rejects_unbounded_or_invalid_input(extra):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llm_bawt.service.routes.task_retros import router
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/v1/task-retros/read", json={"user_id": "fixture-user", **extra})
    assert response.status_code == 422


def test_quota_uses_only_existing_snapshot_in_its_actual_reset_window(monkeypatch):
    from types import SimpleNamespace

    from llm_bawt.service import usage
    from llm_bawt.service.task_retro_read import cached_claude_quota

    snapshot = SimpleNamespace(status="ok", fetched_at=1_000_000, limits=[
        SimpleNamespace(id="weekly_all", resets_at=1_100_000, used_pct=65),
    ])
    calls = []

    def cached(provider):
        calls.append(provider)
        return snapshot

    monkeypatch.setattr(usage, "get_cached", cached)
    assert cached_claude_quota() == {
        "start": 495200, "end": 1100000, "fetched_at": 1000000, "utilization_pct": 65,
    }
    snapshot.fetched_at = 1_100_000
    assert cached_claude_quota() is None
    snapshot.fetched_at = 495199
    assert cached_claude_quota() is None
    snapshot.status = "error"
    assert cached_claude_quota() is None
    assert calls == ["claude"] * 4
