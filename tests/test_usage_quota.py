"""Tri-state quota from cached usage (TASK-1249, design Blocker 3).

``available`` needs an ok, fresh snapshot whose every applicable window has a
known percentage under 100. Anything less is ``unknown``; a maxed applicable
window that hasn't reset is ``exhausted``. Fixture snapshots only.
"""
from __future__ import annotations

import pytest

from llm_bawt.service.usage.adapters.claude import _from_limit_item, _from_toplevel
from llm_bawt.service.usage.canonical import ProviderUsage, UsageLimit
from llm_bawt.service.usage.quota import QUOTA_FRESH_SECONDS, limit_applies, quota_state

NOW = 1_800_000_000
MODEL = "claude-opus-5-5"


def _limit(pct, *, scope=None, resets=None, lid="w"):
    return UsageLimit(id=lid, label=lid, used_pct=pct, model_scope=scope, resets_at=resets)


def _snap(*limits, status="ok", age=60, kind="subscription_windows"):
    return ProviderUsage(provider="fixture", display_name="Fixture", status=status, kind=kind,
                         available=status == "ok", fetched_at=NOW - age, limits=list(limits))


def _state(snapshot, model=MODEL):
    return quota_state(snapshot, model, now=NOW).state


def test_ok_fresh_and_under_every_window_is_available():
    quota = quota_state(_snap(_limit(10), _limit(88)), MODEL, now=NOW)

    assert (quota.state, quota.checked_at) == ("available", NOW - 60)


def test_a_maxed_unscoped_window_is_exhausted():
    assert _state(_snap(_limit(10), _limit(100, resets=NOW + 60))) == "exhausted"
    assert _state(_snap(_limit(100.0))) == "exhausted"


def test_a_window_that_already_reset_is_not_exhausted():
    assert _state(_snap(_limit(100, resets=NOW - 1), _limit(5))) == "available"


def test_only_another_models_window_is_unknown():
    assert _state(_snap(_limit(100, scope=["fable"]))) == "unknown"


def test_another_models_maxed_window_does_not_block_this_model():
    assert _state(_snap(_limit(100, scope=["fable"]), _limit(40))) == "available"
    assert _state(_snap(_limit(100, scope=["fable"]), _limit(40)), "claude-fable-5-1") == "exhausted"


def test_an_unknown_percentage_on_an_applicable_window_is_unknown():
    assert _state(_snap(_limit(10), _limit(None))) == "unknown"
    # A known maxed window still decides it.
    assert _state(_snap(_limit(None), _limit(100))) == "exhausted"


@pytest.mark.parametrize("snapshot", [
    None,
    _snap(_limit(10), status="usage_stale"),
    _snap(_limit(10), status="stale"),
    _snap(_limit(10), status="not_implemented"),
    _snap(_limit(10), status="rate_limited"),
    _snap(_limit(10), age=QUOTA_FRESH_SECONDS + 1),
    _snap(kind="spend"),  # API spend: no quota windows
])
def test_anything_short_of_a_fresh_ok_snapshot_is_unknown(snapshot):
    assert _state(snapshot) == "unknown"


def test_a_snapshot_exactly_at_the_freshness_limit_still_counts():
    assert _state(_snap(_limit(10), age=QUOTA_FRESH_SECONDS)) == "available"


@pytest.mark.parametrize("scope, model, applies", [
    (None, MODEL, True),
    (None, None, True),
    (["fable"], "claude-fable-5-1", True),
    (["fable"], MODEL, False),
    (["claude-opus-5-5"], MODEL, True),
    (["opus"], "claude-opus-5-5-1m", True),
    (["fable"], None, False),
])
def test_scope_matches_a_model_key_or_one_of_its_parts(scope, model, applies):
    assert limit_applies(_limit(1, scope=scope), model) is applies


def test_the_claude_adapter_scopes_named_weekly_windows_only():
    scoped = _from_limit_item({"kind": "weekly_scoped", "percent": 36,
                               "scope": {"model": {"id": None, "display_name": "Fable"}}}, 2)
    unnamed = _from_limit_item({"kind": "weekly_scoped", "percent": 36, "scope": None}, 2)
    weekly = _from_limit_item({"kind": "weekly_all", "percent": 88}, 1)

    assert (scoped.id, scoped.model_scope) == ("weekly_fable", ["fable"])
    assert unnamed.model_scope is None
    assert weekly.model_scope is None


def test_the_claude_toplevel_fallback_scopes_its_model_windows():
    limits = {lim.id: lim.model_scope for lim in _from_toplevel({
        "five_hour": {"utilization": 0.1}, "seven_day_sonnet": {"utilization": 0.2},
        "seven_day_opus": {"utilization": 0.3}})}

    assert limits == {"session_5h": None, "weekly_sonnet": ["sonnet"], "weekly_opus": ["opus"]}
