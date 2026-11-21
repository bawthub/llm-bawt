"""The one-way cache protection veto on a disposable PostgreSQL database."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from agent_bridge.prompt_cache_clock import sdk_session_ref
from llm_bawt.service.job_parameters import IdleContextResetParameters
from llm_bawt.service.jobs.cache_reset_protection import CACHE_WRITE_HOLD
from llm_bawt.service.jobs.idle_context_candidates import IdleContextCandidates
from llm_bawt.service.jobs.idle_context_reset import IdleContextResetJob
from llm_bawt.service.prompt_stale import STALE_KEY
from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle
from llm_bawt.service.session_lifecycle.intent import Outcome
from tests.session_lifecycle_pg_fixtures import scheduled_intent

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]
_DOMAIN = "fixture-clock"
_START = 10_000_000
_WINDOW = 3_600_000
_SDK_ID = "sdk-fixture"


@pytest.fixture
def clock(monkeypatch):
    current = SimpleNamespace(sample=(_DOMAIN, _START + 1))
    monkeypatch.setattr("llm_bawt.service.jobs.cache_reset_protection.prompt_cache_clock", lambda: current.sample)
    return current


def bind(rig, *, sdk_id=_SDK_ID, stale=False):
    metadata = {"agent_session_keys": {"claude_code": sdk_id}}
    if stale:
        stamp = (rig.now - timedelta(minutes=25)).isoformat()
        metadata[STALE_KEY] = {"since": stamp, "changed_at": stamp, "sections": ["Rules"]}
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET session_metadata=CAST(:m AS jsonb) WHERE id=:id"),
                     {"m": json.dumps(metadata), "id": rig.session_id})


def receipt(observed=_START, *, sdk_id=_SDK_ID, **overrides):
    return {"route": "native", "access_path": "anthropic-oauth", "sdk_ttl": "1h",
            "sdk_session_ref": sdk_session_ref(sdk_id), "protection_write": {
                "source": "native_oauth_relay_1h_write", "clock_domain": _DOMAIN,
                "observed_ms": observed, "model": "claude-fixture", "write_tokens": 50,
            }, **overrides}


def usage(rig, cache, *, size=100_001, turn_id=None):
    data = {"resident_tokens": size}
    if cache is not None:
        data["prompt_cache"] = cache
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET token_usage_json=:u WHERE id=:id"),
                     {"u": json.dumps(data), "id": turn_id or rig.turn_id})


def newer_turn(rig, *, offset=1):
    message, turn = str(uuid.uuid4()), str(uuid.uuid4())
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :message, 'user', 'fixture', 2, :session)
        """), {"bot": rig.bot, "message": message, "session": rig.session_id})
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, ended_at, path, stream,
                bot_id, user_id, status, user_prompt, response_text, trigger_message_id)
            VALUES (:id, :created, :ended, '/test', false, :bot, :owner, 'ok', '', '', :message)
        """), {"id": turn, "created": rig.now - timedelta(minutes=20) + timedelta(seconds=offset),
               "ended": rig.now - timedelta(minutes=16) + timedelta(seconds=offset), "bot": rig.bot, "owner": rig.user, "message": message})
    return turn


def hold(rig):
    candidates = IdleContextCandidates(rig.engine)
    row, = candidates.read(rig.bot)
    return candidates.read_cache_hold_reason(row)


@pytest.mark.parametrize("later", ["short", "read", "missing"])
def test_later_turn_does_not_erase_same_session_write(rig, clock, later):
    bind(rig)
    usage(rig, receipt())
    new = newer_turn(rig)
    next_cache = {
        "short": {"route": "native", "sdk_ttl": "5m", "sdk_session_ref": sdk_session_ref(_SDK_ID)},
        "read": {"route": "native", "sdk_ttl": "1h", "read_tokens": 50,
                 "sdk_session_ref": sdk_session_ref(_SDK_ID)},
        "missing": None,
    }[later]
    usage(rig, next_cache, turn_id=new)
    clock.sample = (_DOMAIN, _START + _WINDOW - 1)
    assert hold(rig) == CACHE_WRITE_HOLD
    clock.sample = (_DOMAIN, _START + _WINDOW)
    assert hold(rig) is None


def test_later_qualifying_write_extends_hold_but_replacement_does_not_inherit(rig, clock):
    bind(rig)
    usage(rig, receipt())
    usage(rig, receipt(_START + 200), turn_id=newer_turn(rig))
    clock.sample = (_DOMAIN, _START + _WINDOW)
    assert hold(rig) == CACHE_WRITE_HOLD
    bind(rig, sdk_id="replacement")
    assert hold(rig) is None
    bind(rig)
    # An SDK result that names a replacement can arrive without updating the
    # durable resume pointer; the conflicting identities prove no hold.
    usage(rig, {"sdk_session_ref": sdk_session_ref("replacement")}, turn_id=newer_turn(rig, offset=2))
    assert hold(rig) is None
    usage(rig, None, turn_id=newer_turn(rig, offset=3))
    assert hold(rig) is None
    clock.sample = (_DOMAIN, _START + _WINDOW + 200)
    assert hold(rig) is None


@pytest.mark.parametrize("sample", [None, ("other-domain", _START + 1), (_DOMAIN, _START - 1)])
def test_unknown_clock_is_not_evidence(rig, clock, sample):
    bind(rig)
    usage(rig, receipt())
    clock.sample = sample
    assert hold(rig) is None


@pytest.mark.parametrize("overrides", [
    {"route": "proxy"}, {"access_path": "anthropic-api"}, {"protection_write": None},
    {"protection_write": {"source": "native_oauth_relay_1h_write", "observed_ms": "garbage"}},
])
def test_unknown_route_or_bad_evidence_is_not_evidence(rig, clock, overrides):
    bind(rig)
    usage(rig, receipt(**overrides))
    assert hold(rig) is None


@pytest.mark.parametrize("stale", [False, True])
def test_both_baseline_reasons_are_vetoed_at_scan_and_locked_admission(rig, real_lifecycle, clock, stale):
    bind(rig, stale=stale)
    usage(rig, receipt(), size=10 if stale else 100_001)
    parameters = IdleContextResetParameters(reset_when_prompt_stale=stale)
    fake = SimpleNamespace(submit=AsyncMock())
    job = IdleContextResetJob(rig.engine, None, lifecycle=fake, parameters=parameters)
    results = asyncio.run(job.run(job_id="fixture", run_id="fixture", target_bot=rig.bot))
    assert results["bots"][rig.bot]["detail"] == CACHE_WRITE_HOLD
    fake.submit.assert_not_awaited()
    intent = replace(scheduled_intent(rig), scheduler_parameters=parameters)
    admission = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert admission.outcome is Outcome.SKIPPED and admission.detail == CACHE_WRITE_HOLD
    assert admission.delivery_id is None
    clock.sample = (_DOMAIN, _START + _WINDOW)
    assert asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0)).delivery_id is not None


def test_veto_at_claim_releases_only_unaccepted_key_for_next_sweep(rig, real_lifecycle, clock):
    bind(rig)
    intent = scheduled_intent(rig)
    first = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert first.delivery_id
    # Enrichment races admission, but does not create a different measured turn.
    usage(rig, receipt())
    store = real_lifecycle.store
    assert store.claim_next(rig.bot, claim_owner="fixture", steer_capable=False) is None
    skipped = store.get(first.delivery_id)
    assert skipped.status == "FAILED" and skipped.transport_accepted_at is None
    assert skipped.metadata["new_session_skip_reason"] == CACHE_WRITE_HOLD
    assert skipped.idempotency_key is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM turn_logs WHERE id=:id"), {"id": first.turn_id}).scalar_one() == 0
    assert asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0)).detail == CACHE_WRITE_HOLD
    clock.sample = (_DOMAIN, _START + _WINDOW)
    second = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert second.delivery_id and second.delivery_id != first.delivery_id
    assert store.claim_next(rig.bot, claim_owner="fixture", steer_capable=False).id == second.delivery_id


def test_hold_is_not_eligibility_and_unpinned_eviction_is_unchanged(rig, clock):
    bind(rig)
    usage(rig, receipt(), size=100_000)
    candidates = IdleContextCandidates(rig.engine)
    row, = candidates.read(rig.bot)
    assert candidates.skip_reason(row, now=rig.now) == "resident context within threshold"
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET house='fixture', house_pinned=false WHERE slug=:bot"), {"bot": rig.bot})
        assert NewSessionLifecycle.scheduled_skip_reason(conn, scheduled_intent(rig)) is None
