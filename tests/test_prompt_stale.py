"""Stale pinned prompt helpers and the idle-reset stale trigger (TASK-1771)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.service.job_parameters import IdleContextResetParameters
from llm_bawt.service.jobs import idle_context_reset
from llm_bawt.service.jobs.idle_context_candidates import IdleContextCandidates
from llm_bawt.service.jobs.idle_context_reset import IdleContextResetJob
from llm_bawt.service.prompt_stale import PREAMBLE, STALE_KEY, changed_sections, public_view, settled
from llm_bawt.service.session_lifecycle.intent import NewSessionResult, Outcome, Stage
from tests.bot_tag_fixtures import static_tags

NOW = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
PROMPT = "intro\n## Rules\nnever release\n### Skill: git\nuse worktrees\n"


def test_changed_sections_names_only_what_differs():
    edited = PROMPT.replace("never release", "release via ops cards")
    assert changed_sections(PROMPT, edited) == ["Rules"]
    assert changed_sections(PROMPT, PROMPT) == []
    assert changed_sections(PROMPT, PROMPT + "## New\nx\n") == ["New"]
    assert changed_sections(PROMPT, PROMPT.replace("### Skill: git\nuse worktrees\n", "")) == ["Skill: git"]
    assert changed_sections(PROMPT, "other intro" + PROMPT[5:]) == [PREAMBLE]
    # Names only: no section content leaks into the marker.
    assert "release via ops cards" not in json.dumps(changed_sections(PROMPT, edited))


def marker(minutes_ago, **extra):
    stamp = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    return {"since": stamp, "changed_at": stamp, "rendered_sha256": "b", "pinned_sha256": "a",
            "sections": ["Rules"], **extra}


def test_settled_waits_for_the_last_change():
    assert settled(marker(11), now=NOW, settle_minutes=10)
    assert not settled(marker(9), now=NOW, settle_minutes=10)
    assert settled(json.dumps(marker(11)), now=NOW, settle_minutes=10)
    assert not settled(None, now=NOW, settle_minutes=0)
    assert not settled({"sections": []}, now=NOW, settle_minutes=0)
    assert public_view(marker(1)) == {"since": marker(1)["since"], "changed_at": marker(1)["changed_at"],
                                      "sections": ["Rules"]}
    assert public_view({}) is None


@pytest.fixture(autouse=True)
def tags(monkeypatch):
    fake = static_tags(private={"fixture-private"}, inactive={"fixture-inactive"})
    monkeypatch.setattr(idle_context_reset, "BotTags", fake)
    return fake


def candidate(stale=None, **overrides):
    metadata = {STALE_KEY: stale} if stale is not None else {}
    return {
        "bot_id": "disposable", "session_id": "session", "user_id": "owner",
        "usage_turn_id": "turn", "latest_turn_id": "turn",
        "ended_at": NOW - timedelta(minutes=16), "last_completed_at": NOW - timedelta(minutes=16),
        # Small context: the size threshold alone would skip this session.
        "token_usage_json": json.dumps({"resident_tokens": 5000}),
        "session_metadata": metadata, **overrides,
    }


ON = IdleContextResetParameters(reset_when_prompt_stale=True, prompt_stale_settle_minutes=10)


@pytest.mark.parametrize("row,parameters,expected", [
    (candidate(marker(11)), ON, None),
    (candidate(marker(11)), IdleContextResetParameters(), "resident context within threshold"),
    (candidate(marker(9)), ON, "resident context within threshold"),
    (candidate(), ON, "resident context within threshold"),
    # The other fences still apply first.
    (candidate(marker(11), last_completed_at=NOW - timedelta(minutes=5)), ON, "idle less than threshold"),
    (candidate(marker(11), latest_turn_id="later"), ON, "newer bot activity invalidated measurement"),
    (candidate(marker(11), ended_at=None), ON, "missing completed context measurement"),
    (candidate(marker(11), session_id=None), ON, "no valid accepted session"),
    # A stale marker bypasses size even when resident fields are unusable.
    (candidate(marker(11), token_usage_json="{"), ON, None),
    (candidate(marker(11), session_metadata=json.dumps({STALE_KEY: marker(11)})), ON, None),
])
def test_stale_trigger_replaces_only_the_size_threshold(row, parameters, expected):
    assert IdleContextCandidates.skip_reason(row, now=NOW, parameters=parameters) == expected


def job_with(rows, *, protected=()):
    lifecycle = SimpleNamespace(submit=AsyncMock(return_value=NewSessionResult(
        outcome=Outcome.COMPLETED, stage=Stage.RECEIPT, old_session_id="session",
        new_session_id="new", receipt_message_id="receipt", delivery_id="d", turn_id="t")))
    job = IdleContextResetJob(None, object(), lifecycle=lifecycle, protected_bot_ids=protected,
                              parameters=ON)
    job.candidates.read = Mock(return_value=rows)
    return job, lifecycle


def run(job):
    return asyncio.run(job.run(job_id="job", run_id="run", target_bot="*"))


def test_stale_reset_is_keyed_by_the_render_change(monkeypatch):
    monkeypatch.setattr(idle_context_reset, "datetime", SimpleNamespace(now=lambda tz=None: NOW))
    stale = marker(11)
    job, lifecycle = job_with([candidate(stale)])
    result = run(job)
    assert result["reset"] == 1 and result["bots"]["disposable"]["trigger"] == "prompt_stale"
    first_key = lifecycle.submit.await_args.args[0].idempotency_key
    assert first_key.endswith(f":stale:{stale['changed_at']}")
    # The next sweep over the same edit burst reattaches to the same reset.
    run(job)
    assert lifecycle.submit.await_args.args[0].idempotency_key == first_key


@pytest.mark.parametrize("bot,protected", [("fixture-private", ()), ("fixture-inactive", ()),
                                           ("guarded", ("guarded",))])
def test_stale_trigger_never_reaches_lifecycle_for_excluded_bots(monkeypatch, bot, protected):
    monkeypatch.setattr(idle_context_reset, "datetime", SimpleNamespace(now=lambda tz=None: NOW))
    job, lifecycle = job_with([candidate(marker(11), bot_id=bot)], protected=protected)
    assert run(job)["skipped"] == 1
    lifecycle.submit.assert_not_awaited()
