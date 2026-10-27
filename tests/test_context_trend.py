"""First SDK-turn context chart eligibility and user scoping."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine

from llm_bawt.service.context_trend import _snapshot, load_context_trend
from llm_bawt.service.turn_log_schema import TurnLog


def _usage(start=None, *, tokens=50, deferred=False):
    breakdown = {
        "cli_total_tokens": tokens,
        "categories": [
            {"name": "Skills", "tokens": tokens, "isDeferred": deferred},
        ],
        "message_breakdown": {"toolResultTokens": 7},
    }
    if start is not None:
        breakdown["session_start"] = start
    return json.dumps({
        "resident_tokens": tokens + 3,
        "resident_source": "claude_sdk_context",
        "context_breakdown": breakdown,
    })


def test_trend_only_uses_marked_first_turns_and_never_crosses_users():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__])
    now = datetime.now(timezone.utc)
    rows = [
        ("first", "alpha", "owner", 3, _usage("clean", tokens=40)),
        ("second-session", "alpha", "owner", 1, _usage("seeded", tokens=60)),
        ("resumed", "alpha", "owner", 0, _usage(tokens=999)),
        ("other-bot", "beta", "owner", 2, _usage("clean", tokens=70)),
        ("other-user", "alpha", "stranger", 0, _usage("clean", tokens=900)),
        ("old", "alpha", "owner", 100, _usage("clean", tokens=1000)),
        ("malformed", "gamma", "owner", 0, '{"session_start": "clean"'),
    ]
    with Session(engine) as session:
        for turn_id, bot, user, age, usage in rows:
            when = now - timedelta(days=age)
            session.add(TurnLog(
                id=turn_id, bot_id=bot, user_id=user, created_at=when,
                ended_at=when, token_usage_json=usage, user_prompt="private prompt",
            ))
        session.commit()

    result = load_context_trend(engine, user_id="owner", bot_id="alpha", days=90, limit=60)
    assert [row["turn_id"] for row in result["series"]] == ["first", "second-session"]
    assert [(row["bot_id"], row["turn_id"]) for row in result["latest"]] == [
        ("alpha", "second-session"), ("beta", "other-bot"),
    ]
    assert result["series"][1]["session_start"] == "seeded"
    assert result["series"][1]["tool_result_tokens"] == 7
    assert result["series"][1]["categories"] == [
        {"name": "Skills", "tokens": 60, "is_deferred": False},
    ]
    assert "private prompt" not in str(result)
    assert [row["turn_id"] for row in load_context_trend(
        engine, user_id="owner", bot_id="alpha", days=90, limit=1,
    )["series"]] == ["second-session"]


@pytest.mark.parametrize("bad_start", [[], {}])
def test_malformed_session_start_is_rejected(bad_start):
    row = {
        "id": "bad", "bot_id": "alpha", "created_at": datetime.now(timezone.utc),
        "token_usage_json": _usage(bad_start),
    }
    assert _snapshot(row) is None


@pytest.mark.parametrize("bad_usage", [
    '{"session_start": "clean"',
    _usage("unknown"),
    json.dumps({"unrelated": "session_start"}),
    _usage([]),
    _usage({}),
])
def test_latest_and_limit_are_applied_after_validating_snapshots(bad_usage):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__])
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        for turn_id, when, usage in [
            ("valid", now - timedelta(days=1), _usage("clean")),
            ("invalid", now, bad_usage),
        ]:
            session.add(TurnLog(
                id=turn_id, bot_id="alpha", user_id="owner", created_at=when,
                ended_at=when, token_usage_json=usage,
            ))
        session.commit()
    result = load_context_trend(engine, user_id="owner", bot_id="alpha", days=90, limit=1)
    assert [row["turn_id"] for row in result["latest"]] == ["valid"]
    assert [row["turn_id"] for row in result["series"]] == ["valid"]
    assert result["truncated"] is False


def test_candidate_cap_reports_truncation_without_claiming_all_bots(monkeypatch):
    from llm_bawt.service import context_trend

    monkeypatch.setattr(context_trend, "MAX_CANDIDATES", 2)
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__])
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        for i in range(3):
            when = now - timedelta(minutes=i)
            session.add(TurnLog(
                id=f"turn-{i}", bot_id="alpha", user_id="owner", created_at=when,
                ended_at=when, token_usage_json=_usage("clean"),
            ))
        session.commit()
    result = load_context_trend(engine, user_id="owner", bot_id="alpha", days=90, limit=1)
    assert result["truncated"] is True
    assert [row["turn_id"] for row in result["series"]] == ["turn-0"]
