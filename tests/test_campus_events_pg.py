"""TASK-2120: tenancy events leave the real profile store only after commit."""
from __future__ import annotations

import pytest
from sqlmodel import Session

from llm_bawt import campus_events
from tests.test_bot_tag_api_pg import api  # noqa: F401
from tests.test_bot_tags_pg import rig  # noqa: F401

pytestmark = pytest.mark.integration

BOT = "fixture-bot"
AGENT = {"slug": BOT, "bot_type": "agent", "agent_backend": "claude-code"}


@pytest.fixture
def sent(monkeypatch):
    out: list = []
    monkeypatch.setattr(campus_events, "emit_tenancy_changes",
                        lambda pending, config=None: out.extend(pending))
    return out


def test_owned_write_emits_once_after_commit(api, sent):  # noqa: F811
    api.store.upsert(AGENT, require_existing=True)
    sent.clear()
    api.store.upsert({"slug": BOT, "house": "h1"}, require_existing=True)
    assert [(s[0], s[1]["house"], s[2]["house"], s[3]) for s in sent] == [(BOT, None, "h1", "")]
    api.store.upsert({"slug": BOT, "description": "unrelated"}, require_existing=True)
    assert len(sent) == 1  # no tenancy change, no event


def test_borrowed_session_emits_on_callers_commit_not_on_rollback(api, sent):  # noqa: F811
    api.store.upsert(AGENT, require_existing=True)
    sent.clear()
    with Session(api.store.engine) as session:
        api.store.upsert({"slug": BOT, "house": "h1", "_location_reason": "move_in"},
                         session=session, require_existing=True)
        assert sent == []
        session.rollback()
    assert sent == [] and api.store.get(BOT).house is None
    with Session(api.store.engine) as session:
        api.store.upsert({"slug": BOT, "house": "h1", "_location_reason": "move_in"},
                         session=session, require_existing=True)
        assert sent == []
        session.commit()
    assert [(s[0], s[3]) for s in sent] == [(BOT, "move_in")]


def test_deleting_a_housed_bot_moves_it_out(api, sent):  # noqa: F811
    api.store.upsert({**AGENT, "house": "h1"}, require_existing=True)
    sent.clear()
    assert api.store.delete(BOT)
    assert [(s[0], s[1]["house"], s[2], s[3]) for s in sent] == [(BOT, "h1", {}, "move_out")]
