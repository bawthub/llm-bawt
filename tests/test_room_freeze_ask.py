"""TASK-1692: only an affirmative ask unlocks git, and a failed lookup locks it."""
from __future__ import annotations

import pytest

from agent_bridge.room_freeze import ASKED, LOCKED, UNKNOWN
from claude_code_bridge.skill_hooks import INJECTED_SKILLS_ENV, git_gate, room_freeze_env
from llm_bawt import room_freeze_notice
from llm_bawt.room_freeze_notice import asks_to_commit, room_freeze_state, with_room_freeze_notice


@pytest.mark.parametrize("text", [
    "commit it",
    "ok, commit that and push it",
    "Looks good. Go ahead and push.",
    "please ship it",
    "yes, commit",
    "fix the typo, then commit",
])
def test_affirmative_asks(text):
    assert asks_to_commit(text)


@pytest.mark.parametrize("text", [
    "Do not commit or push anything yet.",
    "Show me the last commit.",
    "Why did you push that?",
    "don't push",
    "commit?",
    "hold off, commit later",
    "the push failed earlier",
    "Keep editing.",
    "",
])
def test_mentions_questions_and_prohibitions_stay_locked(text):
    assert not asks_to_commit(text)


def _push():
    return {"session_id": "s1", "tool_input": {"command": "git commit -m x"}}


def test_a_failed_room_lookup_locks_git_without_a_notice(monkeypatch):
    def broken(engine, bot_id):
        raise OSError("db blip")

    monkeypatch.setattr(room_freeze_notice, "in_room", broken)
    state = room_freeze_state("commit it", engine=object(), bot_id="vex")
    assert state == UNKNOWN
    assert with_room_freeze_notice("Keep editing.", state, bot_id="vex") == "Keep editing."
    env = {INJECTED_SKILLS_ENV: "git-commits", **room_freeze_env(state)}
    assert git_gate(_push(), env)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_live_room_states(monkeypatch):
    monkeypatch.setattr(room_freeze_notice, "in_room", lambda engine, bot_id: True)
    assert room_freeze_state("Do not commit.", engine=object(), bot_id="vex") == LOCKED
    assert room_freeze_state("commit it", engine=object(), bot_id="vex") == ASKED
