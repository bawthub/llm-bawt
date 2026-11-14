"""TASK-1691: the skill hooks count injected skills as loaded."""
from __future__ import annotations

import io
import json

import pytest

from agent_bridge.skill_selection import SkillSelectionError, injected_field
from claude_code_bridge import skill_hooks
from claude_code_bridge.skill_hooks import (
    INJECTED_SKILLS_ENV,
    git_gate,
    hook_command,
    injected_skills_env,
    install_settings,
    main,
    record,
    resume_reminder,
    room_freeze_env,
)

@pytest.fixture(autouse=True)
def _markers(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_hooks, "MARKER_DIR", tmp_path)


def _push(sid="s1"):
    return {"session_id": sid, "tool_input": {"command": "git push origin abc:master"}}


def test_the_command_field_names_injected_skills_into_the_env():
    field = injected_field(["git-commits", "peer-review", "git-commits"])
    assert field == "git-commits,peer-review"
    assert injected_skills_env(field) == {INJECTED_SKILLS_ENV: "git-commits,peer-review"}
    assert injected_skills_env(None) == {} and injected_skills_env("") == {}
    with pytest.raises(SkillSelectionError):
        injected_skills_env("git-commits,../etc")


def test_gate_denies_a_mutating_git_command_with_no_skill_in_context():
    decision = git_gate(_push(), {})
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_gate_allows_an_injected_git_skill_without_a_skill_load():
    assert git_gate(_push(), {INJECTED_SKILLS_ENV: "peer-review,git-commits"}) is None


def test_gate_ignores_injected_skills_without_git_rules():
    assert git_gate(_push(), {INJECTED_SKILLS_ENV: "peer-review"}) is not None


def test_gate_allows_a_skill_loaded_this_session_and_only_that_session():
    record({"session_id": "s1", "tool_input": {"skill": "bawthub"}}, {})
    assert git_gate(_push("s1"), {}) is None
    assert git_gate(_push("s2"), {}) is not None


def test_room_freeze_locks_commits_even_with_a_git_skill_in_context():
    """TASK-1692: while the user is in the room, commit/push waits for their ask."""
    from agent_bridge.room_freeze import ROOM_FREEZE_ENV
    from claude_code_bridge.skill_hooks import ROOM_FREEZE_DENY_REASON, room_freeze_env

    skilled = {INJECTED_SKILLS_ENV: "git-commits"}
    locked = git_gate(_push(), {**skilled, **room_freeze_env("locked")})
    assert locked["hookSpecificOutput"]["permissionDecisionReason"] == ROOM_FREEZE_DENY_REASON
    assert git_gate(_push(), {**skilled, **room_freeze_env("asked")}) is None
    status = {"session_id": "s1", "tool_input": {"command": "git status"}}
    assert git_gate(status, {**skilled, **room_freeze_env("locked")}) is None
    assert room_freeze_env(None) == {} and room_freeze_env("bogus") == {}
    assert room_freeze_env("locked") == {ROOM_FREEZE_ENV: "locked"}


def test_room_freeze_rides_the_command_into_the_request():
    from agent_bridge.room_freeze import ROOM_FREEZE_FIELD
    from claude_code_bridge.send_request import SendRequest

    assert ROOM_FREEZE_FIELD == "room_freeze"
    assert "room_freeze" in SendRequest.__dataclass_fields__


def test_gate_passes_read_only_git():
    assert git_gate({"session_id": "s1", "tool_input": {"command": "git status"}}, {}) is None


def _cmd(command):
    return {"session_id": "s1", "tool_input": {"command": command}}


@pytest.mark.parametrize("command", [
    "git log --grep merge --oneline",
    "git show HEAD:README.md | grep commit",
    "git log --format='%s' -- push.py",
    "git diff origin/master -- src/reset.py",
    "echo 'git push' > notes.txt",
    "grep -r 'git commit' docs/",
    "git -C /r status && echo merge",
    "git branch --merged master",
    "ls; cat push.log",
    "if git diff --quiet; then echo then git push; fi",
    "{ git status; echo do git commit; }",
    "echo '$(git push)'",
    "grep -n '`git commit`' docs/x.md",
])
def test_gate_passes_read_only_commands_mentioning_mutation_words(command):
    """TASK-2272: operands and other commands are not git subcommands."""
    frozen = room_freeze_env("locked")
    assert git_gate(_cmd(command), frozen) is None


@pytest.mark.parametrize("command", [
    "git push origin abc:master",
    "(cd /r && git push)",
    "git push;",
    "git stash&&git pull",
    "/usr/bin/git commit -m x",
    "git pull --rebase",
    "git am < patch.mbox",
    "git -C /r -c user.name=x commit -m y",
    "git --no-pager --git-dir /r/.git rebase master",
    "GIT_EDITOR=true git merge topic",
    "cd /r\ngit reset --hard",
    "echo $(git cherry-pick abc)",
    "bash -c 'cd /r && git revert HEAD'",
    "sudo -u x git push",
    "git status | cat || git push",
    "git commit -m 'unbalanced",
    "if test -d .git; then git push origin HEAD; fi",
    "git status && { git push origin HEAD; }",
    "for b in a; do git push origin $b; done",
    "! git push",
    'echo "$(git push origin HEAD)"',
    'echo "`git push`"',
    'echo "$(printf \')\'; git push)"',
    'echo "$( (cd /r) ; git push)"',
    'echo "$(echo \"a)\"; git push)"',
])
def test_gate_denies_mutating_git_in_any_shell_position(command):
    """TASK-2286: subshells, adjacency, paths, pull/am and shells are caught."""
    assert git_gate(_cmd(command), room_freeze_env("locked")) is not None
    assert git_gate(_cmd(command), {}) is not None


def test_reminder_names_injected_skills_as_still_in_context():
    text = resume_reminder({"source": "compact"}, {INJECTED_SKILLS_ENV: "git-commits,llm-bawt"})
    context = text["hookSpecificOutput"]["additionalContext"]
    assert "still in context: git-commits, llm-bawt" in context
    assert "Any other skill" in context


def test_reminder_without_injection_keeps_the_all_skills_warning():
    text = resume_reminder({"source": "resume"}, {})
    assert text["hookSpecificOutput"]["additionalContext"] == skill_hooks.REMINDER_ALL
    assert resume_reminder({"source": "startup"}, {}) is None


def test_main_prints_one_decision_and_never_fails():
    out = io.StringIO()
    assert main(["git-gate"], stdin=io.StringIO(json.dumps(_push())), env={}, stdout=out) == 0
    assert json.loads(out.getvalue())["hookSpecificOutput"]["permissionDecision"] == "deny"
    out = io.StringIO()
    assert main(["git-gate"], stdin=io.StringIO("not json"), env={}, stdout=out) == 0
    assert out.getvalue() == ""


def test_install_replaces_legacy_scripts_and_keeps_other_hooks():
    guard = {"type": "command", "command": "/opt/agent-skills/bin/git-guard", "if": "Bash(git *)"}
    legacy = {"type": "command", "command": "/home/bridge/.claude/hooks/git-skill-gate.sh",
              "if": "Bash(git *)"}
    settings = {"hooks": {
        "PreToolUse": [{"matcher": "Bash", "hooks": [guard, legacy]}],
        "PostToolUse": [{"matcher": "Skill", "hooks": [
            {"type": "command", "command": "/home/bridge/.claude/hooks/skill-load-record.sh"}]}],
        "SessionStart": [{"hooks": [
            {"type": "command", "command": "/home/bridge/.claude/hooks/session-resume-reminder.sh"}]}],
        "PreCompact": [{"hooks": [{"type": "command", "command": "compact"}]}],
    }, "attribution": {"commit": ""}}
    once = install_settings(settings)
    assert install_settings(once) == once
    hooks = once["hooks"]
    commands = [h["command"] for event in hooks.values() for g in event for h in g["hooks"]]
    assert not any(".claude/hooks/" in c for c in commands)
    assert hooks["PreToolUse"][0]["hooks"] == [guard]
    assert hooks["PreToolUse"][1] == {"matcher": "Bash", "hooks": [
        {"type": "command", "command": hook_command("git-gate"), "timeout": 10}]}
    assert hooks["PostToolUse"] == [{"matcher": "Skill", "hooks": [
        {"type": "command", "command": hook_command("record"), "timeout": 10}]}]
    assert hooks["SessionStart"][-1]["hooks"][0]["command"] == hook_command("resume-reminder")
    assert hooks["PreCompact"] == settings["hooks"]["PreCompact"]
    assert once["attribution"] == {"commit": ""}


def test_compact_invalidates_loaded_skills_but_resume_keeps_them():
    """TASK-2272: compaction discards tool-loaded skill bodies."""
    load = {"session_id": "s1", "tool_input": {"skill": "git-commits"}}
    record(load, {})
    resume_reminder({"session_id": "s1", "source": "resume"}, {})
    assert git_gate(_push("s1"), {}) is None
    resume_reminder({"session_id": "s1", "source": "compact"}, {})
    assert git_gate(_push("s1"), {}) is not None
    assert git_gate(_push("s1"), {INJECTED_SKILLS_ENV: "git-commits"}) is None
    record(load, {})
    assert git_gate(_push("s1"), {}) is None
