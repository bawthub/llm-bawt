"""TASK-1691: the skill hooks count injected skills as loaded."""
from __future__ import annotations

import io
import json

import pytest

from agent_bridge.skill_selection import ManagedSelection, SkillSelectionError
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
)

MANIFEST = {
    "manifest_sha": "abc",
    "entries": [{"ref": "git-commits", "mode": "inject"}, {"ref": "llm-bawt", "mode": "enable"},
                {"ref": "peer-review", "mode": "inject"}],
}


@pytest.fixture(autouse=True)
def _markers(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_hooks, "MARKER_DIR", tmp_path)


def _push(sid="s1"):
    return {"session_id": sid, "tool_input": {"command": "git push origin abc:master"}}


def test_selection_keeps_injected_refs_through_the_command_field():
    selection = ManagedSelection.from_manifest(MANIFEST)
    assert selection.injected == ("git-commits", "peer-review")
    assert ManagedSelection.from_field(selection.to_field()) == selection


def test_env_names_injected_skills_only_for_a_managed_selection():
    field = ManagedSelection.from_manifest(MANIFEST).to_field()
    assert injected_skills_env(field) == {INJECTED_SKILLS_ENV: "git-commits,peer-review"}
    assert injected_skills_env(None) == {}
    plain = ManagedSelection.from_manifest({**MANIFEST, "entries": [{"ref": "llm-bawt"}]})
    assert injected_skills_env(plain.to_field()) == {}
    with pytest.raises(SkillSelectionError):
        injected_skills_env("{not json")


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


def test_gate_passes_read_only_git():
    assert git_gate({"session_id": "s1", "tool_input": {"command": "git status"}}, {}) is None


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
        {"type": "command", "command": hook_command("git-gate"), "timeout": 10,
         "if": "Bash(git *)"}]}
    assert hooks["PostToolUse"] == [{"matcher": "Skill", "hooks": [
        {"type": "command", "command": hook_command("record"), "timeout": 10}]}]
    assert hooks["SessionStart"][-1]["hooks"][0]["command"] == hook_command("resume-reminder")
    assert hooks["PreCompact"] == settings["hooks"]["PreCompact"]
    assert once["attribution"] == {"commit": ""}
