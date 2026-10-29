"""Claude CLI hooks that track which skills a session has in context (TASK-1691).

Three hooks, one command (``python -m claude_code_bridge.skill_hooks <name>``):

* ``record`` (PostToolUse, Skill): appends the loaded skill to a per-session
  marker file.
* ``git-gate`` (PreToolUse, Bash git): denies a mutating git command until a
  git-relevant skill is in context: loaded with the Skill tool this session,
  or injected into this turn's system prompt.
* ``resume-reminder`` (SessionStart, resume/compact): warns that skills loaded
  with the Skill tool did not survive the lossy summary. Injected skills did:
  the system prompt is sent again on every turn.

The bridge names the injected skills in :data:`INJECTED_SKILLS_ENV` on the CLI
process, which hook commands inherit. Every hook prints at most one JSON
decision and always exits 0, so a hook fault never blocks a turn by crashing.
"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

#: Comma-separated refs whose SKILL.md bodies this turn's system prompt carries.
INJECTED_SKILLS_ENV = "LLM_BAWT_INJECTED_SKILLS"
#: Per-session marker of skills loaded with the Skill tool.
MARKER_DIR = Path("/tmp")
MARKER_PREFIX = "claude-skills-loaded-"
#: Skills that carry the git rules for a repo this harness commits to.
GIT_SKILLS = frozenset({"git-commits", "skill-creation", "llm-bawt", "bawthub"})
#: Mutating git subcommands. Read-only ones (status/diff/log/show) pass.
_MUTATING_GIT = re.compile(
    r"(^|\s)git(\s.*)?\s(commit|push|reset|rebase|merge|stash|cherry-pick|revert)(\s|$)")
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")

GATE_DENY_REASON = (
    "BLOCKED: no git skill loaded this session. Seeded session summaries do NOT carry "
    "skill content — the rules are not in your context even if a summary says you "
    "loaded them. Invoke the Skill tool with git-commits (or the repo skill: llm-bawt / "
    "bawthub, or skill-creation for the agent-skills repo) BEFORE any git "
    "commit/push/reset/rebase/merge/stash, then retry this command."
)
REMINDER_ALL = (
    "CONTEXT WAS RESTORED FROM LOSSY SUMMARIES. Any skill you appear to have loaded "
    "earlier is NOT in context now — the rules, commands, paths, and git workflow are "
    "gone even though a summary may say you loaded them. Before acting on prior work "
    "(especially git, deploys, or repo-specific edits), re-invoke the relevant Skill via "
    "the Skill tool. Do not operate on a summary as if it were the skill."
)
REMINDER_INJECTED = (
    "CONTEXT WAS RESTORED FROM LOSSY SUMMARIES. Skills injected into your system prompt "
    "are still in context: {injected}. Any other skill you appear to have loaded earlier "
    "is NOT in context now, even though a summary may say you loaded it. Before acting "
    "on prior work that needs one, re-invoke it via the Skill tool. Do not operate on a "
    "summary as if it were the skill."
)


def injected_skills_env(skill_selection: str | None) -> dict[str, str]:
    """The env naming a managed selection's injected skills; ``{}`` when none.

    A malformed selection raises ``SkillSelectionError``, as it does wherever
    else the bridge applies it, so the turn fails visibly.
    """
    from agent_bridge.skill_selection import ManagedSelection

    selection = ManagedSelection.from_field(skill_selection)
    if selection is None or not selection.injected:
        return {}
    return {INJECTED_SKILLS_ENV: ",".join(selection.injected)}


def injected_skills(env: Mapping[str, str]) -> tuple[str, ...]:
    raw = env.get(INJECTED_SKILLS_ENV) or ""
    return tuple(ref for ref in (part.strip() for part in raw.split(",")) if ref)


def _marker(data: Mapping) -> Path:
    sid = str(data.get("session_id") or "")
    return MARKER_DIR / f"{MARKER_PREFIX}{sid if _SESSION_ID.fullmatch(sid) else 'nosession'}"


def record(data: Mapping, env: Mapping[str, str]) -> dict | None:
    skill = (data.get("tool_input") or {}).get("skill")
    if isinstance(skill, str) and skill.strip():
        with _marker(data).open("a") as handle:
            handle.write(skill.strip() + "\n")
    return None


def _loaded(data: Mapping) -> set[str]:
    try:
        return set(_marker(data).read_text().split())
    except OSError:
        return set()


def git_gate(data: Mapping, env: Mapping[str, str]) -> dict | None:
    command = (data.get("tool_input") or {}).get("command")
    if not isinstance(command, str) or not _MUTATING_GIT.search(command):
        return None
    if GIT_SKILLS & (set(injected_skills(env)) | _loaded(data)):
        return None
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": GATE_DENY_REASON}}


def resume_reminder(data: Mapping, env: Mapping[str, str]) -> dict | None:
    if data.get("source") not in ("resume", "compact"):
        return None
    injected = injected_skills(env)
    text = REMINDER_INJECTED.replace("{injected}", ", ".join(injected)) if injected \
        else REMINDER_ALL
    return {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                   "additionalContext": text}}


HOOKS = {"record": record, "git-gate": git_gate, "resume-reminder": resume_reminder}

#: (event, matcher, ``if`` rule, hook name) for each settings entry we own.
_ENTRIES = (
    ("PostToolUse", "Skill", None, "record"),
    ("PreToolUse", "Bash", "Bash(git *)", "git-gate"),
    ("SessionStart", None, None, "resume-reminder"),
)
#: The untracked host scripts these hooks replace; their entries are removed.
LEGACY_SCRIPTS = ("skill-load-record.sh", "git-skill-gate.sh", "session-resume-reminder.sh")


def hook_command(name: str) -> str:
    import shlex

    return shlex.join([sys.executable, "-m", "claude_code_bridge.skill_hooks", name])


def _is_ours(hook: object) -> bool:
    command = hook.get("command") if isinstance(hook, dict) else None
    if not isinstance(command, str):
        return False
    return (command in {hook_command(name) for name in HOOKS}
            or any(command.endswith("/.claude/hooks/" + script) for script in LEGACY_SCRIPTS))


def install_settings(settings: dict) -> dict:
    """Install the skill hooks in CLI settings, replacing the legacy scripts.

    Other hooks (the agent-skills git-guard, the compact hook, user hooks) are
    kept in place; a group left empty by removing ours is dropped.
    Idempotent. Unexpected shapes are left unchanged.
    """
    hooks = settings.get("hooks") if settings.get("hooks") is not None else {}
    if not isinstance(hooks, dict) or not all(
            isinstance(hooks.get(event, []), list) for event, *_ in _ENTRIES):
        return dict(settings)
    updated = dict(hooks)
    for event, matcher, rule, name in _ENTRIES:
        groups = []
        for group in updated.get(event, []):
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                groups.append(group)
                continue
            kept = [hook for hook in group["hooks"] if not _is_ours(hook)]
            if kept or not group["hooks"]:
                groups.append({**group, "hooks": kept})
        hook = {"type": "command", "command": hook_command(name), "timeout": 10}
        if rule:
            hook["if"] = rule
        group = {"hooks": [hook]}
        if matcher:
            group["matcher"] = matcher
        updated[event] = [*groups, group]
    return {**settings, "hooks": updated}


def main(argv: list[str] | None = None, *, stdin=None, env: Mapping[str, str] | None = None,
         stdout=None) -> int:
    import os

    args = sys.argv[1:] if argv is None else argv
    hook = HOOKS.get(args[0]) if args else None
    if hook is None:
        return 0
    try:
        data = json.load(stdin or sys.stdin)
        decision = hook(data if isinstance(data, Mapping) else {},
                        os.environ if env is None else env)
    except Exception:  # noqa: BLE001 - a hook must never crash the CLI's tool call
        return 0
    if decision is not None:
        (stdout or sys.stdout).write(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
