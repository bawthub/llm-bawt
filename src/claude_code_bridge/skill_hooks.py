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
MUTATING_GIT_SUBCOMMANDS = frozenset({
    "commit", "push", "pull", "am", "reset", "rebase", "merge", "stash",
    "cherry-pick", "revert"})
#: Git global options that consume the following word as their value.
_GIT_OPTS_WITH_ARG = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace",
                                "--super-prefix", "--config-env", "--list-cmds"})
#: Words that run the rest of a segment as a command.
_COMMAND_PREFIXES = frozenset({"sudo", "env", "command", "exec", "nohup", "time", "nice",
                               "builtin", "xargs"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash"})
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
#: Fallback when the command does not tokenize (unbalanced quotes): any
#: mutating subcommand word anywhere after a ``git`` word.
_FALLBACK = re.compile(
    r"(?:^|[^\w-])git\b.*?(?<![\w-])(" + "|".join(sorted(MUTATING_GIT_SUBCOMMANDS))
    + r")(?![\w-])", re.S)
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")

GATE_DENY_REASON = (
    "BLOCKED: no git skill loaded this session. Seeded session summaries do NOT carry "
    "skill content — the rules are not in your context even if a summary says you "
    "loaded them. Invoke the Skill tool with git-commits (or the repo skill: llm-bawt / "
    "bawthub, or skill-creation for the agent-skills repo) BEFORE any git "
    "commit/push/reset/rebase/merge/stash, then retry this command."
)
ROOM_FREEZE_DENY_REASON = (
    "BLOCKED: the user is in the room (operator freeze). Do not commit, push, reset, "
    "rebase, merge or stash unless the user's message explicitly asks you to. Leave "
    "the work uncommitted; that is expected during the freeze."
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


def injected_skills_env(field: str | None) -> dict[str, str]:
    """The env naming the command's injected skills; ``{}`` when none.

    ``field`` is the command's ``INJECTED_FIELD``. An invalid ref raises
    ``SkillSelectionError`` so the turn fails visibly.
    """
    from agent_bridge.skill_selection import injected_field

    refs = [ref for ref in (field or "").split(",") if ref]
    return {INJECTED_SKILLS_ENV: injected_field(refs)} if refs else {}


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


def room_freeze_env(state: str | None) -> dict[str, str]:
    """The env carrying the command's room freeze state; ``{}`` when none.

    An unknown value is ignored rather than failing the user's turn.
    """
    from agent_bridge.room_freeze import ROOM_FREEZE_ENV, ROOM_FREEZE_STATES

    return {ROOM_FREEZE_ENV: state} if state in ROOM_FREEZE_STATES else {}


def _deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def _segments(command: str) -> list[list[str]]:
    """Split a shell command into simple-command word lists.

    Separators (``;``, ``&``, ``|``, newlines, parentheses for subshells and
    ``$(...)``) end a segment. Quoted words stay whole. Raises ``ValueError``
    when the command does not tokenize.
    """
    import shlex

    text = command.replace("`", " ; ").replace("\n", " ; ")
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token and set(token) <= set(";&|()"):
            segments.append([])
        elif token != "$":
            segments[-1].append(token)
    return [seg for seg in segments if seg]


def _git_subcommand(words: list[str]) -> str | None:
    """The subcommand of ``git [global options] <sub>``, or None."""
    i = 1
    while i < len(words):
        word = words[i]
        if word in _GIT_OPTS_WITH_ARG:
            i += 2
        elif word.startswith("-"):
            i += 1
        else:
            return word
    return None


def _segment_mutates(words: list[str], depth: int) -> bool:
    i = 0
    while i < len(words) and _ASSIGNMENT.match(words[i]):
        i += 1
    if i < len(words) and words[i] in _COMMAND_PREFIXES:
        # Prefix options may take values (``sudo -u x``): run from the first
        # git or shell word after the prefix.
        rest = [j for j in range(i + 1, len(words))
                if words[j].rsplit("/", 1)[-1] in _SHELLS | {"git"}]
        i = rest[0] if rest else len(words)
    words = words[i:]
    if not words:
        return False
    program = words[0].rsplit("/", 1)[-1]
    if program in _SHELLS and depth < 3:
        for j, word in enumerate(words[1:-1], start=1):
            if word.startswith("-") and not word.startswith("--") and "c" in word:
                return _command_mutates(words[j + 1], depth + 1)
        return False
    return program == "git" and _git_subcommand(words) in MUTATING_GIT_SUBCOMMANDS


def _command_mutates(command: str, depth: int = 0) -> bool:
    """Whether a shell command runs a mutating git subcommand (TASK-2272/2286)."""
    try:
        segments = _segments(command)
    except ValueError:
        return bool(_FALLBACK.search(command))
    return any(_segment_mutates(seg, depth) for seg in segments)


def git_gate(data: Mapping, env: Mapping[str, str]) -> dict | None:
    from agent_bridge.room_freeze import GIT_LOCKED_STATES, ROOM_FREEZE_ENV

    command = (data.get("tool_input") or {}).get("command")
    if not isinstance(command, str) or not _command_mutates(command):
        return None
    if env.get(ROOM_FREEZE_ENV) in GIT_LOCKED_STATES:
        return _deny(ROOM_FREEZE_DENY_REASON)
    if GIT_SKILLS & (set(injected_skills(env)) | _loaded(data)):
        return None
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": GATE_DENY_REASON}}


def resume_reminder(data: Mapping, env: Mapping[str, str]) -> dict | None:
    if data.get("source") not in ("resume", "compact"):
        return None
    if data.get("source") == "compact":
        # TASK-2272: compaction drops tool-loaded skill bodies, so their
        # markers no longer prove the skill is in context. Resume is lossless.
        try:
            _marker(data).unlink()
        except OSError:
            pass
    injected = injected_skills(env)
    text = REMINDER_INJECTED.replace("{injected}", ", ".join(injected)) if injected \
        else REMINDER_ALL
    return {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                   "additionalContext": text}}


HOOKS = {"record": record, "git-gate": git_gate, "resume-reminder": resume_reminder}

#: (event, matcher, ``if`` rule, hook name) for each settings entry we own.
_ENTRIES = (
    ("PostToolUse", "Skill", None, "record"),
    # No ``if`` rule: git also runs inside subshells, ``cd x && git ...`` and
    # ``bash -c``; the gate parses every Bash command itself.
    ("PreToolUse", "Bash", None, "git-gate"),
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
