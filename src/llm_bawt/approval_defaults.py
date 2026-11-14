"""Starter approval policy catalog."""
from typing import Any

# Conservative default rule set (TASK-296). High/critical, shell-destructive only.
_OPS_DEFAULT_POLICIES: list[dict[str, Any]] = [
    {
        "id": "seed-agent-deny-local-prod-build",
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        # Anchor at a command invocation: search-text in rg/tests is not a release.
        # Covers direct make and the common SSH+cd shell wrapper.
        "pattern": r"(?i)^\s*(?:ssh\s+(?:-[a-z]+\s+)*[^\s;&|]+\s+['\"]?)?(?:cd\s+[^;&|]+\s*&&\s*)?make\s+(?:-o\s+version-release\s+)?(?:rebuild-prod|snapshot-rebuild)\b",
        "action": "deny", "severity": "critical", "category": "deploy", "order": 1,
    },
    {
        "id": "seed-agent-deny-direct-prod-container-mutation",
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"(?i)^\s*(?:ssh\s+(?:-[a-z]+\s+)*[^\s;&|]+\s+['\"]?)?(?:cd\s+[^;&|]+\s*&&\s*)?docker\s+(?:(?:container\s+)?(?:stop|start|restart|rm|kill)\s+[^;&|]*\bbawthub-frontend-prod-1\b|compose\s+(?:-f\s+[^\s]+\s+)*(?:up|down|stop|restart|rm|build)\s+[^;&|]*\bfrontend-prod\b)",
        "action": "deny", "severity": "critical", "category": "deploy", "order": 2,
    },
    {
        "id": "seed-ops-run-bridge-restart",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=llm-bawt.restart-bridge ",
        "action": "require_approval",
        "severity": "high",
        "category": "restart",
        "approval_prompt": "Restarting an agent bridge kills its active sessions. Approve?",
        "order": 5,
    },
    {
        "id": "seed-ops-run-redis-restart",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=llm-bawt.restart-redis ",
        "action": "require_approval",
        "severity": "critical",
        "category": "restart",
        "approval_prompt": "Restarting Redis interrupts the shared agent event bus. Approve?",
        "order": 6,
    },
    {
        # TASK-997: separate from the build approval (GitHub workflow_dispatch).
        "id": "seed-ops-run-bawthub-deploy-prod-image",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.deploy-prod-image ",
        "action": "require_approval",
        "severity": "high",
        "category": "deploy",
        "approval_prompt": "Deploy this verified BawtHub release digest to production (app.bawthub.com)? Approve?",
        "order": 7,
    },
    {
        "id": "seed-ops-run-bawthub-rollback-prod-image",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.rollback-prod-image ",
        "action": "require_approval",
        "severity": "critical",
        "category": "deploy",
        "approval_prompt": "Roll BawtHub production back to the recorded last-known-good image? Approve?",
        "order": 8,
    },
    {
        "id": "seed-ops-run-bawthub-release-prod",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.release-prod ",
        "action": "require_approval",
        "severity": "high",
        "category": "deploy",
        "approval_prompt": "Build this BawtHub release and deploy the verified image to production?",
        "order": 9,
    },
    {
        "id": "seed-ops-run-bawthub-reconcile-prod-routing",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.reconcile-prod-routing ",
        "action": "require_approval",
        "severity": "high",
        "category": "configuration",
        "approval_prompt": "Repair production health routing using the same pinned image (brief frontend interruption)?",
        "order": 10,
    },
    {
        "id": "seed-ops-run-bawthub-reconcile-prod-config",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.reconcile-prod-config ",
        "action": "require_approval",
        "severity": "high",
        "category": "configuration",
        "approval_prompt": "Reconcile production configuration to its Compose service using the same pinned image (brief frontend interruption)?",
        "order": 11,
    },
    {
        # TASK-1801: the Pocket TTS worker image (CI publish binding).
        "id": "seed-ops-run-bawthub-deploy-pocket-tts",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.deploy-pocket-tts ",
        "action": "require_approval",
        "severity": "high",
        "category": "deploy",
        "approval_prompt": "Deploy this verified Pocket TTS worker image digest? Approve?",
        "order": 12,
    },
    {
        "id": "seed-ops-run-bawthub-rollback-pocket-tts",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.rollback-pocket-tts ",
        "action": "require_approval",
        "severity": "critical",
        "category": "deploy",
        "approval_prompt": "Roll the Pocket TTS worker back to the recorded last-known-good image? Approve?",
        "order": 13,
    },
    {
        "id": "seed-ops-run-bawthub-reconcile-pocket-tts-config",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=bawthub.reconcile-pocket-tts-config ",
        "action": "require_approval",
        "severity": "high",
        "category": "configuration",
        "approval_prompt": "Reconcile the Pocket TTS worker to its Compose service using the same pinned image?",
        "order": 14,
    },
    {
        # TASK-2100: also enforced by the ops service (card-only).
        "id": "seed-ops-run-house-restart",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "prefix",
        "pattern": "operation=llm-bawt.restart-house ",
        "action": "require_approval",
        "severity": "high",
        "category": "restart",
        "approval_prompt": "Restart this agent house? It refuses while occupied unless override_occupied is set.",
        "order": 15,
    },
    {
        "id": "seed-ops-run-require-approval",
        "backend_scope": "*",
        "tool_name": "ops_run",
        "matcher_type": "always",
        "pattern": "",
        "action": "require_approval",
        "severity": "medium",
        "category": "operations",
        "approval_prompt": "Run this operator-configured operation?",
        "order": 100,
    },
]


# Match literal shell words as units, so separators/flags inside quoted data do
# not become command positions. Possessive repetition prevents splitting a word
# (or exponential backtracking) when a later command does not match.
_SHELL_WORD = r"""(?:[^\s'"\\;&|()<>#]|'[^']*'|"(?:\\.|[^"\\])*"|\\[^\n])++"""
_COMMAND_SEPARATOR = r"(?:&&|\|\||[;|&\n])"
_BIN_PATH = r"(?:/(?:usr/)?bin/)?"
_SSH = (
    rf"{_BIN_PATH}ssh[ \t]+"
    r"(?:(?:-[46AaCfGgKkMNnqsTtVvXxYy]+|-[BbcDEeFIiJLlmOopQRSWw][ \t]*[^\s'\";&|]+)[ \t]+)*"
    r"[^\s'\"\\;&|()\-][^\s'\"\\;&|()]*[ \t]+"
)
_WRAPPER = (
    rf"(?:{_BIN_PATH}(?:command(?:[ \t]+--)?|exec)[ \t]+|"
    rf"{_BIN_PATH}env[ \t]+(?:[A-Za-z_][A-Za-z_0-9]*={_SHELL_WORD}[ \t]+)*|"
    rf"{_BIN_PATH}sudo[ \t]+(?:(?:-[nEHPS]+|-[ug][ \t]+{_SHELL_WORD})[ \t]+)*|{_SSH})"
)


def _shell_command_pattern(command: str) -> str:
    """Anchor the two starter rules at literal commands, not arbitrary text.

    Supports command lists, common transparent wrappers and literal SSH/sh -c
    payloads. This is deliberately not a general shell parser (expansions,
    heredocs and dynamically constructed commands are outside this grammar).
    """
    def prefix(word: str) -> str:
        preceding = rf"{word}(?:[ \t]+{word})*[ \t]*{_COMMAND_SEPARATOR}[ \t]*"
        assignment = rf"[A-Za-z_][A-Za-z_0-9]*=(?:{word})?[ \t]+"
        # These programs execute a following argv. Consume only their options,
        # not arbitrary arguments (which would turn echo/grep data into code).
        xargs_option = (
            rf"(?:--(?:arg-file|delimiter|eof|max-lines|max-args|max-procs|replace|max-chars)"
            rf"(?:=|[ \t]+){word}|--[a-z-]+|-[adEILnPs](?:[ \t]*{word})|-[0oprtx]+)"
        )
        timeout_option = rf"(?:--(?:signal|kill-after)(?:=|[ \t]+){word}|-[sk][ \t]*{word}|--[a-z-]+|-[v]+)"
        argv_wrapper = (
            rf"{_BIN_PATH}(?:"
            rf"xargs[ \t]+(?:{xargs_option}[ \t]+)*(?:--[ \t]+)?|"
            rf"timeout[ \t]+(?:{timeout_option}[ \t]+)*(?:--[ \t]+)?[0-9]+(?:\.[0-9]+)?[smhd]?[ \t]+|"
            rf"nice[ \t]+(?:(?:-n[ \t]*[+-]?[0-9]+|--adjustment(?:=|[ \t]+)[+-]?[0-9]+|-[0-9]+)[ \t]+)?(?:--[ \t]+)?|"
            rf"(?:nohup|time)[ \t]+(?:-p[ \t]+)?(?:--[ \t]+)?|"
            rf"stdbuf[ \t]+(?:(?:-[ioe][ \t]*{word}|--(?:input|output|error)(?:=|[ \t]+){word})[ \t]+)+(?:--[ \t]+)?"
            rf")"
        )
        find_exec = rf"{_BIN_PATH}find[ \t]+(?:{word}[ \t]+)*?-(?:exec|execdir)[ \t]+"
        syntax = r"(?:\([ \t]*|\{[ \t]+|(?:do|then|else|elif)[ \t]+)"
        return rf"[ \t]*(?:{preceding}|{_WRAPPER}|{assignment}|{argv_wrapper}|{find_exec}|{syntax})*"

    def body(word: str, end: str) -> str:
        return prefix(word) + command.format(word=word, end=end)

    end = rf"(?=[ \t]|{_COMMAND_SEPARATOR}|\Z)"
    direct = body(_SHELL_WORD, end)
    # A quoted remote payload is shell code; a quoted echo/grep argument isn't.
    # Only literal payloads are handled here, without escaped outer quotes.
    remote = rf"(?:{_SSH}|{_BIN_PATH}(?:ba)?sh[ \t]+-[a-z]*c[ \t]+)"
    quoted = []
    for quote in ("'", '"'):
        other = '"' if quote == "'" else "'"
        word = rf"""(?:[^\s'"\\;&|()<>#]|{other}[^{other}{quote}]*{other})++"""
        quoted.append(remote + quote + body(word, rf"(?=[ \t]|{_COMMAND_SEPARATOR}|{quote})"))
    # The same command-list/wrapper prefix applies before the quoted wrapper.
    return rf"\A(?:{direct}|{prefix(_SHELL_WORD)}(?:{'|'.join(quoted)}))"


_RM_OPTION = r"(?:-[a-zA-Z]*[rRf][a-zA-Z]*|--recursive|--force)"


_DEFAULT_POLICIES: list[dict[str, Any]] = [
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": _shell_command_pattern(
            rf"{_BIN_PATH}rm[ \t]+"
            r"(?:(?!(?:--|'--'|\"--\"){end}){word}[ \t]+)*"
            rf"(?:{_RM_OPTION}|'{_RM_OPTION}'|\"{_RM_OPTION}\"){{end}}"
        ),
        "action": "require_approval", "severity": "high", "category": "filesystem",
        "approval_prompt": "This will recursively/forcibly delete files. Approve?",
        "order": 10,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": _shell_command_pattern(rf"{_BIN_PATH}sudo{{end}}"),
        "action": "require_approval", "severity": "high",
        "category": "privilege", "order": 20,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"git\s+push\b.*(--force|-f)\b",
        "action": "require_approval", "severity": "high", "category": "git",
        "approval_prompt": "Force-push can overwrite remote history. Approve?",
        "order": 30,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"(?i)\b(DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE)\b",
        "action": "require_approval", "severity": "critical", "category": "database",
        "approval_prompt": "Destructive SQL. Approve?", "order": 40,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"(mkfs|dd\s+.*of=/dev/|>\s*/dev/sd|chmod\s+-R\s+777|:\(\)\s*\{)",
        "action": "require_approval", "severity": "critical", "category": "system",
        "approval_prompt": "Potentially system-destroying command. Approve?", "order": 50,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"(curl|wget)\b.*\|\s*(sudo\s+)?(ba)?sh\b",
        "action": "require_approval", "severity": "high", "category": "network",
        "approval_prompt": "Piping a remote script straight into a shell. Approve?",
        "order": 60,
    },
]
