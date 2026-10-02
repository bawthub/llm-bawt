"""Starter approval policy catalog."""
from typing import Any

# Conservative default rule set (TASK-296). High/critical, shell-destructive only.
_OPS_DEFAULT_POLICIES: list[dict[str, Any]] = [
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


_DEFAULT_POLICIES: list[dict[str, Any]] = [
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "regex",
        "pattern": r"\brm\b\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rRf][a-zA-Z]*",
        "action": "require_approval", "severity": "high", "category": "filesystem",
        "approval_prompt": "This will recursively/forcibly delete files. Approve?",
        "order": 10,
    },
    {
        "backend_scope": "*", "tool_name": "Bash", "matcher_type": "prefix",
        "pattern": "sudo ", "action": "require_approval", "severity": "high",
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
