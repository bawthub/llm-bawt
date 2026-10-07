"""Seeded agent-path guardrails; never mistake documentation for a command."""
import re

import pytest

from llm_bawt.approval_defaults import _OPS_DEFAULT_POLICIES


@pytest.mark.parametrize("command", [
    "make rebuild-prod",
    "cd /home/nick/dev/bawthub && make rebuild-prod",
    "cd /home/nick/dev/bawthub && docker compose up -d --build frontend-prod",
    "make snapshot-rebuild",
    "make -o version-release snapshot-rebuild",
    "ssh nick@172.18.0.1 'cd /home/nick/dev/bawthub && make rebuild-prod'",
    "docker stop bawthub-frontend-prod-1",
    "ssh nick@172.18.0.1 'docker restart bawthub-frontend-prod-1'",
    "docker compose up -d --build frontend-prod",
])
def test_seeded_agent_release_mutations_are_denied(command):
    rules = [p for p in _OPS_DEFAULT_POLICIES if p["tool_name"] == "Bash"]
    assert any(re.search(rule["pattern"], command) for rule in rules), command
    assert all(rule["action"] == "deny" for rule in rules)


@pytest.mark.parametrize("command", [
    'rg -n "make rebuild-prod" references/prod-release.md',
    'python3 -c "print(\'docker stop bawthub-frontend-prod-1\')"',
    'docker inspect bawthub-frontend-prod-1',
    'docker stop local-test-container',
    'make mode-status',
])
def test_seeded_guards_do_not_match_read_only_commands(command):
    rules = [p for p in _OPS_DEFAULT_POLICIES if p["tool_name"] == "Bash"]
    assert not any(re.search(rule["pattern"], command) for rule in rules), command
