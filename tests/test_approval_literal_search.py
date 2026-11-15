"""Regression coverage for inert search data and exact approval replay."""
import pytest

from agent_bridge.approval import matchable_subject
from agent_bridge.events import AgentEventKind
from claude_code_bridge.event_ops import ClaudeEventMixin


@pytest.mark.parametrize('command', [
    "rg -n 'docker compose restart' /tmp",
    "pwd; git -C /home/bridge/dev status --short; rg -n 'ops_run|nohup|docker compose restart' /tmp --glob '*.md'",
    'grep "docker compose restart" /tmp/notes',
])
def test_literal_search_data_is_not_a_command(command):
    assert 'docker compose restart' not in matchable_subject('Bash', command)


@pytest.mark.parametrize('command', [
    "rg foo /tmp; docker compose restart app",
    "rg foo /tmp && docker compose restart app",
    "ssh host \"docker compose restart app\"",
    "bash -c 'docker compose restart app'",
    "rg \"$(docker compose restart app)\" /tmp",
    "rg `docker compose restart app` /tmp",
    "rg 'docker compose restart app' /tmp | bash",
    "rg --pre 'docker compose restart app' foo /tmp",
    "rg --pre='docker compose restart app' foo /tmp",
    "rg --p're' 'docker compose restart app' foo /tmp",
    "rg foo <(docker compose restart app)",
    "rg foo /tmp\ndocker compose restart app",
    "env rg 'docker compose restart app' /tmp",
    "rg 'unterminated docker compose restart app",
])
def test_executable_or_ambiguous_commands_remain_visible(command):
    assert 'docker compose restart' in matchable_subject('Bash', command)


def test_approval_event_preserves_exact_arguments():
    events = []
    class Bridge(ClaudeEventMixin):
        _backend_name = 'claude-code'
        _trigger_message_ids = {}
        def _publish_run_event_with_changed_file(self, request_id, event):
            events.append(event)
    arguments = {'command': "printf '%s' /home/bridge/dev", 'timeout': 10000,
                 'nested': {'path': '/home/bridge/a'}}
    Bridge()._publish_event('request', 'snark:operator', 1,
                            kind=AgentEventKind.APPROVAL_REQUIRED,
                            tool_name='Bash', tool_arguments=arguments)
    assert events[0].tool_arguments == arguments
    assert '/home/bridge/dev' in events[0].tool_arguments['command']
