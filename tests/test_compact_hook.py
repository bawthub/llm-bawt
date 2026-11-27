"""TASK-1071: CLI command-hook output never overrides caller instructions."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from claude_code_bridge.bootstrap import _ensure_compact_hook
from claude_code_bridge.compact_hook import guidance_for_hook
from llm_bawt.chat_prompt_defaults import COMPACT_SUMMARY_DEFAULT


@pytest.mark.parametrize("trigger", ["auto", "manual"])
@pytest.mark.parametrize("custom", [None, "", "  \n "])
def test_default_only_without_custom_instructions(trigger, custom):
    assert guidance_for_hook(
        {"trigger": trigger, "custom_instructions": custom},
        lookup=lambda url: COMPACT_SUMMARY_DEFAULT,
    ) == COMPACT_SUMMARY_DEFAULT


@pytest.mark.parametrize("custom", ["keep everything", "\n保留所有细节\n", "Write 20,000 tokens", {"bad": "shape"}])
def test_custom_instructions_disable_default(custom):
    assert guidance_for_hook({"trigger": "manual", "custom_instructions": custom}) == ""
    assert guidance_for_hook({"trigger": "auto", "custom_instructions": custom}) == ""


def test_unknown_hook_input_does_not_inject():
    for data in (None, [], {}, {"trigger": "unknown"}, {"trigger": "manual", "custom_instructions": 1}):
        assert guidance_for_hook(data) == ""


def test_bootstrap_preserves_existing_hook_and_is_idempotent():
    previous = {"hooks": {
        "PreCompact": [{"hooks": [{"type": "command", "command": "existing-hook"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "stop-hook"}]}],
    }}
    settings = _ensure_compact_hook(previous)
    assert previous["hooks"]["PreCompact"] == [{"hooks": [{"type": "command", "command": "existing-hook"}]}]
    assert settings["hooks"]["PreCompact"][0] == previous["hooks"]["PreCompact"][0]
    assert settings["hooks"]["Stop"] == previous["hooks"]["Stop"]
    assert _ensure_compact_hook(settings) == settings


@pytest.fixture
def registry_server():
    state = {"body": COMPACT_SUMMARY_DEFAULT, "requests": 0, "status": 200, "keys": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"] += 1
            state["keys"].append(self.headers.get("X-LLM-Bawt-Key"))
            assert self.path == "/v1/prompts/agents.compaction_summary"
            content = json.dumps({"key": "agents.compaction_summary", "body": state["body"],
                                  "source": "db_override", "updated_at": "v1"}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()
        server.server_close()


@pytest.mark.real_process(reason="runs the real command hook script as Claude Code would")
def test_command_hook_fetches_each_time_and_skips_custom_and_outage(registry_server):
    url, state = registry_server
    command = _ensure_compact_hook({})["hooks"]["PreCompact"][0]["hooks"][0]["command"]

    def invoke(data):
        return subprocess.run(
            shlex.split(command), input=json.dumps(data), text=True, capture_output=True,
            check=True, timeout=10, env={**os.environ, "LLM_BAWT_API_URL": url},
        )

    first = invoke({"trigger": "auto", "custom_instructions": None})
    assert first.stdout == COMPACT_SUMMARY_DEFAULT + "\n"
    assert "key=agents.compaction_summary" in first.stderr
    assert COMPACT_SUMMARY_DEFAULT not in first.stderr
    state["body"] = "Second edit: preserve newer goals, not old constraints."
    second = invoke({"trigger": "auto", "custom_instructions": None})
    assert second.stdout == state["body"] + "\n"
    custom = invoke({"trigger": "manual", "custom_instructions": "user\ncustom"})
    assert custom.stdout == ""
    assert state["requests"] == 2  # User instructions skip lookup entirely.
    state["status"] = 503
    failure = invoke({"trigger": "auto", "custom_instructions": None})
    assert failure.stdout == ""
    assert "unavailable" in failure.stderr
    assert state["requests"] == 3


@pytest.mark.real_process(reason="the CLI hook process must send the bridge's API key itself")
def test_command_hook_sends_api_key(registry_server, tmp_path):
    url, state = registry_server
    key_file = tmp_path / "api-key"
    key_file.write_text("sample-bridge.secretvalue\n", encoding="utf-8")
    command = _ensure_compact_hook({})["hooks"]["PreCompact"][0]["hooks"][0]["command"]
    env = {**os.environ, "LLM_BAWT_API_URL": url, "LLM_BAWT_API_KEY_FILE": str(key_file)}
    result = subprocess.run(shlex.split(command), input=json.dumps({"trigger": "auto"}), text=True,
                            capture_output=True, check=True, timeout=10, env=env)
    assert result.stdout == COMPACT_SUMMARY_DEFAULT + "\n"
    assert state["keys"] == ["sample-bridge.secretvalue"]
    assert "secretvalue" not in result.stderr


@pytest.mark.real_process(reason="the CLI hook process must not carry its API key across a redirect")
def test_command_hook_does_not_follow_redirect_with_key(registry_server, tmp_path):
    url, other_state = registry_server  # Stands in for a different origin.
    hits = {"requests": 0, "keys": []}

    class Redirector(BaseHTTPRequestHandler):
        def do_GET(self):
            hits["requests"] += 1
            hits["keys"].append(self.headers.get("X-LLM-Bawt-Key"))
            self.send_response(302)
            self.send_header("Location", f"{url}{self.path}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        key_file = tmp_path / "api-key"
        key_file.write_text("sample-bridge.secretvalue\n", encoding="utf-8")
        command = _ensure_compact_hook({})["hooks"]["PreCompact"][0]["hooks"][0]["command"]
        env = {**os.environ, "LLM_BAWT_API_URL": f"http://127.0.0.1:{server.server_port}",
               "LLM_BAWT_API_KEY_FILE": str(key_file)}
        result = subprocess.run(shlex.split(command), input=json.dumps({"trigger": "auto"}), text=True,
                                capture_output=True, check=True, timeout=10, env=env)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert hits["keys"] == ["sample-bridge.secretvalue"]  # Configured origin is still authenticated.
    assert other_state["requests"] == 0  # Redirect target never contacted.
    assert result.stdout == ""
    assert "unavailable" in result.stderr
    assert "secretvalue" not in result.stderr
