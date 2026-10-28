"""TASK-1477 phase 2: bridges send the derived X-Bridge-Token from its file.

Precedence for each sender: readable non-empty file, then the legacy env
value, then no header at all. With no file this is the pre-1477 behaviour.
"""

from __future__ import annotations

import ast
import http.server
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent_bridge import bridge_token
from agent_bridge.bridge_token import BROKER, GIT_CREDENTIAL, HEADER

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "docker" / "git-credential-bawthub"
DERIVED = "d" * 64
LEGACY = "l" * 40


def _file(tmp_path: Path, content: str) -> str:
    path = tmp_path / "token"
    path.write_text(content)
    return str(path)


# --- precedence ------------------------------------------------------------
@pytest.mark.parametrize("sender", [BROKER, GIT_CREDENTIAL], ids=["broker", "git"])
def test_file_wins_then_env_then_nothing(sender, tmp_path):
    path = _file(tmp_path, DERIVED + "\n")
    both = {sender.file_env: path, sender.legacy_env: LEGACY}
    assert bridge_token.headers(sender, both) == {HEADER: DERIVED}
    assert bridge_token.headers(sender, {sender.legacy_env: LEGACY}) == {HEADER: LEGACY}
    assert bridge_token.headers(sender, {}) == {}
    assert bridge_token.headers(sender, {sender.legacy_env: "  "}) == {}


@pytest.mark.parametrize("sender", [BROKER, GIT_CREDENTIAL], ids=["broker", "git"])
def test_empty_or_unreadable_file_falls_back_to_env(sender, tmp_path):
    empty = _file(tmp_path, "\n")
    absent = str(tmp_path / "absent")
    for path in (empty, absent):
        env = {sender.file_env: path, sender.legacy_env: LEGACY}
        assert bridge_token.token(sender, env) == LEGACY
        assert bridge_token.headers(sender, {sender.file_env: path}) == {}


def test_undecodable_file_falls_back_to_env(tmp_path):
    path = tmp_path / "token"
    path.write_bytes(b"\xff\xfe\x80")
    env = {BROKER.file_env: str(path), BROKER.legacy_env: LEGACY}
    assert bridge_token.token(BROKER, env) == LEGACY


def test_senders_do_not_cross_gates(tmp_path):
    env = {BROKER.file_env: _file(tmp_path, DERIVED), BROKER.legacy_env: LEGACY}
    assert bridge_token.token(GIT_CREDENTIAL, env) is None


def test_file_is_reread_after_rotation(tmp_path):
    path = _file(tmp_path, DERIVED)
    env = {BROKER.file_env: path}
    assert bridge_token.token(BROKER, env) == DERIVED
    Path(path).write_text("r" * 64)
    assert bridge_token.token(BROKER, env) == "r" * 64


def test_an_unreadable_file_is_logged_by_name_only(tmp_path, caplog):
    path = str(tmp_path / "absent")
    bridge_token._warned.discard(path)
    with caplog.at_level("WARNING"):
        bridge_token.token(BROKER, {BROKER.file_env: path, BROKER.legacy_env: LEGACY})
        bridge_token.token(BROKER, {BROKER.file_env: path, BROKER.legacy_env: LEGACY})
    assert len(caplog.records) == 1 and LEGACY not in caplog.text


# --- call sites ------------------------------------------------------------
def test_provider_key_broker_sends_the_file_token(monkeypatch, tmp_path):
    from claude_code_bridge import provider_key_broker

    seen = {}

    def fake_get(url, headers=None, **_kwargs):
        seen["headers"] = headers
        return SimpleNamespace(is_error=False, json=lambda: {"access_token": "k"})

    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app:8642")
    monkeypatch.setenv(BROKER.file_env, _file(tmp_path, DERIVED))
    monkeypatch.setenv(BROKER.legacy_env, LEGACY)
    monkeypatch.setattr(httpx, "get", fake_get)
    assert provider_key_broker.fetch_provider_api_key("openai-api") == "k"
    assert seen["headers"] == {HEADER: DERIVED}


_SENDER_PACKAGES = ("claude_code_bridge", "codex_bridge", "agent_bridge")
_GATE_NAMES = {
    "x-bridge-token",
    "bridge_claude_token_secret",
    "bridge_git_credential_token",
    "bridge_broker_token_file",
    "bridge_git_credential_token_file",
}


def test_bridge_code_reads_gate_tokens_only_through_bridge_token():
    """Guard: no bridge builds the header or reads a gate env var itself."""
    own = (ROOT / "src/agent_bridge/bridge_token.py").resolve()
    offenders = []
    for package in _SENDER_PACKAGES:
        for path in (ROOT / "src" / package).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if path.resolve() == own or not any(n in source.lower() for n in _GATE_NAMES):
                continue
            offenders += [
                f"{path.relative_to(ROOT)}:{node.lineno}"
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.strip().lower() in _GATE_NAMES
            ]
    assert offenders == []


# --- git credential helper (shell + curl) ------------------------------------
class _Recorder(http.server.BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_GET(self):  # noqa: N802 - http.server API
        _Recorder.requests.append({"path": self.path, "token": self.headers.get(HEADER)})
        body = b"username=x-access-token\npassword=t\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def app_stub():
    _Recorder.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", _Recorder.requests
    server.shutdown()
    server.server_close()


def _helper(op: str, url: str, **env: str) -> subprocess.CompletedProcess[str]:
    base = {"PATH": os.environ["PATH"], "LLM_BAWT_API_URL": url}
    return subprocess.run([str(HELPER), op], env={**base, **env}, input="protocol=https\n\n",
                          text=True, capture_output=True, timeout=10)


@pytest.mark.real_process(reason="runs the real git credential helper with curl")
def test_git_helper_token_precedence(app_stub, tmp_path):
    url, requests = app_stub
    derived = _file(tmp_path, DERIVED + "\n")
    empty = str(tmp_path / "empty")
    Path(empty).write_text("")
    cases = [
        ({"BRIDGE_GIT_CREDENTIAL_TOKEN_FILE": derived, "BRIDGE_GIT_CREDENTIAL_TOKEN": LEGACY}, DERIVED),
        ({"BRIDGE_GIT_CREDENTIAL_TOKEN_FILE": empty, "BRIDGE_GIT_CREDENTIAL_TOKEN": LEGACY}, LEGACY),
        ({"BRIDGE_GIT_CREDENTIAL_TOKEN_FILE": str(tmp_path / "absent"),
          "BRIDGE_GIT_CREDENTIAL_TOKEN": LEGACY}, LEGACY),
        ({}, None),  # no token: the header is omitted, not sent empty
    ]
    for env, expected in cases:
        result = _helper("get", url, **env)
        assert result.returncode == 0, result.stderr
        assert "password=t" in result.stdout
        assert requests[-1] == {"path": "/v1/providers/github/git-credential", "token": expected}
    assert len(requests) == len(cases)


@pytest.mark.real_process(reason="runs the real git credential helper")
@pytest.mark.parametrize("op", ["store", "erase"])
def test_git_helper_ignores_non_get_ops(app_stub, op):
    url, requests = app_stub
    result = _helper(op, url, BRIDGE_GIT_CREDENTIAL_TOKEN=LEGACY)
    assert result.returncode == 0 and requests == []
