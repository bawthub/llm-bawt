"""TASK-1472: llm-bawt sends the BawtHub service bearer on every BawtHub call."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from llm_bawt.bawthub_auth import (
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    BawtHubServiceAuth,
    ServiceTokenSource,
    TokenCheck,
)

TOKEN = "fixture-service-token-0123456789"


# --- token source -----------------------------------------------------------

def test_env_token_is_trimmed():
    assert ServiceTokenSource({TOKEN_ENV: f"  {TOKEN}\n"}).read() == TOKEN


def test_file_wins_over_env_and_is_trimmed(tmp_path):
    secret = tmp_path / "token"
    secret.write_text(f"{TOKEN}\n")
    source = ServiceTokenSource({TOKEN_ENV: "env-value", TOKEN_FILE_ENV: str(secret)})
    assert source.read() == TOKEN


def test_file_is_reread_so_rotation_needs_no_restart(tmp_path):
    secret = tmp_path / "token"
    secret.write_text("first")
    source = ServiceTokenSource({TOKEN_FILE_ENV: str(secret)})
    assert source.read() == "first"
    secret.write_text("second")
    assert source.read() == "second"


@pytest.mark.parametrize("environ", [
    {},
    {TOKEN_ENV: "   "},
])
def test_no_token_reads_as_none(environ):
    assert ServiceTokenSource(environ).read() is None


def test_unreadable_or_empty_file_never_falls_back_to_env(tmp_path):
    empty = tmp_path / "empty"
    empty.write_text("\n")
    for path in (tmp_path / "missing", empty):
        assert ServiceTokenSource({TOKEN_ENV: TOKEN, TOKEN_FILE_ENV: str(path)}).read() is None


def test_missing_token_is_logged_once_and_value_never_logged(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="llm_bawt.bawthub_auth.service_token")
    environ: dict[str, str] = {}
    source = ServiceTokenSource(environ)
    for _ in range(3):
        source.read()
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    environ[TOKEN_ENV] = TOKEN
    source.read()
    environ.pop(TOKEN_ENV)
    source.read()  # a later regression is reported again
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert TOKEN not in caplog.text


@pytest.mark.parametrize("header,expected", [
    (f"Bearer {TOKEN}", TokenCheck.OK),
    (f"bearer   {TOKEN}  ", TokenCheck.OK),
    (None, TokenCheck.MISSING),
    ("", TokenCheck.MISSING),
    ("Bearer ", TokenCheck.MISSING),
    (f"Basic {TOKEN}", TokenCheck.MISSING),
    (TOKEN, TokenCheck.MISSING),
    ("Bearer wrong-token", TokenCheck.INVALID),
    (f"Bearer {TOKEN}x", TokenCheck.INVALID),
    (f"Bearer {TOKEN[:-1]}", TokenCheck.INVALID),
])
def test_check(header, expected):
    assert ServiceTokenSource({TOKEN_ENV: TOKEN}).check(header) is expected


def test_check_fails_closed_when_this_process_has_no_token():
    source = ServiceTokenSource({})
    assert source.check(None) is TokenCheck.UNCONFIGURED
    assert source.check("Bearer anything") is TokenCheck.UNCONFIGURED


def test_auth_adds_bearer_only_when_configured():
    def flow(source):
        request = httpx.Request("GET", "http://bawthub.test/api/tasks/tasks")
        return next(BawtHubServiceAuth(source).auth_flow(request))

    assert flow(ServiceTokenSource({TOKEN_ENV: TOKEN})).headers["Authorization"] == f"Bearer {TOKEN}"
    assert "Authorization" not in flow(ServiceTokenSource({})).headers


# --- every BawtHub caller ---------------------------------------------------

@pytest.fixture
def bawthub(monkeypatch):
    """Route every httpx client to a recorder; configure the process token."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith("/associations") and request.method == "PUT":
            return httpx.Response(200, json={"association": {"id": "a1"}})
        if path.endswith("/initiative/events"):
            return httpx.Response(200, json={"events": []})
        if path.endswith("/commit-evidence"):
            return httpx.Response(200, json={"evidence": {}})
        if path.endswith("/search/semantic"):
            return httpx.Response(200, json={"mode": "semantic", "results": []})
        return httpx.Response(200, json={"generation": 1, "tasks": []})

    transport = httpx.MockTransport(handle)
    real_async, real_sync = httpx.AsyncClient, httpx.Client
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(transport=transport, **kw))
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_sync(transport=transport, **kw))
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.delenv(TOKEN_FILE_ENV, raising=False)
    return seen


def _assert_bearer(seen: list[httpx.Request], expected: str | None = f"Bearer {TOKEN}"):
    assert seen, "the caller made no BawtHub request"
    assert {r.headers.get("Authorization") for r in seen} == {expected}


@pytest.fixture
def task_api(monkeypatch):
    from llm_bawt.mcp_server import task_api

    monkeypatch.setattr(task_api, "_client", None)
    yield task_api
    task_api._client = None


def test_task_api_sends_bearer_with_bot_id(bawthub, task_api):
    asyncio.run(task_api.api_post("/tasks", {"title": "x"}, headers=task_api.headers("fixture-bot")))
    _assert_bearer(bawthub)
    assert bawthub[0].headers["X-Agent-Bot-Id"] == "fixture-bot"


def test_task_api_sends_no_authorization_without_a_token(bawthub, task_api, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV)
    asyncio.run(task_api.api_get("/tasks"))
    _assert_bearer(bawthub, None)


def test_task_association_sends_bearer(bawthub, monkeypatch):
    from llm_bawt.mcp_server import task_association
    from llm_bawt.task_turn_context import TaskTurnContext

    monkeypatch.setattr(task_association, "current_task_turn_context", lambda: TaskTurnContext(
        session_id="sess-1", turn_id="turn-" + "a" * 32, trigger_message_id="msg-1",
        bot_id="fixture-bot", user_id="fixture-user", issued_at=0))
    asyncio.run(task_association.associate_current_task("TASK-1"))
    _assert_bearer(bawthub)


def test_initiative_state_sends_bearer(bawthub):
    from llm_bawt.mcp_server import initiative_tools

    asyncio.run(initiative_tools.initiative_state(task_id="TASK-1", action="get"))
    _assert_bearer(bawthub)


def test_recall_task_source_sends_bearer(bawthub):
    from llm_bawt.recall.contracts import RecallScope
    from llm_bawt.recall.task_source import TaskSource

    scope = RecallScope("fixture-bot", "fixture-user", "sess-1")
    asyncio.run(TaskSource().search("anything", scope, ()))
    _assert_bearer(bawthub)


def test_changed_file_commit_evidence_sends_bearer(bawthub):
    from llm_bawt.service.changed_file_commits import reconcile_files

    reconcile_files([{"path": "repo/file", "in_repo": True}])
    _assert_bearer(bawthub)
