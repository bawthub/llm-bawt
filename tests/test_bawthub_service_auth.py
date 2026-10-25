"""TASK-1472: llm-bawt sends the BawtHub service bearer on every BawtHub call.

The conftest fixture ``no_derived_bawthub_service_token`` makes derivation
fail by default, so "no override" means "no token" unless a test opts in with
``derived`` (a fixed token) or ``master_key`` (real derivation, fixture key).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging

import httpx
import pytest
from cryptography.fernet import Fernet

from llm_bawt.bawthub_auth import (
    DERIVATION_LABEL,
    EXPORT_FILE_ENV,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    BawtHubServiceAuth,
    ServiceTokenSource,
    TokenCheck,
    service_token,
)
from llm_bawt.service.providers import crypto

TOKEN = "fixture-service-token-0123456789"
DERIVED = "fixture-derived-token-" + "d" * 42
FIXTURE_KEY = Fernet.generate_key().decode()


@pytest.fixture
def derived(monkeypatch):
    """Derivation succeeds with a fixed token."""
    monkeypatch.setattr(service_token, "derive_service_token", lambda: DERIVED)
    return DERIVED


@pytest.fixture
def master_key(monkeypatch):
    """Real derivation from a fixture master key (never the live key file).

    ``LLM_BAWT_SECRET_KEY`` is the first key source, so the key file is never read.
    """
    monkeypatch.setattr(service_token, "derive_service_token", _REAL_DERIVE)
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", FIXTURE_KEY)
    return FIXTURE_KEY


# Captured at import, before the conftest fixture swaps it out per test.
_REAL_DERIVE = service_token.derive_service_token


def _expected_token(key: str) -> str:
    raw = base64.urlsafe_b64decode(key.encode())
    return hmac.new(raw, DERIVATION_LABEL.encode(), hashlib.sha256).hexdigest()


# --- derivation ---------------------------------------------------------------

def test_derived_token_is_hmac_of_the_master_key_and_stable(master_key):
    first = ServiceTokenSource({}).read()
    assert first == _expected_token(master_key)
    assert len(first) == 64  # BawtHub refuses tokens under 32 chars
    assert ServiceTokenSource({}).read() == first  # same key, same token, new process


def test_derived_token_changes_with_the_master_key(master_key, monkeypatch):
    first = ServiceTokenSource({}).read()
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", Fernet.generate_key().decode())
    assert ServiceTokenSource({}).read() != first


def test_derivation_is_domain_separated_by_label(master_key):
    assert crypto.derive_key(DERIVATION_LABEL) != crypto.derive_key("some-other-use/v1")
    with pytest.raises(ValueError):
        crypto.derive_key("")


@pytest.mark.parametrize("bad_key", ["not-a-fernet-key", base64.urlsafe_b64encode(b"short").decode()])
def test_unusable_master_key_fails_closed(master_key, monkeypatch, bad_key):
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", bad_key)
    source = ServiceTokenSource({})
    assert source.read() is None
    assert source.check(f"Bearer {_expected_token(master_key)}") is TokenCheck.UNCONFIGURED


@pytest.fixture
def key_file(monkeypatch, tmp_path):
    """Real derivation from the key-file source, at a scratch path (never the live one)."""
    monkeypatch.setattr(service_token, "derive_service_token", _REAL_DERIVE)
    monkeypatch.delenv("LLM_BAWT_SECRET_KEY", raising=False)
    path = tmp_path / "config" / "secret.key"
    monkeypatch.setattr(crypto, "_key_file", lambda: path)
    return path


def test_unpersistable_master_key_fails_closed(key_file, monkeypatch, tmp_path):
    """A key generated per call would derive a different token each time."""
    blocker = tmp_path / "a-file-not-a-directory"
    blocker.write_text("")
    monkeypatch.setattr(crypto, "_key_file", lambda: blocker / "secret.key")
    for _ in range(2):
        with pytest.raises(crypto.EphemeralKeyError):
            crypto.derive_key(DERIVATION_LABEL)
    source = ServiceTokenSource({})
    assert source.read() is None
    assert source.check("Bearer anything-at-all-0123456789") is TokenCheck.UNCONFIGURED
    Fernet(crypto._load_or_create_key())  # encryption alone still gets an in-process key


def test_first_use_persists_one_key_and_derives_from_it(key_file):
    first = ServiceTokenSource({}).read()
    assert key_file.exists()
    assert oct(key_file.stat().st_mode & 0o777) == "0o600"
    stored = key_file.read_text().strip()
    assert first == _expected_token(stored)
    assert crypto._load_or_create_key().decode() == stored  # what Fernet uses too
    assert ServiceTokenSource({}).read() == first  # a second process, or after restart
    assert sorted(p.name for p in key_file.parent.iterdir()) == ["secret.key"]


def test_racing_key_creation_adopts_the_key_on_disk(key_file, monkeypatch):
    """Two processes bootstrapping at once must end up on one key."""
    winner = Fernet.generate_key()
    real_link = crypto.os.link

    def lose_the_race(src, dst):
        key_file.write_bytes(winner)  # the other process got there first
        real_link(src, dst)

    monkeypatch.setattr(crypto.os, "link", lose_the_race)
    assert ServiceTokenSource({}).read() == _expected_token(winner.decode())
    assert key_file.read_bytes() == winner
    assert sorted(p.name for p in key_file.parent.iterdir()) == ["secret.key"]


def test_no_override_uses_the_derived_token(derived):
    assert ServiceTokenSource({}).read() == DERIVED
    assert ServiceTokenSource({}).check(f"Bearer {DERIVED}") is TokenCheck.OK


def test_explicit_env_overrides_the_derived_token(derived):
    source = ServiceTokenSource({TOKEN_ENV: TOKEN})
    assert source.read() == TOKEN
    assert source.check(f"Bearer {DERIVED}") is TokenCheck.INVALID


def test_explicit_file_overrides_the_derived_token(derived, tmp_path):
    secret = tmp_path / "token"
    secret.write_text(TOKEN)
    assert ServiceTokenSource({TOKEN_FILE_ENV: str(secret)}).read() == TOKEN


def test_own_export_file_is_not_an_override(derived, tmp_path):
    """Reading back our own export would pin a stale token across key rotation."""
    export = tmp_path / "export" / "token"
    export.parent.mkdir()
    export.write_text("stale-token-from-an-old-master-key-000000")
    environ = {TOKEN_FILE_ENV: str(export), EXPORT_FILE_ENV: str(tmp_path / "export" / ".." / "export" / "token")}
    assert ServiceTokenSource(environ).read() == DERIVED


def test_derivation_failure_is_logged_once_and_value_never_logged(caplog, master_key, monkeypatch):
    caplog.set_level(logging.INFO, logger="llm_bawt.bawthub_auth.service_token")
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", "not-a-fernet-key")
    source = ServiceTokenSource({})
    for _ in range(3):
        assert source.read() is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", master_key)
    token = source.read()
    assert token == _expected_token(master_key)
    assert token not in caplog.text


# --- override token source ----------------------------------------------------

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


def test_unreadable_or_empty_file_never_falls_back_to_env_or_derived(tmp_path, derived):
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

    def routed(client_cls):
        # Patch __init__, not the class: replacing the class with a function
        # breaks any module first imported under the patch that uses it in an
        # annotation (``httpx.AsyncClient | None``) or isinstance check.
        real_init = client_cls.__init__

        def init(self, *args, **kwargs):
            real_init(self, *args, **{**kwargs, "transport": transport})

        monkeypatch.setattr(client_cls, "__init__", init)

    routed(httpx.AsyncClient)
    routed(httpx.Client)
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


def test_task_api_sends_the_derived_token_with_no_override(bawthub, task_api, monkeypatch, master_key):
    """The zero-config path: the process-wide source derives from the master key."""
    monkeypatch.delenv(TOKEN_ENV)
    asyncio.run(task_api.api_get("/tasks"))
    _assert_bearer(bawthub, f"Bearer {_expected_token(master_key)}")


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
