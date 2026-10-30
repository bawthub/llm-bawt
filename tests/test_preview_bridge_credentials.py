"""Preview-role bridge credentials come only from the preview gateway (TASK-1802).

Every refusal path must fail closed: no prod broker, no token file, no legacy
bundle, no env token, no stale token, no API-key fallback.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_bridge import preview_credentials as pc
from claude_code_bridge import _bridge_helpers as helpers
from claude_code_bridge.native_auth import AUTH_ENV_KEYS
from claude_code_bridge.provider_key_broker import (
    BrokerKeyError,
    BrokerKeyNotInstalled,
    fetch_provider_api_key,
)
from claude_code_bridge.proxy.adapters.openai import OpenAIAdapter
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.adapters.openrouter import OpenRouterAdapter

GATEWAY = "http://preview-gw.test:8642"
NOW = 1_800_000_000.0  # epoch seconds
ENV = {pc.ENABLED_ENV: "1", pc.URL_ENV: GATEWAY}


def _resp(status: int, body: object = None, headers: dict | None = None) -> httpx.Response:
    content = b"" if body is None else json.dumps(body).encode()
    return httpx.Response(status, content=content, headers=headers or {})


class _Get:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _claude_ok(**over):
    body = {"access_token": "tok-c", "expires_at": int((NOW + 3600) * 1000), "state": "ok"}
    body.update(over)
    return body


def _chatgpt_ok(**over):
    body = {"access_token": "tok-o", "expires_at": int(NOW + 3600), "state": "ok", "account_id": "acct"}
    body.update(over)
    return body


def _fetch(provider, response, env=ENV):
    get = _Get(response)
    return pc.fetch(provider, environ=env, get=get, now=lambda: NOW), get


# ── accepted shapes ────────────────────────────────────────────────────────

def test_claude_ms_expiry_accepted_and_request_is_bare():
    cred, get = _fetch("claude", _resp(200, _claude_ok()))
    assert cred.access_token == "tok-c"
    assert cred.expires_at_s == pytest.approx(NOW + 3600)
    assert cred.account_id is None
    [(url, kwargs)] = get.calls
    assert url == f"{GATEWAY}/v1/providers/claude/token"
    assert kwargs["headers"] == {}
    assert kwargs["follow_redirects"] is False
    assert "params" not in kwargs


def test_chatgpt_seconds_expiry_and_account_id_accepted():
    cred, get = _fetch("openai_chatgpt", _resp(200, _chatgpt_ok()))
    assert (cred.access_token, cred.account_id) == ("tok-o", "acct")
    assert cred.expires_at_s == pytest.approx(NOW + 3600)
    assert get.calls[0][0] == f"{GATEWAY}/v1/providers/openai_chatgpt/token"


# ── refusals ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("provider", "body"),
    [
        ("claude", _claude_ok(expires_at=int(NOW + 3600))),  # seconds, wants ms
        ("openai_chatgpt", _chatgpt_ok(expires_at=int((NOW + 3600) * 1000))),  # ms, wants s
        ("claude", _claude_ok(expires_at="soon")),
        ("claude", _claude_ok(expires_at=True)),
        ("claude", _claude_ok(expires_at=None)),
    ],
)
def test_wrong_unit_or_non_numeric_expiry_refused(provider, body):
    with pytest.raises(pc.PreviewCredentialError):
        _fetch(provider, _resp(200, body))


@pytest.mark.parametrize("account_id", [None, "", 7])
def test_chatgpt_missing_account_id_refused(account_id):
    body = _chatgpt_ok()
    if account_id is None:
        del body["account_id"]
    else:
        body["account_id"] = account_id
    with pytest.raises(pc.PreviewCredentialError, match="account_id"):
        _fetch("openai_chatgpt", _resp(200, body))


@pytest.mark.parametrize("status", [403, 503, 500, 401, 204])
def test_non_200_refused(status):
    with pytest.raises(pc.PreviewCredentialError, match=str(status)):
        _fetch("claude", _resp(status, _claude_ok()))


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirect_refused_not_followed(status):
    with pytest.raises(pc.PreviewCredentialError, match=str(status)):
        _fetch("claude", _resp(status, None, {"location": "http://app:8642/v1/providers/claude/token"}))


@pytest.mark.parametrize("state", ["stale", "refreshing", None, "OK"])
def test_non_ok_state_refused(state):
    with pytest.raises(pc.PreviewCredentialError, match="state"):
        _fetch("claude", _resp(200, _claude_ok(state=state)))


def test_under_sixty_seconds_refused():
    body = _claude_ok(expires_at=int((NOW + 59) * 1000))
    with pytest.raises(pc.PreviewCredentialError, match="60s"):
        _fetch("claude", _resp(200, body))


@pytest.mark.parametrize("token", [None, "", 5])
def test_missing_token_refused(token):
    with pytest.raises(pc.PreviewCredentialError, match="access_token"):
        _fetch("claude", _resp(200, _claude_ok(access_token=token)))


def test_invalid_json_and_non_object_refused():
    with pytest.raises(pc.PreviewCredentialError):
        _fetch("claude", httpx.Response(200, content=b"not json"))
    with pytest.raises(pc.PreviewCredentialError):
        _fetch("claude", _resp(200, ["ok"]))


def test_transport_error_refused():
    def boom(url, **kwargs):
        raise httpx.ConnectError("down")

    with pytest.raises(pc.PreviewCredentialError, match="unreachable"):
        pc.fetch("claude", environ=ENV, get=boom, now=lambda: NOW)


@pytest.mark.parametrize("url", ["", "   ", "preview-gw:8642", "file:///etc/passwd"])
def test_unset_or_non_http_gateway_url_refused(url):
    with pytest.raises(pc.PreviewCredentialError, match=pc.URL_ENV):
        _fetch("claude", _resp(200, _claude_ok()), env={pc.ENABLED_ENV: "1", pc.URL_ENV: url})


def test_unsupported_provider_refused_without_request():
    get = _Get(_resp(200, _claude_ok()))
    with pytest.raises(pc.PreviewCredentialError, match="not served"):
        pc.fetch("openrouter", environ=ENV, get=get, now=lambda: NOW)
    assert get.calls == []


def test_client_refuses_outside_preview_role():
    with pytest.raises(pc.PreviewCredentialError, match="outside preview"):
        _fetch("claude", _resp(200, _claude_ok()), env={pc.URL_ENV: GATEWAY})


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_only_exact_one_enables_preview(value):
    assert pc.enabled({pc.ENABLED_ENV: value}) is False
    assert pc.enabled({pc.ENABLED_ENV: "1"}) is True


# ── bridge wiring: no fallback anywhere in preview ─────────────────────────

@pytest.fixture
def preview_env(monkeypatch, tmp_path):
    """Preview role with every prod/legacy source planted, so a fallback would show."""
    monkeypatch.setenv(pc.ENABLED_ENV, "1")
    monkeypatch.setenv(pc.URL_ENV, GATEWAY)
    monkeypatch.setenv("LLM_BAWT_API_URL", "http://prod-app.test:8642")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    token_file = tmp_path / "broker"
    token_file.write_text("bridge-secret")
    monkeypatch.setenv("BRIDGE_BROKER_TOKEN_FILE", str(token_file))
    bundle = tmp_path / ".credentials.json"
    bundle.write_text(json.dumps({"claudeAiOauth": {"accessToken": "legacy", "expiresAt": 9_999_999_999_999}}))
    monkeypatch.setattr(helpers, "_LEGACY_CREDENTIALS_PATH", bundle)
    monkeypatch.setattr(pc.time, "time", lambda: NOW)
    calls: list[str] = []

    def install(response: httpx.Response):
        def fake_get(url, **kwargs):
            calls.append(url)
            assert url.startswith(GATEWAY), f"preview bridge reached {url}"
            assert "X-Bridge-Token" not in (kwargs.get("headers") or {})
            assert not kwargs.get("params")
            return response

        monkeypatch.setattr(httpx, "get", fake_get)

    return install, calls


def test_claude_preview_uses_gateway_and_never_forwards_force(preview_env):
    install, calls = preview_env
    install(_resp(200, _claude_ok()))
    token, expires_ms = helpers._fetch_broker_token(force=True, rejected_token_sha256="ab")
    assert token == "tok-c"
    assert expires_ms == int((NOW + 3600) * 1000)
    assert helpers._get_fresh_oauth_token(force_refresh=True) == "tok-c"
    assert calls == [f"{GATEWAY}/v1/providers/claude/token"] * 2


@pytest.mark.parametrize("status", [403, 503])
def test_claude_preview_refusal_has_no_bundle_env_or_stale_fallback(preview_env, status):
    install, _ = preview_env
    install(_resp(status, {"detail": "no"}))
    assert helpers._fetch_broker_token() == (None, None)
    assert helpers._get_fresh_oauth_token() is None


def test_chatgpt_preview_uses_gateway(preview_env):
    install, calls = preview_env
    install(_resp(200, _chatgpt_ok()))
    adapter = OpenAIChatGPTAdapter()
    assert adapter._fetch_broker_token(force=True) == ("tok-o", "acct", pytest.approx(NOW + 3600))
    assert calls == [f"{GATEWAY}/v1/providers/openai_chatgpt/token"]


def test_chatgpt_preview_refusal_raises(preview_env):
    install, _ = preview_env
    install(_resp(200, _chatgpt_ok(expires_at=int((NOW + 3600) * 1000))))
    with pytest.raises(RuntimeError, match="Preview ChatGPT credential refused"):
        OpenAIChatGPTAdapter()._fetch_broker_token()


def test_chatgpt_preview_never_serves_cached_bearer_after_refusal(preview_env):
    # Codex r1 B2: a persistent adapter must consult the gateway every request.
    import asyncio

    install, calls = preview_env
    adapter = OpenAIChatGPTAdapter()
    install(_resp(200, _chatgpt_ok()))
    token, _base = asyncio.run(adapter.authorize())
    assert token == "tok-o"
    install(_resp(403, {"detail": "revoked"}))
    with pytest.raises(RuntimeError, match="Preview ChatGPT credential refused"):
        asyncio.run(adapter.authorize())
    assert len(calls) == 2


def test_api_key_brokers_refuse_in_preview_without_requests(preview_env):
    install, calls = preview_env
    install(_resp(200, {"access_token": "k"}))
    with pytest.raises(BrokerKeyError) as exc:
        fetch_provider_api_key("xai")
    assert not isinstance(exc.value, BrokerKeyNotInstalled)  # would unlock an env-key fallback
    with pytest.raises(RuntimeError, match="not served in preview"):
        OpenAIAdapter._fetch_broker_key()
    with pytest.raises(RuntimeError, match="not served in preview"):
        OpenRouterAdapter._fetch_broker_key()
    assert calls == []


# ── startup: no local Claude credential may be visible ─────────────────────

def _bundle(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / pc.BUNDLE_NAME
    path.write_text("{}")
    return str(path)


def _local(env, home):
    return pc.local_claude_credentials(AUTH_ENV_KEYS, env, home=str(home))


def test_clean_preview_home_has_no_local_credentials(tmp_path):
    (tmp_path / ".claude").mkdir()
    assert _local({}, tmp_path) == []


def test_home_bundle_detected(tmp_path):
    path = _bundle(tmp_path / ".claude")
    assert _local({}, tmp_path) == [path]


@pytest.mark.parametrize("env_name", pc.CONFIG_DIR_ENVS)
def test_config_dir_bundle_detected(tmp_path, env_name):
    path = _bundle(tmp_path / "cfg")
    assert _local({env_name: str(tmp_path / "cfg")}, tmp_path / "h") == [path]


def test_dangling_bundle_symlink_detected(tmp_path):
    (tmp_path / ".claude").mkdir()
    link = tmp_path / ".claude" / pc.BUNDLE_NAME
    link.symlink_to(tmp_path / "missing")
    assert _local({}, tmp_path) == [str(link)]


def test_auth_env_keys_cover_inherited_auth_token():
    # Codex r1 B1: LegacyNativeRoute would let an inherited ANTHROPIC_AUTH_TOKEN through.
    assert {"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"} <= set(AUTH_ENV_KEYS)


@pytest.mark.parametrize("env_name", AUTH_ENV_KEYS)
def test_credential_env_detected_but_blank_ignored(tmp_path, env_name):
    assert _local({env_name: "x"}, tmp_path) == [env_name]
    assert _local({env_name: "  "}, tmp_path) == []


class _BootstrapReached(Exception):
    pass


@pytest.fixture
def bridge_main(monkeypatch, tmp_path):
    from claude_code_bridge import __main__ as bridge_main_mod

    def reached():
        raise _BootstrapReached

    monkeypatch.setattr(bridge_main_mod, "bootstrap_claude_home", reached)
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (*AUTH_ENV_KEYS, *pc.CONFIG_DIR_ENVS):
        monkeypatch.delenv(name, raising=False)
    return bridge_main_mod, tmp_path


@pytest.mark.parametrize("plant", ["home_bundle", "config_dir", *AUTH_ENV_KEYS])
def test_preview_bridge_refuses_to_start_with_local_credentials(bridge_main, monkeypatch, caplog, plant):
    mod, home = bridge_main
    monkeypatch.setenv(pc.ENABLED_ENV, "1")
    if plant == "home_bundle":
        _bundle(home / ".claude")
    elif plant == "config_dir":
        _bundle(home / "cfg")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / "cfg"))
    else:
        monkeypatch.setenv(plant, "secret-value")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1
    assert "Preview bridge refuses to start" in caplog.text
    assert "secret-value" not in caplog.text  # names only, never values


def test_clean_preview_bridge_proceeds_to_bootstrap(bridge_main, monkeypatch):
    mod, _ = bridge_main
    monkeypatch.setenv(pc.ENABLED_ENV, "1")
    with pytest.raises(_BootstrapReached):
        mod.main()


def test_non_preview_bridge_ignores_local_credentials(bridge_main, monkeypatch):
    mod, home = bridge_main
    monkeypatch.delenv(pc.ENABLED_ENV, raising=False)
    _bundle(home / ".claude")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "x")
    with pytest.raises(_BootstrapReached):
        mod.main()
