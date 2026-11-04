"""TASK-1280: offline repository credentials and encrypted App configuration.

Unmarked hermetic tests are selected automatically by the canonical unit suite.
All upstream calls, provider storage and encryption roots are private fakes.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi import FastAPI

from llm_bawt.service.providers import crypto, github
from llm_bawt.service.providers.base import ConnectionRecord, CredentialStore
from llm_bawt.service.routes import providers


class MemoryStore:
    """Exercise real encrypted serialization without a DB or a real root key."""
    def __init__(self):
        self.rows = {}
        self.writes = []

    def load(self, provider):
        raw = self.rows.get(provider)
        return CredentialStore._record(provider, copy.deepcopy(raw)) if raw else None

    def save(self, record):
        self.rows[record.provider] = CredentialStore._payload(record)
        self.writes.append(record.provider)

    def durable_key_problem(self):
        return None


@pytest.fixture(scope="module")
def app_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return key, pem


@pytest.fixture
def adapter(monkeypatch):
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: signer)
    for name in ("GITHUB_APP_ID", "GITHUB_APP_CLIENT_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_APP_PRIVATE_KEY_PATH"):
        monkeypatch.delenv(name, raising=False)
    # A missed upstream mock fails, never reaches GitHub.
    monkeypatch.setattr(github.httpx, "get", Mock(side_effect=AssertionError("unexpected upstream GET")))
    monkeypatch.setattr(github.httpx, "post", Mock(side_effect=AssertionError("unexpected upstream POST")))
    result = object.__new__(github.GitHubAdapter)
    result.store = MemoryStore()
    return result


def upstream(monkeypatch, *, status=200, installation=82):
    get = Mock(return_value=httpx.Response(
        status, json={"id": installation},
        request=httpx.Request("GET", "https://api.github.com/repos/Other/Project/installation"),
    ))
    post = Mock(return_value=httpx.Response(
        201, json={"token": "test-installation-token"},
        request=httpx.Request("POST", "https://api.github.com/app/installations/82/access_tokens"),
    ))
    monkeypatch.setattr(github.httpx, "get", get)
    monkeypatch.setattr(github.httpx, "post", post)
    return get, post


def connected(adapter):
    adapter.store.save(ConnectionRecord(
        provider="github", status="connected", auth_method="device_oauth",
        meta={"installation_id": 17, "repos": ["Existing/Repository"]},
        secret={"user_token": "test-user-token"},
    ))


def configure(adapter, app_key):
    adapter.configure_app("123456", "test-client-id", app_key[1])


def assert_app_jwt(headers, key):
    assert headers["Authorization"].startswith("Bearer ")
    token = headers["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(token, key.public_key(), algorithms=["RS256"])
    assert claims["iss"] == "123456"
    assert claims["exp"] - claims["iat"] == 600
    return token


def test_repository_uses_app_lookup_and_restricts_token_without_global_mutation(adapter, app_key, monkeypatch):
    configure(adapter, app_key)
    connected(adapter)
    before, writes = copy.deepcopy(adapter.store.rows), list(adapter.store.writes)
    get, post = upstream(monkeypatch)
    assert adapter.mint_installation_token(repository="Other/Project") == "test-installation-token"
    assert get.call_args.args == ("https://api.github.com/repos/Other/Project/installation",)
    assert post.call_args.args == ("https://api.github.com/app/installations/82/access_tokens",)
    assert post.call_args.kwargs["json"] == {"repositories": ["Project"]}
    assert get.call_args.kwargs["headers"] == post.call_args.kwargs["headers"]
    assert_app_jwt(get.call_args.kwargs["headers"], app_key[0])
    assert adapter.store.rows == before and adapter.store.writes == writes


def test_repository_does_not_require_device_oauth_connection(adapter, app_key, monkeypatch):
    configure(adapter, app_key)
    upstream(monkeypatch)
    assert adapter.store.load("github") is None
    assert adapter.mint_installation_token(repository="Other/Project") == "test-installation-token"
    assert adapter.store.load("github") is None


@pytest.mark.parametrize("status", [403, 404, 500])
def test_repository_lookup_failure_never_falls_back_to_selected_installation(adapter, app_key, monkeypatch, status):
    configure(adapter, app_key)
    connected(adapter)
    before = copy.deepcopy(adapter.store.rows)
    _, post = upstream(monkeypatch, status=status)
    with pytest.raises(httpx.HTTPStatusError):
        adapter.mint_installation_token(repository="Other/Project")
    post.assert_not_called()
    assert adapter.store.rows == before


@pytest.mark.parametrize("installation", [None, "82", 0, -1, True])
def test_invalid_installation_response_never_mints(adapter, app_key, monkeypatch, installation):
    configure(adapter, app_key)
    _, post = upstream(monkeypatch, installation=installation)
    with pytest.raises(RuntimeError):
        adapter.mint_installation_token(repository="Other/Project")
    post.assert_not_called()


def test_legacy_omitted_repository_retains_selected_installation_and_env_fallback(adapter, app_key, monkeypatch):
    connected(adapter)
    monkeypatch.setenv("GITHUB_APP_ID", "123456")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", app_key[1])
    get, post = upstream(monkeypatch)
    assert adapter.mint_installation_token() == "test-installation-token"
    get.assert_not_called()
    assert post.call_args.args == ("https://api.github.com/app/installations/17/access_tokens",)
    assert "json" not in post.call_args.kwargs
    assert_app_jwt(post.call_args.kwargs["headers"], app_key[0])


@pytest.mark.parametrize("token", [None, "", "token\nextra=value", 42])
def test_malformed_token_response_fails_closed(adapter, app_key, monkeypatch, token):
    configure(adapter, app_key)
    _, post = upstream(monkeypatch)
    post.return_value = httpx.Response(
        201, json={"token": token}, request=httpx.Request("POST", github._API),
    )
    with pytest.raises(RuntimeError, match="token response"):
        adapter.mint_installation_token(repository="Other/Project")


def test_missing_app_configuration_is_not_an_upstream_call(adapter):
    with pytest.raises(github.GitHubConfigError):
        adapter.mint_installation_token(repository="Other/Project")
    github.httpx.get.assert_not_called()
    github.httpx.post.assert_not_called()


def test_mint_post_failure_does_not_persist_or_use_oauth_fallback(adapter, app_key, monkeypatch):
    configure(adapter, app_key)
    connected(adapter)
    before = copy.deepcopy(adapter.store.rows)
    get, post = upstream(monkeypatch)
    post.return_value = httpx.Response(
        403, json={"message": "denied"}, request=httpx.Request("POST", github._API),
    )
    with pytest.raises(httpx.HTTPStatusError):
        adapter.mint_installation_token(repository="Other/Project")
    assert get.call_count == 1 and post.call_count == 1
    assert adapter.store.rows == before


def test_legacy_still_requires_connected_selected_installation(adapter):
    with pytest.raises(RuntimeError, match="not connected"):
        adapter.mint_installation_token()
    adapter.store.save(ConnectionRecord(provider="github", status="connected"))
    with pytest.raises(RuntimeError, match="installation selected"):
        adapter.mint_installation_token()


@pytest.mark.parametrize("repository", [
    "", "Owner", "/Repo", "Owner/", "https://github.com/Owner/Repo", "Owner/Repo/extra",
    "Owner/..", "Owner/.", "Owner/Repo?x=y", "Owner/Repo#x", "Owner/Repo\n", " Owner/Repo",
    "Owner%2fRepo", "Owner/Repo%2fextra", "Owner/Repo\\extra", "../Repo", "O_wner/Repo",
])
def test_invalid_repository_rejected_before_any_storage_or_upstream(adapter, repository):
    adapter.store.load = Mock(side_effect=AssertionError("unexpected storage read"))
    with pytest.raises(ValueError, match="OWNER/REPO"):
        adapter.mint_installation_token(repository=repository)
    adapter.store.load.assert_not_called()


def test_config_is_encrypted_separate_and_stored_values_override_env(adapter, app_key, monkeypatch):
    connected(adapter)
    before = copy.deepcopy(adapter.store.rows["github"])
    configure(adapter, app_key)
    raw = json.dumps(adapter.store.rows["github-app"])
    for secret in ("123456", "test-client-id", "PRIVATE KEY", app_key[1]):
        assert secret not in raw
    stored = adapter.store.load("github-app")
    assert stored.secret == {"app_id": "123456", "client_id": "test-client-id", "private_key": app_key[1].strip()}
    assert stored.public()["meta"] == {} and "secret" not in stored.public()
    assert adapter.store.rows["github"] == before
    monkeypatch.setenv("GITHUB_APP_ID", "999")
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "wrong-env-client")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "wrong-env-key")
    assert adapter._device_client_id() == "test-client-id"
    assert_app_jwt({"Authorization": "Bearer " + adapter._installation_jwt()}, app_key[0])
    assert adapter.is_configured()


def test_device_flow_start_and_poll_use_stored_client_id(adapter, app_key, monkeypatch):
    configure(adapter, app_key)
    post = Mock(side_effect=[
        httpx.Response(200, json={"user_code": "test-code", "device_code": "test-device"},
                       request=httpx.Request("POST", github._DEVICE_CODE_URL)),
        httpx.Response(200, json={"error": "authorization_pending"}),
    ])
    monkeypatch.setattr(github.httpx, "post", post)
    assert adapter.start_device_flow().device_code == "test-device"
    assert adapter.poll_device_flow("test-device").status == "pending"
    assert all(call.kwargs["data"]["client_id"] == "test-client-id" for call in post.call_args_list)


def test_unusable_stored_config_does_not_fall_back_to_env(adapter, monkeypatch):
    monkeypatch.setenv("GITHUB_APP_CLIENT_ID", "legacy-client")
    adapter.store.save(ConnectionRecord(provider="github-app", status="connected"))
    with pytest.raises(github.GitHubConfigError):
        adapter._device_client_id()
    with pytest.raises(github.GitHubConfigError):
        adapter._installation_jwt()
    assert not adapter.is_configured()


def test_configuration_requires_durable_encryption_before_writing(adapter, app_key):
    adapter.store.durable_key_problem = lambda: "ephemeral_root"
    with pytest.raises(github.GitHubConfigError):
        configure(adapter, app_key)
    assert adapter.store.rows == {}


@pytest.mark.parametrize("bad_field", ["app_id", "client_id", "private_key"])
def test_invalid_configuration_preserves_previous_credentials(adapter, app_key, bad_field):
    configure(adapter, app_key)
    before = copy.deepcopy(adapter.store.rows)
    values = {"app_id": "123456", "client_id": "test-client-id", "private_key": app_key[1]}
    values[bad_field] = "invalid input"
    with pytest.raises(ValueError):
        adapter.configure_app(**values)
    assert adapter.store.rows == before


def test_configuration_rejects_non_rsa_key(adapter):
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    with pytest.raises(ValueError, match="RSA"):
        adapter.configure_app("123456", "test-client-id", pem)
    assert adapter.store.rows == {}


@pytest_asyncio.fixture
async def client(adapter, monkeypatch):
    monkeypatch.setattr(providers, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(providers, "get_adapter", lambda config, provider: adapter)
    gates = providers.bridge_gate.BridgeGates(
        lambda: {providers.bridge_gate.GIT_CREDENTIAL.setting_key: True},
        environ={}, derive=lambda label: f"test-token-{label}",
    )
    monkeypatch.setattr(providers.bridge_gate, "_GATES", gates)
    app = FastAPI()
    app.include_router(providers.router)
    headers = {providers.bridge_gate.HEADER: gates.derived_token(providers.bridge_gate.GIT_CREDENTIAL)}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers,
    ) as result:
        yield result


@pytest.mark.asyncio
async def test_route_optional_query_preserves_protocol_and_does_not_cache(client, adapter, monkeypatch):
    mint = Mock(return_value="test-installation-token")
    monkeypatch.setattr(adapter, "git_credential_token", mint)
    for params, kwargs in ((None, {}), ({"repository": "Other/Project"}, {"repository": "Other/Project"})):
        response = await client.get("/v1/providers/github/git-credential", params=params)
        assert response.status_code == 200
        assert response.text == "username=x-access-token\npassword=test-installation-token\n"
        assert response.headers["cache-control"] == "no-store"
        mint.assert_called_with(**kwargs)
    gates = providers.bridge_gate.bridge_gates()
    assert gates.counts(providers.bridge_gate.GIT_CREDENTIAL)["derived"] == 2
    assert gates.counts(providers.bridge_gate.BROKER)["derived"] == 0


@pytest.mark.asyncio
async def test_route_scopes_each_request_independently_and_preserves_selection(client, adapter, app_key, monkeypatch):
    configure(adapter, app_key)
    connected(adapter)
    before, writes = copy.deepcopy(adapter.store.rows), list(adapter.store.writes)
    get, post = upstream(monkeypatch)
    for repository, installation_id in (("Other/Project", 82), ("Another/Second", 93)):
        get.return_value = httpx.Response(
            200, json={"id": installation_id}, request=httpx.Request("GET", github._API),
        )
        response = await client.get("/v1/providers/github/git-credential", params={"repository": repository})
        assert response.status_code == 200
        assert response.text == "username=x-access-token\npassword=test-installation-token\n"
        assert response.headers["cache-control"] == "no-store"
        assert get.call_args.args == (f"https://api.github.com/repos/{repository}/installation",)
        assert post.call_args.args == (f"https://api.github.com/app/installations/{installation_id}/access_tokens",)
        assert post.call_args.kwargs["json"] == {"repositories": [repository.split("/")[1]]}
        assert_app_jwt(get.call_args.kwargs["headers"], app_key[0])
        assert adapter.store.rows == before and adapter.store.writes == writes

    # The request-local choices must not become the no-query default.
    legacy = await client.get("/v1/providers/github/git-credential")
    assert legacy.status_code == 200
    assert get.call_count == 2 and post.call_count == 3
    assert post.call_args.args == ("https://api.github.com/app/installations/17/access_tokens",)
    assert "json" not in post.call_args.kwargs
    assert adapter.store.rows == before and adapter.store.writes == writes


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["", "Owner/Repo/extra", "Owner/Repo?x=secret"])
async def test_route_bad_query_is_400_without_mint(client, adapter, monkeypatch, repository):
    mint = Mock()
    monkeypatch.setattr(adapter, "git_credential_token", mint)
    response = await client.get("/v1/providers/github/git-credential", params={"repository": repository})
    assert response.status_code == 400 and response.headers["cache-control"] == "no-store"
    assert response.json() == {"detail": "repository must be OWNER/REPO"}
    mint.assert_not_called()


@pytest.mark.asyncio
async def test_git_gate_is_checked_before_repository_validation_or_mint(client, adapter, monkeypatch):
    mint = Mock()
    monkeypatch.setattr(adapter, "git_credential_token", mint)
    response = await client.get(
        "/v1/providers/github/git-credential?repository=",
        headers={providers.bridge_gate.HEADER: "bad-test-token"},
    )
    assert response.status_code == 401 and response.headers["cache-control"] == "no-store"
    mint.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error, status", [(github.GitHubConfigError, 503), (RuntimeError, 502)])
async def test_mint_errors_do_not_leak_secret_to_response_or_logs(client, adapter, monkeypatch, caplog, error, status):
    monkeypatch.setattr(adapter, "git_credential_token", Mock(side_effect=error("test-sensitive-value")))
    response = await client.get("/v1/providers/github/git-credential?repository=Other/Project")
    assert response.status_code == status and response.headers["cache-control"] == "no-store"
    assert "test-sensitive-value" not in response.text + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", ["missing", "bad", "other-gate"])
@pytest.mark.parametrize("malformed_json", [False, True])
async def test_config_enforced_gate_refuses_before_parse_or_mutation(
    client, adapter, app_key, monkeypatch, credential, malformed_json, caplog,
):
    configure(adapter, app_key)
    connected(adapter)
    before, writes = copy.deepcopy(adapter.store.rows), list(adapter.store.writes)
    gates = providers.bridge_gate.bridge_gates()
    assert gates.enforced(providers.bridge_gate.GIT_CREDENTIAL)
    client.headers.pop(providers.bridge_gate.HEADER)
    if credential == "bad":
        client.headers[providers.bridge_gate.HEADER] = "bad-test-token"
    elif credential == "other-gate":
        client.headers[providers.bridge_gate.HEADER] = gates.derived_token(providers.bridge_gate.BROKER)
    get_adapter = Mock(side_effect=AssertionError("unauthorized adapter access"))
    configure_app = Mock(side_effect=AssertionError("unauthorized configuration write"))
    monkeypatch.setattr(providers, "get_adapter", get_adapter)
    monkeypatch.setattr(adapter, "configure_app", configure_app)
    payload = '{"private_key": "test-sensitive-value",' if malformed_json else json.dumps({
        "app_id": "654321", "client_id": "replacement-client", "private_key": app_key[1],
    })
    response = await client.put(
        "/v1/providers/github/app-config", content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401 and response.headers["cache-control"] == "no-store"
    assert response.json() == {"detail": "bad bridge token"}
    for secret in ("test-sensitive-value", "replacement-client", "PRIVATE KEY", "bad-test-token"):
        assert secret not in response.text + caplog.text
    assert adapter.store.rows == before and adapter.store.writes == writes
    get_adapter.assert_not_called()
    configure_app.assert_not_called()
    outcome = "missing" if credential == "missing" else "bad"
    assert gates.counts(providers.bridge_gate.GIT_CREDENTIAL)[outcome] == 1


@pytest.mark.asyncio
async def test_config_route_returns_no_fields_or_secrets_and_does_not_touch_oauth(client, adapter, app_key, caplog):
    connected(adapter)
    before = copy.deepcopy(adapter.store.rows["github"])
    response = await client.put("/v1/providers/github/app-config", json={
        "app_id": "123456", "client_id": "test-client-id", "private_key": app_key[1],
    })
    assert response.status_code == 200 and response.json() == {"configured": True}
    assert response.headers["cache-control"] == "no-store"
    assert adapter.store.rows["github"] == before
    for secret in ("123456", "test-client-id", "PRIVATE KEY"):
        assert secret not in response.text + caplog.text


def test_no_app_descriptor_reuses_pat_form_without_inference_balance(adapter):
    descriptor = adapter.descriptor()
    assert descriptor["id"] == "github"
    assert descriptor["auth_methods"] == ["api_key"]
    assert descriptor["category"] == "git" and adapter.is_configured()
    assert "Contents: Read" in descriptor["credential_help"]
    assert "personal access token" in descriptor["credential_label"]


def pat_probe(monkeypatch, status=200):
    get = Mock(return_value=httpx.Response(
        status, json={"login": "test-user"},
        request=httpx.Request("GET", "https://api.github.com/user"),
    ))
    monkeypatch.setattr(github.httpx, "get", get)
    return get


@pytest.mark.asyncio
async def test_no_app_pat_connect_encrypted_broker_and_validate(client, adapter, monkeypatch, caplog):
    get = pat_probe(monkeypatch)
    token = "test-private-pat"
    response = await client.post("/v1/providers/github/connect/api-key", json={"api_key": f" {token} "})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["connection"]["auth_method"] == "api_key"
    assert response.json()["account"] == "test-user"
    assert get.call_args.kwargs["headers"]["Authorization"] == f"Bearer {token}"
    record = adapter.store.load("github")
    assert record.secret == {"api_key": token}
    assert adapter.validate(record).ok
    assert token not in json.dumps(adapter.store.rows) + response.text + caplog.text
    get.reset_mock()
    for params in ({}, {"repository": "Owner/Private"}):
        credentials = await client.get("/v1/providers/github/git-credential", params=params)
        assert credentials.status_code == 200
        assert credentials.headers["cache-control"] == "no-store"
        assert credentials.text == f"username=x-access-token\npassword={token}\n"
    get.assert_not_called()
    github.httpx.post.assert_not_called()


@pytest.mark.parametrize("key", ["", " \n ", "test\ninjection=value", "test\tpat", "test\x00pat", "test\x7fpat"])
def test_pat_protocol_injection_is_rejected_before_probe_or_storage(adapter, key):
    assert not adapter.set_api_key(key).ok
    assert adapter.store.rows == {}
    github.httpx.get.assert_not_called()


def test_pat_requires_persistent_encryption_before_probe(adapter):
    adapter.store.durable_key_problem = lambda: "ephemeral_root"
    assert not adapter.set_api_key("test-private-pat").ok
    assert adapter.store.rows == {}
    github.httpx.get.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 500])
def test_rejected_pat_is_not_saved(adapter, monkeypatch, status):
    pat_probe(monkeypatch, status)
    result = adapter.set_api_key("test-private-pat")
    assert not result.ok and "test-private-pat" not in result.detail
    assert adapter.store.rows == {}


@pytest.mark.parametrize("name", ["GITHUB_APP_ID", "GITHUB_APP_CLIENT_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_APP_PRIVATE_KEY_PATH"])
def test_any_app_config_presence_prevents_pat_fallback(adapter, monkeypatch, name):
    adapter.store.save(ConnectionRecord(
        provider="github", status="connected", auth_method="api_key", secret={"api_key": "test-private-pat"},
    ))
    monkeypatch.setenv(name, "unusable-test-app-config")
    assert adapter.auth_methods == ("device_oauth",)
    assert not adapter.set_api_key("replacement-pat").ok
    assert "PAT inactive" in adapter.health()["detail"]
    with pytest.raises(github.GitHubConfigError):
        adapter.git_credential_token(repository="Owner/Private")


def test_stored_app_outranks_pat_and_does_not_use_it_on_app_failure(adapter, app_key, monkeypatch):
    pat_probe(monkeypatch)
    assert adapter.set_api_key("test-private-pat").ok
    configure(adapter, app_key)
    _, post = upstream(monkeypatch, status=403)
    with pytest.raises(httpx.HTTPStatusError):
        adapter.git_credential_token(repository="Owner/Private")
    post.assert_not_called()
    assert adapter.auth_methods == ("device_oauth",)


@pytest.mark.asyncio
@pytest.mark.parametrize("record", [
    None,
    ConnectionRecord(provider="github", status="disconnected", auth_method="api_key"),
    ConnectionRecord(provider="github", status="connected", auth_method="device_oauth", secret={"user_token": "test-oauth-token"}),
    ConnectionRecord(provider="github", status="connected", auth_method="api_key", secret={"api_key": "test\ninjection=value"}),
    ConnectionRecord(provider="github", status="connected", auth_method="api_key"),
])
async def test_no_usable_pat_returns_clear_disconnected_state(client, adapter, record):
    if record:
        adapter.store.save(record)
    response = await client.get("/v1/providers/github/git-credential")
    assert response.status_code == 503 and "GitHub not connected" in response.json()["detail"]
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["gitlab.com", "github.com.attacker.example", "github.com:443", ""])
async def test_broker_rejects_foreign_host_before_credential_read(client, adapter, monkeypatch, host):
    read = Mock(side_effect=AssertionError("must not read credential"))
    monkeypatch.setattr(adapter, "git_credential_token", read)
    response = await client.get("/v1/providers/github/git-credential", params={"host": host})
    assert response.status_code == 400 and response.headers["cache-control"] == "no-store"
    read.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    '{"api_key": {"secret": "test-private-pat"}}',
    '{"api_key": "test-private-pat",',
    '["test-private-pat"]',
])
async def test_pat_invalid_input_never_echoed(client, adapter, caplog, payload):
    response = await client.post(
        "/v1/providers/github/connect/api-key", content=payload, headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400 and response.headers["cache-control"] == "no-store"
    assert "test-private-pat" not in response.text + caplog.text
    assert adapter.store.rows == {}


@pytest.mark.asyncio
async def test_pat_storage_error_never_logged(client, adapter, monkeypatch, caplog):
    pat_probe(monkeypatch)
    monkeypatch.setattr(adapter.store, "save", Mock(side_effect=RuntimeError("test-private-pat")))
    response = await client.post("/v1/providers/github/connect/api-key", json={"api_key": "test-private-pat"})
    assert response.status_code == 502
    assert "test-private-pat" not in response.text + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"app_id": "123", "client_id": "test", "private_key": {"secret": "test-sensitive-value"}},
    {"app_id": "123", "client_id": "test", "private_key": "test-sensitive-value"},
    {"app_id": "123", "client_id": "test", "unexpected": "test-sensitive-value"},
    ["test-sensitive-value"],
])
async def test_config_validation_never_echoes_input(client, adapter, body, caplog):
    response = await client.put("/v1/providers/github/app-config", json=body)
    assert response.status_code == 400 and response.headers["cache-control"] == "no-store"
    assert response.json() == {"detail": "Invalid GitHub App configuration"}
    assert "test-sensitive-value" not in response.text + caplog.text
    assert adapter.store.rows == {}


@pytest.mark.asyncio
async def test_config_malformed_json_never_echoes_input(client, adapter, caplog):
    response = await client.put(
        "/v1/providers/github/app-config", content='{"private_key": "test-sensitive-value",',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400 and response.headers["cache-control"] == "no-store"
    assert response.json() == {"detail": "Invalid GitHub App configuration"}
    assert "test-sensitive-value" not in response.text + caplog.text
    assert adapter.store.rows == {}


@pytest.mark.asyncio
async def test_config_rotation_keeps_oauth_selection_and_uses_new_app(client, adapter, app_key, monkeypatch):
    configure(adapter, app_key)
    connected(adapter)
    before = copy.deepcopy(adapter.store.rows["github"])
    response = await client.put("/v1/providers/github/app-config", json={
        "app_id": "654321", "client_id": "replacement-client", "private_key": app_key[1],
    })
    assert response.status_code == 200 and response.json() == {"configured": True}
    assert adapter.store.rows["github"] == before
    assert adapter._device_client_id() == "replacement-client"
    get, _ = upstream(monkeypatch)
    adapter.mint_installation_token(repository="Other/Project")
    token = get.call_args.kwargs["headers"]["Authorization"].removeprefix("Bearer ")
    assert jwt.decode(token, app_key[0].public_key(), algorithms=["RS256"])["iss"] == "654321"


@pytest.mark.asyncio
async def test_config_storage_failure_is_sanitized(client, adapter, app_key, monkeypatch, caplog):
    monkeypatch.setattr(adapter.store, "save", Mock(side_effect=RuntimeError("test-sensitive-value")))
    response = await client.put("/v1/providers/github/app-config", json={
        "app_id": "123456", "client_id": "test-client-id", "private_key": app_key[1],
    })
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
    assert "test-sensitive-value" not in response.text + caplog.text
