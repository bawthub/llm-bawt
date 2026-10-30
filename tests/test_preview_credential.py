"""Preview broker serves only current access tokens under an active scoped grant."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service import preview_credential as broker

NOW = 2_000_000_000
RUN = "00000000-0000-4000-8000-000000000001"


class _Engine:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, statement, params):
        assert "scope_type='global'" in str(statement)
        self.queries.append(params["key"])
        return SimpleNamespace(scalar_one_or_none=lambda: json.dumps(self.rows[params["key"]])
                               if params["key"] in self.rows else None)


def _endpoint(provider="claude", endpoint_id=7):
    anthropic = provider == "claude"
    return ModelEndpoint(
        id=endpoint_id,
        model=ModelIdentity(1, "fixture-model", "anthropic" if anthropic else "openai", "Fixture",
                            None, None, None, None),
        access_path=AccessPath(
            1, "anthropic-oauth" if anthropic else "openai-oauth",
            "anthropic" if anthropic else "openai", "anthropic-messages" if anthropic else "responses",
            None, "oauth", None, {},
        ),
        upstream_model_id="fixture-upstream",
    )


def _jwt(expires):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expires}).encode()).decode().rstrip("=")
    return f"fixture.{payload}.signature"


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(broker.time, "time", lambda: NOW)
    monkeypatch.setattr(broker.crypto, "derive_key", lambda label: b"fixture-key-not-a-secret")
    monkeypatch.setattr(broker.crypto, "decrypt", lambda encrypted: encrypted)
    from llm_bawt.service import dependencies
    from llm_bawt.utils import db

    config = SimpleNamespace(model_catalog=ModelCatalog([_endpoint("claude"),
                                                          _endpoint("openai_chatgpt", 8)]))
    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(config=config))
    grant = {"run_id": RUN, "provider": "claude", "endpoint_id": 7, "expires_at": NOW + 3600}
    engine = _Engine({
        broker.GRANT_KEY: grant,
        "provider_connection:claude": {
            "status": "connected", "secret_enc": json.dumps({"claudeAiOauth": {
                "accessToken": "fixture-access", "refreshToken": "NEVER_RETURN",
                "expiresAt": (NOW + 1200) * 1000,
            }}),
        },
        "provider_connection:openai_chatgpt": {
            "status": "connected", "secret_enc": json.dumps({"codex_auth": {"tokens": {
                "access_token": _jwt(NOW + 1200), "refresh_token": "NEVER_RETURN",
                "account_id": "fixture-account",
            }}}),
        },
    })
    monkeypatch.setattr(db, "get_shared_engine", lambda _: engine)
    app = FastAPI()
    app.include_router(broker.router)
    with TestClient(app) as client:
        yield client, engine, grant, config


def _cap(grant):
    return broker.issue_capability(broker.PreviewGrant.parse(grant, NOW), now=NOW)


def _get(client, grant, **kwargs):
    return client.get("/v1/preview/credential",
                      headers={broker.CAPABILITY_HEADER: _cap(grant)}, **kwargs)


def test_only_scoped_current_token_is_returned_without_refresh(server, monkeypatch):
    client, engine, grant, _ = server
    from llm_bawt.service.usage import claude_oauth, codex_oauth

    monkeypatch.setattr(claude_oauth, "get_access_token", Mock(side_effect=AssertionError("refresh")))
    monkeypatch.setattr(claude_oauth, "_load", Mock(side_effect=AssertionError("migrate")))
    monkeypatch.setattr(codex_oauth, "get_access_token", Mock(side_effect=AssertionError("refresh")))
    monkeypatch.setattr(codex_oauth, "_load", Mock(side_effect=AssertionError("migrate")))
    response = _get(client, grant)
    assert response.status_code == 200
    assert response.json() == {"access_token": "fixture-access", "expires_at": (NOW + 1200) * 1000,
                               "state": "ok"}
    assert response.headers["cache-control"] == "no-store"
    assert engine.queries == [broker.GRANT_KEY, "provider_connection:claude"]
    assert "NEVER_RETURN" not in response.text
    for fn in (claude_oauth.get_access_token, claude_oauth._load,
               codex_oauth.get_access_token, codex_oauth._load):
        fn.assert_not_called()


def test_chatgpt_returns_account_and_expires_at_seconds(server):
    client, engine, grant, _ = server
    grant.update(provider="openai_chatgpt", endpoint_id=8)
    response = _get(client, grant)
    assert response.status_code == 200
    assert response.json() == {"access_token": _jwt(NOW + 1200),
                               "account_id": "fixture-account", "expires_at": NOW + 1200,
                               "state": "ok"}
    assert engine.queries == [broker.GRANT_KEY, "provider_connection:openai_chatgpt"]
    assert "NEVER_RETURN" not in response.text


@pytest.mark.parametrize("headers,params", [
    ({}, None),
    ({"X-Bridge-Token": "prod-token"}, None),
    ({}, {"force": "true"}),
    ({}, {"rejected_token_sha256": "abc"}),
    ({}, {"provider": "claude"}),
])
def test_missing_broad_or_query_authority_never_reads_provider(server, headers, params):
    client, engine, grant, _ = server
    headers = ({broker.CAPABILITY_HEADER: _cap(grant), **headers}
               if headers or params is not None else {})
    response = client.get("/v1/preview/credential", headers=headers, params=params)
    assert response.status_code == 403
    assert "provider_connection:claude" not in engine.queries


@pytest.mark.parametrize("change", [
    {"run_id": "00000000-0000-4000-8000-000000000002"},
    {"endpoint_id": 8, "provider": "openai_chatgpt"},
    {"expires_at": NOW + 10},
])
def test_changed_or_revoked_grant_refuses_old_capability(server, change):
    client, engine, grant, _ = server
    capability = _cap(grant)
    grant.update(change)
    assert client.get("/v1/preview/credential",
                      headers={broker.CAPABILITY_HEADER: capability}).status_code == 403
    assert not any(key.startswith("provider_connection:") for key in engine.queries)


def test_malformed_grant_refuses_without_credential_read(server):
    client, engine, grant, _ = server
    grant["provider"] = ["claude"]
    response = client.get("/v1/preview/credential",
                          headers={broker.CAPABILITY_HEADER: "opaque"})
    assert response.status_code == 403
    assert engine.queries == [broker.GRANT_KEY]


def test_removed_grant_refuses_with_no_provider_read(server):
    client, engine, grant, _ = server
    capability = _cap(grant)
    del engine.rows[broker.GRANT_KEY]
    assert client.get("/v1/preview/credential",
                      headers={broker.CAPABILITY_HEADER: capability}).status_code == 403
    assert engine.queries == [broker.GRANT_KEY]


def test_malformed_capability_and_endpoint_refuse_before_credential_read(server):
    client, engine, grant, config = server
    for token in ["bad", "abc.***", "aa.#", "a" * 1025, _cap(grant)[:-2] + "aa"]:
        response = client.get("/v1/preview/credential",
                              headers={broker.CAPABILITY_HEADER: token})
        assert response.status_code == 403
    grant["endpoint_id"] = 999
    response = _get(client, grant)
    assert response.status_code == 403
    assert not any(key.startswith("provider_connection:") for key in engine.queries)


@pytest.mark.parametrize("change", [
    {"status": "disconnected"},
    {"secret_enc": "{}"},
])
def test_disconnected_or_missing_bundle_does_not_fallback(server, change):
    client, engine, grant, _ = server
    engine.rows["provider_connection:claude"].update(change)
    assert _get(client, grant).status_code == 503


def test_key_derivation_failure_is_visible_unavailable(server, monkeypatch):
    client, engine, grant, _ = server
    capability = _cap(grant)
    monkeypatch.setattr(broker.crypto, "derive_key", Mock(side_effect=RuntimeError("missing owner key")))
    response = client.get("/v1/preview/credential",
                          headers={broker.CAPABILITY_HEADER: capability})
    assert response.status_code == 503
    assert "missing owner key" not in response.text
    assert "provider_connection:claude" not in engine.queries


def test_non_object_decrypted_secret_fails_visibly(server):
    client, engine, grant, _ = server
    engine.rows["provider_connection:claude"]["secret_enc"] = "[]"
    assert _get(client, grant).status_code == 503


def test_unreadable_or_malformed_credential_fails_visibly(server):
    client, engine, grant, _ = server
    engine.rows["provider_connection:claude"]["secret_enc"] = "not-json"
    response = _get(client, grant)
    assert response.status_code == 503
    assert "not-json" not in response.text


def test_expiring_token_is_refused_without_renewal(server):
    client, engine, grant, _ = server
    encrypted = engine.rows["provider_connection:claude"]
    data = json.loads(encrypted["secret_enc"])
    data["claudeAiOauth"]["expiresAt"] = (NOW + 30) * 1000
    encrypted["secret_enc"] = json.dumps(data)
    assert _get(client, grant).status_code == 503


def test_broker_route_only_on_production_router_collection():
    from llm_bawt.service.routes import all_routers
    from llm_bawt.service.preview_entrypoint import create_preview_app

    assert broker.router in all_routers
    assert "/v1/preview/credential" not in {route.path for route in create_preview_app().routes}


def test_capability_issuer_rejects_expired_grants():
    grant = broker.PreviewGrant(RUN, "claude", 7, NOW - 1)
    with pytest.raises(ValueError):
        broker.issue_capability(grant, now=NOW)
