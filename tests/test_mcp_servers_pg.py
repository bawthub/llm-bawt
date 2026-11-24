"""TASK-2610: remote MCP server rows, bot/tag grants and per-bot resolution on private PostgreSQL."""
from __future__ import annotations

import uuid

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.integrations.mcp_servers.oauth import store_id
from llm_bawt.integrations.mcp_servers.service import McpServers
from llm_bawt.integrations.mcp_servers.store import McpServerError
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import ConnectionRecord
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def servers(private_pg, monkeypatch):
    name = f"mcp_servers_{uuid.uuid4().hex[:12]}_test"
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.exec_driver_sql(f'CREATE DATABASE "{name}"')
    engine = create_engine(private_url(private_pg, name))
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "encrypt", lambda value: cipher.encrypt(value.encode()).decode())
    monkeypatch.setattr(crypto, "decrypt", lambda value: cipher.decrypt(value.encode()).decode())
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.engine = engine
    credentials = object.__new__(IntegrationStore)
    credentials._store = runtime
    try:
        yield McpServers(None, engine=engine, credentials=credentials)
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def add(servers, slug, auth_kind="none", **kw):
    return servers.create(slug=slug, name=kw.pop("name", slug.title()), url=f"https://{slug}.example.net/mcp",
                          transport="streamable_http", auth_kind=auth_kind, actor="fixture-admin", **kw)


def test_grants_resolve_by_bot_and_by_tag(servers):
    add(servers, "alpha", bearer_token=None)
    add(servers, "beta", auth_kind="bearer", bearer_token="beta-token")
    add(servers, "gamma", bearer_token=None)
    servers.set_grants("alpha", [{"scope_type": "bot", "scope_id": "fixture-one"}], "fixture-admin")
    servers.set_grants("beta", [{"scope_type": "tag", "scope_id": "fixture_role"}], "fixture-admin")
    servers.set_grants("gamma", [{"scope_type": "bot", "scope_id": "fixture-two"}], "fixture-admin")

    resolved = {s.slug: s for s in servers.resolve_for_bot("fixture-one", ["fixture_role"])}
    assert set(resolved) == {"alpha", "beta"}
    assert resolved["alpha"].headers == {}
    assert resolved["beta"].headers == {"Authorization": "Bearer beta-token"}
    assert {s.slug for s in servers.resolve_for_bot("fixture-two", [])} == {"gamma"}
    assert servers.resolve_for_bot("fixture-three", ["other"]) == []


def test_disabled_server_and_missing_credentials_are_skipped(servers):
    add(servers, "alpha", bearer_token=None)
    add(servers, "oauthy", auth_kind="oauth", bearer_token=None)
    for slug in ("alpha", "oauthy"):
        servers.set_grants(slug, [{"scope_type": "bot", "scope_id": "fixture-one"}], "fixture-admin")
    servers.update("alpha", name=None, enabled=False, actor="fixture-admin")
    # Not signed in yet: skipped rather than failing the bot's other tools.
    assert servers.resolve_for_bot("fixture-one", []) == []


def test_listing_never_returns_secrets(servers):
    add(servers, "beta", auth_kind="bearer", bearer_token="beta-token")
    listed = servers.list()
    assert listed[0]["connected"] is True and "beta-token" not in str(listed)


def test_duplicate_slug_keeps_the_existing_server_and_its_secret(servers):
    add(servers, "beta", auth_kind="bearer", bearer_token="beta-token")
    with pytest.raises(McpServerError, match="slug_taken"):
        add(servers, "beta", auth_kind="bearer", bearer_token="other")
    assert servers.credentials.load(store_id("beta")).secret == {"token": "beta-token"}


def test_delete_cascades_grants_and_forgets_secrets(servers):
    add(servers, "beta", auth_kind="bearer", bearer_token="beta-token")
    servers.credentials.save(ConnectionRecord(provider=f"client:{store_id('beta')}", status="configured",
                                              meta={"client_id": "c"}))
    servers.set_grants("beta", [{"scope_type": "bot", "scope_id": "fixture-one"}], "fixture-admin")
    assert servers.delete("beta") is True
    assert servers.credentials.load(store_id("beta")) is None
    assert servers.credentials.load(f"client:{store_id('beta')}") is None
    add(servers, "beta", bearer_token=None)
    assert servers.store.grants("beta") == []


def test_grants_validate_scope_and_server(servers):
    add(servers, "alpha", bearer_token=None)
    with pytest.raises(McpServerError, match="invalid_field"):
        servers.set_grants("alpha", [{"scope_type": "user", "scope_id": "x"}], None)
    with pytest.raises(McpServerError, match="not_found"):
        servers.set_grants("missing", [{"scope_type": "bot", "scope_id": "x"}], None)


def test_bearer_kind_requires_a_token(servers):
    with pytest.raises(McpServerError, match="bearer_token_required"):
        add(servers, "beta", auth_kind="bearer", bearer_token="  ")
    assert servers.store.get("beta") is None
