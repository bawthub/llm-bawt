"""TASK-1476: env→record import never overwrites, verifies before commit, never leaks values."""

from __future__ import annotations

import json
import logging

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text

from llm_bawt import env_import
from llm_bawt.env_import import SPECS, EnvImporter, remove_names
from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import (
    AUTH_API_KEY,
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    ConnectionRecord,
    CredentialStore,
)
from llm_bawt.utils.db import get_shared_engine
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

ZAI = "zai-fixture-value-aaaa1111"
KIMI = "kimi-fixture-value-bbbb2222"
BRAVE = "brave-fixture-value-cccc3333"
NEWS = "news-fixture-value-dddd4444"
CLAW = "claw-fixture-value-eeee5555"
CLAW_OTHER = "claw-fixture-value-ffff6666"
S3_ACCESS = "s3-access-fixture-gggg7777"
S3_SECRET = "s3-secret-fixture-hhhh8888"
SONARR = "sonarr-fixture-value-iiii9999"
RADARR = "radarr-fixture-value-jjjj0000"
SAB = "sab-fixture-value-kkkk1111"
UNMAPPED = "openai-fixture-value-llll2222"
SECRETS = (ZAI, KIMI, BRAVE, NEWS, CLAW, CLAW_OTHER, S3_ACCESS, S3_SECRET, SONARR, RADARR, SAB, UNMAPPED)

FULL_ENV = {
    "ZAI_API_KEY": ZAI,
    "KIMI_CODING_API_KEY": KIMI,
    "LLM_BAWT_BRAVE_API_KEY": BRAVE,
    "LLM_BAWT_NEWSAPI_API_KEY": NEWS,
    "OPENCLAW_GATEWAY_TOKEN": CLAW,
    "LLM_BAWT_OPENCLAW_GATEWAY_TOKEN": CLAW,
    "LLM_BAWT_S3_ENDPOINT": "https://s3.example",
    "LLM_BAWT_S3_REGION": "us-east-1",
    "LLM_BAWT_S3_BUCKET": "media",
    "LLM_BAWT_S3_ACCESS_KEY": S3_ACCESS,
    "LLM_BAWT_S3_SECRET_KEY": S3_SECRET,
    "LLM_BAWT_SONARR_URL": "https://sonarr.example",
    "LLM_BAWT_SONARR_API_KEY": SONARR,
    "LLM_BAWT_RADARR_URL": "https://radarr.example",
    "LLM_BAWT_RADARR_API_KEY": RADARR,
    "LLM_BAWT_SABNZBD_URL": "https://sabnzbd.example",
    "LLM_BAWT_SABNZBD_API_KEY": SAB,
    "OPENAI_API_KEY": UNMAPPED,  # set but not adoptable: never read
}


@pytest.fixture
def env(monkeypatch):
    root = Fernet.generate_key()
    fernet = Fernet(root)
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", root.decode())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    config = require_test_db_config()
    engine = get_shared_engine(config)
    assert engine is not None
    providers, integrations = CredentialStore(config), IntegrationStore(config)
    yield engine, providers, integrations
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM runtime_settings WHERE key LIKE 'provider_connection:%' "
                          "OR key LIKE 'integration_connection:%'"))


def run(providers, integrations, environ, *, check=False):
    return {o.destination: o for o in EnvImporter(providers, integrations, environ).run(check=check)}


def row_count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM runtime_settings WHERE key LIKE 'provider_connection:%' "
                                 "OR key LIKE 'integration_connection:%'")).scalar_one()


def test_mapping_is_closed():
    assert {name for spec in SPECS for name in spec.env_names} == set(FULL_ENV) - {"OPENAI_API_KEY"}
    assert {(spec.destination, spec.kind) for spec in SPECS} == {
        ("zai", "provider"), ("kimi-coding", "provider"), ("brave", "provider"), ("newsapi", "provider"),
        ("openclaw", "integration"), ("object-store", "integration"), ("sonarr", "integration"),
        ("radarr", "integration"), ("sabnzbd", "integration")}


def test_every_destination_imports_verified_and_is_removable(env):
    engine, providers, integrations = env
    result = run(providers, integrations, FULL_ENV)
    assert {d: o.outcome for d, o in result.items()} == {d: "imported" for d in result}
    assert all(o.removable for o in result.values())
    zai = providers.load("zai")
    assert (zai.status, zai.auth_method, zai.secret) == (STATUS_CONNECTED, AUTH_API_KEY, {"api_key": ZAI})
    assert providers.load("kimi-coding").secret == {"api_key": KIMI}
    store = integrations.load("object-store")
    assert store.meta == {"endpoint": "https://s3.example", "region": "us-east-1", "bucket": "media"}
    assert store.secret == {"access_key_id": S3_ACCESS, "secret_access_key": S3_SECRET}
    assert integrations.load("openclaw").secret == {"token": CLAW}
    sab = integrations.load("sabnzbd")
    assert (sab.meta, sab.secret) == ({"url": "https://sabnzbd.example"}, {"api_key": SAB})
    assert remove_names(list(result.values())) == sorted(set(FULL_ENV) - {"OPENAI_API_KEY"})
    # A rerun with the lines still present finds matching records: still removable, no writes.
    again = run(providers, integrations, FULL_ENV)
    assert {o.outcome for o in again.values()} == {"already_present"}
    assert all(o.removable for o in again.values())


def test_existing_rows_are_never_overwritten(env):
    engine, providers, integrations = env
    providers.save(ConnectionRecord(provider="zai", status=STATUS_DISCONNECTED, auth_method=AUTH_API_KEY))
    providers.save(ConnectionRecord(provider="brave", status=STATUS_CONNECTED, auth_method=AUTH_API_KEY,
                                    secret={"api_key": "operator-entered"}))
    integrations.save(ConnectionRecord(provider="sonarr", status=STATUS_CONNECTED, auth_method="oauth",
                                       meta={"url": "https://sonarr.example"}, secret={"api_key": SONARR}))
    before = {p: providers.load(p) for p in ("zai", "brave")}
    result = run(providers, integrations, FULL_ENV)
    assert (result["zai"].outcome, result["zai"].removable) == ("disconnected", False)
    assert (result["brave"].outcome, result["brave"].removable) == ("record_differs", False)
    assert (result["sonarr"].outcome, result["sonarr"].removable) == ("managed_elsewhere", False)
    assert providers.load("zai").status == STATUS_DISCONNECTED
    assert providers.load("brave").secret == before["brave"].secret
    names = remove_names(list(result.values()))
    assert not {"ZAI_API_KEY", "LLM_BAWT_BRAVE_API_KEY", "LLM_BAWT_SONARR_URL", "LLM_BAWT_SONARR_API_KEY"} & set(names)


def test_unreadable_row_is_left_alone(env):
    engine, providers, integrations = env
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO runtime_settings(scope_type, scope_id, key, value_json, updated_at) "
                          "VALUES ('global', '*', 'provider_connection:newsapi', '\"not-an-object\"', now())"))
    result = run(providers, integrations, {"LLM_BAWT_NEWSAPI_API_KEY": NEWS})
    assert (result["newsapi"].outcome, result["newsapi"].detail_code) == ("managed_elsewhere", "unreadable_record")
    assert not result["newsapi"].removable


def test_alias_conflict_writes_nothing_and_agreeing_aliases_import_once(env):
    engine, providers, integrations = env
    split = {"OPENCLAW_GATEWAY_TOKEN": CLAW, "LLM_BAWT_OPENCLAW_GATEWAY_TOKEN": CLAW_OTHER}
    result = run(providers, integrations, split)
    assert (result["openclaw"].outcome, result["openclaw"].removable) == ("conflict", False)
    assert row_count(engine) == 0
    result = run(providers, integrations, {"LLM_BAWT_OPENCLAW_GATEWAY_TOKEN": CLAW})
    assert result["openclaw"].outcome == "imported"
    assert result["openclaw"].env_names == ("LLM_BAWT_OPENCLAW_GATEWAY_TOKEN",)


def test_partial_integration_is_incomplete_and_not_written(env):
    engine, providers, integrations = env
    partial = {k: v for k, v in FULL_ENV.items() if k.startswith("LLM_BAWT_S3_") and k != "LLM_BAWT_S3_BUCKET"}
    result = run(providers, integrations, partial)
    assert (result["object-store"].outcome, result["object-store"].detail_code) == (
        "incomplete", "missing_required_field")
    assert row_count(engine) == 0


def test_invalid_url_is_incomplete(env):
    engine, providers, integrations = env
    result = run(providers, integrations, {"LLM_BAWT_RADARR_URL": "radarr:7878", "LLM_BAWT_RADARR_API_KEY": RADARR})
    assert (result["radarr"].outcome, result["radarr"].detail_code) == ("incomplete", "invalid_url")
    assert row_count(engine) == 0


def test_read_back_mismatch_rolls_the_insert_back(env, monkeypatch):
    engine, providers, integrations = env
    monkeypatch.setattr(integrations, "load_in", lambda conn, provider: ConnectionRecord(
        provider=provider, status=STATUS_CONNECTED, auth_method="credentials", secret={"token": "other"}))
    result = run(providers, integrations, {"OPENCLAW_GATEWAY_TOKEN": CLAW})
    assert (result["openclaw"].outcome, result["openclaw"].detail_code) == ("failed", "verify_mismatch")
    assert not result["openclaw"].removable
    assert row_count(engine) == 0


def test_ephemeral_key_writes_nothing(env, monkeypatch):
    engine, providers, integrations = env
    monkeypatch.setattr(CredentialStore, "durable_key_problem", staticmethod(lambda: "ephemeral_root"))
    result = run(providers, integrations, FULL_ENV)
    assert {(o.outcome, o.detail_code) for o in result.values()} == {("failed", "ephemeral_root")}
    assert remove_names(list(result.values())) == []
    assert row_count(engine) == 0


def test_check_mode_writes_nothing(env):
    engine, providers, integrations = env
    result = run(providers, integrations, FULL_ENV, check=True)
    assert {o.outcome for o in result.values()} == {"would_import"}
    assert row_count(engine) == 0


def test_unset_names_are_no_source(env):
    engine, providers, integrations = env
    result = run(providers, integrations, {"ZAI_API_KEY": "  ", "OPENAI_API_KEY": UNMAPPED})
    assert {o.outcome for o in result.values()} == {"no_source"}
    assert row_count(engine) == 0


def test_main_only_limits_the_run(env, monkeypatch, capsys):
    engine, providers, integrations = env
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    assert env_import.main(["--only", "zai", "--only", "sonarr"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["destination"] for line in lines[:-1]] == ["zai", "sonarr"]
    assert lines[-1]["remove_names"] == ["LLM_BAWT_SONARR_API_KEY", "LLM_BAWT_SONARR_URL", "ZAI_API_KEY"]
    assert integrations.load("openclaw") is None
    assert row_count(engine) == 2
    with pytest.raises(SystemExit):
        env_import.main(["--only", "anthropic-api"])  # not adoptable


def test_main_reports_names_and_codes_never_values(env, monkeypatch, capsys, caplog):
    engine, providers, integrations = env
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_BAWT_OPENCLAW_GATEWAY_TOKEN", CLAW_OTHER)  # one unresolved conflict
    with caplog.at_level(logging.DEBUG):
        code = env_import.main([])
    assert code == 1
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines()]
    summary = lines[-1]
    by_dest = {line["destination"]: line for line in lines[:-1]}
    assert by_dest["openclaw"]["outcome"] == "conflict"
    assert by_dest["zai"] == {"destination": "zai", "env_names": ["ZAI_API_KEY"], "outcome": "imported",
                              "detail_code": None, "removable": True}
    assert "OPENCLAW_GATEWAY_TOKEN" not in summary["remove_names"]
    assert "LLM_BAWT_S3_SECRET_KEY" in summary["remove_names"]
    assert summary["check"] is False
    for value in SECRETS:
        assert value not in out
        assert value not in caplog.text
