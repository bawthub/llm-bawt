"""TASK-1477: the pgcrypto vault import never overwrites, verifies before recording, never leaks values."""

from __future__ import annotations

import json
import logging

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text

from llm_bawt import vault_import
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import (
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    ConnectionRecord,
    CredentialStore,
)
from llm_bawt.utils.db import get_shared_engine
from llm_bawt.vault_import import VaultImporter
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

KEY = "fixture-master-key-0123456789"
OTHER_KEY = "some-other-master-key-9876543210"
ANTHROPIC = "sk-ant-fixture-value-aaaa1111"
OPENAI = "sk-openai-fixture-value-bbbb2222"
XAI_ONE = "xai-fixture-value-cccc3333"
XAI_TWO = "xai-fixture-value-dddd4444"
GEMINI = "gemini-fixture-value-eeee5555"
VALUES = (ANTHROPIC, OPENAI, XAI_ONE, XAI_TWO, GEMINI, KEY, OTHER_KEY)


@pytest.fixture
def env(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    config = require_test_db_config()
    engine = get_shared_engine(config)
    assert engine is not None
    store = CredentialStore(config)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS vault"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS vault.secrets (name text PRIMARY KEY, "
                          "ciphertext bytea NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())"))
    yield engine, store
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS vault CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS vault_import_ledger"))
        conn.execute(text("DELETE FROM runtime_settings WHERE key LIKE 'provider_connection:%'"))
    VaultImporter._guard.reset_for_tests(engine)


def vault_set(engine, name: str, value: str, key: str = KEY) -> None:
    """What the operator `vault set` CLI does, with bound parameters."""
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO vault.secrets(name, ciphertext, updated_at) VALUES (:n, pgp_sym_encrypt(:v, :k), now()) "
            "ON CONFLICT (name) DO UPDATE SET ciphertext = EXCLUDED.ciphertext, updated_at = now()"),
            {"n": name, "v": value, "k": key})


def vault_rm(engine, name: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM vault.secrets WHERE name = :n"), {"n": name})


def row_json(engine, provider: str) -> str | None:
    with engine.connect() as conn:
        return conn.execute(text(
            "SELECT value_json FROM runtime_settings WHERE scope_type='global' AND scope_id='*' AND key=:k"),
            {"k": f"provider_connection:{provider}"}).scalar_one_or_none()


def by_name(outcomes) -> dict[str, vault_import.Outcome]:
    return {outcome.name: outcome for outcome in outcomes}


def ledger_rows(engine) -> dict[str, dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM vault_import_ledger")).mappings().all()
    return {row["name"]: dict(row) for row in rows}


def run(engine, store, key: str | None = KEY):
    return by_name(VaultImporter(engine, store, key).run())


def test_mapped_names_import_and_verify(env):
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", f"  {ANTHROPIC}\n")
    vault_set(engine, "OPENAI_API_KEY", OPENAI)
    result = run(engine, store)
    assert {n: o.outcome for n, o in result.items()} == {"ANTHROPIC_API_KEY": "imported", "OPENAI_API_KEY": "imported"}
    anthropic = store.load("anthropic-api")
    assert anthropic.status == STATUS_CONNECTED and anthropic.secret == {"api_key": ANTHROPIC}
    assert anthropic.meta == {"source": "vault-import"}
    assert store.load("openai-api").secret == {"api_key": OPENAI}
    assert {n: r["outcome"] for n, r in ledger_rows(engine).items()} == {
        "ANTHROPIC_API_KEY": "imported", "OPENAI_API_KEY": "imported"}
    # A rerun changes nothing.
    again = run(engine, store)
    assert all(not o.processed and o.outcome == "imported" for o in again.values())


def test_unmapped_names_are_never_decrypted_or_imported(env):
    engine, store = env
    vault_set(engine, "GEMINI_API_KEY", GEMINI, key=OTHER_KEY)  # undecryptable with KEY
    result = run(engine, store)
    assert result["GEMINI_API_KEY"].outcome == "unmapped"
    assert result["GEMINI_API_KEY"].destination is None
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM runtime_settings WHERE key LIKE 'provider_connection:%'")
                            ).scalar_one() == 0


def test_two_names_with_different_keys_conflict_until_one_is_removed(env):
    engine, store = env
    vault_set(engine, "XAI_API_KEY", XAI_ONE)
    vault_set(engine, "LLM_BAWT_XAI_API_KEY", XAI_TWO)
    result = run(engine, store)
    assert {o.outcome for o in result.values()} == {"conflict"}
    assert row_json(engine, "xai") is None
    vault_rm(engine, "LLM_BAWT_XAI_API_KEY")
    result = run(engine, store)
    assert result["XAI_API_KEY"].outcome == "imported"
    assert store.load("xai").secret == {"api_key": XAI_ONE}


def test_two_names_with_the_same_key_import_once(env):
    engine, store = env
    vault_set(engine, "XAI_API_KEY", XAI_ONE)
    vault_set(engine, "LLM_BAWT_XAI_API_KEY", XAI_ONE)
    result = run(engine, store)
    assert {o.outcome for o in result.values()} == {"imported"}
    assert store.load("xai").secret == {"api_key": XAI_ONE}


def test_existing_destination_in_any_status_is_never_overwritten(env):
    engine, store = env
    store.save(ConnectionRecord(provider="anthropic-api", status=STATUS_DISCONNECTED))
    before = row_json(engine, "anthropic-api")
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    result = run(engine, store)
    assert result["ANTHROPIC_API_KEY"].outcome == "skipped_exists"
    assert row_json(engine, "anthropic-api") == before


def test_create_if_absent_inserts_once_and_never_replaces(env):
    engine, store = env
    first = ConnectionRecord(provider="openai-api", status=STATUS_CONNECTED, secret={"api_key": OPENAI})
    assert store.create_if_absent(first) is True
    stored = row_json(engine, "openai-api")
    second = ConnectionRecord(provider="openai-api", status=STATUS_CONNECTED, secret={"api_key": "replacement"})
    assert store.create_if_absent(second) is False
    assert row_json(engine, "openai-api") == stored


def test_rotate_then_retry_leaves_the_imported_record_untouched_and_retries_only_failed(env):
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    vault_set(engine, "OPENAI_API_KEY", OPENAI, key=OTHER_KEY)  # fails to decrypt with KEY
    first = run(engine, store)
    assert first["ANTHROPIC_API_KEY"].outcome == "imported"
    assert first["OPENAI_API_KEY"].outcome == "failed"
    assert first["OPENAI_API_KEY"].detail_code.startswith("error:")
    assert row_json(engine, "openai-api") is None
    imported = row_json(engine, "anthropic-api")
    vault_set(engine, "ANTHROPIC_API_KEY", "sk-ant-rotated-value")
    vault_set(engine, "OPENAI_API_KEY", OPENAI)  # operator fixed it
    second = run(engine, store)
    assert second["ANTHROPIC_API_KEY"].processed is False
    assert row_json(engine, "anthropic-api") == imported
    assert second["OPENAI_API_KEY"].outcome == "imported"
    assert store.load("openai-api").secret == {"api_key": OPENAI}


def test_a_disconnected_import_is_not_brought_back_by_a_rerun(env):
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    run(engine, store)
    assert store.delete("anthropic-api")
    result = run(engine, store)
    assert result["ANTHROPIC_API_KEY"].processed is False
    assert row_json(engine, "anthropic-api") is None


def test_verify_mismatch_rolls_back_the_insert(env, monkeypatch):
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    monkeypatch.setattr(store, "load_in", lambda conn, provider: ConnectionRecord(
        provider=provider, status=STATUS_CONNECTED, secret={"api_key": "something-else"}))
    result = run(engine, store)
    assert result["ANTHROPIC_API_KEY"].outcome == "failed"
    assert result["ANTHROPIC_API_KEY"].detail_code == "verify_mismatch"
    assert row_json(engine, "anthropic-api") is None
    assert ledger_rows(engine)["ANTHROPIC_API_KEY"]["outcome"] == "failed"


def test_missing_master_key_fails_mapped_names_only(env):
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    vault_set(engine, "GEMINI_API_KEY", GEMINI)
    result = run(engine, store, key=None)
    assert (result["ANTHROPIC_API_KEY"].outcome, result["ANTHROPIC_API_KEY"].detail_code) == ("failed", "no_master_key")
    assert result["GEMINI_API_KEY"].outcome == "unmapped"


def test_main_reports_names_and_codes_never_values(env, tmp_path, capsys, caplog):
    engine, _store = env
    key_file = tmp_path / "master-key"
    key_file.write_text(KEY)
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    vault_set(engine, "OPENAI_API_KEY", OPENAI, key=OTHER_KEY)
    vault_set(engine, "XAI_API_KEY", XAI_ONE)
    vault_set(engine, "LLM_BAWT_XAI_API_KEY", XAI_TWO)
    vault_set(engine, "GEMINI_API_KEY", GEMINI)
    with caplog.at_level(logging.DEBUG):
        code = vault_import.main(["--master-key-file", str(key_file)])
    assert code == 1  # OPENAI failed
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines()]
    assert {line["name"]: line["outcome"] for line in lines} == {
        "ANTHROPIC_API_KEY": "imported", "OPENAI_API_KEY": "failed", "XAI_API_KEY": "conflict",
        "LLM_BAWT_XAI_API_KEY": "conflict", "GEMINI_API_KEY": "unmapped"}
    ledger_text = json.dumps(ledger_rows(engine), default=str)
    for value in VALUES:
        assert value not in out
        assert value not in caplog.text
        assert value not in ledger_text


def test_main_with_no_vault_is_a_noop(env, capsys):
    engine, _store = env
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA vault CASCADE"))
    assert vault_import.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {"vault": "absent"}
