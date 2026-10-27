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
    root = Fernet.generate_key()
    fernet = Fernet(root)
    # The active key is the persisted (env) root, as on a real install.
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", root.decode())
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


def test_stale_importer_cannot_resurrect_a_disconnected_import(env, monkeypatch):
    # Loopy r1 B1: importer A reads the ledger, importer B imports, the operator
    # disconnects, then A resumes from its stale snapshot.
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    stale = VaultImporter(engine, store, KEY)
    read_ledger = stale._ledger

    def pause_after_ledger_read():
        snapshot = read_ledger()
        assert run(engine, store)["ANTHROPIC_API_KEY"].outcome == "imported"
        assert store.delete("anthropic-api")
        return snapshot

    monkeypatch.setattr(stale, "_ledger", pause_after_ledger_read)
    result = by_name(stale.run())
    assert result["ANTHROPIC_API_KEY"].processed is False
    assert row_json(engine, "anthropic-api") is None
    assert ledger_rows(engine)["ANTHROPIC_API_KEY"]["outcome"] == "imported"


def test_stale_failure_does_not_overwrite_a_concurrent_import(env, monkeypatch):
    # A stale decision of any kind (here a decrypt failure) must not replace
    # the ledger's "imported", or a later rerun would retry and resurrect.
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    stale = VaultImporter(engine, store, OTHER_KEY)  # its decrypt fails
    read_ledger = stale._ledger

    def pause_after_ledger_read():
        snapshot = read_ledger()
        assert run(engine, store)["ANTHROPIC_API_KEY"].outcome == "imported"
        assert store.delete("anthropic-api")
        return snapshot

    monkeypatch.setattr(stale, "_ledger", pause_after_ledger_read)
    stale.run()
    assert ledger_rows(engine)["ANTHROPIC_API_KEY"]["outcome"] == "imported"
    assert run(engine, store)["ANTHROPIC_API_KEY"].processed is False
    assert row_json(engine, "anthropic-api") is None


def _rotate_import_and_disconnect(engine, store):
    """Another process: the vault rotates, it imports the new key, the operator disconnects."""
    vault_set(engine, "ANTHROPIC_API_KEY", "fixture-rotated-provider-key")
    assert run(engine, store)["ANTHROPIC_API_KEY"].outcome == "imported"
    assert store.delete("anthropic-api")


def test_stale_import_of_a_rotated_source_cannot_restore_the_old_key(env, monkeypatch):
    # Loopy r2 (cmvq4flel00gf0ms3xcrtera5): A decrypted h1 and paused; the vault
    # rotated to h2, B imported h2, the operator disconnected. A must not import h1.
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    stale = VaultImporter(engine, store, KEY)
    commit = stale._commit
    raced = []

    def pause_before_commit(group, outcome, detail, value=None, **kwargs):
        if not raced:
            raced.append(outcome)
            _rotate_import_and_disconnect(engine, store)
        return commit(group, outcome, detail, value, **kwargs)

    monkeypatch.setattr(stale, "_commit", pause_before_commit)
    result = by_name(stale.run())
    assert raced == [vault_import.IMPORTED]
    assert (result["ANTHROPIC_API_KEY"].outcome, result["ANTHROPIC_API_KEY"].detail_code,
            result["ANTHROPIC_API_KEY"].processed) == ("failed", "source_changed", False)
    assert row_json(engine, "anthropic-api") is None
    assert ledger_rows(engine)["ANTHROPIC_API_KEY"]["outcome"] == "imported"
    assert run(engine, store)["ANTHROPIC_API_KEY"].processed is False
    assert row_json(engine, "anthropic-api") is None


def test_stale_failure_on_a_rotated_source_does_not_reopen_the_newer_import(env, monkeypatch):
    # Same race ending in a failure: A's "failed" for h1 would overwrite the
    # ledger's "imported" h2 and make the next run re-import the disconnected key.
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    stale = VaultImporter(engine, store, OTHER_KEY)  # its decrypt fails
    commit = stale._commit
    raced = []

    def pause_before_commit(group, outcome, detail, value=None, **kwargs):
        if not raced:
            raced.append(outcome)
            _rotate_import_and_disconnect(engine, store)
        return commit(group, outcome, detail, value, **kwargs)

    monkeypatch.setattr(stale, "_commit", pause_before_commit)
    stale.run()
    assert raced == [vault_import.FAILED]
    assert ledger_rows(engine)["ANTHROPIC_API_KEY"]["outcome"] == "imported"
    assert run(engine, store)["ANTHROPIC_API_KEY"].processed is False
    assert row_json(engine, "anthropic-api") is None


def _vault_write_waits(engine, name: str, value: str) -> bool:
    """True when a vault write from another session can't take its lock now."""
    from sqlalchemy.exc import OperationalError

    try:
        with engine.begin() as conn:
            conn.execute(text("SET LOCAL lock_timeout = '200ms'"))
            conn.execute(text(
                "INSERT INTO vault.secrets(name, ciphertext, updated_at) VALUES (:n, pgp_sym_encrypt(:v, :k), now()) "
                "ON CONFLICT (name) DO UPDATE SET ciphertext = EXCLUDED.ciphertext, updated_at = now()"),
                {"n": name, "v": value, "k": KEY})
    except OperationalError as error:
        assert getattr(error.orig, "pgcode", None) == "55P03"  # lock_not_available
        return True
    return False


def test_a_rotation_after_the_source_check_waits_for_the_import(env, monkeypatch):
    # Codex r1 (cmvq563a400gq0ms3j8u1vrb4) B1: the source check and the
    # commit must be atomic with respect to vault writers. A rotation that
    # can't land before the commit lands after it, and is then an ordinary
    # post-import change the never-overwrite contract skips.
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    importer = VaultImporter(engine, store, KEY)
    check = importer._source_changed
    waited = []

    def rotate_after_check(conn, group):
        changed = check(conn, group)
        waited.append(_vault_write_waits(engine, "ANTHROPIC_API_KEY", "fixture-rotated-provider-key"))
        return changed

    monkeypatch.setattr(importer, "_source_changed", rotate_after_check)
    assert by_name(importer.run())["ANTHROPIC_API_KEY"].outcome == "imported"
    assert waited == [True]
    assert store.load("anthropic-api").secret == {"api_key": ANTHROPIC}  # the source that was checked
    vault_set(engine, "ANTHROPIC_API_KEY", "fixture-rotated-provider-key")
    assert run(engine, store)["ANTHROPIC_API_KEY"].processed is False
    assert store.load("anthropic-api").secret == {"api_key": ANTHROPIC}


def test_an_alias_added_during_the_import_waits_and_cannot_skip_the_conflict(env, monkeypatch):
    # Codex r1 B2: an alias inserted after the check must wait for the commit.
    engine, store = env
    vault_set(engine, "XAI_API_KEY", XAI_ONE)
    importer = VaultImporter(engine, store, KEY)
    check = importer._source_changed
    waited = []

    def add_alias_after_check(conn, group):
        changed = check(conn, group)
        waited.append(_vault_write_waits(engine, "LLM_BAWT_XAI_API_KEY", XAI_TWO))
        return changed

    monkeypatch.setattr(importer, "_source_changed", add_alias_after_check)
    importer.run()
    assert waited == [True]
    assert "LLM_BAWT_XAI_API_KEY" not in ledger_rows(engine)


def test_an_alias_added_after_listing_is_a_source_change_then_a_conflict(env, monkeypatch):
    # Codex r1 B2, the window before the fence: the alias exists at commit but
    # wasn't listed. The group is stale, so nothing is written, and the next
    # run sees both aliases and reports the conflict.
    engine, store = env
    vault_set(engine, "XAI_API_KEY", XAI_ONE)
    importer = VaultImporter(engine, store, KEY)
    commit = importer._commit
    added = []

    def add_alias_before_commit(group, outcome, detail, value=None, **kwargs):
        if not added:
            added.append(outcome)
            vault_set(engine, "LLM_BAWT_XAI_API_KEY", XAI_TWO)
        return commit(group, outcome, detail, value, **kwargs)

    monkeypatch.setattr(importer, "_commit", add_alias_before_commit)
    first = by_name(importer.run())
    assert added == [vault_import.IMPORTED]
    assert (first["XAI_API_KEY"].outcome, first["XAI_API_KEY"].detail_code) == ("failed", "source_changed")
    assert row_json(engine, "xai") is None
    second = run(engine, store)
    assert second["XAI_API_KEY"].outcome == second["LLM_BAWT_XAI_API_KEY"].outcome == "conflict"
    assert row_json(engine, "xai") is None


def test_import_refuses_a_root_key_that_was_not_persisted(env, monkeypatch, tmp_path):
    # Loopy r1 B2: no env root and a key file that can't be written. Load the
    # real crypto module separately from the fixture's in-memory Fernet.
    import importlib.util

    from llm_bawt.service.providers import base

    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    spec = importlib.util.spec_from_file_location("isolated_crypto", crypto.__file__)
    fresh_crypto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh_crypto)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("fixture")
    monkeypatch.delenv("LLM_BAWT_SECRET_KEY", raising=False)
    monkeypatch.setattr(fresh_crypto, "_key_file", lambda: blocker / "secret.key")
    monkeypatch.setattr(base, "crypto", fresh_crypto)
    result = run(engine, store)
    assert (result["ANTHROPIC_API_KEY"].outcome, result["ANTHROPIC_API_KEY"].detail_code) == (
        "failed", "ephemeral_root")
    assert row_json(engine, "anthropic-api") is None


def test_import_refuses_a_cached_key_that_differs_from_the_persisted_root(env, monkeypatch):
    engine, store = env
    vault_set(engine, "ANTHROPIC_API_KEY", ANTHROPIC)
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", Fernet.generate_key().decode())  # not the active key
    result = run(engine, store)
    assert result["ANTHROPIC_API_KEY"].detail_code == "ephemeral_root"
    assert row_json(engine, "anthropic-api") is None


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
