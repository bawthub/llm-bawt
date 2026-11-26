"""RuntimeSettingsStore.set_values survives a concurrent first insert (TASK-2661).

A second writer (the installer's default-voice seed, another browser tab) can
insert the same key between set_values' read and its insert. The explicit save
must still win instead of failing with a unique-key conflict. One private DB
per test.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import create_engine, event, text

from llm_bawt.runtime_settings import RuntimeSettingsStore
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration

# The installer's conditional seed: fills the key only when it is unset.
SEED = text(
    "INSERT INTO runtime_settings (scope_type, scope_id, key, value_json, updated_at) "
    "VALUES ('global', '*', 'default_voice', :value, now()) "
    "ON CONFLICT (scope_type, scope_id, key) DO UPDATE "
    "SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at "
    "WHERE runtime_settings.value_json IN ('null', '\"\"')"
)
SEEDED = json.dumps("seed:voice")


@pytest.fixture
def db(private_pg, monkeypatch):
    name = f"settings_upsert_{uuid.uuid4().hex[:12]}_test"
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.exec_driver_sql(f'CREATE DATABASE "{name}"')
    engine = create_engine(private_url(private_pg, name))
    other = create_engine(private_url(private_pg, name))
    store = RuntimeSettingsStore.__new__(RuntimeSettingsStore)
    store.config, store.engine = None, engine
    store._ensure_tables_exist()
    try:
        yield store, other
    finally:
        engine.dispose()
        other.dispose()
        with admin.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def _seed(other) -> None:
    with other.begin() as conn:
        conn.execute(SEED, {"value": SEEDED})


def _stored(other) -> list[str]:
    with other.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT value_json FROM runtime_settings "
            "WHERE scope_type='global' AND scope_id='*' AND key='default_voice'"
        ))]


def test_seed_lands_between_read_and_insert(db):
    store, other = db
    fired = []

    def seed_after_first_read(_conn, _cursor, statement, _params, _context, _many):
        if not fired and statement.lstrip().upper().startswith("SELECT") and "runtime_settings" in statement:
            fired.append(True)
            _seed(other)

    event.listen(store.engine, "after_cursor_execute", seed_after_first_read)
    try:
        store.set_values([
            ("global", "*", "default_voice", "chosen:voice"),
            ("global", "*", "unrelated", 1),
        ])
    finally:
        event.remove(store.engine, "after_cursor_execute", seed_after_first_read)

    assert fired, "the seed never ran inside the read/insert window"
    assert _stored(other) == [json.dumps("chosen:voice")]
    assert store.get_value("global", "*", "unrelated") == 1


def test_explicit_save_first_then_seed_keeps_the_choice(db):
    store, other = db
    store.set_value("global", "*", "default_voice", "chosen:voice")
    _seed(other)
    assert _stored(other) == [json.dumps("chosen:voice")]


def test_seed_first_then_explicit_save_replaces_it(db):
    store, other = db
    _seed(other)
    store.set_value("global", "*", "default_voice", "chosen:voice")
    assert _stored(other) == [json.dumps("chosen:voice")]


def test_batch_stays_all_or_nothing_and_repeated_keys_update(db):
    store, other = db
    store.set_values([
        ("global", "*", "default_voice", "first"),
        ("global", "*", "default_voice", "second"),
    ])
    assert _stored(other) == [json.dumps("second")]

    with pytest.raises(Exception):
        store.set_values([
            ("global", "*", "default_voice", "third"),
            ("global", "*", "x" * 1000, "too long for the key column"),
        ])
    assert _stored(other) == [json.dumps("second")]
