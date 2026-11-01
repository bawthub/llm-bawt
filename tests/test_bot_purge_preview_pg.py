"""Preview on a test-owned PostgreSQL schema, never any live bot/store."""
from __future__ import annotations

import pytest
from sqlalchemy import event

from llm_bawt import bot_purge_preview as preview
from tests.test_bot_data_purge_pg import purge_db, seed  # noqa: F401 (shared isolated fixture)

pytestmark = pytest.mark.integration


@pytest.fixture
def preview_db(request):
    engine, _unused_executor = request.getfixturevalue('purge_db')
    seed(engine, keep_profile=True)
    with engine.begin() as conn:
        conn.exec_driver_sql('CREATE TABLE bot_profile_tags (bot_slug TEXT)')
    return engine


def by_key(manifest):
    return {row['key']: row for row in manifest['inventory']}


def test_real_postgresql_readonly_repeatable_snapshot_and_no_mutations(preview_db):
    statements, modes = [], []

    def observe(conn, cursor, statement, parameters, context, many):
        statements.append(statement)
        other = conn.connection.cursor()
        try:
            other.execute('SHOW transaction_read_only')
            readonly = other.fetchone()[0]
            other.execute('SHOW transaction_isolation')
            modes.append((readonly, other.fetchone()[0]))
        finally:
            other.close()

    event.listen(preview_db, 'before_cursor_execute', observe)
    try:
        result = preview.preview_bot_purge(preview_db, 'source')
    finally:
        event.remove(preview_db, 'before_cursor_execute', observe)
    assert statements and all(s.lstrip().upper().startswith('SELECT') for s in statements)
    assert set(modes) == {('on', 'repeatable read')}
    rows = by_key(result)
    assert rows['messages']['count'] == 1
    assert rows['sessions']['count'] == 3
    assert rows['entity_profiles']['count'] == 1
    assert rows['job_runs']['count'] == 1
    assert rows['shared_job_runs']['count'] == 2
    assert rows['tool_call_result_payloads']['count'] == 1
    assert result['execution_enabled'] is False
    with preview_db.connect() as conn:
        assert conn.exec_driver_sql('SELECT count(*) FROM sessions').scalar_one() == 4
        assert conn.exec_driver_sql('SELECT count(*) FROM tool_call_records').scalar_one() == 2
        assert conn.exec_driver_sql('SELECT count(*) FROM bot_profiles').scalar_one() == 2


def test_profile_absent_and_native_enum_case_are_truthful(preview_db):
    with preview_db.begin() as conn:
        conn.exec_driver_sql("DELETE FROM bot_profiles WHERE slug='source'")
        conn.exec_driver_sql("INSERT INTO entity_profiles VALUES ('bot','source'),('user','source')")
    result = preview.preview_bot_purge(preview_db, 'source')
    assert result['profile_version']['state'] == 'absent'
    assert by_key(result)['entity_profiles']['count'] == 2
    assert by_key(result)['messages']['disposition'] == 'unknown'
    assert by_key(result)['messages']['count'] is None
    assert by_key(result)['messages']['observed_storage_count'] == 1


def test_durable_profileless_alias_evidence_blocks_partition_attribution(preview_db):
    with preview_db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO bot_profile_tags VALUES ('so-urce')")
    result = preview.preview_bot_purge(preview_db, 'source')
    assert result['storage_ownership']['status'] == 'ambiguous'
    assert by_key(result)['messages']['count'] is None
    assert by_key(result)['sessions']['count'] == 3


def test_database_refuses_an_accidental_collector_write(preview_db, monkeypatch):
    def accidental_write(conn, slug):
        conn.exec_driver_sql("DELETE FROM sessions WHERE bot_id='source'")
    monkeypatch.setattr(preview, 'collect_preview', accidental_write)
    with pytest.raises(preview.PreviewUnavailable) as caught:
        preview.preview_bot_purge(preview_db, 'source')
    assert caught.value.__cause__.orig.pgcode == '25006'
    with preview_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM sessions WHERE bot_id='source'").scalar_one() == 3


def test_snapshot_stays_coherent_when_independent_writer_commits(preview_db, monkeypatch):
    original = preview.collect_preview
    saved = {}

    def interleaved(conn, slug):
        first = original(conn, slug)
        with preview_db.begin() as writer:
            writer.exec_driver_sql("INSERT INTO messages VALUES ('new-message','source','s-active')")
        second = original(conn, slug)
        assert second == first
        saved['first'] = first
        return second

    with monkeypatch.context() as patch:
        patch.setattr(preview, 'collect_preview', interleaved)
        result = preview.preview_bot_purge(preview_db, 'source')
    assert result == saved['first']
    newer = preview.preview_bot_purge(preview_db, 'source')
    assert by_key(newer)['messages']['count'] == 2
    assert newer['manifest_digest'] != result['manifest_digest']


def test_foreign_profile_and_tool_pointer_impacts_are_retained_not_deleted(preview_db):
    with preview_db.begin() as conn:
        conn.exec_driver_sql("UPDATE bot_profiles SET last_session_id='s-active' WHERE slug='dash'")
        conn.exec_driver_sql("UPDATE tool_call_records SET turn_id='t-source' WHERE bot_id='dash'")
    rows = by_key(preview.preview_bot_purge(preview_db, 'source'))
    assert rows['profile_session_pointer_effects']['count'] == 2
    assert rows['profile_session_pointer_effects']['disposition'] == 'retained'
    assert rows['tool_turn_pointer_effects']['count'] == 1
    assert rows['tool_turn_pointer_effects']['disposition'] == 'retained'
    with preview_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT last_session_id FROM bot_profiles WHERE slug='dash'").scalar_one() == 's-active'


def test_absent_optional_table_remains_unknown_without_bootstrap(preview_db):
    result = preview.preview_bot_purge(preview_db, 'source')
    assert by_key(result)['working_memory_items']['count'] is None
    assert by_key(result)['working_memory_items']['coverage'] == 'missing_table_or_columns'
    with preview_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT to_regclass('working_memory_items')").scalar_one() is None
