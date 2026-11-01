"""Read-only inventory truthfulness, not an execution-readiness test."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from llm_bawt.bot_purge_inventory import DELETED, OWNER_SOURCES, RETAINED, UNKNOWN
from llm_bawt.bot_purge_preview import collect_preview, fingerprint, normalize_slug
from llm_bawt.service.routes import bot_purge_preview as routes


@pytest.fixture
def db():
    """Disposable in-memory fixtures, never application or configured stores."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    schema = {}
    for spec in (*DELETED, *RETAINED):
        for table, cols in ((spec.table, spec.columns), *spec.dependencies):
            schema.setdefault(table, set()).update(cols)
    for table, cols, _, _ in OWNER_SOURCES:
        schema.setdefault(table, set()).update(cols)
    schema['bot_profiles'].update(('updated_at', 'system_prompt'))
    schema['job_runs'].add('status')
    with engine.begin() as conn:
        for table, cols in schema.items():
            conn.exec_driver_sql(f'CREATE TABLE "{table}" ({", ".join(c + " TEXT" for c in sorted(cols))})')
        for slug in ('source', 'other'):
            conn.execute(text("INSERT INTO bot_profiles(slug,updated_at,system_prompt) VALUES (:slug,'v1','private-prompt')"), {'slug': slug})
        for table in ('messages', 'memories', 'forgotten_messages', 'sessions', 'turn_logs',
                      'tool_call_records', 'working_memory_items', 'turn_changed_files', 'tool_approval_requests'):
            conn.exec_driver_sql(f"INSERT INTO {table}(bot_id) VALUES ('source'),('other')")
        for table in ('runtime_settings', 'prompt_templates', 'prompt_template_versions', 'skill_assignments'):
            conn.exec_driver_sql(f"INSERT INTO {table}(scope_type,scope_id) VALUES ('bot','source'),('user','source'),('bot','other')")
        for table in ('entity_profiles', 'entity_profile_attributes'):
            conn.exec_driver_sql(f"INSERT INTO {table}(entity_type,entity_id) VALUES ('BOT','source'),('bot','source'),('USER','source'),('user','source'),('BOT','other')")
        conn.exec_driver_sql("INSERT INTO scheduled_jobs(id,bot_id) VALUES ('own','source'),('shared','*'),('else','other')")
        conn.exec_driver_sql("INSERT INTO job_runs(job_id,bot_id,status) VALUES ('own','source','SUCCESS'),('own','other','RUNNING'),('shared','source','RUNNING'),('else','source','SUCCESS')")
        for table in ('prompt_occurrences', 'prompt_schedules'):
            conn.exec_driver_sql(f"INSERT INTO {table}(job_id) VALUES ('own'),('shared')")
        conn.exec_driver_sql("INSERT INTO inter_bot_deliveries(sender_bot_id,target_bot_id,status) VALUES ('other','source','DELIVERED'),('source','other','RUNNING')")
        conn.exec_driver_sql("INSERT INTO new_session_event_outbox(bot_id,published_at) VALUES ('source','2020-01-01'),('source',NULL),('other',NULL)")
        conn.exec_driver_sql("INSERT INTO bot_profile_changes(bot_slug) VALUES ('source'),('other')")
    yield engine
    engine.dispose()


def snapshot(engine, slug='source'):
    with engine.connect() as conn:
        return collect_preview(conn, slug)


def entries(manifest):
    return {row['key']: row for row in manifest['inventory']}


def test_current_purge_ownership_retention_and_operational_subsets(db):
    result = snapshot(db)
    rows = entries(result)
    assert rows['messages']['count'] == rows['sessions']['count'] == 1
    assert rows['entity_profiles']['count'] == rows['entity_profile_attributes']['count'] == 2
    assert rows['runtime_settings']['count'] == 1
    assert rows['job_runs']['count'] == 2  # ownership via parent, not run bot_id/status
    assert rows['shared_job_runs']['count'] == 2
    assert rows['shared_job_runs']['disposition'] == 'retained'
    assert rows['inter_bot_deliveries']['count'] == 2
    assert rows['nonterminal_deliveries']['count'] == 1
    assert rows['new_session_event_outbox']['count'] == 2
    assert rows['pending_session_events']['count'] == 1
    for key in ('bot_profiles', 'bot_profile_changes', 'working_memory_items',
                'turn_changed_files', 'tool_approval_requests'):
        assert rows[key]['disposition'] == 'retained'
        assert rows[key]['count'] == 1
    assert result['execution_enabled'] is False
    assert result['storage_ownership']['status'] == 'observed_unique'
    assert 'safe_to_execute' not in result
    assert 'private-prompt' not in json.dumps(result)


@pytest.mark.parametrize('key', list(UNKNOWN))
def test_external_and_unproven_counts_are_null_not_zero(db, key):
    row = entries(snapshot(db))[key]
    assert row['count'] is None and row['disposition'] == 'unknown'
    assert row['coverage'] == 'not_inspected'


def test_legacy_payloads_count_only_through_owned_record(db):
    with db.begin() as conn:
        conn.exec_driver_sql("UPDATE tool_call_records SET id='1' WHERE bot_id='source'")
        conn.exec_driver_sql("UPDATE tool_call_records SET id='2' WHERE bot_id='other'")
        conn.exec_driver_sql("INSERT INTO tool_call_result_payloads(tool_call_record_id) VALUES ('1'),('2'),('orphan')")
    assert entries(snapshot(db))['tool_call_result_payloads']['count'] == 1


@pytest.mark.parametrize('table,column,extras', [
    ('bot_profiles', 'slug', {}), ('bot_profile_tags', 'bot_slug', {}),
    ('sessions', 'bot_id', {}), ('runtime_settings', 'scope_id', {'scope_type': 'bot'}),
    ('entity_profiles', 'entity_id', {'entity_type': 'BOT'}),
])
def test_registered_and_profileless_raw_alias_evidence_is_not_target_owned(db, table, column, extras):
    values = {column: 'so-urce', **extras}
    with db.begin() as conn:
        conn.execute(text(f'INSERT INTO {table} ({",".join(values)}) VALUES ({",".join(":"+k for k in values)})'), values)
    result = snapshot(db)
    assert result['storage_ownership']['status'] == 'ambiguous'
    for key in ('messages', 'memories', 'forgotten_messages'):
        row = entries(result)[key]
        assert row['disposition'] == 'unknown' and row['count'] is None
        assert row['observed_storage_count'] == 1
    assert 'so-urce' not in json.dumps(result)  # roster fingerprint, not identity disclosure


def test_profileless_preview_is_supported_but_partition_attribution_unproven(db):
    with db.begin() as conn:
        conn.exec_driver_sql("DELETE FROM bot_profiles WHERE slug='source'")
    result = snapshot(db)
    assert result['profile_version'] == {'state': 'absent', 'fingerprint': None}
    assert result['storage_ownership']['status'] == 'unproven_profileless'
    assert entries(result)['messages']['count'] is None
    assert entries(result)['sessions']['count'] == 1


@pytest.mark.parametrize('table', ['sessions', 'bot_profile_tags', 'tool_call_result_payloads', 'scheduled_jobs'])
def test_missing_tables_do_not_fabricate_zero_or_create_schema(db, table):
    with db.begin() as conn:
        conn.exec_driver_sql(f'DROP TABLE {table}')
    result = snapshot(db)
    row = entries(result)[table]
    assert row['disposition'] == 'unknown' and row['count'] is None
    assert row['coverage'] == 'missing_table_or_columns'
    if table == 'scheduled_jobs':
        assert entries(result)['job_runs']['count'] is None
    if table in {'sessions', 'bot_profile_tags'}:
        assert result['storage_ownership']['status'] == 'incomplete'
    with db.connect() as conn:
        assert table not in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").scalars().all()


def test_missing_columns_are_unknown(db):
    with db.begin() as conn:
        conn.exec_driver_sql('DROP TABLE turn_changed_files')
        conn.exec_driver_sql('CREATE TABLE turn_changed_files (unrelated TEXT)')
    assert entries(snapshot(db))['turn_changed_files']['count'] is None


def test_scope_profile_and_inventory_digest_are_deterministic_and_recomputed(db):
    one = snapshot(db)
    assert snapshot(db, ' SOURCE ') == one
    assert fingerprint({k: v for k, v in one.items() if k != 'manifest_digest'}) == one['manifest_digest']
    with db.begin() as conn:
        conn.exec_driver_sql("UPDATE bot_profiles SET system_prompt='changed-private-prompt' WHERE slug='source'")
    two = snapshot(db)
    assert one['profile_version'] != two['profile_version']
    assert one['manifest_digest'] != two['manifest_digest']
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages(bot_id) VALUES ('source')")
    three = snapshot(db)
    assert three['manifest_digest'] != two['manifest_digest']
    with db.begin() as conn:
        conn.exec_driver_sql("UPDATE bot_profiles SET system_prompt='unrelated' WHERE slug='other'")
    assert snapshot(db) == three


def test_collection_issues_only_reads_and_does_not_commit_caller_transaction(db):
    statements = []
    event.listen(db, 'before_cursor_execute', lambda conn, cur, stmt, params, ctx, many: statements.append(stmt))
    with db.connect() as conn:
        tx = conn.begin()
        collect_preview(conn, 'source')
        assert tx.is_active
        tx.rollback()
    assert statements
    assert all(s.lstrip().upper().startswith(('SELECT', 'PRAGMA')) for s in statements)


@pytest.mark.parametrize('slug', ['', '  ', '\n', 'a\x00b', 'x'*129])
def test_invalid_slugs_are_rejected_without_db(slug):
    with pytest.raises(ValueError):
        normalize_slug(slug)


def test_one_character_and_legacy_punctuation_are_supported():
    assert normalize_slug(' X ') == 'x'
    assert normalize_slug(' Test.Bot-1 ') == 'test.bot-1'


@pytest.fixture
def client(monkeypatch):
    service = SimpleNamespace(config=object())
    monkeypatch.setattr(routes, 'get_service', lambda: service)
    engine = Mock()
    monkeypatch.setattr(routes, 'get_shared_engine', lambda config: engine)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as c:
        yield c


def test_get_route_is_no_store_and_never_authorizes_execution(client, monkeypatch, db):
    call = Mock(side_effect=lambda engine, slug: snapshot(db, slug))
    monkeypatch.setattr(routes, 'preview_bot_purge', call)
    response = client.get('/v1/bots/SOURCE/purge-preview')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json()['execution_enabled'] is False
    assert call.call_args.args[1] == 'source'
    assert client.post('/v1/bots/source/purge-preview').status_code == 405


def test_invalid_route_slug_is_rejected_before_service_or_engine(client, monkeypatch):
    lookup = Mock(side_effect=AssertionError('must not read service'))
    monkeypatch.setattr(routes, 'get_service', lookup)
    response = client.get('/v1/bots/%20/purge-preview')
    assert response.status_code == 422
    assert response.headers['cache-control'] == 'no-store'
    lookup.assert_not_called()


@pytest.mark.parametrize('missing', [False, True])
def test_route_unavailable_is_no_store_503_without_private_driver_detail(client, monkeypatch, missing):
    if missing:
        monkeypatch.setattr(routes, 'get_shared_engine', lambda config: None)
    else:
        monkeypatch.setattr(routes, 'get_shared_engine', Mock(side_effect=OperationalError('private sql', {'secret': 'private'}, Exception('private driver'))))
    response = client.get('/v1/bots/source/purge-preview')
    assert response.status_code == 503
    assert response.headers['cache-control'] == 'no-store'
    assert 'private' not in response.text
    assert 'no writes were attempted' in response.text
