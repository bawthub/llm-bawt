"""Real PostgreSQL admission ordering and schema-owned exposure ledgers."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import ProgrammingError

from llm_bawt.content_provenance import ContentProvenanceStore, ContentTurn, _metadata
from tests.pg_lock_wait import wait_blocked_by
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(private_pg, private_module_db):
    result = create_engine(private_url(private_pg, private_module_db))
    yield result
    result.dispose()


def admit(store, turn):
    return store.admit(
        turn, tool_name="memory_store", write_kind="memory", invocation_hash="a" * 64,
    )


@pytest.mark.parametrize("schema", ["tenant", 'tenant"quoted'])
def test_public_ledgers_do_not_mask_tenant_bootstrap_or_runtime(engine, schema):
    public = ContentProvenanceStore(engine)
    turn = ContentTurn("fixture-reader", "fixture-owner", schema, "trigger")
    public.expose(turn)
    quoted = engine.dialect.identifier_preparer.quote_identifier(schema)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA {quoted}")
    tenant_engine = create_engine(engine.url)

    def choose_schema(conn, *_args):
        conn.exec_driver_sql(f"SET search_path TO {quoted}, public")

    event.listen(tenant_engine, "begin", choose_schema)
    try:
        tenant = ContentProvenanceStore(tenant_engine)
        assert not admit(tenant, turn).held
        tenant.expose(turn)
        assert admit(ContentProvenanceStore(tenant_engine), turn).held
        with tenant_engine.connect() as conn:
            assert conn.execute(text(
                "SELECT count(*) FROM agent_content_write_decisions",
            )).scalar_one() == 2
            for table in _metadata.tables:
                assert conn.execute(text(
                    "SELECT n.nspname FROM pg_class c JOIN pg_namespace n "
                    "ON n.oid=c.relnamespace WHERE c.oid=to_regclass(:name)",
                ), {"name": table}).scalar_one() == schema
    finally:
        event.remove(tenant_engine, "begin", choose_schema)
        tenant_engine.dispose()
    assert public.seen(turn)
    with engine.connect() as conn:
        assert conn.execute(text(
            "SELECT count(*) FROM agent_content_write_decisions",
        )).scalar_one() == 0


def test_write_waits_for_committing_exposure_and_records_hold(engine):
    store = ContentProvenanceStore(engine)
    turn = ContentTurn("fixture-racer", "fixture-owner", "parallel", "trigger")
    assert not store.seen(turn)
    with engine.connect() as holder, ThreadPoolExecutor(1) as pool:
        holder.execute(text(
            "UPDATE agent_content_turns SET untrusted_seen=true WHERE id=:id",
        ), {"id": turn.key})
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        worker = pool.submit(admit, ContentProvenanceStore(engine), turn)
        try:
            wait_blocked_by(engine, pid, worker)
        finally:
            holder.commit()
        assert worker.result(timeout=10).held
    restarted = ContentProvenanceStore(engine)
    assert admit(restarted, turn).held
    assert not admit(restarted, ContentTurn(
        turn.bot_id, turn.user_id, "fresh-human-turn", "fresh-trigger",
    )).held


def test_transitive_lineage_cycles_and_identity_isolation_survive_restart(engine):
    store = ContentProvenanceStore(engine)
    turns = [ContentTurn(f"fixture-{n}", "fixture-owner", "lineage", "trigger")
             for n in range(3)]
    store.inherit(turns[0], turns[1])
    store.inherit(turns[1], turns[2])
    store.inherit(turns[2], turns[0])
    store.expose(turns[0])
    restarted = ContentProvenanceStore(engine)
    assert admit(restarted, turns[2]).held
    assert not admit(restarted, ContentTurn(
        turns[2].bot_id, "other-owner", turns[2].turn_id, turns[2].trigger_message_id,
    )).held
    with pytest.raises(ValueError, match="trigger identity changed"):
        restarted.expose(ContentTurn(
            turns[0].bot_id, turns[0].user_id, turns[0].turn_id, "forged-trigger",
        ))


def test_missing_current_schema_cannot_fall_back_to_public(engine):
    ContentProvenanceStore(engine)
    missing = create_engine(engine.url)

    def choose_schema(conn, *_args):
        conn.exec_driver_sql("SET search_path TO missing_provenance_schema")

    event.listen(missing, "begin", choose_schema)
    try:
        with pytest.raises(ProgrammingError, match="no schema has been selected"):
            ContentProvenanceStore(missing)
    finally:
        missing.dispose()
