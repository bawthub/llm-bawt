"""Isolated PostgreSQL acceptance for missing-memory-vector repair."""

from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import event, text

from llm_bawt.memory import embeddings
from llm_bawt.memory import embedding_repair
from llm_bawt.memory.embedding_repair import EmbeddingRepair
from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service import embedding_repair_job
from tests.private_postgres import private_app_pg

pytestmark = pytest.mark.integration


class EmbedServer:
    def __init__(self):
        self.code = 503
        self.failed_texts = set()
        self.calls = []

    def post(self, _path, *, json):
        payload = json["texts"][0]
        self.calls.append(payload)
        code = 400 if payload in self.failed_texts else self.code

        class Response:
            status_code = code

            def json(self):
                return {"embeddings": [[0.05] * embeddings.EMBED_DIM]}

        return Response()


@pytest.fixture
def memory_rig(private_app_pg, monkeypatch):
    config = private_app_pg
    slug = f"repair-{uuid.uuid4().hex[:12]}"
    server = EmbedServer()
    monkeypatch.setattr(embeddings, "_get_client", lambda: server)
    store = BotProfileStore(config)
    store.upsert({"slug": slug, "name": slug, "system_prompt": "fixture",
                  "include_in_global_search": True})
    backend = PostgreSQLMemoryBackend(config, bot_id=slug)
    repair = EmbeddingRepair(config, backend.engine)
    return slug, server, backend, repair, store


def _row(backend, memory_id):
    with backend.engine.connect() as conn:
        return conn.execute(text(f"""
            SELECT embedding IS NOT NULL AS content_ok,
                   meaning_embedding IS NOT NULL AS meaning_ok
            FROM {backend._memories_table_name} WHERE id=:id
        """), {"id": memory_id}).one()


def test_outage_write_recovers_both_vectors_and_stats(memory_rig):
    _slug, server, backend, repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "coffee and croissants", intent="a treat")
    assert tuple(_row(backend, memory_id)) == (False, False)
    before = backend.stats()["memories"]
    assert before["missing_content_embeddings"] == 1
    assert before["missing_meaning_embeddings"] == 1
    assert before["missing_any_embeddings"] == 1

    server.code = 200
    result = repair.repair_bot(backend.bot_id)
    assert result["repaired_content"] == result["repaired_meaning"] == 1
    assert tuple(_row(backend, memory_id)) == (True, True)
    assert backend.stats()["memories"]["missing_any_embeddings"] == 0


def test_changed_memory_during_outage_discards_stale_vectors(memory_rig):
    _slug, server, backend, repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    server.code = 200
    backend.add_memory(memory_id, "old content", intent="old intent")
    assert tuple(_row(backend, memory_id)) == (True, True)
    server.code = 503
    backend.add_memory(memory_id, "new content", intent="new intent")
    assert tuple(_row(backend, memory_id)) == (False, False)
    server.code = 200
    repaired = repair.repair_bot(backend.bot_id)
    assert repaired["repaired_content"] == repaired["repaired_meaning"] == 1
    assert tuple(_row(backend, memory_id)) == (True, True)


def test_only_missing_meaning_is_repaired(memory_rig):
    _slug, server, backend, repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    server.code = 200
    backend.add_memory(memory_id, "a fact", embedding=[0.04] * embeddings.EMBED_DIM,
                       meaning_embedding=None, stakes="remember")
    # Meaning generation succeeded; recreate the failure without changing content.
    with backend.engine.begin() as conn:
        conn.execute(text(f"UPDATE {backend._memories_table_name} "
                          "SET meaning_embedding=NULL WHERE id=:id"), {"id": memory_id})
    server.calls.clear()
    result = repair.repair_bot(backend.bot_id)
    assert result["repaired_content"] == 0
    assert result["repaired_meaning"] == 1
    assert server.calls == ["remember"]


def test_permanent_failures_do_not_starve_and_cursor_survives_instances(memory_rig):
    _slug, server, backend, repair, _store = memory_rig
    ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(1, 5)]
    server.code = 503
    for i, memory_id in enumerate(ids, start=1):
        backend.add_memory(memory_id, f"candidate {i}")
    server.code = 200
    server.failed_texts = {f"candidate {i}" for i in range(1, 4)}
    first = repair.repair_bot(backend.bot_id, limit=3)
    assert first["permanent_failures"] >= 3
    second = EmbeddingRepair(backend.config, backend.engine).repair_bot(backend.bot_id, limit=1)
    assert second["repaired_content"] == second["repaired_meaning"] == 1
    assert tuple(_row(backend, ids[3])) == (True, True)


def test_repeated_outage_is_bounded_and_leaves_vectors_null(memory_rig):
    _slug, server, backend, repair, _store = memory_rig
    for i in range(4):
        backend.add_memory(str(uuid.uuid4()), f"outage candidate {i}")
    server.calls.clear()
    result = repair.repair_bot(backend.bot_id)
    assert result["stopped"] and result["transient_failures"] >= 3
    assert result["examined"] <= 3
    assert len(server.calls) <= 6  # 3 vector attempts, two HTTP retries each.


def test_new_private_assignment_during_embed_refuses_update(memory_rig, monkeypatch):
    slug, server, backend, repair, store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "private transition")
    server.code = 200
    real_embed = embedding_repair.embed_with_status

    def private_during_call(texts, model):
        store.upsert({"slug": slug, "include_in_global_search": False})
        return real_embed(texts, model)

    monkeypatch.setattr(embedding_repair, "embed_with_status", private_during_call)
    result = repair.repair_bot(slug)
    assert result["stopped"]
    assert result["repaired_content"] == result["repaired_meaning"] == 0
    assert tuple(_row(backend, memory_id)) == (False, False)


def test_changed_content_is_not_overwritten_by_stale_vector(memory_rig, monkeypatch):
    _slug, server, backend, repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "source before edit")
    server.code = 200
    original_fill = repair._fill

    def edit_then_fill(conn, table, params, row, field, vector):
        with backend.engine.begin() as writer:
            writer.execute(text(f"UPDATE {table} SET content='source after edit' WHERE id=:id"),
                           {"id": memory_id})
        return original_fill(conn, table, params, row, field, vector)

    monkeypatch.setattr(repair, "_fill", edit_then_fill)
    result = repair.repair_bot(backend.bot_id)
    assert result["repaired_content"] == result["repaired_meaning"] == 0
    assert tuple(_row(backend, memory_id)) == (False, False)


def test_privacy_change_at_update_is_rejected_by_sql_guard(memory_rig, monkeypatch):
    slug, server, backend, repair, store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "privacy at update")
    server.code = 200
    original_fill = repair._fill

    def privatize_then_fill(conn, table, params, row, field, vector):
        store.upsert({"slug": slug, "include_in_global_search": False})
        return original_fill(conn, table, params, row, field, vector)

    monkeypatch.setattr(repair, "_fill", privatize_then_fill)
    result = repair.repair_bot(slug)
    assert result["repaired_content"] == result["repaired_meaning"] == 0
    assert tuple(_row(backend, memory_id)) == (False, False)


def test_duplicate_runner_skips_without_reading_memory(memory_rig):
    slug, server, backend, repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "busy repair")
    server.calls.clear()
    with backend.engine.connect() as conn:
        assert conn.execute(text(embedding_repair._LOCK_SQL)).scalar()
        conn.commit()
        try:
            assert repair.repair_bot(slug)["busy"]
            assert not server.calls
        finally:
            conn.execute(text(embedding_repair._UNLOCK_SQL))
            conn.commit()


def test_report_mode_is_read_only_and_does_not_call_repair(memory_rig, monkeypatch):
    _slug, server, backend, _repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "report only")
    server.calls.clear()
    monkeypatch.setattr(embedding_repair_job, "resolve_global_runtime_setting",
                        lambda _config, _key: "report")
    monkeypatch.setattr(EmbeddingRepair, "repair_bot",
                        lambda *_args, **_kwargs: pytest.fail("report called repair"))
    assert not embedding_repair_job.run_pass(backend.config, backend.engine,
                                             threading.Event())
    assert not server.calls
    assert tuple(_row(backend, memory_id)) == (False, False)


def test_dimension_mismatch_refuses_writes(memory_rig, monkeypatch):
    _slug, server, backend, repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "wrong dimension")
    server.calls.clear()
    monkeypatch.setattr(embedding_repair, "EMBED_DIM", embeddings.EMBED_DIM + 1)
    result = repair.repair_bot(backend.bot_id)
    assert result["dimension_mismatch"] and result["stopped"]
    assert not server.calls
    assert tuple(_row(backend, memory_id)) == (False, False)


def test_stop_event_prevents_vector_writes(memory_rig):
    _slug, server, backend, repair, _store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "stop first")
    server.calls.clear()
    stop = threading.Event()
    stop.set()
    assert repair.repair_bot(backend.bot_id, stop=stop)["stopped"]
    assert not server.calls
    assert tuple(_row(backend, memory_id)) == (False, False)


def test_private_bot_is_unread_and_report_never_creates_progress(memory_rig):
    slug, server, backend, repair, store = memory_rig
    memory_id = str(uuid.uuid4())
    backend.add_memory(memory_id, "protected fixture")
    store.upsert({"slug": slug, "include_in_global_search": False})
    server.calls.clear()
    statements = []

    def observed(_conn, _cursor, statement, _params, _context, _many):
        statements.append(statement)
    event.listen(backend.engine, "before_cursor_execute", observed)
    try:
        assert slug not in repair.report()
    finally:
        event.remove(backend.engine, "before_cursor_execute", observed)
    assert not any("CREATE TABLE" in statement.upper() for statement in statements)
    result = repair.repair_bot(slug)
    assert result["excluded"] and not server.calls
    assert tuple(_row(backend, memory_id)) == (False, False)
    with backend.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT count(*) FROM memory_embedding_repair_progress WHERE bot_id=:bot_id
        """), {"bot_id": slug}).scalar_one() == 0
