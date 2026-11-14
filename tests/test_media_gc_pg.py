"""TASK-2224: media GC reachability against a real, private PostgreSQL."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.service.jobs import media_gc
from llm_bawt.service.jobs.media_gc import collect_local_references, run_media_gc
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


def _id(tag: str) -> str:
    return "ma_" + tag.upper().ljust(26, "0")[:26]


ORPHAN = _id("orphan")
ATTACHED = _id("attached")
TOOL_CARD = _id("toolcard")
INLINE_A = _id("inlinea")
INLINE_B = _id("inlineb")
ARRAY_REF = _id("arrayref")
YOUNG = _id("young")
EXPIRED_REF = _id("expiredref")
NAMED = _id("named")


@pytest.fixture(scope="module")
def engine(private_pg, private_module_db):
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE media_assets (
                id text PRIMARY KEY, sha256 text NOT NULL, size_bytes bigint,
                storage_key text, expires_at timestamptz,
                created_at timestamptz NOT NULL DEFAULT now())
        """))
        conn.execute(text("CREATE TABLE messages (id serial, content text, attachments jsonb)"))
        # A stopped turn: tool card saved, no assistant message.
        conn.execute(text("CREATE TABLE tool_call_records (id serial, attachments_json jsonb, result_text text)"))
        # Mixed-case identifiers, as Prisma creates them in BawtHub.
        conn.execute(text('CREATE TABLE "AgentNote" (id serial, "imageUrls" text[], "Body" varchar(500))'))
        old = "now() - interval '30 days'"
        for asset in (ORPHAN, ATTACHED, TOOL_CARD, INLINE_A, INLINE_B, ARRAY_REF):
            conn.execute(text(f"INSERT INTO media_assets (id, sha256, size_bytes, created_at) VALUES (:i, :i, 1, {old})"), {"i": asset})
        conn.execute(text("INSERT INTO media_assets (id, sha256, size_bytes) VALUES (:i, :i, 1)"), {"i": YOUNG})
        conn.execute(text("INSERT INTO media_assets (id, sha256, size_bytes, expires_at) VALUES (:i, :i, 1, now() - interval '1 day')"), {"i": EXPIRED_REF})
        conn.execute(text(f"INSERT INTO media_assets (id, sha256, size_bytes, storage_key, created_at) VALUES (:i, :i, 1, 'named/x', {old})"), {"i": NAMED})

        conn.execute(text("INSERT INTO messages (content, attachments) VALUES ('hi', :a)"),
                     {"a": f'[{{"asset_id": "{ATTACHED}", "kind": "image"}}]'})
        conn.execute(text("INSERT INTO tool_call_records (attachments_json, result_text) VALUES (:a, 'done')"),
                     {"a": f'[{{"asset_id": "{TOOL_CARD}"}}]'})
        # Two ids in one value must both count.
        conn.execute(text("INSERT INTO messages (content) VALUES (:c)"),
                     {"c": f"![a](/api/chat/uploads/{INLINE_A}) and ![b](/api/chat/uploads/{INLINE_B})"})
        conn.execute(text('INSERT INTO "AgentNote" ("imageUrls", "Body") VALUES (:u, :b)'),
                     {"u": [f"/api/chat/uploads/{ARRAY_REF}"], "b": f"see {EXPIRED_REF}"})
    yield engine
    engine.dispose()


class _Store:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, asset_id: str) -> None:
        self.deleted.append(asset_id)


@pytest.fixture
def wired(engine, monkeypatch):
    import llm_bawt.utils.db as utils_db

    monkeypatch.setattr(utils_db, "get_shared_engine", lambda config, **k: engine)
    return engine


def _no_remote() -> tuple[set[str], int]:
    return set(), 1


def test_sweep_finds_every_holder(engine):
    with engine.connect() as conn, conn.begin():
        ids, columns = collect_local_references(conn, deadline=float("inf"))
    assert {ATTACHED, TOOL_CARD, INLINE_A, INLINE_B, ARRAY_REF, EXPIRED_REF} <= ids
    assert ORPHAN not in ids
    # media_assets is never a reference source.
    assert YOUNG not in ids
    assert columns >= 6


def test_only_true_orphan_is_collected(wired):
    store = _Store()
    result = run_media_gc(object(), media_store=store, fetch_remote=_no_remote, max_candidates=10)
    assert store.deleted == [ORPHAN]
    assert result["deleted_count"] == 1


def test_remote_reference_protects(wired):
    store = _Store()
    run_media_gc(object(), media_store=store, fetch_remote=lambda: ({ORPHAN}, 4), max_candidates=10)
    assert store.deleted == []


def test_late_column_failure_deletes_nothing(wired, monkeypatch):
    """Earlier columns succeed, the last one errors: nothing is deleted."""
    real = media_gc._column_ids_sql
    seen: list[Any] = []

    def failing_last(table: str, column: str) -> str:
        seen.append((table, column))
        if (table, column) == ("tool_call_records", "result_text"):
            return "SELECT no_such_column FROM tool_call_records"
        return real(table, column)

    monkeypatch.setattr(media_gc, "_column_ids_sql", failing_last)
    store = _Store()
    result = run_media_gc(object(), media_store=store, fetch_remote=_no_remote, max_candidates=10)

    assert len(seen) > 1
    assert store.deleted == []
    assert "tool_call_records.result_text" in result["aborted_reason"]
