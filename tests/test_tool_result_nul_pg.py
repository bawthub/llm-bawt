"""TASK-1443: a tool result containing NUL (0x00) must persist on real Postgres.

Postgres TEXT rejects NUL, so ``save_result`` used to fail at flush and the tool
card lost its output. The preview column now holds a same-length U+FFFD copy and
the exact bytes take the overflow path (object store, else a lossy TEXT row).
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy import text
from sqlmodel import Session

from agent_bridge.tool_results import ToolResultPayload
from llm_bawt.service import tool_call_store as tcs
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord, ToolCallStore
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

NUL_RESULT = "header\x00\x00binary\x00tail"


class FakeBlobs:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self.objects

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    def get(self, key: str) -> bytes:
        return self.objects[key]


@pytest.fixture
def store():
    from llm_bawt.service.turn_logs import TurnLogStore

    engine = TurnLogStore(require_test_db_config()).engine
    assert engine is not None
    store = ToolCallStore(engine)
    store.ensure_schema()
    return store


def _blob_backend(monkeypatch, backend) -> None:
    monkeypatch.setattr(tcs, "_TOOL_BLOB_RESOLVED", True)
    monkeypatch.setattr(tcs, "_TOOL_BLOB_BACKEND", backend)


def _save(store: ToolCallStore, content: str):
    turn = f"turn-nul-{uuid.uuid4().hex[:12]}"
    with store.engine.begin() as conn:
        conn.execute(
            text("INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt)"
                 " VALUES (:id, now(), '/v1/chat/completions', false, 'test-nul-bot', 'nick', 'ok', 'probe')"),
            {"id": turn},
        )
    return store.save_result(
        turn_id=turn, call_id="call-1", tool_use_id="tu-1", tool_name="Read",
        bot_id="test-nul-bot", user_id="nick", payload=ToolResultPayload.from_value(content),
        ended_at=None, is_error=False,
    )


def test_nul_result_persists_with_exact_bytes_in_object_store(store, monkeypatch):
    blobs = FakeBlobs()
    _blob_backend(monkeypatch, blobs)

    record_id, available = _save(store, NUL_RESULT)

    assert record_id is not None and available is True
    with Session(store.engine) as session:
        row = session.get(ToolCallRecord, record_id)
        assert row.result_text == NUL_RESULT.replace("\x00", "�")
        assert row.result_total_chars == len(NUL_RESULT)
        assert row.result_sha256 == hashlib.sha256(NUL_RESULT.encode()).hexdigest()
        assert row.result_blob_key == row.result_sha256
        assert session.get(ToolCallResultPayloadRecord, record_id) is None
    page = store.page(record_id, bot_id="test-nul-bot", user_id="nick", offset=0, limit=100)
    assert page is not None and page.content == NUL_RESULT


def test_nul_result_persists_without_object_store(store, monkeypatch):
    _blob_backend(monkeypatch, None)

    record_id, available = _save(store, NUL_RESULT)

    assert record_id is not None and available is True
    page = store.page(record_id, bot_id="test-nul-bot", user_id="nick", offset=0, limit=100)
    assert page is not None
    assert page.content == NUL_RESULT.replace("\x00", "�")
    assert page.total_chars == len(NUL_RESULT)


def test_plain_small_result_still_stays_preview_only(store, monkeypatch):
    _blob_backend(monkeypatch, FakeBlobs())

    record_id, available = _save(store, "plain output")

    assert record_id is not None and available is False
    with Session(store.engine) as session:
        assert session.get(ToolCallRecord, record_id).result_text == "plain output"
