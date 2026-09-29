"""TASK-977: per-tool-call media refs survive tool_end -> refresh -> reload.

Sequence under test: the live tool_end carries enriched envelopes (the flash);
the canonical tool_call_records row must keep the refs so /v1/tool-calls can
rebuild the same card after activity refresh or a hard reload (the disappear).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlmodel import Session, SQLModel, create_engine

from agent_bridge.tool_results import ToolResultPayload
from llm_bawt.media.serializers import attachment_refs
from llm_bawt.service import dependencies
from llm_bawt.service.routes import turn_logs as routes
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord, ToolCallStore
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.turn_logs import TurnLog, TurnLogStore


def _envelope(asset_id: str) -> dict:
    return {
        "asset_id": asset_id,
        "kind": "image",
        "mime_type": "image/webp",
        "urls": {"original": f"/v1/uploads/{asset_id}"},
    }


class FakeAssets:
    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.calls: list[list[str]] = []

    def get_many(self, ids):
        self.calls.append(list(ids))
        return [
            {"id": aid, "kind": "image", "mime_type": "image/webp", "width": 8, "height": 8,
             "filename": f"{aid}.webp", "size_bytes": 10}
            for aid in ids if aid in self.known
        ]

    def get_by_id(self, aid):
        return None


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__],
    )
    with Session(engine) as session:
        session.add(TurnLog(id="turn", bot_id="al", user_id="nick", status="ok",
                            trigger_message_id="msg-1", ended_at=datetime.now(timezone.utc)))
        session.commit()
    return engine


@pytest.fixture
def assets(monkeypatch):
    fake = FakeAssets({"ma_a", "ma_b"})
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(dependencies, "get_media_asset_store", lambda _config: fake)
    return fake


def _end(coordinator: ToolEventCoordinator, call_id: str, attachments, *, is_error=False) -> dict:
    return coordinator.end({
        "turn_id": "turn", "call_id": call_id, "tool_use_id": f"tu-{call_id}",
        "tool_name": "generate_image", "bot_id": "al", "user_id": "nick",
        "result": "ok", "ts": 150.0, "is_error": is_error, "attachments": attachments,
    })


def _row(engine, call_id: str) -> ToolCallRecord:
    with Session(engine) as session:
        from sqlmodel import select
        return session.exec(select(ToolCallRecord).where(ToolCallRecord.call_id == call_id)).one()


def test_attachment_refs_reduces_envelopes_dedupes_and_drops_idless():
    refs = attachment_refs([_envelope("ma_a"), _envelope("ma_a"), {"kind": "image"}, "junk",
                            {"asset_id": "ma_b"}])
    assert refs == [{"asset_id": "ma_a", "kind": "image"}, {"asset_id": "ma_b", "kind": "image"}]
    assert attachment_refs(None) == []


def test_tool_end_persists_refs_and_keeps_live_envelopes(engine):
    public = _end(ToolEventCoordinator(engine), "c1", [_envelope("ma_a")])
    # Live SSE still gets the full envelope (inline preview renders immediately).
    assert public["attachments"] == [_envelope("ma_a")]
    # The durable row keeps only the tiny ref — URLs are re-derived on read.
    assert json.loads(_row(engine, "c1").attachments_json) == [{"asset_id": "ma_a", "kind": "image"}]


def test_replayed_tool_end_without_media_preserves_refs(engine):
    coordinator = ToolEventCoordinator(engine)
    _end(coordinator, "c1", [_envelope("ma_a")])
    # Approval resolution / replay rewrites the result without attachments.
    ToolCallStore(engine).save_result(
        turn_id="turn", call_id="c1", tool_use_id="tu-c1", tool_name="generate_image",
        bot_id="al", user_id="nick", payload=ToolResultPayload.from_value("ok again"),
        ended_at=160.0, is_error=False,
    )
    assert json.loads(_row(engine, "c1").attachments_json) == [{"asset_id": "ma_a", "kind": "image"}]


def test_tool_end_without_media_stores_nothing(engine):
    _end(ToolEventCoordinator(engine), "c1", None)
    assert _row(engine, "c1").attachments_json is None
    assert routes._records_to_calls([_row(engine, "c1")])[0]["attachments"] == []


def test_tool_calls_api_rehydrates_each_card_once(engine, assets, monkeypatch):
    coordinator = ToolEventCoordinator(engine)
    _end(coordinator, "c1", [_envelope("ma_a")])
    _end(coordinator, "c2", [_envelope("ma_b"), _envelope("ma_gone")])
    _end(coordinator, "c3", None, is_error=True)  # failed image call: no media

    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = engine
    store.ttl_hours = None
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: store)

    response = routes.get_tool_call_events(
        bot_id="al", user_id="nick", message_id=None, message_ids=None,
        after=None, before=None, since_hours=168, limit=200,
    )
    calls = {c["call_id"]: c for c in response.model_dump()["events"][0]["tool_calls"]}
    assert [a["asset_id"] for a in calls["c1"]["attachments"]] == ["ma_a"]
    # Deleted asset drops out; the surviving one keeps its URL envelope.
    assert [a["asset_id"] for a in calls["c2"]["attachments"]] == ["ma_b"]
    assert calls["c2"]["attachments"][0]["urls"]["original"].endswith("ma_b")
    assert calls["c3"]["attachments"] == []
    # One batched asset lookup for the whole listing.
    assert len(assets.calls) == 1


def test_enrichment_failure_degrades_to_no_media(monkeypatch):
    def boom(_config):
        raise RuntimeError("media db down")

    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(dependencies, "get_media_asset_store", boom)
    calls = [{"call_id": "c1", "attachments": [{"asset_id": "ma_a", "kind": "image"}]}]
    routes._enrich_call_attachments(calls)
    assert calls[0]["attachments"] == []
