"""Hermetic registry browser tests; never touches Garage or real media."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from llm_bawt.media.assets import MediaAssetStore
from llm_bawt.service.routes import uploads


@pytest.fixture
def store():
    store = MediaAssetStore.__new__(MediaAssetStore)
    store.engine = create_engine("sqlite://")
    with store.engine.begin() as conn:
        conn.execute(text("""CREATE TABLE media_assets (
            id TEXT, filename TEXT, mime_type TEXT, storage_key TEXT,
            owner_user_id TEXT, kind TEXT, source TEXT, size_bytes INTEGER, created_at TEXT)
        """))
        for row in [
            ("a", "screen_100%.png", "image/webp", None, "operator", "image", "chat_upload", 20),
            ("b", "screen_1000.png", "image/webp", None, "operator", "image", "tool_generated", 30),
            ("c", "notice.mp3", "audio/mpeg", "audio/notice.mp3", None, "file", "agent_attachment", 50),
        ]:
            conn.execute(text("INSERT INTO media_assets VALUES (:id,:name,:mime,:key,:owner,:kind,:source,:size,'2026-09-23')"),
                         dict(zip(["id", "name", "mime", "key", "owner", "kind", "source", "size"], row)))
    yield store
    store.engine.dispose()


def test_browse_order_pagination_and_totals(store):
    page = store.browse(limit=2)
    assert [x["id"] for x in page["items"]] == ["c", "b"]
    assert page["total"] == 3 and page["total_bytes"] == 100
    assert [x["id"] for x in store.browse(limit=2, offset=2)["items"]] == ["a"]
    assert store.browse(offset=100)["total"] == 3
    assert store.browse(offset=100)["items"] == []


def test_browse_filters_literal_search_and_injection(store):
    assert store.browse(q="SCREEN_100%") ["total"] == 1
    assert store.browse(q="audio/notice")["total"] == 1
    assert store.browse(q="operator")["total"] == 2
    assert store.browse(q="' OR 1=1 --")["total"] == 0
    assert store.browse(kind="image", source="tool_generated")["total"] == 1
    assert store.browse(kind="file")["total_bytes"] == 50
    assert store.browse(q="missing")["total_bytes"] == 0


@pytest.mark.parametrize("params", [{"limit":0}, {"limit":101}, {"offset":-1}, {"kind":"sql"}, {"source":"sql"}])
def test_store_rejects_invalid_filters(store, params):
    with pytest.raises(ValueError):
        store.browse(**params)


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(uploads.router)
    row = {"id":"ma_test", "kind":"image", "filename":"test.png", "mime_type":"image/webp",
           "size_bytes":100, "source":"chat_upload", "owner_user_id":"operator",
           "created_at":datetime(2026,9,23,tzinfo=timezone.utc)}
    db = SimpleNamespace(engine=True, browse=lambda **kwargs: {"items":[row], "total":1,"total_bytes":100, **kwargs})
    monkeypatch.setattr(uploads, "_store", lambda: SimpleNamespace(db=db))
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", "https://example.test")
    return TestClient(app)


def test_list_route_serializes_canonical_links_and_metadata(client):
    response = client.get("/v1/uploads?limit=12&offset=0&q=test")
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 12
    asset = body["items"][0]
    assert asset["asset_id"] == "ma_test"
    assert asset["public_url"] == "https://example.test/api/chat/uploads/ma_test"
    assert asset["urls"]["thumb"] == "/v1/uploads/ma_test/thumb"
    assert asset["owner_user_id"] == "operator"
    assert asset["created_at"].startswith("2026-09-23")


@pytest.mark.parametrize("query", ["limit=101", "offset=-1", "kind=video", "source=unknown", "q=" + "x"*201])
def test_list_route_validates_query(client, query):
    assert client.get("/v1/uploads?" + query).status_code == 422


def test_missing_registry_is_not_an_empty_library(client, monkeypatch):
    monkeypatch.setattr(uploads, "_store", lambda: SimpleNamespace(db=None))
    assert client.get("/v1/uploads").status_code == 503
