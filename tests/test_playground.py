"""Playground artifacts (TASK-2190): own blobs, one-time open, expiry, routes, tool."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import update

from llm_bawt.bot_access import BotAccessError
from llm_bawt.media.object_store import BlobBackendUnavailable, FsBlobBackend
from llm_bawt.mcp_server import playground_tools
from llm_bawt.playground import service as pg_service
from llm_bawt.playground.service import (
    Playground,
    PlaygroundError,
    Source,
    kind_for_mime,
    source_from_asset,
    source_from_inline,
)
from llm_bawt.playground.store import OpenRefused, PlaygroundStore, artifacts, public_view
from llm_bawt.service.routes import playground as routes
from tests.sqlite_support import fast_sqlite_engine

USER = "user-1"
PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


@pytest.fixture
def pg(tmp_path):
    store = PlaygroundStore(fast_sqlite_engine(tmp_path / "pg.db"))
    return Playground(store, FsBlobBackend(tmp_path / "blobs"))


def _send_png(pg, user=USER, bot="vex", title="A picture"):
    return pg.send(user_id=user, bot_id=bot, title=title, note=None,
                   source=Source(kind="image", mime="image/png", filename="a.png", data=PNG))


def _blob_path(pg, row):
    return pg.backend.root / row["blob_key"]


def test_send_writes_its_own_blob_and_lists_metadata_only(pg):
    row = _send_png(pg)
    assert row["id"].startswith("pg_") and row["blob_key"] == f"playground/{row['id']}"
    assert _blob_path(pg, row).read_bytes() == PNG
    [item] = [public_view(r) for r in pg.store.list(USER)]
    assert item["status"] == "unopened" and item["size_bytes"] == len(PNG)
    assert "blob_key" not in item and "inline_content" not in item
    assert pg.store.list("someone-else") == []


def test_open_once_then_close_deletes_bytes_and_keeps_history(pg):
    row = _send_png(pg)
    path = _blob_path(pg, row)
    opened, token = pg.open(row["id"], USER)
    assert opened["opened_at"] is not None
    live = pg.store.viewable(row["id"], USER, token)
    assert pg.read(live, None).data == PNG
    assert pg.store.viewable(row["id"], USER, "wrong") is None

    closed = pg.expire(row["id"], USER)
    assert closed["expired_at"] is not None and not path.exists()
    stored = pg.store.get(row["id"], USER)
    assert stored["blob_key"] is None and stored["view_token_hash"] is None
    assert pg.store.viewable(row["id"], USER, token) is None
    with pytest.raises(OpenRefused) as refused:
        pg.open(row["id"], USER)
    assert refused.value.code == "expired"
    assert public_view(stored)["status"] == "expired"


def test_second_open_expires_the_artifact(pg):
    row = _send_png(pg)
    pg.open(row["id"], USER)
    with pytest.raises(OpenRefused):
        pg.open(row["id"], USER)
    assert pg.store.get(row["id"], USER)["expired_at"] is not None
    assert not _blob_path(pg, row).exists()


def test_open_is_scoped_to_the_user(pg):
    row = _send_png(pg)
    with pytest.raises(OpenRefused) as refused:
        pg.open(row["id"], "intruder")
    assert refused.value.code == "not_found"
    assert pg.expire(row["id"], "intruder") is None
    assert _blob_path(pg, row).exists()


def test_inline_content_is_cleared_on_expiry(pg):
    row = pg.send(user_id=USER, bot_id="vex", title="Page", note="hi",
                  source=source_from_inline("<h1>hi</h1>", "html"))
    assert row["blob_key"] is None and row["kind"] == "html"
    _, token = pg.open(row["id"], USER)
    assert pg.read(pg.store.viewable(row["id"], USER, token), None) == "<h1>hi</h1>"
    pg.expire(row["id"], USER)
    assert pg.store.get(row["id"], USER)["inline_content"] is None


def test_sweep_expires_only_overdue_opened_artifacts(pg):
    stale, fresh, unopened = _send_png(pg), _send_png(pg), _send_png(pg)
    pg.open(stale["id"], USER)
    pg.open(fresh["id"], USER)
    with pg.store.engine.begin() as conn:
        conn.execute(update(artifacts).where(artifacts.c.id == stale["id"]).values(
            opened_at=datetime.now(timezone.utc) - timedelta(minutes=45)))
    assert pg.sweep(30) == {"expired": 1, "blobs_deleted": 1, "blob_delete_failures": 0}
    assert not _blob_path(pg, stale).exists()
    assert _blob_path(pg, fresh).exists() and _blob_path(pg, unopened).exists()
    assert pg.store.get(unopened["id"], USER)["expired_at"] is None


def test_failed_blob_delete_is_retried_by_the_sweep(pg, monkeypatch):
    row = _send_png(pg)
    real_delete = pg.backend.delete

    def down(key):
        raise BlobBackendUnavailable("down")

    monkeypatch.setattr(pg.backend, "delete", down)
    pg.expire(row["id"], USER)
    assert pg.store.get(row["id"], USER)["blob_key"] == row["blob_key"]
    monkeypatch.setattr(pg.backend, "delete", real_delete)
    assert pg.sweep(30)["blobs_deleted"] == 1
    assert pg.store.get(row["id"], USER)["blob_key"] is None
    assert not _blob_path(pg, row).exists()


def test_failed_insert_removes_the_new_blob(pg, monkeypatch):
    def boom(row):
        raise RuntimeError("db down")

    monkeypatch.setattr(pg.store, "insert", boom)
    with pytest.raises(RuntimeError):
        _send_png(pg)
    assert not list((pg.backend.root / "playground").glob("*"))


class FakeMediaStore:
    def __init__(self, data, mime, filename=None):
        self.data, self.mime, self.filename = data, mime, filename
        self.reads = 0

    def stat(self, asset_id):
        if asset_id != "ma_1":
            return None
        return SimpleNamespace(size_bytes=len(self.data), filename=self.filename)

    def read_variant(self, asset_id, variant):
        assert variant == "original"
        self.reads += 1
        return self.data, self.mime


def test_asset_source_copies_bytes_and_leaves_the_source_alone(pg):
    media = FakeMediaStore(b"RIFFwav", "audio/wav", "clip.wav")
    source = source_from_asset(media, "ma_1", max_bytes=1000)
    row = pg.send(user_id=USER, bot_id="vex", title="Clip", note=None, source=source)
    assert row["kind"] == "audio" and row["filename"] == "clip.wav"
    pg.expire(row["id"], USER)
    assert media.data == b"RIFFwav"  # nothing on the source side is touched
    with pytest.raises(PlaygroundError):
        source_from_asset(media, "ma_missing", max_bytes=1000)
    with pytest.raises(PlaygroundError):
        source_from_asset(media, "ma_1", max_bytes=3)


@pytest.mark.parametrize("mime,kind", [
    ("image/webp", "image"), ("video/mp4", "video"), ("audio/mpeg", "audio"),
    ("application/pdf", "pdf"), ("text/html; charset=utf-8", "html"),
    ("text/markdown", "markdown"), ("text/x-python", "text"),
    ("application/zip", "file"),
])
def test_kind_for_mime(mime, kind):
    assert kind_for_mime(mime) == kind


def test_inline_validation():
    with pytest.raises(PlaygroundError):
        source_from_inline("x", "pdf")
    with pytest.raises(PlaygroundError):
        source_from_inline("   ", "text")
    with pytest.raises(PlaygroundError):
        source_from_inline("x" * (pg_service.MAX_INLINE_BYTES + 1), "text")


# -- REST -------------------------------------------------------------------


@pytest.fixture
def client(pg, monkeypatch):
    monkeypatch.setattr(routes, "_playground", lambda: pg)
    monkeypatch.setattr(routes, "_resolve_user", lambda user: user or USER)
    monkeypatch.setattr(routes, "view_timeout_minutes", lambda config: 30)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_rest_open_read_close_flow(client, pg):
    row = _send_png(pg)
    listed = client.get("/v1/playground").json()
    assert [i["id"] for i in listed["items"]] == [row["id"]] and listed["senders"] == ["vex"]

    opened = client.post(f"/v1/playground/{row['id']}/open").json()
    content = client.get(opened["content_path"])
    assert content.status_code == 200 and content.content == PNG
    assert content.headers["content-type"] == "image/png"
    assert content.headers["content-security-policy"] == "sandbox"
    assert content.headers["cache-control"] == "no-store"
    ranged = client.get(opened["content_path"], headers={"Range": "bytes=0-3"})
    assert ranged.status_code == 206 and ranged.content == PNG[:4]
    assert client.get(f"/v1/playground/{row['id']}/content?token=nope").status_code == 403

    closed = client.post(f"/v1/playground/{row['id']}/close").json()
    assert closed["status"] == "expired"
    assert client.get(opened["content_path"]).status_code == 410
    assert client.post(f"/v1/playground/{row['id']}/open").status_code == 410
    assert client.get(f"/v1/playground/{row['id']}").json()["opened_at"]


def test_rest_html_is_served_as_text_and_unknown_is_404(client, pg):
    row = pg.send(user_id=USER, bot_id="vex", title="Page", note=None,
                  source=source_from_inline("<script>x()</script>", "html"))
    opened = client.post(f"/v1/playground/{row['id']}/open").json()
    content = client.get(opened["content_path"])
    assert content.headers["content-type"].startswith("text/plain")
    assert content.headers["x-playground-kind"] == "html"
    assert client.post("/v1/playground/pg_nope/open").status_code == 404
    assert client.post(f"/v1/playground/{row['id']}/open?user=other").status_code == 404


# -- MCP tool -----------------------------------------------------------------


def _run(**kwargs):
    return asyncio.run(playground_tools.playground_send(**kwargs))


@pytest.fixture
def tool(pg, monkeypatch):
    bells = []

    async def bell(row):
        bells.append(row["id"])

    monkeypatch.setattr(pg_service, "_instance", pg)
    monkeypatch.setattr(playground_tools, "_bell", bell)
    return bells


def _caller(monkeypatch, caller=None, error=False):
    from llm_bawt.mcp_server import bot_access_identity

    def trusted(name, arguments):
        if error:
            raise BotAccessError("caller_unavailable")
        return caller

    monkeypatch.setattr(bot_access_identity, "trusted_caller", trusted)


def test_tool_sends_inline_for_the_signed_caller(tool, pg, monkeypatch):
    _caller(monkeypatch, SimpleNamespace(bot_id="vex", user_id=USER))
    result = _run(bot_id="Vex", title="Notes", content="# Hi", format="markdown")
    assert result["status"] == "unopened" and result["kind"] == "markdown"
    assert tool == [result["id"]]
    assert pg.store.get(result["id"], USER)["sender_bot_id"] == "vex"


@pytest.mark.parametrize("caller,error,expected", [
    (None, False, "caller_unverified"),
    (None, True, "caller_unverified"),
    (SimpleNamespace(bot_id="byte", user_id=USER), False, "caller_mismatch"),
])
def test_tool_refuses_unverified_or_mismatched_callers(tool, pg, monkeypatch, caller, error,
                                                       expected):
    _caller(monkeypatch, caller, error)
    assert _run(bot_id="vex", title="x", content="y", format="text") == {"error": expected}
    assert pg.store.list(USER) == [] and tool == []


def test_tool_requires_exactly_one_source(tool, monkeypatch):
    _caller(monkeypatch, SimpleNamespace(bot_id="vex", user_id=USER))
    assert "exactly one" in _run(bot_id="vex", title="x")["error"]
    assert "exactly one" in _run(bot_id="vex", title="x", asset_id="ma_1", content="y",
                                 format="text")["error"]
    assert "format" in _run(bot_id="vex", title="x", content="y")["error"]


def test_tool_validates_title_and_note_lengths(tool, monkeypatch):
    _caller(monkeypatch, SimpleNamespace(bot_id="vex", user_id=USER))
    assert "title" in _run(bot_id="vex", title=" ", content="y", format="text")["error"]
    assert "title" in _run(bot_id="vex", title="t" * 201, content="y", format="text")["error"]
    assert "note" in _run(bot_id="vex", title="t", note="n" * 301, content="y",
                          format="text")["error"]
