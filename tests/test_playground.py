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


def test_open_marks_read_and_keeps_the_artifact_for_reopening(pg):
    """TASK-2195: opening never deletes; it can be reopened any number of times."""
    row = _send_png(pg)
    path = _blob_path(pg, row)
    opened, token = pg.open(row["id"], USER)
    first_read = opened["opened_at"]
    assert first_read is not None and public_view(opened)["status"] == "opened"
    assert pg.read(pg.store.viewable(row["id"], USER, token), None).data == PNG
    assert pg.store.viewable(row["id"], USER, "wrong") is None

    again, token2 = pg.open(row["id"], USER)
    assert again["opened_at"] == first_read and path.exists()
    assert pg.store.viewable(row["id"], USER, token2) is not None


def test_mailbox_actions_move_between_views(pg):
    row = _send_png(pg)
    def ids(view):
        return [r["id"] for r in pg.store.list(USER, view=view)]

    assert ids(None) == ids("inbox") == [row["id"]] and ids("archive") == ids("trash") == []

    pg.store.set_state(row["id"], USER, "read")
    assert public_view(pg.store.get(row["id"], USER))["status"] == "opened"
    pg.store.set_state(row["id"], USER, "unread")
    assert pg.store.list(USER, status="unopened")[0]["id"] == row["id"]

    view = public_view(pg.store.set_state(row["id"], USER, "archive"))
    assert view["folder"] == "archive" and ids("inbox") == [] and ids("archive") == [row["id"]]
    pg.store.set_state(row["id"], USER, "unarchive")
    assert ids("inbox") == [row["id"]]

    view = public_view(pg.store.set_state(row["id"], USER, "trash"))
    assert view["folder"] == "trash" and ids("inbox") == [] and ids("trash") == [row["id"]]
    assert _blob_path(pg, row).exists()
    pg.store.set_state(row["id"], USER, "restore")
    assert ids("inbox") == [row["id"]] and ids("all") == [row["id"]]
    assert pg.store.set_state(row["id"], "intruder", "trash") is None


def test_search_matches_title_note_and_sender_literally(pg):
    a = pg.send(user_id=USER, bot_id="nyx", title="Sunset photo", note="from the deck",
                source=source_from_inline("x", "text"))
    b = pg.send(user_id=USER, bot_id="vex", title="100% done", note=None,
                source=source_from_inline("x", "text"))
    def find(q):
        return {r["id"] for r in pg.store.list(USER, query=q)}

    assert find("SUNSET") == {a["id"]} and find("deck") == {a["id"]}
    assert find("vex") == {b["id"]} and find("100%") == {b["id"]} and find("%") == {b["id"]}
    assert find("nothing") == set()


def test_open_is_scoped_to_the_user(pg):
    row = _send_png(pg)
    with pytest.raises(OpenRefused) as refused:
        pg.open(row["id"], "intruder")
    assert refused.value.code == "not_found"
    assert pg.expire(row["id"], "intruder") is None
    assert _blob_path(pg, row).exists()


def test_delete_forever_clears_content_and_keeps_a_tombstone(pg):
    row = pg.send(user_id=USER, bot_id="vex", title="Page", note="hi",
                  source=source_from_inline("<h1>hi</h1>", "html"))
    png = _send_png(pg)
    _, token = pg.open(row["id"], USER)
    pg.expire(row["id"], USER)
    pg.expire(png["id"], USER)
    stored = pg.store.get(row["id"], USER)
    assert stored["inline_content"] is None and stored["view_token_hash"] is None
    assert pg.store.viewable(row["id"], USER, token) is None
    assert not _blob_path(pg, png).exists()
    assert public_view(stored)["folder"] == "deleted"
    with pytest.raises(OpenRefused) as refused:
        pg.open(row["id"], USER)
    assert refused.value.code == "expired"
    assert pg.store.set_state(row["id"], USER, "restore")["expired_at"] is not None


def test_sweep_purges_only_old_trash(pg):
    old, recent, read = _send_png(pg), _send_png(pg), _send_png(pg)
    pg.store.set_state(old["id"], USER, "trash")
    pg.store.set_state(recent["id"], USER, "trash")
    pg.open(read["id"], USER)
    with pg.store.engine.begin() as conn:
        conn.execute(update(artifacts).where(artifacts.c.id.in_([old["id"], read["id"]])).values(
            trashed_at=None, opened_at=datetime.now(timezone.utc) - timedelta(days=90)))
        conn.execute(update(artifacts).where(artifacts.c.id == old["id"]).values(
            trashed_at=datetime.now(timezone.utc) - timedelta(days=31)))
    assert pg.sweep(30) == {"purged": 1, "blobs_deleted": 0, "blob_delete_failures": 0}
    assert not _blob_path(pg, old).exists()
    assert _blob_path(pg, recent).exists() and _blob_path(pg, read).exists()
    assert pg.store.get(read["id"], USER)["expired_at"] is None


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
    monkeypatch.setattr(routes, "trash_days", lambda config: 30)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_rest_open_read_close_flow(client, pg):
    row = _send_png(pg)
    listed = client.get("/v1/playground").json()
    assert [i["id"] for i in listed["items"]] == [row["id"]] and listed["senders"] == ["vex"]
    assert listed["trash_days"] == 30

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
    assert closed["status"] == "opened" and closed["folder"] == "inbox"
    reopened = client.post(f"/v1/playground/{row['id']}/open").json()
    assert client.get(reopened["content_path"]).content == PNG


def test_rest_mailbox_search_and_delete_forever(client, pg):
    row = _send_png(pg, title="Sunset")
    def state(action):
        return client.post(f"/v1/playground/{row['id']}/state", json={"action": action})

    assert state("bogus").status_code == 422
    assert client.post(f"/v1/playground/{row['id']}/delete").status_code == 409
    assert state("archive").json()["folder"] == "archive"
    assert client.get("/v1/playground").json()["items"] == []
    assert [i["id"] for i in client.get("/v1/playground?view=archive").json()["items"]] == [row["id"]]
    assert [i["id"] for i in client.get("/v1/playground?view=all&q=suns").json()["items"]] == [row["id"]]
    assert client.get("/v1/playground?view=all&q=nope").json()["items"] == []
    assert state("trash").json()["folder"] == "trash"
    deleted = client.post(f"/v1/playground/{row['id']}/delete").json()
    assert deleted["status"] == "expired" and deleted["folder"] == "deleted"
    assert not _blob_path(pg, row).exists()
    assert client.post(f"/v1/playground/{row['id']}/open").status_code == 410
    assert client.get(f"/v1/playground/{row['id']}/preview").status_code == 410
    assert client.post("/v1/playground/pg_nope/state", json={"action": "read"}).status_code == 404


def test_rest_html_is_served_as_text_and_unknown_is_404(client, pg):
    row = pg.send(user_id=USER, bot_id="vex", title="Page", note=None,
                  source=source_from_inline("<script>x()</script>", "html"))
    opened = client.post(f"/v1/playground/{row['id']}/open").json()
    content = client.get(opened["content_path"])
    assert content.headers["content-type"].startswith("text/plain")
    assert content.headers["x-playground-kind"] == "html"
    assert client.post("/v1/playground/pg_nope/open").status_code == 404
    assert client.post(f"/v1/playground/{row['id']}/open?user=other").status_code == 404


def test_preview_never_marks_read_and_ends_with_delete(client, pg):
    """TASK-2195: the chat card's preview leaves the artifact unread."""
    row = _send_png(pg)
    for _ in range(2):
        preview = client.get(f"/v1/playground/{row['id']}/preview")
        assert preview.status_code == 200 and preview.content == PNG
        assert preview.headers["content-security-policy"] == "sandbox"
        assert preview.headers["cache-control"] == "no-store"
    assert client.get(f"/v1/playground/{row['id']}").json()["status"] == "unopened"
    assert client.get(f"/v1/playground/{row['id']}/preview?user=other").status_code == 404

    client.post(f"/v1/playground/{row['id']}/open")
    assert client.get(f"/v1/playground/{row['id']}/preview").status_code == 200
    pg.expire(row["id"], USER)
    assert client.get(f"/v1/playground/{row['id']}/preview").status_code == 410


def test_preview_truncates_text_and_skips_other_kinds(client, pg):
    text = pg.send(user_id=USER, bot_id="vex", title="Notes", note=None,
                   source=source_from_inline("line\n" * 1000, "markdown"))
    preview = client.get(f"/v1/playground/{text['id']}/preview")
    assert preview.headers["content-type"].startswith("text/plain")
    assert preview.headers["x-playground-kind"] == "markdown"
    assert len(preview.text) == routes.PREVIEW_TEXT_CHARS["markdown"]
    pdf = pg.send(user_id=USER, bot_id="vex", title="Doc", note=None,
                  source=Source(kind="pdf", mime="application/pdf", filename="a.pdf", data=b"%PDF"))
    assert client.get(f"/v1/playground/{pdf['id']}/preview").status_code == 415
    assert client.get(f"/v1/playground/{pdf['id']}").json()["status"] == "unopened"


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
    assert result["title"] == "Notes" and result["sender_bot_id"] == "vex"
    assert "Don't open, fetch or screenshot" in result["notice"]
    assert "Don't open, fetch or screenshot" in playground_tools.playground_send.__doc__
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
