"""Tests for ``/v1/history`` attachment hydration (TASK-226).

These tests build a minimal FastAPI app that mounts only the history
router and replaces the global service singleton with a hand-rolled
fake that knows just enough to exercise the attachment-hydration code
path. We keep it hermetic — no Postgres, no real ``MediaAssetStore``,
no MCP server.

Coverage:

- Empty messages list -> route still returns 200, no errors.
- Messages without attachments -> every row has ``attachments: []``.
- Messages with one attachment -> row carries the full URL block.
- Same asset referenced by two messages -> only one
  ``MediaAssetStore.get_many`` call (no per-message fan-out).
- Orphan ``asset_id`` (deleted media row) -> silently dropped, message
  still rendered.
- Mixed page (some rows have media, some don't) keeps shape consistent.
- Pagination (``before=``) hydrates per-page only — older rows that
  fall outside the window don't contribute id lookups.
- ``/v1/history/search`` returns the same shape (acceptance criterion).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service import dependencies as service_deps
from llm_bawt.service.routes import history as history_routes
from llm_bawt.service.routes import history_pages


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeBackend:
    """Stands in for the focused history-hydration backend reads."""

    def __init__(self, attachments_by_msg: dict[str, list[dict]]):
        self._attachments_by_msg = attachments_by_msg
        self._interrupt_anchors_by_msg: dict[str, tuple[str, int]] = {}
        self.call_count = 0
        self.last_ids: list[str] = []

    def get_attachments_for_message_ids(
        self, message_ids: list[str]
    ) -> dict[str, list[dict]]:
        self.call_count += 1
        self.last_ids = list(message_ids)
        return {mid: list(self._attachments_by_msg.get(mid, [])) for mid in message_ids}

    def get_reasoning_for_message_ids(
        self, message_ids: list[str]
    ) -> dict[str, str | None]:
        return {mid: None for mid in message_ids}

    def get_interrupt_anchors_for_message_ids(
        self, message_ids: list[str]
    ) -> dict[str, tuple[str, int] | None]:
        return {mid: self._interrupt_anchors_by_msg.get(mid) for mid in message_ids}


class FakeShortTermManager:
    def __init__(self, backend: FakeBackend):
        self._backend = backend


class FakeMemoryClient:
    def __init__(self, messages: list[dict], backend: FakeBackend):
        self._messages = messages
        self._stm = FakeShortTermManager(backend)

    def get_messages(self, since_seconds: int | None = None) -> list[dict]:
        return list(self._messages)

    def get_short_term_manager(self) -> FakeShortTermManager:
        return self._stm


class FakeMediaAssetStore:
    """Records ``get_many`` invocations so tests can assert call shape."""

    def __init__(self, rows_by_id: dict[str, dict[str, Any]]):
        self._rows = rows_by_id
        self.get_many_calls: list[list[str]] = []

    def get_many(self, ids: list[str]) -> list[dict[str, Any]]:
        self.get_many_calls.append(list(ids))
        return [self._rows[i] for i in ids if i in self._rows]

    # serializer fallback path uses get_by_id — we override get_many to
    # return rows, so this should not get exercised. Keep a stub anyway.
    def get_by_id(self, asset_id: str) -> dict[str, Any] | None:
        return self._rows.get(asset_id)


class FakeService:
    """Bare-minimum stand-in for ``BackgroundService``."""

    def __init__(self, client: FakeMemoryClient, *, default_bot: str = "nova"):
        self._client = client
        self._default_bot = default_bot
        # The cached media accessor is patched in fixtures below, so config
        # itself is never inspected.
        self.config = object()

    def get_memory_client(self, bot_id: str | None = None) -> FakeMemoryClient:
        return self._client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(asset_id: str) -> dict[str, Any]:
    """media_assets row mirroring what MediaAssetStore would return."""
    return {
        "id": asset_id,
        "sha256": f"sha-{asset_id}",
        "mime_type": "image/webp",
        "original_mime_type": "image/png",
        "size_bytes": 12345,
        "width": 1568,
        "height": 882,
        "source": "chat_upload",
        "owner_user_id": "u1",
    }


def _msg(mid: str, role: str = "user", content: str = "hi", ts: float = 100.0) -> dict:
    return {"id": mid, "role": role, "content": content, "timestamp": ts}


@pytest.fixture
def app_factory(monkeypatch):
    """Build a TestClient bound to the configured fake service.

    Returns a callable ``make(messages, attachments_by_msg, asset_rows)``
    that wires up the fakes and returns ``(client, fake_store)``.
    """

    def make(
        messages: list[dict],
        attachments_by_msg: dict[str, list[dict]] | None = None,
        asset_rows: dict[str, dict[str, Any]] | None = None,
    ):
        backend = FakeBackend(attachments_by_msg or {})
        client = FakeMemoryClient(messages, backend)
        svc = FakeService(client)
        service_deps.set_service(svc)

        fake_store = FakeMediaAssetStore(asset_rows or {})

        monkeypatch.setattr(
            history_pages,
            "get_media_asset_store",
            lambda *a, **kw: fake_store,
        )

        app = FastAPI()
        app.include_router(history_routes.router)
        return TestClient(app), fake_store, backend

    yield make
    service_deps.set_service(None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_history_returns_empty_messages_list(app_factory):
    client, _, _ = app_factory([])
    resp = client.get("/v1/history")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages"] == []
    assert data["total_count"] == 0


def test_message_without_attachments_still_carries_empty_list(app_factory):
    msgs = [_msg("m1", content="hello", ts=1.0)]
    client, _, _ = app_factory(msgs, attachments_by_msg={"m1": []})
    resp = client.get("/v1/history")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["attachments"] == []


def test_system_authored_user_role_keeps_history_visible_and_provenance(app_factory):
    system = _msg("m1", content="continuation notice", ts=1.0)
    system["author"] = {
        "entity_type": "system", "entity_id": "approval-continuation:mcp",
        "status": "canonical", "display_name": "System", "color": "slate",
    }
    human = _msg("m2", content="hello", ts=2.0)
    client, _, _ = app_factory([system, human])

    response = client.get("/v1/history")
    assert response.status_code == 200, response.text
    assert [row["id"] for row in response.json()["messages"]] == ["m1", "m2"]
    assert response.json()["messages"][0]["role"] == "user"
    assert response.json()["messages"][0]["author"] == system["author"] | {
        "avatar": None, "avatar_render": None,
    }

    search = client.post("/v1/history/search?query=continuation")
    assert search.status_code == 200, search.text
    assert search.json()["messages"][0]["author"]["entity_type"] == "system"


def test_history_returns_persisted_interrupt_anchor(app_factory):
    msgs = [
        _msg("source-1", content="start", ts=1.0),
        _msg("interrupt-1", content="change direction", ts=2.0),
    ]
    client, _, backend = app_factory(msgs)
    backend._interrupt_anchors_by_msg["interrupt-1"] = ("source-1", 42)

    resp = client.get("/v1/history")

    assert resp.status_code == 200, resp.text
    interrupt = next(message for message in resp.json()["messages"] if message["id"] == "interrupt-1")
    assert interrupt["interrupt_source_message_id"] == "source-1"
    assert interrupt["interrupt_content_offset"] == 42


def test_message_with_attachment_returns_full_url_block(app_factory):
    msgs = [_msg("m1", content="see this", ts=2.0)]
    refs = {"m1": [{"asset_id": "ma_AAA", "kind": "image"}]}
    rows = {"ma_AAA": _row("ma_AAA")}
    client, store, _ = app_factory(msgs, attachments_by_msg=refs, asset_rows=rows)
    resp = client.get("/v1/history")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    atts = data["messages"][0]["attachments"]
    assert len(atts) == 1
    att = atts[0]
    assert att["asset_id"] == "ma_AAA"
    assert att["kind"] == "image"
    assert att["mime_type"] == "image/webp"
    assert att["width"] == 1568
    assert att["height"] == 882
    assert att["urls"] == {
        "thumb": "/v1/uploads/ma_AAA/thumb",
        "preview": "/v1/uploads/ma_AAA/preview",
        "original": "/v1/uploads/ma_AAA",
    }
    # The serializer batched the single id into one call.
    assert store.get_many_calls == [["ma_AAA"]]


def test_orphan_asset_id_is_silently_dropped(app_factory):
    msgs = [_msg("m1", content="orphan", ts=3.0)]
    refs = {"m1": [{"asset_id": "ma_GONE", "kind": "image"}]}
    # No rows -> serializer drops the ref, msg still renders.
    client, _, _ = app_factory(msgs, attachments_by_msg=refs, asset_rows={})
    resp = client.get("/v1/history")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages"][0]["attachments"] == []


def test_same_asset_across_messages_single_query(app_factory):
    """One get_many call for the page even if N messages reference the same id."""
    msgs = [
        _msg("m1", content="first", ts=1.0),
        _msg("m2", content="second", ts=2.0),
        _msg("m3", content="third", ts=3.0),
    ]
    refs = {
        "m1": [{"asset_id": "ma_X", "kind": "image"}],
        "m2": [{"asset_id": "ma_X", "kind": "image"}],
        "m3": [{"asset_id": "ma_X", "kind": "image"}],
    }
    rows = {"ma_X": _row("ma_X")}
    client, store, backend = app_factory(msgs, attachments_by_msg=refs, asset_rows=rows)
    resp = client.get("/v1/history")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # All three rows resolved.
    for m in data["messages"]:
        assert len(m["attachments"]) == 1
        assert m["attachments"][0]["asset_id"] == "ma_X"
    # Backend hit once for the page, asset store hit once with deduped ids.
    assert backend.call_count == 1
    assert store.get_many_calls == [["ma_X"]]


def test_mixed_page_some_messages_have_no_media(app_factory):
    msgs = [
        _msg("m1", content="text only", ts=1.0),
        _msg("m2", content="with image", ts=2.0),
        _msg("m3", content="text only", ts=3.0),
    ]
    refs = {
        "m1": [],
        "m2": [{"asset_id": "ma_Y", "kind": "image"}],
        "m3": [],
    }
    rows = {"ma_Y": _row("ma_Y")}
    client, _, _ = app_factory(msgs, attachments_by_msg=refs, asset_rows=rows)
    resp = client.get("/v1/history")
    assert resp.status_code == 200, resp.text
    out = {m["id"]: m["attachments"] for m in resp.json()["messages"]}
    assert out["m1"] == []
    assert out["m3"] == []
    assert len(out["m2"]) == 1
    assert out["m2"][0]["asset_id"] == "ma_Y"


def test_pagination_hydrates_only_the_returned_page(app_factory):
    # Six messages, oldest at ts=1 -> ts=6. Page with limit=2 returns last two
    # in chronological order (m5, m6). Backend should be asked only for those.
    msgs = [_msg(f"m{i}", ts=float(i)) for i in range(1, 7)]
    refs = {f"m{i}": [{"asset_id": f"ma_{i}", "kind": "image"}] for i in range(1, 7)}
    rows = {f"ma_{i}": _row(f"ma_{i}") for i in range(1, 7)}
    client, store, backend = app_factory(msgs, attachments_by_msg=refs, asset_rows=rows)

    resp = client.get("/v1/history?limit=2")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [m["id"] for m in data["messages"]] == ["m5", "m6"]
    # Only the page's ids made it into the backend lookup.
    assert sorted(backend.last_ids) == ["m5", "m6"]
    # And exactly those two assets were resolved against the asset store.
    assert sorted(store.get_many_calls[0]) == ["ma_5", "ma_6"]
    assert data["has_more"] is True

    # Now request the page before m5 — should return [m3, m4] and re-hit
    # with only that page's ids.
    resp2 = client.get("/v1/history?limit=2&before=m5")
    assert resp2.status_code == 200, resp2.text
    page2 = resp2.json()
    assert [m["id"] for m in page2["messages"]] == ["m3", "m4"]
    assert sorted(backend.last_ids) == ["m3", "m4"]


def test_search_route_carries_same_attachment_shape(app_factory):
    msgs = [
        _msg("m1", content="hello world", ts=1.0),
        _msg("m2", content="see this image", ts=2.0),
    ]
    refs = {
        "m1": [],
        "m2": [{"asset_id": "ma_S", "kind": "image"}],
    }
    rows = {"ma_S": _row("ma_S")}
    client, _, _ = app_factory(msgs, attachments_by_msg=refs, asset_rows=rows)

    resp = client.post("/v1/history/search?query=image")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["id"] == "m2"
    atts = data["messages"][0]["attachments"]
    assert len(atts) == 1
    assert atts[0]["asset_id"] == "ma_S"
    assert atts[0]["urls"]["original"] == "/v1/uploads/ma_S"


def test_attachment_lookup_failure_drops_to_empty_safely(app_factory, monkeypatch):
    """Even if the backend call blows up, the route returns 200 with the
    messages list (attachments coerced to [])."""
    msgs = [_msg("m1", content="hi", ts=1.0)]
    client, _, backend = app_factory(msgs, attachments_by_msg={"m1": []})

    def boom(_ids):
        raise RuntimeError("DB down")

    monkeypatch.setattr(backend, "get_attachments_for_message_ids", boom)

    resp = client.get("/v1/history")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages"][0]["attachments"] == []
