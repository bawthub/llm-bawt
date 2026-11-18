"""TASK-1115: saved reply audio — lifecycle, reaping and the Range route."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from llm_bawt.media import response_audio as ra
from llm_bawt.service.routes import response_audio as route
from llm_bawt.service.routes import service_auth
from llm_bawt.service.routes.service_auth import TokenCheck
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration

OWNER = "user-a"
OTHER = "user-b"


@pytest.fixture()
def engine(private_pg, private_module_db):
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS response_audio, messages, forgotten_messages"))
        conn.execute(text("CREATE TABLE messages (bot_id text, id text, PRIMARY KEY (bot_id, id))"))
        conn.execute(text("CREATE TABLE forgotten_messages (bot_id text, id text, PRIMARY KEY (bot_id, id))"))
    ra.ResponseAudioStore._schema_guard = type(ra.ResponseAudioStore._schema_guard)()
    yield engine
    engine.dispose()


@pytest.fixture()
def store(engine):
    return ra.ResponseAudioStore(engine)


def _message(engine, mid: str, table: str = "messages") -> None:
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO {table} VALUES ('bot', :m)"), {"m": mid})


def _age(engine, hours: int) -> None:
    with engine.begin() as conn:
        conn.execute(text("UPDATE response_audio SET created_at = created_at - make_interval(hours => :h), "
                          "updated_at = updated_at - make_interval(hours => :h)"), {"h": hours})


def _reserve(store, turn="t1", msg=None, owner=OWNER):
    return store.reserve(turn_id=turn, bot_id="bot", owner_user_id=owner, session_id="s", assistant_message_id=msg)


def test_reserve_is_idempotent_and_lease_exclusive(store):
    first = _reserve(store)
    assert first.lease_token and first.row["status"] == "recording"
    again = _reserve(store)
    assert again.lease_token is None, "a live lease is not handed out twice"
    assert again.row["id"] == first.row["id"]
    with pytest.raises(ra.ResponseAudioConflict):
        _reserve(store, owner=OTHER)


def test_expired_lease_is_reissued(store, engine):
    first = _reserve(store)
    with engine.begin() as conn:
        conn.execute(text("UPDATE response_audio SET lease_expires_at = NOW() - interval '1 second'"))
    second = _reserve(store)
    assert second.lease_token and second.lease_token != first.lease_token
    with pytest.raises(ra.ResponseAudioConflict):
        store.finalize(turn_id="t1", lease_token=first.lease_token, status="finalized", media_asset_id="ma_X")


def test_bind_then_finalize_idempotent(store, engine):
    _message(engine, "m1")
    res = _reserve(store)
    store.bind_message(turn_id="t1", owner_user_id=OWNER, assistant_message_id="m1")
    store.bind_message(turn_id="t1", owner_user_id=OWNER, assistant_message_id="m1")
    with pytest.raises(ra.ResponseAudioConflict):
        store.bind_message(turn_id="t1", owner_user_id=OWNER, assistant_message_id="m2")
    row = store.finalize(turn_id="t1", lease_token=res.lease_token, status="finalized",
                         media_asset_id="ma_A", duration_ms=1200, size_bytes=99)
    assert row["status"] == "finalized" and row["assistant_message_id"] == "m1"
    repeat = store.finalize(turn_id="t1", lease_token=res.lease_token, status="finalized", media_asset_id="ma_A")
    assert repeat["id"] == row["id"]
    with pytest.raises(ra.ResponseAudioConflict):
        store.finalize(turn_id="t1", lease_token=res.lease_token, status="partial", media_asset_id="ma_B")
    assert store.get_for_message("m1", OWNER)["media_asset_id"] == "ma_A"
    assert store.get_for_message("m1", OTHER) is None, "owner scoped"


def test_reap_follows_message_lifetime(store, engine):
    for turn, msg in (("live", "m-live"), ("forgot", "m-forgot"), ("gone", "m-gone"), ("pending", None)):
        _reserve(store, turn=turn, msg=msg)
    _message(engine, "m-live")
    _message(engine, "m-forgot", "forgotten_messages")
    with engine.begin() as conn:
        assert ra.reap_orphans(conn) == 0, "fresh bindings may still be committing"
    _age(engine, ra.PENDING_COMMIT_HOURS + 1)
    with engine.begin() as conn:
        assert ra.reap_orphans(conn, dry_run=True) == 1
        assert ra.reap_orphans(conn) == 1
        left = {r[0] for r in conn.execute(text("SELECT turn_id FROM response_audio"))}
    assert left == {"live", "forgot", "pending"}, "recent unbound reservation and forgotten replies keep audio"
    _age(engine, 48)
    with engine.begin() as conn:
        assert ra.reap_orphans(conn) == 1


def test_finalize_after_reap_returns_none(store, engine):
    res = _reserve(store, msg="m-x")
    _age(engine, ra.PENDING_COMMIT_HOURS + 1)
    with engine.begin() as conn:
        ra.reap_orphans(conn)
    assert store.finalize(turn_id="t1", lease_token=res.lease_token, status="finalized", media_asset_id="ma_A") is None


def test_deleted_reply_audio_is_hidden_without_gc(store, engine):
    res = _reserve(store, msg="m-del")
    _message(engine, "m-del")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM messages WHERE id = 'm-del'"))
    store.finalize(turn_id="t1", lease_token=res.lease_token, status="finalized", media_asset_id="ma_A")
    assert store.get_for_message("m-del", OWNER) is None
    assert store.get_many_for_messages(["m-del"], OWNER) == {}


def test_binding_before_commit_survives_gc(store, engine):
    res = _reserve(store, msg="m-late")
    with engine.begin() as conn:
        assert ra.reap_orphans(conn) == 0
    assert store.get_for_message("m-late", OWNER) is None, "not visible until committed"
    _message(engine, "m-late")
    row = store.finalize(turn_id="t1", lease_token=res.lease_token, status="finalized", media_asset_id="ma_A")
    assert row is not None and store.get_for_message("m-late", OWNER)["media_asset_id"] == "ma_A"


def test_concurrent_contradictory_binds_one_conflicts(store):
    import threading

    _reserve(store)
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def bind(mid: str) -> None:
        barrier.wait()
        try:
            store.bind_message(turn_id="t1", owner_user_id=OWNER, assistant_message_id=mid)
            results[mid] = "ok"
        except ra.ResponseAudioConflict:
            results[mid] = "conflict"

    threads = [threading.Thread(target=bind, args=(m,)) for m in ("ma", "mb")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results.values()) == ["conflict", "ok"]


def test_parse_range():
    assert route.parse_range(None, 100) is None
    assert route.parse_range("bytes=0-9", 100) == (0, 9)
    assert route.parse_range("bytes=90-", 100) == (90, 99)
    assert route.parse_range("bytes=-10", 100) == (90, 99)
    assert route.parse_range("bytes=50-500", 100) == (50, 99)
    assert route.parse_range("bytes=100-", 100) == "416"
    assert route.parse_range("bytes=-0", 100) == "416"
    assert route.parse_range("bytes=0-1,5-6", 100) is None


def _wav(n_samples: int = 50) -> bytes:
    data = b"\x01\x00" * n_samples
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16) + b"data" + struct.pack("<I", len(data))
    return header + data


class FakeMedia:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}

    def upload(self, raw, mime, source, owner, **_):
        assert source == "tts_response" and mime == "audio/wav"
        # Content-addressed like the real store: the same bytes, the same asset.
        aid = next((k for k, v in self.blobs.items() if v == raw), None)
        if aid is None:
            aid = "ma_" + str(len(self.blobs)).rjust(26, "0")
            self.blobs[aid] = raw
        return SimpleNamespace(id=aid)

    def stat(self, aid):
        raw = self.blobs.get(aid)
        return None if raw is None else SimpleNamespace(sha256="sha-" + aid, size_bytes=len(raw))

    def read_variant(self, aid, variant):
        return self.blobs[aid], "audio/wav"


@pytest.fixture()
def client(store, monkeypatch):
    media = FakeMedia()
    events: list[dict] = []
    monkeypatch.setattr(route, "get_response_audio_store", lambda: store)
    monkeypatch.setattr(route, "get_media_store", lambda: media)
    monkeypatch.setattr(route, "_publish", lambda row: events.append(ra.public_view(row) | {"turn": row["turn_id"]}))
    monkeypatch.setattr(service_auth, "service_token_source",
                        lambda: SimpleNamespace(check=lambda h: TokenCheck.OK if h == "Bearer ok" else TokenCheck.INVALID))
    app = FastAPI()
    app.include_router(route.router)
    c = TestClient(app)
    c.events = events
    return c


SVC = {"Authorization": "Bearer ok"}


def test_producer_routes_require_service_token(client):
    assert client.post("/v1/media/response-audio/reservations", content=b"{bad").status_code == 401
    assert client.put("/v1/media/response-audio/reservations/t/audio?lease_token=x&duration_ms=1&owner_user_id=u",
                      content=_wav()).status_code == 401


def test_full_flow_with_ranges(client, engine):
    _message(engine, "m1")
    body = {"turn_id": "t1", "bot_id": "bot", "owner_user_id": OWNER, "assistant_message_id": "m1"}
    res = client.post("/v1/media/response-audio/reservations", json=body, headers=SVC).json()
    lease = res["lease_token"]
    wav = _wav()
    url = f"/v1/media/response-audio/reservations/t1/audio?lease_token={lease}&duration_ms=2&owner_user_id={OWNER}"
    up = client.put(url, content=wav, headers=SVC)
    assert up.status_code == 200 and up.json()["reservation"]["available"]
    assert client.put(url, content=wav, headers=SVC).status_code == 200, "repeat upload is idempotent"
    assert [e["status"] for e in client.events] == ["recording", "finalized", "finalized"]

    me = {}
    full = client.get(f"/v1/media/response-audio/m1?user_id={OWNER}", headers=me)
    assert full.status_code == 200 and full.content == wav
    assert full.headers["accept-ranges"] == "bytes"
    etag = full.headers["etag"]

    part = client.get(f"/v1/media/response-audio/m1?user_id={OWNER}", headers={**me, "Range": "bytes=4-11"})
    assert part.status_code == 206 and part.content == wav[4:12]
    assert part.headers["content-range"] == f"bytes 4-11/{len(wav)}"

    head = client.head(f"/v1/media/response-audio/m1?user_id={OWNER}", headers={**me, "Range": "bytes=0-"})
    assert head.status_code == 206 and head.headers["content-length"] == str(len(wav)) and head.content == b""

    bad = client.get(f"/v1/media/response-audio/m1?user_id={OWNER}", headers={**me, "Range": f"bytes={len(wav)}-"})
    assert bad.status_code == 416 and bad.headers["content-range"] == f"bytes */{len(wav)}"

    stale = client.get(f"/v1/media/response-audio/m1?user_id={OWNER}", headers={**me, "Range": "bytes=0-3", "If-Range": '"old"'})
    assert stale.status_code == 200 and stale.content == wav
    fresh = client.get(f"/v1/media/response-audio/m1?user_id={OWNER}", headers={**me, "Range": "bytes=0-3", "If-Range": etag})
    assert fresh.status_code == 206
    assert client.get(f"/v1/media/response-audio/m1?user_id={OWNER}", headers={**me, "If-None-Match": etag}).status_code == 304

    assert client.get(f"/v1/media/response-audio/m1?user_id={OTHER}").status_code == 404
    assert client.get("/v1/media/response-audio/m1").status_code == 422
    assert client.get("/v1/media/response-audio/m1?user_id=%20").status_code == 401

    look = client.get(f"/v1/media/response-audio?message_ids=m1,nope&user_id={OWNER}", headers=me).json()["items"]
    assert set(look) == {"m1"} and look["m1"]["duration_ms"] == 2


def test_upload_rejects_non_wav_and_failure_path(client):
    body = {"turn_id": "t2", "bot_id": "bot", "owner_user_id": OWNER}
    lease = client.post("/v1/media/response-audio/reservations", json=body, headers=SVC).json()["lease_token"]
    url = f"/v1/media/response-audio/reservations/t2/audio?lease_token={lease}&duration_ms=1&owner_user_id={OWNER}"
    assert client.put(url, content=b"not audio", headers=SVC).status_code == 415
    fail = client.post("/v1/media/response-audio/reservations/t2/fail", json={"lease_token": lease}, headers=SVC)
    assert fail.status_code == 200 and fail.json()["reservation"]["status"] == "failed"
    assert not fail.json()["reservation"]["available"]


def test_history_page_hydrates_assistant_rows_only(store, engine, monkeypatch):
    _message(engine, "m-a")
    from llm_bawt.service.routes import history_pages

    res = _reserve(store, msg="m-a")
    store.finalize(turn_id="t1", lease_token=res.lease_token, status="partial", media_asset_id="ma_A", duration_ms=5)
    monkeypatch.setattr(ra, "get_response_audio_store", lambda: store)
    page = [{"id": "m-u", "role": "user"}, {"id": "m-a", "role": "assistant"}]
    out = history_pages._hydrate_response_audio_for_page(page, OWNER)
    assert set(out) == {"m-a"} and out["m-a"]["status"] == "partial" and out["m-a"]["available"]
    assert history_pages._hydrate_response_audio_for_page(page, OTHER) == {}
    assert history_pages._hydrate_response_audio_for_page(page, None) == {}


def test_reader_owner_is_a_declared_query_field():
    """BawtHub's proxy binds only declared user carriers (query/body/path).

    A header-only owner was never filled, so every browser read got 401.
    """
    from llm_bawt.service.routes import response_audio as route

    app = FastAPI()
    app.include_router(route.router)
    paths = app.openapi()["paths"]
    for path, method in (("/v1/media/response-audio", "get"),
                         ("/v1/media/response-audio/{assistant_message_id}", "get"),
                         ("/v1/media/response-audio/{assistant_message_id}", "head")):
        params = {(p["name"], p["in"]) for p in paths[path][method].get("parameters", [])}
        assert ("user_id", "query") in params, (path, method, params)
