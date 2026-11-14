"""Named replacement uses existing upload/read/delete paths without CAS damage."""
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.media.assets import MediaAssetStore, new_asset_id
from llm_bawt.media.object_store import BlobBackendUnavailable, FallbackReadBlobBackend, S3BlobBackend
from llm_bawt.media.store import MediaStore
from llm_bawt.service.routes import uploads
from tests.test_media_store import FakeMediaAssetStore


class NamedFakeDB(FakeMediaAssetStore):
    def replace_named(self, *, storage_key, write_blob, **metadata):
        existing = next((r for r in self.rows.values() if r.get("storage_key") == storage_key), None)
        row = dict(existing or {"id": new_asset_id(), "created_at": datetime.now(timezone.utc), "kind": "file"})
        row.update(metadata, storage_key=storage_key)
        write_blob()
        self.rows[row["id"]] = row
        return dict(row)


@pytest.fixture
def store(tmp_path):
    return MediaStore(root=tmp_path, db=NamedFakeDB())


def upload(store, raw, **kw):
    return store.upload(raw, "audio/wav", "tool_generated", "nick", **kw)


def test_repeated_upload_reuses_one_object_and_row_without_changing_saved_asset(store):
    saved = upload(store, b"one")
    first = upload(store, b"one", replace_key="announcements/current.wav")
    assert saved.id != first.id
    for content in (b"two", b"three", b"four"):
        latest = upload(store, content, replace_key="announcements/current.wav")
        assert latest.id == first.id
        assert store.read_variant(first.id, "original")[0] == content
        assert store.read_variant(saved.id, "original")[0] == b"one"
        assert len(store.db.rows) == 2
        assert len([p for p in store.root.rglob("*") if p.is_file()]) == 2
    # Immutable dedup never resolves to the mutable slot.
    assert upload(store, b"one").id == saved.id
    separate = upload(store, b"four")
    assert separate.id != latest.id
    store.delete(latest.id)
    assert store.stat(latest.id) is None
    assert store.read_variant(separate.id, "original")[0] == b"four"


@pytest.mark.parametrize("key", ["../files/x", "/absolute", "foo/../bar", "foo//bar", "foo/.", ""])
def test_invalid_names_fail_before_writing(store, key):
    with pytest.raises(ValueError):
        upload(store, b"one", replace_key=key)
    assert not store.db.rows


def test_replacement_failure_does_not_update_row(store, monkeypatch):
    first = upload(store, b"one", replace_key="announcements/current.wav")
    monkeypatch.setattr(store.backend, "replace", Mock(side_effect=BlobBackendUnavailable("offline")))
    with pytest.raises(BlobBackendUnavailable):
        upload(store, b"two", replace_key="announcements/current.wav")
    assert store.stat(first.id).sha256 == first.sha256
    assert store.read_variant(first.id, "original")[0] == b"one"


def test_partial_publish_fails_closed_then_heals(store):
    first = upload(store, b"one", replace_key="announcements/current.wav")
    store.backend.replace(first.storage_key, b"uncommitted", "audio/wav")
    with pytest.raises(BlobBackendUnavailable, match="consistent"):
        store.read_variant(first.id, "original")
    upload(store, b"healed", replace_key="announcements/current.wav")
    assert store.read_variant(first.id, "original")[0] == b"healed"


def test_mutable_route_never_uses_immutable_cache_or_304(store, monkeypatch):
    first = upload(store, b"one", replace_key="announcements/current.wav")
    monkeypatch.setattr(uploads, "_store", lambda: store)
    app = FastAPI()
    app.include_router(uploads.router)
    with TestClient(app) as client:
        url = f"/v1/uploads/{first.id}"
        response = client.get(url + "?v=job1", headers={"If-None-Match": f'"{first.sha256}"'})
        assert response.status_code == 200
        assert response.content == b"one"
        assert response.headers["cache-control"] == "no-store"
        assert "etag" not in response.headers
        upload(store, b"two", replace_key="announcements/current.wav")
        assert client.get(url + "?v=job2").content == b"two"
        assert client.get(url + "/thumb").status_code == 404


def test_backend_replacement_contracts():
    s3 = object.__new__(S3BlobBackend)
    s3.put = Mock()
    s3.replace("named/a", b"new", "audio/wav")
    s3.put.assert_called_once_with("named/a", b"new", "audio/wav")
    fallback = Mock()
    wrapper = FallbackReadBlobBackend(s3, fallback)
    wrapper.replace("named/a", b"newer", "audio/wav")
    s3.put.assert_called_with("named/a", b"newer", "audio/wav")
    fallback.replace.assert_not_called()


def test_named_upsert_is_locked_and_reuses_record():
    db = object.__new__(MediaAssetStore)
    conn = Mock()
    transaction = Mock()
    transaction.__enter__ = Mock(return_value=conn)
    transaction.__exit__ = Mock(return_value=False)
    db.engine = Mock()
    db.engine.begin.return_value = transaction
    conn.execute.return_value.mappings.return_value.one.return_value = {"id": "stable"}
    write = Mock()
    assert db.replace_named(storage_key="named/a", write_blob=write, sha256="x")["id"] == "stable"
    queries = [str(call.args[0]) for call in conn.execute.call_args_list]
    assert "pg_advisory_xact_lock" in queries[0]
    assert "ON CONFLICT (storage_key) DO UPDATE" in queries[1]
    assert " id = EXCLUDED.id" not in queries[1]
    write.assert_called_once()


def test_gc_excludes_named_slots():
    from llm_bawt.service.jobs.media_gc import _find_orphan_assets
    conn = Mock()
    conn.execute.return_value.mappings.return_value.all.return_value = []
    _find_orphan_assets(conn, frozenset(), 7)
    sql = " ".join(str(conn.execute.call_args.args[0]).split())
    # Named slots are excluded unconditionally, ahead of every other filter.
    assert "WHERE a.storage_key IS NULL AND " in sql
