"""TASK-2149: the media store recovers from a transient DB failure, and the
health checks report an unreadable media directory instead of hiding it."""
from __future__ import annotations

import errno
import os
import threading

import pytest

from llm_bawt.media import store as media_store
from llm_bawt.service import health_checks
from llm_bawt.service.health_checks import DOWN, OK, check_media_root


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    monkeypatch.setenv("LLM_BAWT_MEDIA_ROOT", str(tmp_path))
    clock = _Clock()
    monkeypatch.setattr(media_store.time, "monotonic", clock)
    media_store.reset_media_store()
    yield clock
    media_store.reset_media_store()


def test_transient_db_failure_is_retried_not_cached(isolated_store, monkeypatch):
    clock = isolated_store
    db = object()
    attempts = {"n": 0}

    def build():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("LockNotAvailable: schema bootstrap")
        return db

    monkeypatch.setattr(media_store, "_build_media_asset_db", build)

    first = media_store.get_media_store()
    assert first.db is None
    # Inside the backoff window: no new attempt, same DB-less store.
    clock.now += 1
    assert media_store.get_media_store() is first
    assert attempts["n"] == 1
    # After the backoff: retried, attached in place to the same singleton.
    clock.now += media_store._DB_RETRY_INITIAL_S
    second = media_store.get_media_store()
    assert second is first
    assert second.db is db
    assert attempts["n"] == 2
    # Healthy store: no further construction.
    media_store.get_media_store()
    assert attempts["n"] == 2


def test_backoff_starts_after_a_slow_failure(isolated_store, monkeypatch):
    clock = isolated_store
    attempts = {"n": 0}

    def slow_failure():
        attempts["n"] += 1
        clock.now += 6  # failing build takes longer than the first delay
        raise RuntimeError("lock timeout")

    monkeypatch.setattr(media_store, "_build_media_asset_db", slow_failure)
    media_store.get_media_store()
    media_store.get_media_store()
    assert attempts["n"] == 1
    clock.now += media_store._DB_RETRY_INITIAL_S
    media_store.get_media_store()
    assert attempts["n"] == 2


def test_retry_backoff_doubles_and_caps(isolated_store, monkeypatch):
    clock = isolated_store
    attempts = []

    def build():
        attempts.append(clock.now)
        raise RuntimeError("db down")

    monkeypatch.setattr(media_store, "_build_media_asset_db", build)
    for _ in range(12):
        media_store.get_media_store()
        clock.now += media_store._DB_RETRY_MAX_S
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    assert all(gap <= media_store._DB_RETRY_MAX_S for gap in gaps)
    assert media_store._db_retry_delay == media_store._DB_RETRY_MAX_S


def test_concurrent_callers_build_db_once(isolated_store, monkeypatch):
    calls = {"n": 0}
    gate = threading.Event()

    def build():
        calls["n"] += 1
        gate.wait(1)
        return object()

    monkeypatch.setattr(media_store, "_build_media_asset_db", build)
    stores = []
    threads = [threading.Thread(target=lambda: stores.append(media_store.get_media_store()))
               for _ in range(4)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(2)
    assert calls["n"] == 1
    assert len({id(s) for s in stores}) == 1


# --- media.storage health ------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_media_probe(monkeypatch):
    monkeypatch.setattr(health_checks, "_media_probe", health_checks._MediaProbe())

def test_media_root_readable_is_ok(tmp_path):
    assert check_media_root(tmp_path) == {"name": "media.storage", "status": OK, "reason": None}


def test_media_root_not_created_yet_is_ok(tmp_path):
    row = check_media_root(tmp_path / "media" / "blobs")
    assert row["status"] == OK
    assert "not created yet" in row["reason"]


def test_stale_mount_reports_down(tmp_path, monkeypatch):
    real_listdir = os.listdir

    def stale(path):
        if str(path) == str(tmp_path):
            raise OSError(errno.ESTALE, os.strerror(errno.ESTALE))
        return real_listdir(path)

    monkeypatch.setattr(health_checks.os, "listdir", stale)
    row = check_media_root(tmp_path)
    assert row["status"] == DOWN
    assert "Stale file handle" in row["reason"]


def test_hung_mount_reports_down_without_piling_up_workers(tmp_path, monkeypatch):
    release = threading.Event()
    calls = {"n": 0}

    def hang(_path):
        calls["n"] += 1
        release.wait(5)
        return []

    monkeypatch.setattr(health_checks.os, "listdir", hang)

    def probe_threads():
        return [t for t in threading.enumerate() if t.name == "media-root-probe" and t.is_alive()]

    try:
        rows = [check_media_root(tmp_path, timeout=0.02) for _ in range(12)]
        assert all(row["status"] == DOWN for row in rows)
        assert "has not responded" in rows[-1]["reason"]
        assert len(probe_threads()) == 1
        assert calls["n"] == 1
    finally:
        release.set()
        for t in probe_threads():
            t.join(2)
    # Once the mount answers again the next poll reports ok.
    monkeypatch.setattr(health_checks.os, "listdir", lambda _p: [])
    assert check_media_root(tmp_path)["status"] == OK


def test_object_storage_without_fs_fallback_skips_probe(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "s3")
    monkeypatch.delenv("LLM_BAWT_S3_FALLBACK_FS", raising=False)
    assert check_media_root()["status"] == OK


def test_fs_backend_probes_configured_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    monkeypatch.setenv("LLM_BAWT_MEDIA_ROOT", str(tmp_path))
    assert check_media_root() == {"name": "media.storage", "status": OK, "reason": None}
