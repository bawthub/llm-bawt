"""TASK-1476: object-store (S3) credentials come from the encrypted store.

The ``integration_connection:object-store`` record wins. The deprecated
``LLM_BAWT_S3_*`` env vars are read only when no record exists; a record that
exists but can't be used never falls back to them. Backends resolve per
operation (behind a short TTL) and rebuild their client on rotation, so a
saved or rotated key needs no restart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.integrations.credential_integrations import CredentialIntegrations
from llm_bawt.media import object_store_credentials as creds
from llm_bawt.media.object_store import BlobBackendUnavailable, S3BlobBackend, S3Config, get_blob_backend
from llm_bawt.media.object_store_credentials import ObjectStoreUnavailable, StoredS3Config, resolve_s3_config
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service import changed_files_store, connection_secrets, tool_call_store
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import STATUS_DISCONNECTED

STORED_SECRET = "stored-s3-secret-do-not-echo"
ROTATED_SECRET = "rotated-s3-secret-do-not-echo"
ENV_SECRET = "env-s3-secret-do-not-echo"
ENV = {
    "LLM_BAWT_S3_ENDPOINT": "http://s3-env.example",
    "LLM_BAWT_S3_REGION": "env-region",
    "LLM_BAWT_S3_BUCKET": "env-bucket",
    "LLM_BAWT_S3_ACCESS_KEY": "env-access",
    "LLM_BAWT_S3_SECRET_KEY": ENV_SECRET,
}
STORED = S3Config("https://s3.example", "us-east-1", "stored-bucket", "stored-access", STORED_SECRET)


@pytest.fixture
def env(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def no_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def store(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    with engine.begin() as conn:
        conn.execute(text("CREATE UNIQUE INDEX idx_runtime_settings_scope_key "
                          "ON runtime_settings (scope_type, scope_id, key)"))
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.config = None
    runtime.engine = engine
    made = object.__new__(IntegrationStore)
    made._store = runtime
    monkeypatch.setattr(connection_secrets, "_store", lambda config, kind: made)
    monkeypatch.setattr(creds, "_env_fallback_warned", False)
    monkeypatch.setattr(creds, "_default_source", None)
    return made


@pytest.fixture
def forms(store):
    return CredentialIntegrations(None, store=store)


def _save(forms, *, bucket="stored-bucket", secret=STORED_SECRET):
    forms.configure("object-store", {
        "endpoint": "https://s3.example", "region": "us-east-1", "bucket": bucket,
        "access_key_id": "stored-access", "secret_access_key": secret,
    })


def _store_down(monkeypatch):
    def down(config, kind):
        raise RuntimeError("db down")

    monkeypatch.setattr(connection_secrets, "_store", down)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def built(monkeypatch):
    """Replace boto3 client construction; record the config of each build."""
    configs: list[S3Config] = []

    def build(self, cfg):
        configs.append(cfg)
        client = Mock(name=f"client-{len(configs)}")
        client.get_object.return_value = {"Body": Mock(read=Mock(return_value=b"bytes"))}
        return client

    monkeypatch.setattr(S3BlobBackend, "_build_client", build)
    return configs


# --- the resolver -----------------------------------------------------------

def test_record_wins_over_env(forms, env):
    _save(forms)
    assert resolve_s3_config(None) == STORED


def test_no_record_falls_back_to_env_with_one_warning_naming_vars_only(store, env, caplog):
    with caplog.at_level(logging.WARNING, logger=creds.logger.name):
        found = resolve_s3_config(None)
        resolve_s3_config(None)
    assert found == S3Config("http://s3-env.example", "env-region", "env-bucket", "env-access", ENV_SECRET)
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert all(name in warnings[0] for name in ENV)
    assert ENV_SECRET not in caplog.text and "env-access" not in caplog.text


def test_no_record_and_no_env_is_not_configured(store, no_env):
    assert resolve_s3_config(None) is None
    with pytest.raises(ObjectStoreUnavailable) as caught:
        StoredS3Config(object()).current()
    assert caught.value.code == "not_configured"


def test_disconnect_through_the_form_never_falls_back_to_env(store, forms, env):
    _save(forms)
    assert forms.disconnect("object-store") is True
    with pytest.raises(ObjectStoreUnavailable) as caught:
        resolve_s3_config(None)
    assert caught.value.code == "not_connected"
    record = store.load("object-store")
    assert record.status == STATUS_DISCONNECTED and not record.secret and not record.meta


def test_undecryptable_record_never_falls_back_to_env(forms, env, monkeypatch):
    _save(forms)
    monkeypatch.setattr(crypto, "_get_fernet", lambda: Fernet(Fernet.generate_key()))  # rotated root
    with pytest.raises(ObjectStoreUnavailable) as caught:
        resolve_s3_config(None)
    assert caught.value.code == "unreadable"


def test_store_down_never_falls_back_to_env(env, monkeypatch):
    _store_down(monkeypatch)
    with pytest.raises(ObjectStoreUnavailable) as caught:
        resolve_s3_config(None)
    assert caught.value.code == "store_unavailable"


def test_unavailable_is_a_backend_unavailable_so_routes_answer_503():
    assert issubclass(ObjectStoreUnavailable, BlobBackendUnavailable)


def test_s3_config_repr_never_shows_keys():
    shown = repr(STORED)
    assert STORED_SECRET not in shown and "stored-access" not in shown and "stored-bucket" in shown


# --- the cached source --------------------------------------------------------

def test_source_reuses_an_answer_within_the_ttl_and_rereads_after(forms, no_env):
    _save(forms)
    clock = Clock()
    source = StoredS3Config(object(), ttl=60, clock=clock)
    assert source.current().secret_key == STORED_SECRET
    _save(forms, secret=ROTATED_SECRET)
    clock.now += 59
    assert source.current().secret_key == STORED_SECRET
    clock.now += 1
    assert source.current().secret_key == ROTATED_SECRET


def test_source_caches_a_disconnect_refusal_for_the_ttl(forms, env):
    _save(forms)
    clock = Clock()
    source = StoredS3Config(object(), ttl=60, clock=clock)
    source.current()
    forms.disconnect("object-store")
    clock.now += 60
    for _ in range(2):
        with pytest.raises(ObjectStoreUnavailable) as caught:
            source.current()
        assert caught.value.code == "not_connected"


def test_source_never_caches_store_unavailable(store, forms, no_env, monkeypatch):
    _save(forms)
    clock = Clock()
    source = StoredS3Config(object(), ttl=60, clock=clock)
    working = connection_secrets._store
    _store_down(monkeypatch)
    with pytest.raises(ObjectStoreUnavailable) as caught:
        source.current()
    assert caught.value.code == "store_unavailable"
    monkeypatch.setattr(connection_secrets, "_store", working)
    assert source.current() == STORED  # same instant: the outage wasn't cached


# --- the backend ----------------------------------------------------------------

def test_backend_with_a_source_rebuilds_its_client_only_when_the_config_changes(forms, no_env, built):
    _save(forms)
    clock = Clock()
    backend = S3BlobBackend(StoredS3Config(object(), ttl=60, clock=clock), key_prefix="blobs/")
    assert backend.get("a") == b"bytes"
    assert backend.get("b") == b"bytes"
    assert built == [STORED]
    _save(forms, bucket="new-bucket", secret=ROTATED_SECRET)
    clock.now += 60
    backend.put("c", b"x", "text/plain")
    assert [cfg.secret_key for cfg in built] == [STORED_SECRET, ROTATED_SECRET]
    assert backend.client.put_object.call_args.kwargs["Bucket"] == "new-bucket"


def test_backend_refuses_every_operation_after_a_disconnect(forms, env, built):
    _save(forms)
    clock = Clock()
    backend = S3BlobBackend(StoredS3Config(object(), ttl=60, clock=clock))
    backend.exists("a")
    forms.disconnect("object-store")
    clock.now += 60
    for call in (lambda: backend.get("a"), lambda: backend.get_range("a", None),
                 lambda: backend.put("a", b"x", "text/plain"), lambda: backend.delete("a"),
                 lambda: backend.exists("a")):
        with pytest.raises(ObjectStoreUnavailable):
            call()
    assert len(built) == 1  # no client was ever built from the env keys


def test_a_reader_never_pairs_a_rebuilt_client_with_its_stale_config():
    """A reader holding config A, overlapping a rebuild for B, uses A's client.

    The writer pauses right after it publishes anything holding client B. The
    reader (which read config A) then runs until it finishes or waits on the
    lock, and only then is the writer released. Every call must send its
    request with the bucket its own client was built for.
    """
    import threading

    old_cfg = S3Config("https://old.example", "region", "bucket-A", "access", "old-secret")
    new_cfg = S3Config("https://new.example", "region", "bucket-B", "access", "new-secret")
    reader_has_old_cfg = threading.Event()
    release_reader = threading.Event()
    published_new = threading.Event()
    release_writer = threading.Event()
    reader_done_or_waiting = threading.Event()
    seen: list[tuple[str, str]] = []
    errors: list[BaseException] = []

    class Source:
        cfg = old_cfg

        def current(self):
            if threading.current_thread().name == "reader":
                reader_has_old_cfg.set()
                assert release_reader.wait(5)  # timing-ok: thread hang guard
                return old_cfg
            return self.cfg

    class FakeClient:
        def __init__(self, cfg):
            self.cfg = cfg

        def head_object(self, *, Bucket, Key):
            seen.append((self.cfg.bucket, Bucket))

    def holds_new_client(value):
        values = value if isinstance(value, tuple) else (value,)
        return any(isinstance(v, FakeClient) and v.cfg == new_cfg for v in values)

    class PausingBackend(S3BlobBackend):
        def __setattr__(self, key, value):
            object.__setattr__(self, key, value)
            if threading.current_thread().name == "writer" and holds_new_client(value):
                published_new.set()
                assert release_writer.wait(5)  # timing-ok: thread hang guard

    class WatchedLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "reader":
                reader_done_or_waiting.set()
            self._lock.acquire()

        def __exit__(self, *exc):
            self._lock.release()

    source = Source()
    backend = PausingBackend(source)
    backend._build_client = lambda cfg: FakeClient(cfg)
    backend.exists("initial")
    backend._client_lock = WatchedLock()

    def read():
        try:
            backend.exists("read")
        except BaseException as exc:  # noqa: BLE001 — surfaced by the assert below
            errors.append(exc)
        finally:
            reader_done_or_waiting.set()

    reader = threading.Thread(target=read, name="reader")
    writer = threading.Thread(target=lambda: backend.exists("write"), name="writer")
    reader.start()
    try:
        assert reader_has_old_cfg.wait(5)  # timing-ok: thread hang guard
        source.cfg = new_cfg
        writer.start()
        assert published_new.wait(5)  # timing-ok: thread hang guard
        release_reader.set()
        assert reader_done_or_waiting.wait(5)  # timing-ok: thread hang guard
    finally:
        release_reader.set()
        release_writer.set()
        reader.join(5)  # timing-ok: thread hang guard
        if writer.ident is not None:
            writer.join(5)  # timing-ok: thread hang guard
    assert not reader.is_alive() and not writer.is_alive()
    assert not errors
    assert len(seen) == 3 and all(client == requested for client, requested in seen), seen


def test_backend_with_a_fixed_config_builds_once(built):
    backend = S3BlobBackend(STORED)
    backend.get("a")
    backend.get("b")
    assert built == [STORED]


# --- the factory and its callers -------------------------------------------------

@pytest.fixture
def s3_selected(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "s3")
    monkeypatch.delenv("LLM_BAWT_S3_FALLBACK_FS", raising=False)


def test_factory_without_a_config_uses_the_stored_record_over_env(forms, env, s3_selected, built, tmp_path):
    _save(forms)
    backend = get_blob_backend("media_store", fs_root=tmp_path)
    assert isinstance(backend, S3BlobBackend) and backend.key_prefix == "blobs/"
    backend.get("a")
    assert built == [STORED]
    assert get_blob_backend("media_storage", fs_root=tmp_path)._source is backend._source


def test_factory_without_any_credentials_builds_lazily_and_refuses_on_use(store, no_env, s3_selected, tmp_path):
    backend = get_blob_backend("media_store", fs_root=tmp_path)
    with pytest.raises(ObjectStoreUnavailable) as caught:
        backend.get("a")
    assert caught.value.code == "not_configured"


@pytest.mark.parametrize("make", [
    pytest.param(lambda root: __import__("llm_bawt.media.store", fromlist=["MediaStore"]).MediaStore(root=root),
                 id="media_store"),
    pytest.param(lambda root: __import__("llm_bawt.media.storage", fromlist=["MediaStorage"]).MediaStorage(
        base_path=str(root)), id="media_storage"),
])
def test_media_callers_read_the_record_not_env(forms, env, s3_selected, built, tmp_path: Path, make):
    _save(forms)
    make(tmp_path).backend.exists("a")
    assert built == [STORED]


@pytest.mark.parametrize("get, reset", [
    (tool_call_store._get_tool_blob_backend, tool_call_store.reset_tool_blob_backend),
    (changed_files_store._get_diff_blob_backend, changed_files_store.reset_diff_blob_backend),
], ids=["tool_results", "turn_diffs"])
def test_blob_offload_callers_read_the_record_not_env(forms, env, s3_selected, built, get, reset):
    _save(forms)
    reset()
    try:
        get().exists("a")
    finally:
        reset()
    assert built == [STORED]
