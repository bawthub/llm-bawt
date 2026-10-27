"""Unit tests for :mod:`llm_bawt.media.object_store` (TASK-266).

Coverage targets:

- **FsBlobBackend** — full put/get/range/delete/exists roundtrip, plus
  the NFS ESTALE-tolerant code paths and the key-safety checks
  (relative paths only, no ``..`` escapes).

- **S3BlobBackend** — same surface against an in-memory moto bucket.
  We also exercise the ``key_prefix`` machinery so the factory's
  per-subsystem isolation is verified end-to-end, and the
  unreachable-backend mapping that the route handlers depend on
  (``BlobBackendUnavailable`` → HTTP 503).

- **FallbackReadBlobBackend** — the cutover safety net: S3 primary +
  FS fallback. Confirms reads fall through on 404, writes/deletes
  hit primary only.

- **Factory** — env-driven backend selection, bad kinds rejected,
  required arguments enforced.

- **Range parsing** — the ``_parse_byte_range`` helper. Honest about
  the spec corners we support ("bytes=N-M", "bytes=N-", "bytes=-N").

The S3 client never touches the network; ``mock_aws`` from moto stands
in for Garage, which lets us run the suite with zero infrastructure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_aws

from llm_bawt.media import object_store as os_mod
from llm_bawt.media.object_store import (
    BlobBackendUnavailable,
    BlobNotFound,
    BlobRange,
    FallbackReadBlobBackend,
    FsBlobBackend,
    S3BlobBackend,
    S3Config,
    _parse_byte_range,
    get_blob_backend,
    s3_config_from_env,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs(tmp_path: Path) -> FsBlobBackend:
    return FsBlobBackend(tmp_path)


@pytest.fixture
def s3_cfg() -> S3Config:
    return S3Config(
        endpoint="http://10.0.2.38:9000",
        region="garage",
        bucket="llmbawt-media",
        access_key="test-key",
        secret_key="test-secret",
    )


@pytest.fixture
def s3(s3_cfg: S3Config):
    """A moto-backed S3 bucket, with the backend pointed at it.

    Caveat: moto 5 only intercepts when the boto3 client is built
    *without* a custom ``endpoint_url`` — passing a non-AWS endpoint
    makes moto step aside (it assumes you're targeting a real custom
    service). The production code path always sets ``endpoint_url`` so
    we swap the client out after construction with a stock AWS-default
    client that moto *will* intercept.

    The other behaviors we test (key prefix, error mapping, exists
    semantics) live in :class:`S3BlobBackend`'s own methods and are
    independent of which exact ``endpoint_url`` the client was built
    with — so the swap is sound.
    """
    import boto3

    with mock_aws():
        backend = S3BlobBackend(s3_cfg, key_prefix="blobs/")
        backend._pair = (boto3.client("s3", region_name="us-east-1"), s3_cfg)
        backend.client.create_bucket(Bucket=s3_cfg.bucket)
        yield backend


# ---------------------------------------------------------------------------
# FS backend
# ---------------------------------------------------------------------------


class TestFsBlobBackend:
    def test_put_get_roundtrip(self, fs: FsBlobBackend) -> None:
        fs.put("originals/aa/bb/abc.webp", b"hello", "image/webp")
        assert fs.get("originals/aa/bb/abc.webp") == b"hello"

    def test_put_is_idempotent(self, fs: FsBlobBackend) -> None:
        key = "originals/aa/bb/abc.webp"
        fs.put(key, b"hello", "image/webp")
        fs.put(key, b"hello", "image/webp")  # second put is a no-op for FS
        assert fs.get(key) == b"hello"

    def test_get_missing_raises_blob_not_found(self, fs: FsBlobBackend) -> None:
        with pytest.raises(BlobNotFound):
            fs.get("originals/aa/bb/nope.webp")

    def test_delete_existing(self, fs: FsBlobBackend) -> None:
        fs.put("a/b/c.bin", b"x", "application/octet-stream")
        assert fs.exists("a/b/c.bin")
        fs.delete("a/b/c.bin")
        assert not fs.exists("a/b/c.bin")

    def test_delete_idempotent_on_missing(self, fs: FsBlobBackend) -> None:
        fs.delete("never/was/here.bin")  # must not raise

    def test_delete_cleans_empty_shard_dirs(
        self, fs: FsBlobBackend, tmp_path: Path
    ) -> None:
        fs.put("shard/aa/bb/x.bin", b"x", "application/octet-stream")
        assert (tmp_path / "shard" / "aa" / "bb").is_dir()
        fs.delete("shard/aa/bb/x.bin")
        assert not (tmp_path / "shard" / "aa" / "bb").exists()
        assert not (tmp_path / "shard" / "aa").exists()
        # The root and the first-level subdir below it shouldn't be removed
        # — they survive multiple unrelated assets coexisting.
        assert tmp_path.is_dir()

    def test_get_range_full_body(self, fs: FsBlobBackend) -> None:
        fs.put("k.bin", b"hello world", "application/octet-stream")
        rng = fs.get_range("k.bin", None)
        assert isinstance(rng, BlobRange)
        assert rng.data == b"hello world"
        assert rng.start == 0
        assert rng.end == 10
        assert rng.total_size == 11
        assert rng.partial is False

    def test_get_range_bytes_N_M(self, fs: FsBlobBackend) -> None:
        fs.put("k.bin", b"hello world", "application/octet-stream")
        rng = fs.get_range("k.bin", "bytes=0-4")
        assert rng.data == b"hello"
        assert rng.partial is True
        assert (rng.start, rng.end, rng.total_size) == (0, 4, 11)

    def test_get_range_suffix(self, fs: FsBlobBackend) -> None:
        fs.put("k.bin", b"hello world", "application/octet-stream")
        rng = fs.get_range("k.bin", "bytes=-5")
        assert rng.data == b"world"
        assert (rng.start, rng.end) == (6, 10)

    def test_get_range_open_ended(self, fs: FsBlobBackend) -> None:
        fs.put("k.bin", b"hello world", "application/octet-stream")
        rng = fs.get_range("k.bin", "bytes=6-")
        assert rng.data == b"world"
        assert (rng.start, rng.end) == (6, 10)

    def test_get_range_missing_raises_blob_not_found(self, fs: FsBlobBackend) -> None:
        with pytest.raises(BlobNotFound):
            fs.get_range("nope.bin", None)

    @pytest.mark.parametrize(
        "bad_key",
        [
            "/abs/path",        # absolute
            "../escape",        # parent escape
            "a/../../b",        # nested .. still escapes
            "",                 # empty
        ],
    )
    def test_unsafe_key_rejected(self, fs: FsBlobBackend, bad_key: str) -> None:
        with pytest.raises(ValueError):
            fs.put(bad_key, b"x", "application/octet-stream")

    def test_exists_returns_true_false(self, fs: FsBlobBackend) -> None:
        assert fs.exists("a.bin") is False
        fs.put("a.bin", b"x", "application/octet-stream")
        assert fs.exists("a.bin") is True


# ---------------------------------------------------------------------------
# S3 backend (moto)
# ---------------------------------------------------------------------------


class TestS3BlobBackend:
    def test_put_get_roundtrip(self, s3: S3BlobBackend) -> None:
        s3.put("originals/aa/bb/abc.webp", b"hello", "image/webp")
        assert s3.get("originals/aa/bb/abc.webp") == b"hello"

    def test_key_prefix_applied(self, s3: S3BlobBackend) -> None:
        s3.put("originals/aa/bb/abc.webp", b"hello", "image/webp")
        # The factory pinned key_prefix="blobs/" — the actual S3 key
        # should carry that prefix, not just the relative key.
        resp = s3.client.list_objects_v2(Bucket=s3.cfg.bucket)
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        assert keys == ["blobs/originals/aa/bb/abc.webp"]

    def test_get_missing_raises_blob_not_found(self, s3: S3BlobBackend) -> None:
        with pytest.raises(BlobNotFound):
            s3.get("originals/aa/bb/nope.webp")

    def test_delete_existing(self, s3: S3BlobBackend) -> None:
        s3.put("k.bin", b"x", "application/octet-stream")
        assert s3.exists("k.bin")
        s3.delete("k.bin")
        assert not s3.exists("k.bin")

    def test_delete_idempotent_on_missing(self, s3: S3BlobBackend) -> None:
        s3.delete("never/was/here.bin")  # NoSuchKey is success on S3

    def test_get_range_full_body(self, s3: S3BlobBackend) -> None:
        s3.put("k.bin", b"hello world", "application/octet-stream")
        rng = s3.get_range("k.bin", None)
        assert rng.data == b"hello world"
        assert rng.partial is False
        assert rng.total_size == 11

    def test_get_range_bytes_N_M(self, s3: S3BlobBackend) -> None:
        s3.put("k.bin", b"hello world", "application/octet-stream")
        rng = s3.get_range("k.bin", "bytes=0-4")
        assert rng.data == b"hello"
        assert rng.partial is True
        # boto3 surfaces 'bytes 0-4/11' which we parse back into the
        # structured fields below.
        assert (rng.start, rng.end, rng.total_size) == (0, 4, 11)

    def test_get_range_missing_raises_blob_not_found(self, s3: S3BlobBackend) -> None:
        with pytest.raises(BlobNotFound):
            s3.get_range("nope.bin", None)

    def test_exists_returns_true_false(self, s3: S3BlobBackend) -> None:
        assert s3.exists("a.bin") is False
        s3.put("a.bin", b"x", "application/octet-stream")
        assert s3.exists("a.bin") is True

    def test_client_is_lazy(self, s3_cfg: S3Config) -> None:
        """Constructing :class:`S3BlobBackend` must not build the boto3 client.

        This is the contract the spec calls out: the app must boot
        cleanly with Garage down. Touching ``.client`` is the only thing
        that should trigger construction.
        """
        backend = S3BlobBackend(s3_cfg, key_prefix="blobs/")
        assert backend._pair is None
        with mock_aws():
            _ = backend.client  # first access builds the client
        assert backend._pair is not None


class TestS3BlobBackendErrorMapping:
    """Backend-unavailable mapping must be precise — the route handlers
    distinguish 404 (BlobNotFound) from 503 (BlobBackendUnavailable).
    """

    def test_endpoint_connection_error_maps_to_unavailable(
        self, s3_cfg: S3Config
    ) -> None:
        backend = S3BlobBackend(s3_cfg, key_prefix="blobs/")
        import boto3
        with mock_aws():
            backend._pair = (boto3.client("s3", region_name="us-east-1"), s3_cfg)
            backend.client.create_bucket(Bucket=s3_cfg.bucket)
            with patch.object(
                backend.client,
                "get_object",
                side_effect=EndpointConnectionError(endpoint_url="http://nope"),
            ):
                with pytest.raises(BlobBackendUnavailable):
                    backend.get("any/key")

    def test_500_client_error_maps_to_unavailable(self, s3_cfg: S3Config) -> None:
        backend = S3BlobBackend(s3_cfg, key_prefix="blobs/")
        # Hand-build a ClientError that looks like a 500 from S3 / Garage.
        boom = ClientError(
            error_response={
                "Error": {"Code": "InternalError", "Message": "boom"},
                "ResponseMetadata": {"HTTPStatusCode": 500},
            },
            operation_name="GetObject",
        )
        import boto3
        with mock_aws():
            backend._pair = (boto3.client("s3", region_name="us-east-1"), s3_cfg)
            backend.client.create_bucket(Bucket=s3_cfg.bucket)
            with patch.object(backend.client, "get_object", side_effect=boom):
                with pytest.raises(BlobBackendUnavailable):
                    backend.get("any/key")

    def test_404_client_error_maps_to_blob_not_found(
        self, s3_cfg: S3Config
    ) -> None:
        backend = S3BlobBackend(s3_cfg, key_prefix="blobs/")
        not_found = ClientError(
            error_response={
                "Error": {"Code": "NoSuchKey", "Message": "missing"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            operation_name="GetObject",
        )
        import boto3
        with mock_aws():
            backend._pair = (boto3.client("s3", region_name="us-east-1"), s3_cfg)
            backend.client.create_bucket(Bucket=s3_cfg.bucket)
            with patch.object(backend.client, "get_object", side_effect=not_found):
                with pytest.raises(BlobNotFound):
                    backend.get("any/key")


# ---------------------------------------------------------------------------
# Fallback wrapper
# ---------------------------------------------------------------------------


class TestFallbackReadBlobBackend:
    def test_get_falls_back_to_fs_on_blob_not_found(
        self, tmp_path: Path, s3: S3BlobBackend
    ) -> None:
        fs = FsBlobBackend(tmp_path)
        fs.put("legacy/key.bin", b"from-fs", "application/octet-stream")
        wrapper = FallbackReadBlobBackend(primary=s3, fs_fallback=fs)
        # S3 doesn't have the key — fallback to FS must return its bytes.
        assert wrapper.get("legacy/key.bin") == b"from-fs"

    def test_get_prefers_primary_when_present(
        self, tmp_path: Path, s3: S3BlobBackend
    ) -> None:
        fs = FsBlobBackend(tmp_path)
        s3.put("dupe.bin", b"primary", "application/octet-stream")
        fs.put("dupe.bin", b"fallback", "application/octet-stream")
        wrapper = FallbackReadBlobBackend(primary=s3, fs_fallback=fs)
        assert wrapper.get("dupe.bin") == b"primary"

    def test_get_range_falls_back_on_blob_not_found(
        self, tmp_path: Path, s3: S3BlobBackend
    ) -> None:
        fs = FsBlobBackend(tmp_path)
        fs.put("video/x.mp4", b"hello world", "video/mp4")
        wrapper = FallbackReadBlobBackend(primary=s3, fs_fallback=fs)
        rng = wrapper.get_range("video/x.mp4", "bytes=0-4")
        assert rng.data == b"hello"
        assert rng.partial

    def test_put_goes_to_primary_only(
        self, tmp_path: Path, s3: S3BlobBackend
    ) -> None:
        fs = FsBlobBackend(tmp_path)
        wrapper = FallbackReadBlobBackend(primary=s3, fs_fallback=fs)
        wrapper.put("new/key.bin", b"primary-only", "application/octet-stream")
        # Primary has it; FS does NOT — we're cutting over, not dual-writing.
        assert s3.exists("new/key.bin")
        assert not fs.exists("new/key.bin")

    def test_delete_goes_to_primary_only(
        self, tmp_path: Path, s3: S3BlobBackend
    ) -> None:
        fs = FsBlobBackend(tmp_path)
        s3.put("k.bin", b"x", "application/octet-stream")
        fs.put("k.bin", b"x", "application/octet-stream")
        wrapper = FallbackReadBlobBackend(primary=s3, fs_fallback=fs)
        wrapper.delete("k.bin")
        # Primary cleared, FS still holds the legacy copy until the
        # post-cutover cleanup pass.
        assert not s3.exists("k.bin")
        assert fs.exists("k.bin")

    def test_exists_unions_both(self, tmp_path: Path, s3: S3BlobBackend) -> None:
        fs = FsBlobBackend(tmp_path)
        wrapper = FallbackReadBlobBackend(primary=s3, fs_fallback=fs)
        assert not wrapper.exists("k.bin")
        fs.put("k.bin", b"x", "application/octet-stream")
        assert wrapper.exists("k.bin")  # via fallback


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetBlobBackend:
    def test_fs_mode_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_BAWT_STORAGE_BACKEND", raising=False)
        be = get_blob_backend("media_store", fs_root=tmp_path)
        assert isinstance(be, FsBlobBackend)

    def test_s3_mode(
        self,
        tmp_path: Path,
        s3_cfg: S3Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "s3")
        be = get_blob_backend(
            "media_store", fs_root=tmp_path, s3_cfg=s3_cfg,
        )
        assert isinstance(be, S3BlobBackend)
        assert be.key_prefix == "blobs/"

    def test_media_storage_uses_media_prefix(
        self,
        tmp_path: Path,
        s3_cfg: S3Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "s3")
        be = get_blob_backend(
            "media_storage", fs_root=tmp_path, s3_cfg=s3_cfg,
        )
        assert isinstance(be, S3BlobBackend)
        assert be.key_prefix == "media/"

    def test_s3_with_fallback(
        self,
        tmp_path: Path,
        s3_cfg: S3Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "s3")
        be = get_blob_backend(
            "media_store",
            fs_root=tmp_path,
            s3_cfg=s3_cfg,
            fallback_fs=True,
        )
        assert isinstance(be, FallbackReadBlobBackend)
        assert isinstance(be.primary, S3BlobBackend)
        assert isinstance(be.fs_fallback, FsBlobBackend)

    def test_fallback_via_env(
        self,
        tmp_path: Path,
        s3_cfg: S3Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "s3")
        monkeypatch.setenv("LLM_BAWT_S3_FALLBACK_FS", "true")
        be = get_blob_backend(
            "media_store", fs_root=tmp_path, s3_cfg=s3_cfg,
        )
        assert isinstance(be, FallbackReadBlobBackend)

    def test_unknown_kind_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown blob backend kind"):
            get_blob_backend("nonsense", fs_root=tmp_path)

    def test_unknown_backend_env_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "redis")
        with pytest.raises(ValueError, match="unknown LLM_BAWT_STORAGE_BACKEND"):
            get_blob_backend("media_store", fs_root=tmp_path)

    def test_fs_requires_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
        with pytest.raises(ValueError, match="fs_root not provided"):
            get_blob_backend("media_store")

    def test_s3_without_cfg_uses_the_stored_connection_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # TASK-1476: no fixed config means the shared stored-connection
        # source; nothing is resolved until the first operation.
        from llm_bawt.media import object_store_credentials as creds

        monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "s3")
        monkeypatch.setattr(creds, "_default_source", None)
        be = get_blob_backend("media_store", fs_root=tmp_path)
        assert isinstance(be, S3BlobBackend)
        assert be._source is creds.default_s3_config_source()


# ---------------------------------------------------------------------------
# Env-driven S3Config builder
# ---------------------------------------------------------------------------


class TestS3ConfigFromEnv:
    def test_complete_env_returns_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_BAWT_S3_ENDPOINT", "http://10.0.2.38:9000")
        monkeypatch.setenv("LLM_BAWT_S3_REGION", "garage")
        monkeypatch.setenv("LLM_BAWT_S3_BUCKET", "llmbawt-media")
        monkeypatch.setenv("LLM_BAWT_S3_ACCESS_KEY", "ak")
        monkeypatch.setenv("LLM_BAWT_S3_SECRET_KEY", "sk")
        cfg = s3_config_from_env()
        assert cfg is not None
        assert cfg.endpoint == "http://10.0.2.38:9000"
        assert cfg.bucket == "llmbawt-media"
        assert cfg.region == "garage"

    @pytest.mark.parametrize(
        "missing",
        [
            "LLM_BAWT_S3_ENDPOINT",
            "LLM_BAWT_S3_REGION",
            "LLM_BAWT_S3_BUCKET",
            "LLM_BAWT_S3_ACCESS_KEY",
            "LLM_BAWT_S3_SECRET_KEY",
        ],
    )
    def test_missing_var_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        for var in (
            "LLM_BAWT_S3_ENDPOINT",
            "LLM_BAWT_S3_REGION",
            "LLM_BAWT_S3_BUCKET",
            "LLM_BAWT_S3_ACCESS_KEY",
            "LLM_BAWT_S3_SECRET_KEY",
        ):
            monkeypatch.setenv(var, "x")
        monkeypatch.delenv(missing, raising=False)
        assert s3_config_from_env() is None


# ---------------------------------------------------------------------------
# Range parser
# ---------------------------------------------------------------------------


class TestParseByteRange:
    def test_explicit_range(self) -> None:
        assert _parse_byte_range("bytes=0-4", total=11) == (0, 4)

    def test_explicit_range_clamped_to_total(self) -> None:
        # If the client asks past the end, we clamp to total-1 per RFC 7233.
        assert _parse_byte_range("bytes=5-100", total=11) == (5, 10)

    def test_suffix_range(self) -> None:
        assert _parse_byte_range("bytes=-5", total=11) == (6, 10)

    def test_open_ended_range(self) -> None:
        assert _parse_byte_range("bytes=6-", total=11) == (6, 10)

    @pytest.mark.parametrize(
        "header",
        [
            "items=0-4",        # wrong unit
            "bytes=foo-bar",    # non-numeric
            "bytes=100-200",    # outside object
            "bytes=5-3",        # start > end
        ],
    )
    def test_malformed_or_unsatisfiable_raises_value_error(self, header: str) -> None:
        with pytest.raises(ValueError):
            _parse_byte_range(header, total=11)


# ---------------------------------------------------------------------------
# Re-export sanity: store.py must keep its public surface
# ---------------------------------------------------------------------------


def test_store_module_reexports_helpers() -> None:
    """The store module re-exports the backend types for back-compat.

    External callers that imported these names from ``llm_bawt.media.store``
    (e.g. older bridges, smoke tests) must still find them.
    """
    from llm_bawt.media import store as store_mod

    for name in (
        "BlobBackend",
        "BlobNotFound",
        "BlobBackendUnavailable",
        "FsBlobBackend",
        "S3Config",
        "get_blob_backend",
        "s3_config_from_env",
        "_write_idempotent",
        "_path_exists_nfs_safe",
        "_ESTALE_RETRY_DELAYS",
    ):
        assert hasattr(store_mod, name), f"store.py is missing re-export: {name}"


def test_module_path_sanity() -> None:
    """A drive-by check: the module we imported as ``os_mod`` *is*
    ``llm_bawt.media.object_store`` (catches silly aliasing mistakes in
    refactors).
    """
    assert os_mod.__name__ == "llm_bawt.media.object_store"
