"""Pinned MiniLM: fetched on first use at a pinned revision, hash-checked, loaded offline."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
from pathlib import Path

import pytest

from local_model_bridge import embed_server, pinned_embed_model

_UPSTREAM = {"modules.json": b"{}", "1_Pooling/config.json": b"{}", "model.safetensors": b"weights"}


def _pin(monkeypatch) -> None:
    monkeypatch.setattr(pinned_embed_model, "FILES", {
        rel: (hashlib.sha256(data).hexdigest(), len(data)) for rel, data in _UPSTREAM.items()})


def _fake_upstream(monkeypatch, served: dict[str, bytes]) -> list[str]:
    """Pin ``_UPSTREAM`` and serve ``served`` from a fake Hugging Face; return requested URLs."""
    _pin(monkeypatch)
    requested: list[str] = []

    def urlopen(url, timeout=None):
        requested.append(url)
        rel = url.split(f"/resolve/{pinned_embed_model.REVISION}/", 1)[1]
        return io.BytesIO(served[rel])

    monkeypatch.setattr(pinned_embed_model.urllib.request, "urlopen", urlopen)
    return requested


class _FakeST:
    calls: list = []

    def __init__(self, name, device=None, local_files_only=False):
        _FakeST.calls.append((name, local_files_only))

    def get_sentence_embedding_dimension(self):
        return 384


def _fake_sentence_transformers(monkeypatch):
    module = type(sys)("sentence_transformers")
    module.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setattr(embed_server, "_model", None)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    _FakeST.calls = []


def test_manifest_matches_cleared_minilm():
    assert pinned_embed_model.REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert len(pinned_embed_model.FILES) == 11
    assert pinned_embed_model.FILES["model.safetensors"] == (
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db", 90868376,
    )
    assert not [p for p in pinned_embed_model.FILES if p.endswith((".bin", ".onnx", ".h5", ".msgpack"))]


def test_verify_accepts_exact_set_only(tmp_path: Path, monkeypatch):
    _pin(monkeypatch)
    for rel, data in _UPSTREAM.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    assert pinned_embed_model.verify(tmp_path) == tmp_path
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(pinned_embed_model.PinnedModelError, match="extra"):
        pinned_embed_model.verify(tmp_path)
    (tmp_path / "pytorch_model.bin").unlink()
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(pinned_embed_model.PinnedModelError, match="mismatch"):
        pinned_embed_model.verify(tmp_path)


def test_first_use_fetches_pinned_revision_then_loads_offline(tmp_path: Path, monkeypatch):
    _fake_sentence_transformers(monkeypatch)
    requested = _fake_upstream(monkeypatch, _UPSTREAM)
    cache = tmp_path / "models" / "minilm"
    monkeypatch.setenv("LOCAL_MODEL_EMBED_PINNED_DIR", str(cache))
    assert embed_server.load_model() is not None
    assert _FakeST.calls == [(str(cache), True)]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert len(requested) == 3
    assert all(f"/{pinned_embed_model.REPO_ID}/resolve/{pinned_embed_model.REVISION}/" in u for u in requested)

    # A warm cache loads with no download.
    monkeypatch.setattr(embed_server, "_model", None)
    assert embed_server.load_model() is not None
    assert len(requested) == 3


def test_wrong_upstream_hash_fails_closed_and_leaves_no_cache(tmp_path: Path, monkeypatch):
    _fake_sentence_transformers(monkeypatch)
    _fake_upstream(monkeypatch, {**_UPSTREAM, "model.safetensors": b"tampered"})
    cache = tmp_path / "minilm"
    with pytest.raises(pinned_embed_model.PinnedModelError, match="pinned SHA-256"):
        pinned_embed_model.ensure(cache)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv("LOCAL_MODEL_EMBED_PINNED_DIR", str(cache))
    assert embed_server.load_model() is None
    assert _FakeST.calls == []  # no named or floating fallback


def test_tampered_cache_is_reported_not_refetched(tmp_path: Path, monkeypatch):
    requested = _fake_upstream(monkeypatch, _UPSTREAM)
    cache = pinned_embed_model.ensure(tmp_path / "minilm")
    (cache / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(pinned_embed_model.PinnedModelError, match="mismatch"):
        pinned_embed_model.ensure(cache)
    assert len(requested) == 3


def test_default_load_keeps_named_model(monkeypatch):
    _fake_sentence_transformers(monkeypatch)
    monkeypatch.delenv("LOCAL_MODEL_EMBED_PINNED_DIR", raising=False)
    assert embed_server.load_model() is not None
    assert _FakeST.calls == [(embed_server.EMBED_MODEL_NAME, True)]


def test_concurrent_first_use_downloads_once_and_both_succeed(tmp_path: Path, monkeypatch):
    _pin(monkeypatch)
    started, release = threading.Event(), threading.Event()
    requested: list[str] = []

    def urlopen(url, timeout=None):
        requested.append(url)
        started.set()
        assert release.wait(10)  # hold the first download until the second caller is waiting
        return io.BytesIO(_UPSTREAM[url.split(f"/resolve/{pinned_embed_model.REVISION}/", 1)[1]])

    monkeypatch.setattr(pinned_embed_model.urllib.request, "urlopen", urlopen)
    cache = tmp_path / "minilm"
    results: list[object] = []

    def call():
        try:
            results.append(pinned_embed_model.ensure(cache))
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            results.append(exc)

    first = threading.Thread(target=call)
    first.start()
    assert started.wait(10)
    second = threading.Thread(target=call)
    second.start()
    release.set()
    first.join(10)
    second.join(10)
    assert results == [cache, cache]
    assert len(requested) == 3  # one download set


def test_losing_publisher_uses_verified_winner_cache(tmp_path: Path, monkeypatch):
    _pin(monkeypatch)
    cache = tmp_path / "minilm"
    served = 0

    def urlopen(url, timeout=None):
        nonlocal served
        served += 1
        if served == len(_UPSTREAM):
            # Another process publishes a valid cache before this one finishes.
            for rel, data in _UPSTREAM.items():
                (cache / rel).parent.mkdir(parents=True, exist_ok=True)
                (cache / rel).write_bytes(data)
        return io.BytesIO(_UPSTREAM[url.split(f"/resolve/{pinned_embed_model.REVISION}/", 1)[1]])

    monkeypatch.setattr(pinned_embed_model.urllib.request, "urlopen", urlopen)
    winner = {rel: (cache / rel) for rel in _UPSTREAM}
    assert pinned_embed_model.fetch(cache) == cache
    assert {rel: path.read_bytes() for rel, path in winner.items()} == _UPSTREAM
    assert [p.name for p in tmp_path.iterdir()] == ["minilm"]  # staging cleaned up
