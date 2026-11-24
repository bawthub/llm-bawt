"""Public text profile: omitted source assembles without media/voice handlers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from llm_bawt.public_text_profile import OMITTED_PATHS, capability_present
from local_model_bridge import embed_server, pinned_embed_model

REPO = Path(__file__).resolve().parents[1]
_REAL_TREE = [pytest.mark.integration, pytest.mark.offline_integration,
              pytest.mark.real_process("imports the app and MCP server from an exported tree")]


def _apply_real_tree(fn):
    for marker in _REAL_TREE:
        fn = marker(fn)
    return fn

OMITTED_TOOLS = {
    "generate_image", "speech_generate", "home_audio_devices", "home_audio_enqueue",
    "home_audio_status", "home_audio_cancel", "media_lookup", "media_add", "media_queue",
    "media_history", "media_pipeline_status", "media_library_stats",
}

_PROBE = textwrap.dedent(
    """
    import asyncio, json, sys
    import llm_bawt, local_model_bridge
    from llm_bawt.service.api import app
    from llm_bawt.mcp_server.server import mcp
    from local_model_bridge import embed_server

    paths = sorted(app.openapi()["paths"])
    routes = sorted({getattr(r, "path", "") for r in app.routes})

    async def tools():
        listed = sorted(t.name for t in await mcp.list_tools())
        try:
            await mcp._tool_manager.call_tool("generate_image", {"prompt": "x"}, None)
            called = "handled"
        except Exception as exc:
            called = f"{type(exc).__name__}: {exc}"
        return listed, called

    listed, called = asyncio.run(tools())
    print(json.dumps({
        "src": [llm_bawt.__file__, local_model_bridge.__file__],
        "openapi": paths, "routes": routes, "tools": listed, "generate_image": called,
        "omitted_loaded": sorted(m for m in sys.modules if m.split(".")[-1] in {
            "media_generation_tools", "home_audio_tools", "media_tools", "generation_service",
            "gpu_preflight", "response_audio", "media_handoff", "video_server", "gpu_telemetry",
        } and not m.startswith("llm_bawt.media.response_audio")),
        "media_clients_loaded": "llm_bawt.media.clients" in sys.modules,
        "pinned_dir": embed_server.pinned_model_dir(),
    }))
    """
)


def _probe(src_root: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "LOCAL_MODEL_EMBED_PINNED_DIR"}
    env["PYTHONPATH"] = str(src_root)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=src_root.parent, env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def public_probe(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("public-text") / "tree"
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in OMITTED_PATHS:
        target = root / rel
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    return _probe(root / "src")


def test_manifest_names_existing_source():
    missing = [rel for rel in OMITTED_PATHS if not (REPO / rel).exists()]
    assert missing == []


@_apply_real_tree
def test_public_tree_omits_media_and_voice_http(public_probe):
    assert all("/tree/src/" in path for path in public_probe["src"])
    for listing in (public_probe["openapi"], public_probe["routes"]):
        assert not [p for p in listing if p.startswith("/v1/media") or "response-audio" in p or "response_audio" in p]
        assert "/v1/uploads" in listing
        assert "/v1/providers" in listing or any(p.startswith("/v1/providers") for p in listing)
    assert public_probe["omitted_loaded"] == []
    assert public_probe["media_clients_loaded"] is False


@_apply_real_tree
def test_public_tree_mcp_has_no_omitted_handlers(public_probe):
    assert OMITTED_TOOLS.isdisjoint(public_probe["tools"])
    assert {"attachment_get", "memory_search", "tasks_get"} <= set(public_probe["tools"])
    assert public_probe["generate_image"] != "handled"
    assert "generate_image" in public_probe["generate_image"]


@_apply_real_tree
def test_public_tree_embeds_only_from_pinned_dir(public_probe):
    assert public_probe["pinned_dir"] == pinned_embed_model.DEFAULT_DIR


@_apply_real_tree
def test_private_tree_keeps_every_capability(monkeypatch):
    monkeypatch.delenv("LOCAL_MODEL_EMBED_PINNED_DIR", raising=False)
    probe = _probe(REPO / "src")
    assert any(p.startswith("/v1/media/generations") for p in probe["openapi"])
    assert OMITTED_TOOLS <= set(probe["tools"])
    assert probe["pinned_dir"] is None
    assert capability_present("llm_bawt.service.routes.media")


def _fake_pinned(monkeypatch, root: Path) -> None:
    files = {"modules.json": b"{}", "1_Pooling/config.json": b"{}", "model.safetensors": b"weights"}
    pinned = {}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        pinned[rel] = (hashlib.sha256(data).hexdigest(), len(data))
    monkeypatch.setattr(pinned_embed_model, "FILES", pinned)


def test_pinned_verify_accepts_exact_set_only(tmp_path, monkeypatch):
    _fake_pinned(monkeypatch, tmp_path)
    assert pinned_embed_model.verify(tmp_path) == tmp_path
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(pinned_embed_model.PinnedModelError, match="extra"):
        pinned_embed_model.verify(tmp_path)
    (tmp_path / "pytorch_model.bin").unlink()
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(pinned_embed_model.PinnedModelError, match="mismatch"):
        pinned_embed_model.verify(tmp_path)
    with pytest.raises(pinned_embed_model.PinnedModelError, match="missing"):
        pinned_embed_model.verify(tmp_path / "absent")


def test_pinned_manifest_matches_cleared_minilm():
    assert pinned_embed_model.REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert len(pinned_embed_model.FILES) == 11
    assert pinned_embed_model.FILES["model.safetensors"] == (
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db", 90868376,
    )
    assert not [p for p in pinned_embed_model.FILES if p.endswith((".bin", ".onnx", ".h5", ".msgpack"))]


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
    _FakeST.calls = []


def test_pinned_load_is_offline_and_fails_closed(tmp_path, monkeypatch):
    _fake_sentence_transformers(monkeypatch)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOCAL_MODEL_EMBED_PINNED_DIR", str(tmp_path / "absent"))
    assert embed_server.load_model() is None
    assert _FakeST.calls == []  # no named/network fallback

    _fake_pinned(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_MODEL_EMBED_PINNED_DIR", str(tmp_path))
    assert embed_server.load_model() is not None
    assert _FakeST.calls == [(str(tmp_path), True)]
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_private_default_load_keeps_named_model(monkeypatch):
    _fake_sentence_transformers(monkeypatch)
    monkeypatch.delenv("LOCAL_MODEL_EMBED_PINNED_DIR", raising=False)
    assert embed_server.load_model() is not None
    assert _FakeST.calls == [(embed_server.EMBED_MODEL_NAME, True)]
