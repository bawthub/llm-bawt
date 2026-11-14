"""TASK-1270: GGUF cold-cache download must use the huggingface-hub 1.x API."""
import sys
import types
from types import SimpleNamespace

from llm_bawt import gguf_handler


def test_cold_cache_download_uses_supported_hub_arguments(tmp_path, monkeypatch):
    calls = {}

    # Strict signature: hub 1.x has no local_dir_use_symlinks and no **kwargs.
    def hf_hub_download(*, repo_id, filename, local_dir=None, revision=None):
        calls.update(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return f"{local_dir}/{filename}"

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr(gguf_handler.importlib.util, "find_spec", lambda name: object())

    config = SimpleNamespace(MODEL_CACHE_DIR=str(tmp_path))
    path = gguf_handler.get_or_download_gguf_model("org/repo", "m.Q4_K_M.gguf", config)

    assert path == f"{tmp_path / 'org/repo'}/m.Q4_K_M.gguf"
    assert calls["local_dir"] == str(tmp_path / "org/repo")
