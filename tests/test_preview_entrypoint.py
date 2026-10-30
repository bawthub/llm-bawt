"""Preview entrypoint never imports or starts the production application."""

import pytest
from fastapi.testclient import TestClient

from llm_bawt.service.preview_entrypoint import create_preview_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PREVIEW_ENABLED", "1")
    with TestClient(create_preview_app()) as preview:
        yield preview


def test_preview_health_is_not_a_production_health_receipt(client):
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"service": "preview", "state": "not_configured"}


@pytest.mark.parametrize("method,path", [
    ("post", "/v1/chat/completions"),
    ("get", "/v1/providers/claude/token"),
    ("get", "/v1/providers/openai_chatgpt/bundle"),
    ("post", "/v1/ops/jobs"),
    ("get", "/v1/bots"),
    ("post", "/v1/tasks"),
    ("get", "/mcp"),
])
def test_preview_does_not_register_production_routes(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 404


def test_preview_refuses_to_start_without_explicit_role(monkeypatch):
    monkeypatch.delenv("LLM_BAWT_PREVIEW_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="explicit preview role"):
        with TestClient(create_preview_app()):
            pass


def test_import_does_not_import_production_app():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, llm_bawt.service.preview_entrypoint; "
         "assert 'llm_bawt.service.api' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
