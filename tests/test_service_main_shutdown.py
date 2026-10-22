"""llm-service bounds uvicorn's graceful shutdown (TASK-1286).

BawtHub keeps an SSE stream open on /v1/ws at all times. Without a graceful
timeout uvicorn waits for that connection forever, so the lifespan teardown
never runs before server.sh (8s) and Docker (10s) kill the process. The
scratch-container proof on TASK-1286 showed exactly that with the kwarg
removed.
"""
from __future__ import annotations

import sys

import uvicorn

from llm_bawt.service import api


def test_uvicorn_is_started_with_a_bounded_graceful_shutdown(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(api, "_is_service_running", lambda host, port: False)
    monkeypatch.setattr(api, "setup_service_logging", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["llm-service", "--host", "127.0.0.1", "--port", "65000"])

    api.main()

    assert len(calls) == 1
    assert calls[0]["timeout_graceful_shutdown"] == 3
