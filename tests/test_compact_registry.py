"""TASK-1071: edited registry prompt reaches both compaction paths without restart."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from claude_code_bridge.bootstrap import _ensure_compact_hook
from claude_code_bridge.compact_hook import guidance_for_hook
from claude_code_bridge.compact_prompt_lookup import COMPACT_PROMPT_KEY, fetch_compact_prompt
from claude_code_bridge.send_preflight import prepare_send
from llm_bawt.chat_prompt_defaults import COMPACT_SUMMARY_DEFAULT
from llm_bawt.prompt_registry import (
    PromptResolver, PromptTemplate, PromptTemplateStore, PromptTemplateVersion,
)
from llm_bawt.service.routes import prompts as prompt_routes


@pytest.fixture
def registry_api(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[PromptTemplate.__table__,
                                                  PromptTemplateVersion.__table__])
    store = PromptTemplateStore.__new__(PromptTemplateStore)
    store.engine = engine
    resolver = PromptResolver.__new__(PromptResolver)
    resolver.store = store
    monkeypatch.setattr(prompt_routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(prompt_routes, "PromptResolver", lambda config: resolver)

    counts = {"single_get": 0}
    app = FastAPI()
    app.include_router(prompt_routes.router)

    @app.middleware("http")
    async def count_gets(request, call_next):
        if request.method == "GET" and request.url.path == f"/v1/prompts/{COMPACT_PROMPT_KEY}":
            counts["single_get"] += 1
        return await call_next(request)

    # Exercise the real FastAPI routes, but adapt urlopen to in-process ASGI.
    # There is no socket, uvicorn thread, subprocess, or elapsed timeout wait.
    with TestClient(app) as client:
        def local_urlopen(request, *, timeout):
            assert timeout == 1.5  # lookup remains bounded even on a registry outage
            response = client.request(
                request.get_method(), urlsplit(request.full_url).path,
                headers=dict(request.header_items()),
                content=request.data,
            )
            if response.status_code >= 400:
                raise HTTPError(request.full_url, response.status_code, "registry unavailable", None, None)
            return io.BytesIO(response.content)

        monkeypatch.setattr("claude_code_bridge.compact_prompt_lookup.urlopen", local_urlopen)
        try:
            yield client, store, counts
        finally:
            engine.dispose()


def _save(client, body):
    response = client.put(f"/v1/prompts/{COMPACT_PROMPT_KEY}", json={
        "body": body, "scope_type": "global", "updated_by": "fixture",
    })
    response.raise_for_status()
    return response.json()


def _hook(*, custom=None):
    return guidance_for_hook({"trigger": "auto", "custom_instructions": custom},
                             lookup=lambda url: fetch_compact_prompt(url),
                             api_url="http://registry.test")


def _preflight(bridge, message):
    return asyncio.run(prepare_send(
        bridge, {"message": message, "model": "fixture-model", "request_id": "same-session"},
        "fixture-redis", SimpleNamespace(xack=AsyncMock()),
    ))


def test_editor_save_immediately_updates_explicit_and_auto_existing_session(registry_api, caplog, monkeypatch):
    caplog.set_level("INFO", logger="claude_code_bridge.compact_prompt_lookup")
    client, store, counts = registry_api
    listing = client.get("/v1/prompts?category=agent_execution&include_defaults=true").json()
    entries = [entry for entry in listing["prompts"] if entry["key"] == COMPACT_PROMPT_KEY]
    assert len(entries) == 1
    assert entries[0]["body"] == COMPACT_SUMMARY_DEFAULT
    assert entries[0]["source"] == "code_default"
    assert "compact_hook" in _ensure_compact_hook({})["hooks"]["PreCompact"][0]["hooks"][0]["command"]

    bridge = SimpleNamespace(
        _app_api_url="http://registry.test", _trigger_message_ids={},
        _preprocess_new_command=AsyncMock(side_effect=lambda text, **kw: text),
    )
    assert _preflight(bridge, "/compact").message == f"/compact {COMPACT_SUMMARY_DEFAULT}"
    first = _save(client, "Edited v1: preserve TASK-4242 and open integration work.")
    assert first["source"] == "db_override" and first["updated_at"]
    assert _preflight(bridge, "/compact\n\t").message == f"/compact {first['body']}"
    assert _hook() == first["body"]

    second = _save(client, "Edited v2: retain exact filenames, risks and next actions.")
    assert second["body"] != first["body"]
    assert _preflight(bridge, "/compact").message == f"/compact {second['body']}"
    assert _hook() == second["body"]
    assert "key=agents.compaction_summary" in caplog.text
    assert "sha256=" in caplog.text
    assert second["body"] not in caplog.text  # no prompt content in logging

    before_custom = counts["single_get"]
    custom = "/compact\n保留 all IDs\nWrite 15,000 tokens.  "
    assert _preflight(bridge, custom).message == custom
    assert _hook(custom="保留 all IDs\nWrite 15,000 tokens.") == ""
    assert counts["single_get"] == before_custom

    def timed_out(_request, *, timeout):
        assert timeout == 1.5
        raise TimeoutError("fixture timeout")

    with monkeypatch.context() as outage:
        outage.setattr("claude_code_bridge.compact_prompt_lookup.urlopen", timed_out)
        assert _preflight(bridge, "/compact").message == "/compact"
        assert _hook() == ""

    store.engine = None  # app must not serve a stale code default over a saved edit
    assert client.get(f"/v1/prompts/{COMPACT_PROMPT_KEY}").status_code == 503
    assert _preflight(bridge, "/compact").message == "/compact"
    assert _hook() == ""  # CLI proceeds with its own default behavior
    assert _preflight(bridge, custom).message == custom
    assert "unavailable" in caplog.text
