"""TASK-1210 bounded utility transport and admission tests; no live models."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from openai import APIStatusError, OpenAI

from llm_bawt.clients.base import LLMClient
from llm_bawt.clients.openai_client import OpenAIClient
from llm_bawt.clients.responses_client import ResponsesClient
from llm_bawt.models.message import Message
from llm_bawt.service import utility_completion as llm
from llm_bawt.service.routes import recall_completion as module
from llm_bawt.service.schemas import RawCompletionRequest


@pytest.fixture
def utility(monkeypatch):
    client = MagicMock()
    client.query_bounded.return_value = '{"findings":[]}'
    service = MagicMock()
    service.config.get_model_context_window.return_value = 8192
    service.config.get_model_max_tokens.return_value = 1600
    service._get_background_client.return_value = (client, "configured-maintenance")
    monkeypatch.setattr(llm, "get_service", lambda: service)
    resolver = MagicMock(return_value="configured-maintenance")
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", resolver)
    return service, client, resolver


def test_bounded_utility_uses_only_explicit_input_and_maintenance_model(utility):
    service, client, resolver = utility
    result = llm.complete_utility(RawCompletionRequest(prompt="selected evidence", system="extract only",
                                                      max_tokens=900, temperature=0.0), timeout_seconds=20)
    resolver.assert_called_once_with(service.config, "maintenance_model")
    service._get_background_client.assert_called_once_with(
        model_override="configured-maintenance", job="recall_completion", allow_fallback=True,
    )
    client.query.assert_not_called()
    kwargs = client.query_bounded.call_args.kwargs
    assert [(m.role, m.content) for m in kwargs.pop("messages")] == [
        ("system", "extract only"), ("user", "selected evidence"),
    ]
    assert kwargs == {"timeout_seconds": 20, "max_tokens": 900, "temperature": 0.0}
    assert result.content == '{"findings":[]}'


@pytest.mark.parametrize("error,status", [(NotImplementedError("secret history"), 422),
                                          (RuntimeError("secret history"), 502),
                                          (AttributeError("secret history"), 502)])
def test_bounded_provider_errors_do_not_echo_source_text(utility, error, status, caplog):
    _, client, _ = utility
    client.query_bounded.side_effect = error
    with pytest.raises(HTTPException) as caught:
        llm.complete_utility(RawCompletionRequest(prompt="evidence", system="extract"), timeout_seconds=20)
    assert caught.value.status_code == status
    assert "secret history" not in str(caught.value.detail)
    assert "secret history" not in caplog.text


@pytest.mark.parametrize("window,output_cap,status", [(512, 1600, 413), (8192, 99, 413),
                                                      (None, 1600, 503), (8192, None, 503)])
def test_bounded_utility_checks_resolved_model_budget_before_query(utility, window, output_cap, status):
    service, client, _ = utility
    service.config.get_model_context_window.return_value = window
    service.config.get_model_max_tokens.return_value = output_cap
    with pytest.raises(HTTPException) as caught:
        llm.complete_utility(RawCompletionRequest(prompt="evidence", system="extract", max_tokens=400),
                             timeout_seconds=20)
    # TASK-1795: too large is the caller's (413); an unreadable budget is the server's (503).
    assert caught.value.status_code == status
    client.query_bounded.assert_not_called()
    service.config.get_model_context_window.assert_called_once_with("configured-maintenance")
    service.config.get_model_max_tokens.assert_called_once_with("configured-maintenance")


class _AmbiguousBareKey(Exception):
    pass


def test_budget_reads_the_built_endpoint_ref_not_the_requested_string(utility):
    """TASK-1795: one model on two access paths made the bare key ambiguous.
    The budget follows the endpoint the client was built from, even when the
    setting itself is a bare key only the background harness disambiguates."""
    service, client, resolver = utility
    resolver.return_value = "shared-model"
    service._get_background_client.return_value = (client, "shared-model@first-path")

    def by_ref(name):
        if name == "shared-model":
            raise _AmbiguousBareKey(name)
        return {"shared-model@first-path": 8192}[name]

    service.config.get_model_context_window.side_effect = by_ref
    service.config.get_model_max_tokens.side_effect = lambda name: by_ref(name) and 1600
    result = llm.complete_utility(RawCompletionRequest(prompt="evidence", system="extract", max_tokens=400),
                                  timeout_seconds=20)
    assert result.content == '{"findings":[]}'
    service.config.get_model_context_window.assert_called_once_with("shared-model@first-path")


def test_unreadable_budget_is_recorded_in_maintenance_health(utility, monkeypatch):
    service, client, _ = utility
    service.config.get_model_context_window.side_effect = _AmbiguousBareKey("x")
    health = MagicMock()
    monkeypatch.setattr("llm_bawt.service.maintenance_health.maintenance_health", lambda: health)
    with pytest.raises(HTTPException) as caught:
        llm.complete_utility(RawCompletionRequest(prompt="evidence", system="extract"), timeout_seconds=20)
    assert (caught.value.status_code, caught.value.detail) == (503, "recall_model_budget_unavailable")
    client.query_bounded.assert_not_called()
    job, error_class = health.record_failure.call_args.args[:2]
    assert (job, error_class) == ("recall_completion", "budget_unavailable")


def _client(cls, handler):
    wrapper = object.__new__(cls)
    wrapper.model = "fixture-model"
    wrapper.model_alias = "fixture-model"
    wrapper.config = SimpleNamespace(SYSTEM_MESSAGE="", TEMPERATURE=0.5, TOP_P=1.0,
                                     get_model_max_tokens=lambda _: 2000)
    wrapper.client = OpenAI(api_key="fixture-not-a-real-key", base_url="https://fixture.invalid/v1",
                            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return wrapper


@pytest.mark.parametrize("cls", [OpenAIClient, ResponsesClient])
def test_sdk_transport_receives_timeout_and_does_not_retry(cls):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(503, json={"error": {"message": "fixture unavailable"}})

    client = _client(cls, handler)
    try:
        with pytest.raises(APIStatusError):
            client.query_bounded([Message(role="system", content="extract"),
                                  Message(role="user", content="evidence")],
                                 timeout_seconds=7.0, max_tokens=100)
        assert len(seen) == 1
        assert set(seen[0].extensions["timeout"].values()) == {7.0}
        assert b'"tools"' not in seen[0].content
        assert client.client.max_retries == 2  # pooled client's policy is untouched
        assert client.client.timeout.read != 7.0
    finally:
        client.client.close()


@pytest.mark.parametrize("cls", [OpenAIClient, ResponsesClient])
def test_bounded_clients_return_text_without_tool_execution(cls):
    def handler(request):
        if cls is OpenAIClient:
            payload = {"id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture-model",
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": "evidence"}}]}
        else:
            payload = {"id": "fixture", "object": "response", "created_at": 0, "model": "fixture-model",
                       "output": [{"id": "message-fixture", "type": "message", "role": "assistant",
                                   "status": "completed", "content": [{"type": "output_text", "text": "evidence",
                                                                        "annotations": []}]}]}
        return httpx.Response(200, json=payload)

    client = _client(cls, handler)
    try:
        assert client.query_bounded([Message(role="system", content="extract"),
                                     Message(role="user", content="evidence")],
                                    timeout_seconds=7.0, max_tokens=100) == "evidence"
    finally:
        client.client.close()


def test_unsupported_client_bounded_contract_fails_closed():
    with pytest.raises(NotImplementedError):
        LLMClient.query_bounded(object(), [], timeout_seconds=1, max_tokens=1)


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(module.router)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("extra", [{"model": "override"}, {"bot_id": "someone"}, {"max_tokens": 1601},
                                   {"max_tokens": True}, {"system": ""}])
def test_recall_route_rejects_override_or_bad_limits_without_echo(api, extra):
    payload = {"prompt": "SOURCE_SECRET_DO_NOT_ECHO", "system": "extract", "max_tokens": 100}
    payload.update(extra)
    response = api.post("/v1/llm/recall-complete", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_recall_request"}
    assert "SOURCE_SECRET" not in response.text


def test_recall_route_caps_raw_request_bytes(api):
    response = api.post("/v1/llm/recall-complete", content=b"x" * (module.MAX_REQUEST_BYTES + 1))
    assert response.status_code == 413
    assert response.json() == {"detail": "recall_input_too_large"}


def test_recall_route_works_with_isolated_input(api, utility):
    response = api.post("/v1/llm/recall-complete", json={"prompt": "evidence", "system": "extract", "max_tokens": 100})
    assert response.status_code == 200
    assert response.json() == {"content": '{"findings":[]}'}


@pytest.mark.parametrize("output", ["ERROR: provider rejected temperature", None])
def test_incompatible_model_output_is_typed_failure(api, utility, output):
    _, client, _ = utility
    client.query_bounded.return_value = output
    response = api.post("/v1/llm/recall-complete", json={"prompt": "evidence", "system": "extract"})
    assert response.status_code == 502
    assert response.json() == {"detail": "recall_completion_unavailable"}
    client.query.assert_not_called()


@pytest.mark.parametrize("raised,status,detail", [
    (HTTPException(status_code=422, detail="x"), 503, "recall_completion_unavailable"),
    (HTTPException(status_code=503, detail="x"), 503, "recall_completion_unavailable"),
    (HTTPException(status_code=413, detail="x"), 413, "recall_input_too_large"),
    (HTTPException(status_code=502, detail="x"), 502, "recall_completion_unavailable"),
])
def test_route_reports_server_faults_as_server_errors(api, monkeypatch, raised, status, detail):
    """TASK-1795: a server-side 422 read as the caller's bad request."""
    def fail(*args, **kwargs):
        raise raised

    monkeypatch.setattr(module, "complete_utility", fail)
    response = api.post("/v1/llm/recall-complete", json={"prompt": "evidence", "system": "extract"})
    assert (response.status_code, response.json()) == (status, {"detail": detail})


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [(413, "input_budget"), (422, "invalid_request"),
                                         (429, "busy"), (504, "timeout"),
                                         (503, "model_unavailable"), (502, "model_unavailable")])
async def test_synthesizer_maps_route_status_to_recall_error_code(monkeypatch, status, code):
    from llm_bawt.recall.completion import UtilitySynthesizer
    from llm_bawt.recall.contracts import Evidence, RecallError

    async def transport(request):
        return httpx.Response(status, json={"detail": "x"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(transport), **kwargs))
    with pytest.raises(RecallError) as caught:
        await UtilitySynthesizer("https://test.invalid").synthesize(
            "system", "query", [Evidence("messages", "m1", "b", "u", "text")], 100)
    assert caught.value.code == code


@pytest.mark.parametrize("output", ["", "   "])
def test_route_reports_empty_completion_as_empty_response(api, utility, output):
    """TASK-1795: an empty model answer was reported as model_unavailable."""
    _, client, _ = utility
    client.query_bounded.return_value = output
    response = api.post("/v1/llm/recall-complete", json={"prompt": "evidence", "system": "extract"})
    assert (response.status_code, response.json()) == (502, {"detail": "recall_empty_completion"})
    client.query.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,detail,code", [
    (502, "recall_empty_completion", "empty_response"),
    (502, "recall_completion_unavailable", "model_unavailable"),
    (502, "x" * 4096, "model_unavailable"),
    (413, "recall_empty_completion", "empty_response"),
])
async def test_synthesizer_maps_empty_completion_detail(monkeypatch, status, detail, code):
    from llm_bawt.recall.completion import UtilitySynthesizer
    from llm_bawt.recall.contracts import Evidence, RecallError
    from llm_bawt.recall.service import ERROR_CODES

    async def transport(request):
        return httpx.Response(status, json={"detail": detail})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(transport), **kwargs))
    with pytest.raises(RecallError) as caught:
        await UtilitySynthesizer("https://test.invalid").synthesize(
            "system", "query", [Evidence("messages", "m1", "b", "u", "text")], 100)
    assert caught.value.code == code and code in ERROR_CODES


@pytest.mark.asyncio
async def test_runner_rejects_model_overrun(utility):
    _, client, _ = utility
    client.query_bounded.return_value = "unbounded output " * 2000
    runner = module.RecallCompletionRunner(capacity=1)
    try:
        with pytest.raises(HTTPException) as caught:
            await runner.run(module.RecallCompletionRequest(prompt="evidence", system="extract", max_tokens=100))
        assert caught.value.status_code == 502
        assert caught.value.detail == "recall_completion_unavailable"
    finally:
        runner._worker._pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_timeout_retains_slot_until_underlying_call_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(timeout=10)
        return SimpleNamespace(content="{}")

    monkeypatch.setattr(module, "complete_utility", blocked)
    monkeypatch.setattr(module, "COMPLETION_TIMEOUT_SECONDS", 0.03)
    runner = module.RecallCompletionRunner(capacity=1)
    request = module.RecallCompletionRequest(prompt="evidence", system="extract")
    try:
        with pytest.raises(HTTPException) as caught:
            await runner.run(request)
        assert started.is_set()
        assert caught.value.status_code == 504
        with pytest.raises(HTTPException) as busy:
            await runner.run(request)
        assert busy.value.status_code == 429
    finally:
        release.set()
        runner._worker._pool.shutdown(wait=True)
    # The worker finally released its slot, not the timed-out coroutine.
    assert runner._worker._slots.acquire(blocking=False)
    runner._worker._slots.release()


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_release_running_slot(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(timeout=10)
        return SimpleNamespace(content="{}")

    monkeypatch.setattr(module, "complete_utility", blocked)
    runner = module.RecallCompletionRunner(capacity=1)
    request = module.RecallCompletionRequest(prompt="evidence", system="extract")
    task = asyncio.create_task(runner.run(request))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(HTTPException) as busy:
            await runner.run(request)
        assert busy.value.status_code == 429
    finally:
        release.set()
        runner._worker._pool.shutdown(wait=True)


def test_query_terms_keep_identifier_parts():
    from llm_bawt.recall.excerpts import fts_query, query_terms
    terms = query_terms("What did Caid say about the maintenance_model and query_bounded for TASK-1210?")
    assert {"maintenance", "model", "query", "bounded", "task", "1210", "caid"} <= set(terms)
    assert fts_query("'; DROP TABLE x --") == "drop | table"
    assert fts_query("?? !!") is None


def test_synthesis_prompt_carries_server_time_speaker_and_supersession_rule():
    from llm_bawt.recall.completion import synthesis_prompt
    from llm_bawt.recall.prompt import DEFAULT_RECALL_SYSTEM_PROMPT as SYSTEM
    from llm_bawt.recall.contracts import Evidence
    prompt = synthesis_prompt("q", [Evidence("messages", "m", "b", "u", "t", at="2026-10-21T10:34:05Z",
                                             speaker="assistant", follows="h")])
    assert '"at":"2026-10-21T10:34:05Z"' in prompt and '"speaker":"assistant"' in prompt
    assert "oldest-first" in SYSTEM and "quote the latest statement" in SYSTEM
