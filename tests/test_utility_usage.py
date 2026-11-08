"""TASK-1538: every utility completion leaves one content-free usage row."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from openai.types.chat import ChatCompletion

from llm_bawt.clients.usage_hook import report_usage
from llm_bawt.service import utility_completion as llm
from llm_bawt.service.schemas import RawCompletionRequest
from llm_bawt.service.usage import utility_usage

SECRET = "SOURCE_SECRET_DO_NOT_STORE"


def _completion(prompt_tokens=120, cached=20, completion_tokens=30) -> ChatCompletion:
    """The real OpenAI SDK response shape the clients hand to report_usage."""
    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    })


@pytest.fixture
def rig(monkeypatch):
    client = MagicMock()
    client.usage_observer = None  # a subscription path: no spend recorder
    client.model = "served-model"
    service = MagicMock()
    service.config.get_model_context_window.return_value = 8192
    service.config.get_model_max_tokens.return_value = 1600
    service._get_background_client.return_value = (client, "maintenance-ref")
    monkeypatch.setattr(llm, "get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda *_: "maintenance-ref")
    monkeypatch.setattr("llm_bawt.service.maintenance_fallback.served_model", lambda c, alias: "served-model")
    rows: list[dict] = []
    monkeypatch.setattr(utility_usage, "record", lambda row, **_: rows.append(row))
    return client, rows


def _reporting(client, text: str, response=None):
    def query_bounded(**_):
        report_usage(client, response if response is not None else _completion())
        return text
    return query_bounded


def test_provider_usage_is_recorded_without_content(rig):
    client, rows = rig
    client.query_bounded.side_effect = _reporting(client, f"{SECRET} answer")
    result = llm.complete_utility(RawCompletionRequest(prompt=SECRET, system=SECRET, max_tokens=400),
                                  timeout_seconds=20, caller="recall", bot_id="caid")
    assert result.tokens == 30  # the provider's output count, not len//4
    [row] = rows
    assert {k: row[k] for k in ("caller", "bot_id", "model", "input_tokens", "cache_read_tokens",
                                "cache_write_tokens", "output_tokens", "estimated", "outcome")} == {
        "caller": "recall", "bot_id": "caid", "model": "served-model", "input_tokens": 100,
        "cache_read_tokens": 20, "cache_write_tokens": 0, "output_tokens": 30,
        "estimated": False, "outcome": "ok",
    }
    assert SECRET not in repr(row)


def test_unreported_usage_is_an_estimate_and_flagged(rig):
    client, rows = rig
    client.query_bounded.return_value = "x" * 40
    result = llm.complete_utility(RawCompletionRequest(prompt="p" * 400, max_tokens=50),
                                  timeout_seconds=20, caller="title")
    assert result.tokens == 10
    [row] = rows
    assert row["estimated"] is True
    assert (row["input_tokens"], row["output_tokens"], row["caller"], row["bot_id"]) == (100, 10, "title", None)


def test_multiple_provider_responses_in_one_call_are_summed(rig):
    client, rows = rig

    def twice(**_):
        report_usage(client, _completion())
        report_usage(client, _completion(prompt_tokens=10, cached=0, completion_tokens=5))
        return "ok"
    client.query_bounded.side_effect = twice
    llm.complete_utility(RawCompletionRequest(prompt="p"), timeout_seconds=20, caller="recall")
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (110, 35)


@pytest.mark.parametrize("error,outcome", [(TimeoutError(), "timeout"), (RuntimeError(SECRET), "error")])
def test_failed_calls_record_an_outcome(rig, error, outcome):
    client, rows = rig
    client.query_bounded.side_effect = error
    with pytest.raises(HTTPException):
        llm.complete_utility(RawCompletionRequest(prompt=SECRET), timeout_seconds=20, caller="recall")
    [row] = rows
    assert (row["outcome"], row["estimated"], row["output_tokens"]) == (outcome, True, 0)
    assert SECRET not in repr(row)


def test_empty_completion_is_its_own_outcome(rig):
    client, rows = rig
    client.query_bounded.side_effect = _reporting(client, "  ")
    with pytest.raises(HTTPException):
        llm.complete_utility(RawCompletionRequest(prompt="p"), timeout_seconds=20, caller="recall")
    assert (rows[0]["outcome"], rows[0]["estimated"]) == ("empty", False)


def test_unconfigured_model_makes_no_call_and_no_row(rig, monkeypatch):
    _, rows = rig
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda *_: None)
    monkeypatch.setattr("llm_bawt.service.maintenance_health.record_unconfigured", lambda *_: None)
    with pytest.raises(HTTPException):
        llm.complete_utility(RawCompletionRequest(prompt="p"))
    assert rows == []


def test_http_caller_label_defaults_and_is_validated(rig):
    client, rows = rig
    client.query.return_value = "done"
    llm.complete_utility(RawCompletionRequest(prompt="p"))
    llm.complete_utility(RawCompletionRequest(prompt="p", caller="summarization"))
    assert [r["caller"] for r in rows] == ["llm_complete", "summarization"]
    with pytest.raises(ValueError):
        RawCompletionRequest(prompt="p", caller="Bad Label")


def test_report_usage_outside_a_call_and_recording_failure_never_raise(monkeypatch):
    report_usage(MagicMock(usage_observer=None), _completion())  # no open call: ignored

    class Broken:
        dialect = MagicMock()
        dialect.name = "postgresql"

        def begin(self):
            raise RuntimeError("db down")
    utility_usage.record({"caller": "recall"}, engine=Broken())


def test_spend_observer_still_receives_the_response(rig):
    client, rows = rig
    observer = MagicMock()
    client.usage_observer = observer
    client.query_bounded.side_effect = _reporting(client, "ok")
    llm.complete_utility(RawCompletionRequest(prompt="p"), timeout_seconds=20, caller="recall")
    observer.assert_called_once()
    assert rows[0]["estimated"] is False


def test_recall_route_passes_caller_and_usage_bot(monkeypatch):
    from llm_bawt.service.routes import recall_completion as route

    seen = {}

    def fake(request, **kwargs):
        seen.update(kwargs)
        return MagicMock(content="{}")
    monkeypatch.setattr(route, "complete_utility", fake)
    route.RecallCompletionRunner()._complete(route.RecallCompletionRequest(
        prompt="p", system="s", max_tokens=100, usage_bot_id="caid"))
    assert (seen["caller"], seen["bot_id"]) == ("recall", "caid")


@pytest.mark.asyncio
async def test_synthesizer_sends_the_calling_bot_as_a_label(monkeypatch):
    from llm_bawt.recall import completion

    sent = {}

    async def fake_json(self, method, url, **kwargs):
        sent.update(kwargs["json"])
        return {"content": "{}"}
    monkeypatch.setattr(completion.BoundedHTTP, "json", fake_json)
    token = completion.CALLING_BOT.set("caid")
    try:
        await completion.UtilitySynthesizer("http://app").synthesize("s", "q", [], 100)
    finally:
        completion.CALLING_BOT.reset(token)
    assert sent["usage_bot_id"] == "caid" and "bot_id" not in sent
