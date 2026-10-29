"""TASK-1729: per-job prompt-cache routing and background reasoning effort."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from llm_bawt.clients.base import LLMClient
from llm_bawt.clients.grok_client import GrokClient
from llm_bawt.clients.responses_client import ResponsesClient
from llm_bawt.models.message import Message
from llm_bawt.service.maintenance_health import MaintenanceHealth, MonitoredClient
from llm_bawt.setting_definitions import SETTING_DEFINITIONS, STORAGE_RUNTIME_SETTING


def _bare(cls):
    sized = type(cls.__name__, (cls,), {"effective_max_tokens": 100})
    client = sized.__new__(sized)
    client.model = "m"
    client.config = SimpleNamespace(TEMPERATURE=0.5)
    client._prepare_input = lambda messages, log_input=True: ([{"role": "user", "content": "x"}], "sys")
    return client


def _payload(cls, **kwargs):
    client = _bare(cls)
    options = client.job_request_options("memory_extraction", **kwargs)
    return client._build_base_payload([Message(role="user", content="x")], **options)


def test_openai_responses_routes_by_prompt_cache_key():
    payload = _payload(ResponsesClient, reasoning_effort="none")
    assert payload["prompt_cache_key"] == "llm-bawt:memory_extraction"
    assert payload["reasoning"] == {"effort": "none"}
    assert "extra_headers" not in payload
    assert "cache_routing_key" not in payload and "reasoning_effort" not in payload


def test_grok_routes_by_conv_id_header():
    payload = _payload(GrokClient)
    assert payload["extra_headers"] == {"x-grok-conv-id": "llm-bawt:memory_extraction"}
    assert "prompt_cache_key" not in payload


def test_unset_effort_is_omitted():
    assert "reasoning" not in _payload(GrokClient, reasoning_effort=None)


def test_plain_payload_has_no_routing_or_effort():
    payload = _bare(ResponsesClient)._build_base_payload([Message(role="user", content="x")])
    assert not {"reasoning", "prompt_cache_key", "extra_headers"} & payload.keys()


def test_base_client_default_is_noop():
    assert LLMClient.job_request_options(MagicMock(), "x", reasoning_effort="none") == {}


def _monitored(inner, effort=lambda: "none"):
    return MonitoredClient(inner, job="history_summarization", model="m", provider="p",
                           credential=None, health=MaintenanceHealth(), reasoning_effort=effort)


def test_monitored_query_injects_job_options_and_caller_wins():
    inner = MagicMock()
    inner.job_request_options = lambda job, reasoning_effort=None: {
        "cache_routing_key": f"llm-bawt:{job}", "reasoning_effort": reasoning_effort}
    inner.query.return_value = "ok"
    _monitored(inner).query([], max_tokens=5, reasoning_effort="low")
    kwargs = inner.query.call_args.kwargs
    assert kwargs["cache_routing_key"] == "llm-bawt:history_summarization"
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_tokens"] == 5


def test_monitored_effort_resolved_per_call():
    seen = []
    inner = MagicMock()
    inner.job_request_options = lambda job, reasoning_effort=None: seen.append(reasoning_effort) or {}
    inner.query.return_value = "ok"
    values = iter(["none", ""])
    client = _monitored(inner, effort=lambda: next(values))
    client.query([])
    client.query([])
    assert seen == ["none", None]


def test_monitored_effort_lookup_failure_omits_effort():
    seen = []
    inner = MagicMock()
    inner.job_request_options = lambda job, reasoning_effort=None: seen.append(reasoning_effort) or {}
    inner.query.return_value = "ok"

    def broken():
        raise RuntimeError("settings unavailable")

    assert _monitored(inner, effort=broken).query([]) == "ok"
    assert seen == [None]


def test_monitored_query_bounded_untouched():
    inner = MagicMock()
    inner.query_bounded.return_value = "ok"
    _monitored(inner).query_bounded([], timeout_seconds=1, max_tokens=5)
    assert inner.query_bounded.call_args.kwargs == {"timeout_seconds": 1, "max_tokens": 5}


def test_setting_is_db_runtime_and_unset_by_default():
    definition = SETTING_DEFINITIONS["maintenance_reasoning_effort"]
    assert definition.storage == STORAGE_RUNTIME_SETTING
    assert definition.default is None
