"""Hermetic adapter, scrubber and catalog contract checks."""
import json
from types import SimpleNamespace

import pytest

from llm_bawt.bot_access import BotAccessError
from llm_bawt.mcp_server import tool_call_tools as tools
from llm_bawt.retrieval.tool_call_scrub import MASK, arguments_view, scrub, scrub_text


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("key", ["token", "nestedSecret", "PASSWORD", "api_key",
                                     "Authorization", "Cookie", "credential"])
def test_nested_secret_keys(key):
    assert scrub({"nested": [{key: "synthetic"}]}, field_chars=100) == {
        "nested": [{key: MASK}]}


@pytest.mark.parametrize("value", [
    "Bearer synthetic-token", "sk-syntheticabcdefgh", "ghp_syntheticabcdef",
    "github_pat_synthetic", "xoxb-synthetic123", "eyJabc.eyJdef.signature",
    "TOKEN=synthetic", "password='some secret'", '"api_key": "synthetic"',
])
def test_free_text_patterns(value):
    assert "synthetic" not in scrub_text("run " + value)


def test_structure_and_per_field_clipping():
    view = arguments_view(json.dumps({"path": "/actual/file", "long": "a" * 200}),
                          field_chars=30, max_chars=300)
    assert view["value"]["path"] == "/actual/file"
    assert len(view["value"]["long"]) == 30 and view["truncated"]
    assert "synthetic" not in str(arguments_view('{"token": "synthetic',
                                               field_chars=100, max_chars=100))


@pytest.mark.anyio
async def test_adapter_requires_trusted_caller(monkeypatch):
    def deny(*args):
        raise BotAccessError("caller_unavailable")
    monkeypatch.setattr(tools, "_caller", deny)
    monkeypatch.setattr(tools, "_runtime", lambda: pytest.fail("caller precedes source"))
    assert json.loads(await tools.tool_calls())["reason"] == "caller_unavailable"


@pytest.mark.anyio
async def test_fresh_resolve_and_forward(monkeypatch):
    resolved, looked_up = [], []
    caller = object()
    monkeypatch.setattr(tools, "_caller", lambda *args: (caller, "trigger"))
    async def resolve(value):
        resolved.append(value)
        return "fresh access"
    def lookup(access, request, *, trigger_id):
        looked_up.append((access, request, trigger_id))
        return '{"status":"ok"}'
    monkeypatch.setattr(tools, "_runtime", lambda: (
        SimpleNamespace(resolve=resolve), SimpleNamespace(lookup=lookup)))
    for _ in range(2):
        assert await tools.tool_calls(id=42, include_result=True) == '{"status":"ok"}'
    assert resolved == [caller, caller]
    assert all(access == "fresh access" and request.id == 42 and request.include_result
               and trigger == "trigger" for access, request, trigger in looked_up)


@pytest.mark.anyio
async def test_sanitized_failure_and_bad_time(monkeypatch):
    monkeypatch.setattr(tools, "_caller", lambda *args: ("caller", None))
    monkeypatch.setattr(tools, "_runtime", lambda: (_ for _ in ()).throw(RuntimeError("secret detail")))
    assert json.loads(await tools.tool_calls())["reason"] == "source_unavailable"
    assert json.loads(await tools.tool_calls(since="bad"))["reason"] == "invalid_request"


@pytest.mark.parametrize("values", [
    {"id": 1, "turn_id": "turn"}, {"id": 1, "since": 1},
    {"id": 1, "errors_only": True}, {"include_result": True}, {"limit": 0},
    {"max_chars": 9000}, {"before": "x" * 200},
])
def test_invalid_combinations_fail_before_io(values):
    from llm_bawt.retrieval.tool_calls import CallRequest, ToolCallLookup
    with pytest.raises(ValueError):
        ToolCallLookup(None).lookup(None, CallRequest(**values))
