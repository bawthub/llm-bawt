"""TASK-2248 regressions for bounded text and URL normalization."""
from datetime import datetime, timezone
import json
from unittest.mock import Mock

import pytest

from llm_bawt.retrieval.context import _anchor_text
from llm_bawt.retrieval.find import FindError, serialize
from llm_bawt.search.multi import _dedupe_key
from llm_bawt.search.tavily_client import TavilyClient
from llm_bawt.working_memory import render


@pytest.mark.parametrize("character", ["\n", "\t", "\x00", '"', "\\", "é"])
def test_anchor_escaped_text_always_advances(character):
    content = character * 100
    offset = 0
    pieces = []
    while offset < len(content):
        text, next_offset = _anchor_text(content, offset, 24)
        assert text
        assert len(serialize(text)) <= 24
        pieces.append(text)
        if next_offset is None:
            break
        assert next_offset > offset
        offset = next_offset
    assert "".join(pieces) == content


def test_anchor_impossible_budget_fails_instead_of_repeating_cursor():
    with pytest.raises(FindError):
        _anchor_text("\x00", 0, 3)


def test_working_memory_budget_includes_omission_marker(monkeypatch):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    item = {"id": "a", "revision": 1, "status": "planned", "objective": "x" * 100,
            "updated_at": now.isoformat(), "last_session_id": "s"}
    line = render.title_line(item, "s", now)
    monkeypatch.setattr(render, "CHAR_BUDGET", len(line) + 1)
    result = render.render_items([item, dict(item, id="b")], "s", now)
    assert len(result) <= render.CHAR_BUDGET
    assert json.loads(result.splitlines()[-1])["omitted"] >= 1


def test_working_memory_exact_fit_needs_no_separator():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    item = {"id": "a", "revision": 1, "status": "planned", "objective": "x",
            "updated_at": now.isoformat(), "last_session_id": "s"}
    result = render.render_items([item], "s", now)
    assert json.loads(result)["id"] == "a"


@pytest.mark.parametrize("left,right", [
    ("https://example.com/Path", "https://example.com/path"),
    ("https://example.com/?token=ABC", "https://example.com/?token=abc"),
    ("https://User@example.com/path", "https://user@example.com/path"),
])
def test_url_dedupe_preserves_case_sensitive_components(left, right):
    assert _dedupe_key(left) != _dedupe_key(right)


def test_url_dedupe_normalizes_host_scheme_fragment_and_trailing_slash():
    assert _dedupe_key(" HTTPS://EXAMPLE.COM/Path/#section ") == _dedupe_key("https://example.com/Path")


@pytest.mark.parametrize("time_range,expected", [("d", "day"), ("w", "week"), ("m", "month"), ("y", "year")])
def test_tavily_news_forwards_time_range(monkeypatch, time_range, expected):
    monkeypatch.setattr("llm_bawt.search.tavily_client.is_tavily_available", lambda: True)
    client = TavilyClient(api_key="synthetic")
    client._client = Mock()
    client._client.search.return_value = {"results": []}
    assert client.search_news("test", time_range=time_range) == []
    assert client._client.search.call_args.kwargs["time_range"] == expected


def test_tavily_news_rejects_invalid_time_range(monkeypatch):
    monkeypatch.setattr("llm_bawt.search.tavily_client.is_tavily_available", lambda: True)
    client = TavilyClient(api_key="synthetic")
    client._client = Mock()
    with pytest.raises(ValueError):
        client.search_news("test", time_range="invalid")
    client._client.search.assert_not_called()
