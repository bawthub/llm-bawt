"""TASK-857: background job model resolution must not borrow a bot identity.

Regression: ``_get_background_client`` resolved the global job model AS nova
(``bot_id="nova"`` + nova's harness). Once nova became a ``claude-proxy``
agent, every job model resolved to ``type="claude-code"`` and the openai/grok
gate rejected it — summaries and extraction silently fell back to heuristics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_bawt.service.background_service import BackgroundService


def _make_service(model_def: dict | None, preferred: str = "grok-4.3") -> BackgroundService:
    svc = BackgroundService.__new__(BackgroundService)
    svc._bg_client_cache = {}
    import threading
    svc._bg_client_lock = threading.Lock()
    config = MagicMock()
    config.resolve_model = MagicMock(return_value=model_def)
    svc.config = config
    return svc


@pytest.fixture
def job_model(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.runtime_settings.resolve_job_model",
        lambda config, key: "grok-4.3" if key == "maintenance_model" else None,
    )


def test_resolves_with_chat_harness_and_no_bot(job_model):
    svc = _make_service({"type": "grok", "model_id": "grok-4.3", "model_key": "grok-4.3"})
    fake_client = MagicMock()
    with patch("llm_bawt.clients.grok_client.GrokClient", return_value=fake_client) as grok_cls, \
         patch("llm_bawt.service.instance_manager.BotManager") as bot_manager:
        client, alias = svc._resolve_background_client()

    assert client is fake_client
    assert alias == "grok-4.3"
    # Resolved directly against the catalog with the plain-client harness…
    svc.config.resolve_model.assert_called_once_with("grok-4.3", harness="chat", default=None)
    # …and never through a bot identity.
    bot_manager.assert_not_called()
    grok_cls.assert_called_once()
    assert grok_cls.call_args.kwargs["model"] == "grok-4.3"


def test_cached_by_canonical_model_key(job_model):
    svc = _make_service({"type": "openai", "model_id": "gpt-x", "model_key": "gpt-x"})
    fake_client = MagicMock()
    with patch("llm_bawt.clients.openai_client.OpenAIClient", return_value=fake_client) as cls:
        first = svc._resolve_background_client()
        second = svc._resolve_background_client()
    assert first == (fake_client, "gpt-x") == second
    assert cls.call_count == 1


def test_agent_only_endpoint_still_rejected(job_model):
    """A model whose chat-harness type is not a plain client still falls back."""
    svc = _make_service({"type": "claude-code", "model_id": "x", "model_key": "x"})
    assert svc._resolve_background_client() == (None, None)


def test_unknown_model_falls_back(job_model):
    svc = _make_service(None)
    assert svc._resolve_background_client() == (None, None)


def test_no_job_model_configured(monkeypatch):
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda c, k: None)
    svc = _make_service({"type": "grok"})
    assert svc._resolve_background_client() == (None, None)
    svc.config.resolve_model.assert_not_called()


# --- summarize_session_with_client -------------------------------------------

def _session(text: str = "hello") -> Any:
    from llm_bawt.memory.summarization import Session
    return Session(
        session_id="s1",
        start_timestamp=0,
        end_timestamp=1,
        messages=[{"role": "user", "content": text, "timestamp": 0}],
        message_ids=["1"],
    )


def _render_stub(monkeypatch):
    resolver = MagicMock()
    resolver.render.return_value = "PROMPT"
    monkeypatch.setattr("llm_bawt.prompt_registry.PromptResolver", lambda cfg: resolver)
    return resolver


def test_summarize_with_client_happy_path(monkeypatch):
    from llm_bawt.memory.summarization import summarize_session_with_client
    _render_stub(monkeypatch)
    client = MagicMock()
    client.query.return_value = "  Summary: a real summary.  "
    assert summarize_session_with_client(_session(), client, config=None) == "Summary: a real summary."
    kwargs = client.query.call_args.kwargs
    assert kwargs["max_tokens"] == 320 and kwargs["stream"] is False
    assert [m.role for m in kwargs["messages"]] == ["system", "user"]


def test_summarize_with_client_none_client():
    from llm_bawt.memory.summarization import summarize_session_with_client
    assert summarize_session_with_client(_session(), None) is None


@pytest.mark.parametrize("bad", ["Error: boom", "", "CUDA error: out of memory"])
def test_summarize_with_client_rejects_error_bodies(monkeypatch, bad):
    from llm_bawt.memory.summarization import summarize_session_with_client
    _render_stub(monkeypatch)
    client = MagicMock()
    client.query.return_value = bad
    assert summarize_session_with_client(_session(), client) is None


def test_summarize_with_client_swallows_exception(monkeypatch):
    from llm_bawt.memory.summarization import summarize_session_with_client
    _render_stub(monkeypatch)
    client = MagicMock()
    client.query.side_effect = RuntimeError("upstream down")
    assert summarize_session_with_client(_session(), client) is None
