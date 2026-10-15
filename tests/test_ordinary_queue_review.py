"""Independent TASK-1082 end-to-end admission regression probes (no live I/O)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_admission import AdmissionClaim
from tests.test_nonstream_turn_execution import _service


def test_retry_of_queued_new_does_not_rotate_durable_thread_twice(monkeypatch):
    """A retried outbox claim must reuse /new's first durable session.

    The outbox retries failed deterministic deliveries. Model two valid claims
    of that same delivery, with a setup failure after rotation on each attempt.
    Admission is mocked here; the real request handler and rotation helper run.
    """
    service = _service(Mock(side_effect=AssertionError("no bridge execution")))
    llm = service._get_llm_bawt.return_value
    service._resolve_active_thread_binding.side_effect = lambda *_: {
        "thread_session_id": f"session-{service._rotate_chat_session.call_count}",
    }
    service._rotate_chat_session = Mock(side_effect=["session-1", "session-2"])
    # Exercise the real rotation guard, not the fixture's Mock.
    del service._maybe_rotate_agent_session
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed", Mock(return_value=[]))
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
        AsyncMock(return_value=("/new", [], [], None)),
    )
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.validate_inter_bot_claim",
        lambda *_: AuthorReference.user("nick"),
    )
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.turn_admission_claim",
        lambda *_: AdmissionClaim(owner="current-claim", fence=lambda _: True),
    )
    monkeypatch.setattr(
        "llm_bawt.task_turn_context.mint_for_agent_turn",
        Mock(side_effect=RuntimeError("transient setup error after rotation")),
    )
    request = ChatCompletionRequest(
        bot_id="test-review", user="nick", model="test-model", stream=True,
        messages=[ChatMessage(role="user", content="/new")],
        user_message_id="00000000-0000-4000-8000-000000000001",
        assistant_message_id="00000000-0000-4000-8000-000000000002",
        inter_bot_delivery_id="delivery-test-review",
        inter_bot_turn_id="turn-delivery-test-review",
        inter_bot_bridge_request_id="req_delivery_test_review",
        inter_bot_claim_token="claim-a",
    )

    async def attempt():
        async for _ in service.chat_completion_stream(request):
            pass

    for token in ("claim-a", "claim-b"):
        request.inter_bot_claim_token = token
        with pytest.raises(RuntimeError, match="transient setup error"):
            asyncio.run(attempt())
    assert service._persist_turn_log.call_count == 2
    assert service._update_turn_log.call_count == 2
    llm.execute_llm_query.assert_not_called()
    assert service._rotate_chat_session.call_count == 1, (
        "same durable /new delivery rotated a second time on retry"
    )
