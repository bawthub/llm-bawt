"""Real boundary contracts missed by the original outbox service doubles."""
import io
import json
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import pytest
from cryptography.fernet import Fernet
from llm_bawt.bot_access import service as service_codec


@pytest.fixture(autouse=True)
def service_signer(monkeypatch):
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(service_codec, "_get_fernet", lambda: signer)

from llm_bawt.mcp_server.client import MemoryClient
from llm_bawt.service.approval_continuations import _continuation_identity


def test_continuation_message_ids_fit_database_and_preserve_grant_identity():
    row = SimpleNamespace(continuation_id="approval-cont-" + "a" * 32)
    identity = _continuation_identity(row)
    assert identity == _continuation_identity(row)
    assert identity["user_message_id"] != identity["assistant_message_id"]
    for key in ("user_message_id", "assistant_message_id"):
        assert len(identity[key]) == 36
        assert str(UUID(identity[key])) == identity[key]
    assert identity["inter_bot_turn_id"] == "turn-" + row.continuation_id
    assert identity["inter_bot_bridge_request_id"] == "req_delivery_approval_" + row.continuation_id
    other = _continuation_identity(SimpleNamespace(continuation_id="approval-cont-" + "b" * 32))
    assert other["user_message_id"] != identity["user_message_id"]


def test_streaming_client_preserves_grant_bound_request_without_leaking():
    from llm_bawt.clients.agent_backend_client import AgentBackendClient
    from llm_bawt.models.message import Message

    captured = []

    class Backend:
        def stream_raw(self, prompt, config, **kwargs):
            captured.append(dict(config))
            yield "ok"

    agent = AgentBackendClient.__new__(AgentBackendClient)
    agent._bot_config = {"bot_id": "loopy"}
    agent._backend = Backend()
    agent.last_result = None
    identity = _continuation_identity(SimpleNamespace(continuation_id="approval-cont-test"))
    messages = [Message(role="user", content="test")]
    assert list(agent.stream_raw(messages, bridge_request_id=identity["inter_bot_bridge_request_id"])) == ["ok"]
    assert captured[0]["request_id"] == identity["inter_bot_bridge_request_id"]
    list(agent.stream_raw(messages))
    assert "request_id" not in captured[1]
    assert "request_id" not in agent._bot_config


def client():
    result = MemoryClient(SimpleNamespace(), server_url="http://unused.test",
                          bot_id="loopy", user_id="nick")
    result._ensure_initialized = lambda: None
    return result


def response(payload):
    return io.BytesIO(json.dumps({"jsonrpc": "2.0", "id": "test", "result": payload}).encode())


def test_mcp_tool_failure_surfaces_original_error_not_string_get():
    with patch("urllib.request.urlopen", return_value=response({
        "isError": True, "content": [{"type": "text", "text": "message persistence failed"}],
    })):
        with pytest.raises(RuntimeError, match="MCP tool messages_add failed: message persistence failed"):
            client().add_message("user", "test")


def test_successful_mcp_message_still_decodes():
    with patch("urllib.request.urlopen", return_value=response({
        "isError": False,
        "structuredContent": {"result": {"id": "message", "role": "user", "content": "test", "bot_id": "loopy"}},
    })):
        assert client().add_message("user", "test").id == "message"
