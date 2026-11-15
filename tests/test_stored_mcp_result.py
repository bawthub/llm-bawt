"""Route and recovery share the legacy stored-result decode contract."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_bawt.approval_models import (
    EXEC_SUCCEEDED, KIND_MCP, ToolApprovalRequest, decode_stored_mcp_result,
)
from llm_bawt.service.recovery import purge_ops
from llm_bawt.service.routes.approval_policies import _stored_mcp_resolution


@pytest.mark.parametrize(("encoded", "decoded"), [
    (None, None), ("", None), ("null", None), ("false", False), ("0", 0),
    ('{"text": "café 雪"}', {"text": "café 雪"}), ("[1, null]", [1, None]),
    ('"text"', "text"), ("not JSON 雪", "not JSON 雪"), (" ", " "),
    ("{broken", "{broken"),
])
def test_decode_and_both_callers_preserve_legacy_values(monkeypatch, encoded, decoded):
    row = ToolApprovalRequest(
        id="r", bot_id="test-bot", user_id="test-user", turn_id="turn",
        tool_name="tool", tool_arguments_json="{}", subject="test", grant_key="g",
        request_kind=KIND_MCP, status="approved", execution_state=EXEC_SUCCEEDED,
        result_json=encoded, result_is_error=False,
    )
    session = MagicMock()
    session.__enter__.return_value.exec.return_value.all.return_value = [row]
    monkeypatch.setattr(purge_ops, "Session", lambda _engine: session)
    assert decode_stored_mcp_result(encoded) == decoded
    assert _stored_mcp_resolution(row)["result"] == decoded
    assert purge_ops.purged_resolutions(object(), ["r"])[0]["result"] == decoded


def test_decode_does_not_hide_non_json_errors():
    with pytest.raises(TypeError):
        decode_stored_mcp_result(object())


def test_empty_purge_does_not_open_a_session(monkeypatch):
    def fail(_engine):
        raise AssertionError("should not read")
    monkeypatch.setattr(purge_ops, "Session", fail)
    assert purge_ops.purged_resolutions(object(), []) == []
