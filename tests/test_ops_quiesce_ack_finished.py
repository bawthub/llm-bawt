"""TASK-2121: the "finished" quiesce ack and Nick's verbatim definition of it.

The resume behavior itself is in tests/test_ops_quiesce_resume_pg.py.
"""
from __future__ import annotations

import asyncio

import pytest

from llm_bawt.mcp_server import ops_tools
from llm_bawt.mcp_server.registry import mcp
from llm_bawt.ops.quiesce import ACK_STATUSES, AckNoteRequired, record_ack
from llm_bawt.ops.quiesce_prompts import (DEFAULT_QUIESCE_REQUEST, FINISHED_DEFINITION,
                                          FINISHED_NOTE_RULE)


def test_finished_is_an_ack_status_beside_the_old_two():
    assert set(ACK_STATUSES) == {"finished", "stopping", "cannot_stop"}


@pytest.mark.parametrize("note", [None, "", "  \n "])
def test_record_ack_refuses_finished_without_a_note(note):
    # Checked before any database work, so no engine is needed.
    with pytest.raises(AckNoteRequired):
        record_ack(None, bot_id="b", turn_id="t", status="finished", note=note)


@pytest.mark.parametrize("note", [None, "", "   "])
def test_the_tool_refuses_finished_without_a_note(note, monkeypatch):
    def no_turn_lookup():
        raise AssertionError("refused before the turn is looked up")

    monkeypatch.setattr(ops_tools, "current_task_turn_context", no_turn_lookup)
    with pytest.raises(AckNoteRequired, match="non-empty note"):
        asyncio.run(ops_tools.ops_quiesce_ack("finished", note))


def test_the_definition_is_verbatim_in_the_quiesce_request():
    assert FINISHED_DEFINITION in DEFAULT_QUIESCE_REQUEST
    assert FINISHED_NOTE_RULE in DEFAULT_QUIESCE_REQUEST
    assert DEFAULT_QUIESCE_REQUEST.startswith("[Ops quiesce request:")
    assert DEFAULT_QUIESCE_REQUEST.endswith("]")
    # Nick's wording, spot-checked so an edit to the constant is deliberate.
    for wording in ('if this bot were never woken again, nothing would be lost, half-applied or forgotten',
                    'When in doubt: "stopping". A wasted resume turn is cheap; silently dropped work is not.',
                    'Never ack "finished" to avoid being resumed'):
        assert wording in FINISHED_DEFINITION


def test_the_definition_is_verbatim_in_the_tool_schema():
    [tool] = [t for t in asyncio.run(mcp.list_tools()) if t.name == "ops_quiesce_ack"]
    status = tool.inputSchema["properties"]["status"]["description"]
    assert FINISHED_DEFINITION in status and FINISHED_NOTE_RULE in status
    assert "finished" in tool.description and "stopping" in tool.description
