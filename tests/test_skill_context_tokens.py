"""Skill badges describe the installed skill, never the tiny launcher ACK."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from claude_agent_sdk import UserMessage
from claude_agent_sdk.types import ToolResultBlock
from sqlmodel import Session, SQLModel, create_engine, select

from agent_bridge.tool_results import ToolResultPayload, count_result_tokens
from claude_code_bridge.send_stream import ClaudeStreamMixin
from claude_code_bridge.skill_context import skill_context_tokens
from llm_bawt.service.routes.turn_logs import _records_to_calls
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator


def test_skill_size_reads_installed_file_only(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    skill = root / "llm-bawt"
    skill.mkdir(parents=True)
    text = "# llm-bawt\n" + "This is skill context.\n" * 150
    (skill / "SKILL.md").write_text(text)
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(root))
    assert skill_context_tokens({"skill": "llm-bawt"}) == count_result_tokens(text)
    assert skill_context_tokens({"skill": "missing"}) is None
    assert skill_context_tokens({"skill": "../secret"}) is None
    assert skill_context_tokens({"skill": "/etc/passwd"}) is None
    assert skill_context_tokens({"skill": "llm-bawt:extra"}) is None
    (root / "linked").symlink_to(tmp_path)
    assert skill_context_tokens({"skill": "linked"}) is None
    assert skill_context_tokens({"skill": "llm-bawt"}, skill_bundle="tenant") is None


def test_multiroot_selection_and_frontmatter(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    skill = second / "legacy"
    skill.mkdir(parents=True)
    body = "# Actual skill body\nDo the right thing.\n"
    (skill / "SKILL.md").write_text("---\nname: legacy\ndescription: hidden\n---\n" + body)
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", f"{first}:{second}")
    assert skill_context_tokens({"skill": "legacy"}) == count_result_tokens(body)
    assert skill_context_tokens({"skill": "legacy"}, skill_bundle="public-bundle") is None


def test_bridge_stamps_successful_skill_result_only(tmp_path, monkeypatch):
    skill = tmp_path / "skills" / "llm-bawt"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("Loaded skill text " * 150)
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(skill.parent))
    emitter = SimpleNamespace(_publish_event=Mock(), _is_image_result_tool=lambda _name: False,
                              _host_parent=lambda _msg, _ledger: None)
    for tool_name, failed, bundle in [
        ("Skill", False, None), ("Skill", True, None), ("Bash", False, None),
        ("Skill", False, "public-bundle"),
    ]:
        message = UserMessage(content=[ToolResultBlock(
            tool_use_id="tool", content="Launching skill: llm-bawt", is_error=failed,
        )])
        asyncio.run(ClaudeStreamMixin._on_user_message_tool_results(
            emitter, message, request_id="req", session_key="session", seq=0,
            tool_names_by_id={"tool": tool_name},
            tool_arguments_by_id={"tool": {"skill": "llm-bawt"}},
            turn_screenshot_assets=[], screenshot_artifacts_by_tool_use_id={},
            skill_bundle=bundle,
        ))
        result = emitter._publish_event.call_args.kwargs["tool_result_payload"]
        assert result.get("skill_context_tokens") == (
            skill_context_tokens({"skill": "llm-bawt"})
            if tool_name == "Skill" and not failed and bundle is None else None
        )


def test_skill_size_persists_in_result_meta_without_replacing_ack_count():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__],
    )
    payload = ToolResultPayload.from_value("Launching skill: llm-bawt")
    event = {
        "turn_id": "turn", "call_id": "call", "tool_use_id": "tool",
        "bot_id": "loopy", "user_id": "nick", "tool_name": "Skill",
        "result": payload.preview,
        "tool_result_payload": {**payload.to_dict(), "skill_context_tokens": 2400},
        "ts": 123.0, "is_error": False,
    }
    public = ToolEventCoordinator(engine).end(event)
    assert public["result_meta"]["total_tokens"] == payload.total_tokens
    assert public["result_meta"]["skill_context_tokens"] == 2400
    # A rolling-deploy replay without the optional size must not erase a prior measurement.
    coordinator = ToolEventCoordinator(engine)
    replay = coordinator.end({**event, "tool_result_payload": payload.to_dict()})
    assert "skill_context_tokens" not in replay["result_meta"]
    with Session(engine) as session:
        row = session.exec(select(ToolCallRecord)).one()
        assert row.skill_context_tokens == 2400
        reloaded = _records_to_calls([row])[0]["result_meta"]
        assert reloaded["skill_context_tokens"] == 2400
        assert reloaded["total_tokens"] == payload.total_tokens


def test_failed_or_unmeasured_skills_do_not_report_context_size():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__],
    )
    coordinator = ToolEventCoordinator(engine)
    for call_id, is_error, reported in [("failed", True, 2400), ("unknown", False, None)]:
        payload = ToolResultPayload.from_value("Launching skill")
        public = coordinator.end({
            "turn_id": "turn", "call_id": call_id, "tool_use_id": call_id,
            "bot_id": "loopy", "user_id": "nick", "tool_name": "Skill",
            "result": payload.preview,
            "tool_result_payload": {**payload.to_dict(), "skill_context_tokens": reported},
            "ts": 123.0, "is_error": is_error,
        })
        assert "skill_context_tokens" not in public["result_meta"]
    with Session(engine) as session:
        assert all(row.skill_context_tokens is None for row in session.exec(select(ToolCallRecord)).all())
