"""Source-owned schema facts that must agree across fresh and additive bootstrap."""

from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, String, Text

from llm_bawt.profiles import EntityProfile
from llm_bawt.runtime_settings import BotProfile, BotProfileStore
from llm_bawt.service.chat_pending_questions import PendingQuestion
from llm_bawt.service.tool_call_store import ToolCallRecord
from llm_bawt.service.turn_log_schema import bootstrap_turn_log_schema


def test_fresh_schema_column_contract():
    bot = BotProfile.__table__.c
    assert isinstance(bot.endpoint_id.type, BigInteger)
    for name, default in (
        ("tts_mode", "false"),
        ("include_summaries", "true"),
        ("include_in_global_search", "true"),
    ):
        column = bot[name]
        assert isinstance(column.type, Boolean)
        assert not column.nullable
        assert str(column.server_default.arg) == default

    summary = EntityProfile.__table__.c.summary
    assert isinstance(summary.type, Text) and summary.nullable
    origin = PendingQuestion.__table__.c.origin_harness
    assert isinstance(origin.type, String) and origin.type.length == 32
    assert not origin.nullable and str(origin.server_default.arg) == "'claude'"
    payload = ToolCallRecord.__table__.c.result_payload_available
    assert not payload.nullable and str(payload.server_default.arg) == "false"
    for name in ("tool_use_id", "parent_tool_use_id", "approval_request_id", "approval_status"):
        column = ToolCallRecord.__table__.c[name]
        assert column.type.length is None  # SQLModel AutoString compiles to unbounded VARCHAR


def test_additive_bot_flags_are_nonnullable_with_matching_defaults():
    class Recorder:
        statements: list[str] = []

        def execute(self, statement):
            self.statements.append(str(statement))

    recorder = Recorder()
    BotProfileStore._migrate_add_columns(None, recorder)
    for name, default in (
        ("tts_mode", "FALSE"),
        ("include_summaries", "TRUE"),
        ("include_in_global_search", "TRUE"),
    ):
        assert (
            f"ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS {name} "
            f"BOOLEAN NOT NULL DEFAULT {default}"
        ) in recorder.statements


def test_additive_tool_call_identifiers_are_unbounded(monkeypatch):
    from llm_bawt.service import turn_log_schema

    monkeypatch.setattr(turn_log_schema.SQLModel.metadata, "create_all", lambda **kwargs: None)

    class Recorder:
        statements: list[str] = []

        def execute(self, statement):
            self.statements.append(str(statement))

    recorder = Recorder()
    bootstrap_turn_log_schema(recorder)
    for name in ("tool_use_id", "parent_tool_use_id", "approval_request_id", "approval_status"):
        assert f"ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS {name} VARCHAR" in recorder.statements
