"""Hermetic proofs of AgentContextStore fixture ordering and complete teardown."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tests import test_agent_context as context_tests
from llm_bawt.memory import postgresql


def test_context_rejects_bad_target_before_store_constructor(monkeypatch):
    constructor = Mock(side_effect=AssertionError("store constructed"))
    monkeypatch.setattr(context_tests, "require_test_db_config",
                        Mock(side_effect=RuntimeError("refused operational DB")))
    monkeypatch.setattr(context_tests, "AgentContextStore", constructor)
    with pytest.raises(RuntimeError, match="refused operational DB"):
        next(context_tests.context_store.__wrapped__())
    constructor.assert_not_called()


def test_context_teardown_drops_all_three_test_owned_children(monkeypatch):
    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value
    store = SimpleNamespace(engine=engine)
    monkeypatch.setattr(context_tests, "require_test_db_config", lambda: object())
    monkeypatch.setattr(context_tests, "AgentContextStore", lambda config: store)
    monkeypatch.setattr(postgresql, "ensure_bot_partitions", Mock())
    profiles = SimpleNamespace(engine=engine, upsert=Mock())
    monkeypatch.setattr("llm_bawt.runtime_settings.BotProfileStore", lambda config: profiles)
    fixture = context_tests.context_store.__wrapped__()
    next(fixture)
    with pytest.raises(StopIteration):
        next(fixture)
    sql = [str(call.args[0]) for call in conn.execute.call_args_list]
    drops = [statement for statement in sql if statement.startswith("DROP TABLE")]
    assert len(drops) == 3
    assert all(any(statement.startswith(f"DROP TABLE IF EXISTS {parent}_p_testcontext")
                   for statement in drops) for parent in postgresql.PARENT_TABLES)
    assert sql.index("DELETE FROM turn_logs WHERE bot_id=:bot") < sql.index(drops[0])
    assert sql.index(drops[-1]) < sql.index("DELETE FROM sessions WHERE bot_id=:bot")
