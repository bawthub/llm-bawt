"""Hermetic guard tests: never connect to PostgreSQL here."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from llm_bawt import inter_bot_delivery
from llm_bawt.service import turn_logs
from tests import pg_isolation


def test_operational_db_refuses_before_any_store_constructor(monkeypatch):
    config_factory = Mock(return_value=SimpleNamespace(POSTGRES_DATABASE="askllm"))
    delivery_store = Mock(side_effect=AssertionError("delivery store constructed"))
    turn_store = Mock(side_effect=AssertionError("turn store constructed"))
    monkeypatch.setattr(pg_isolation, "Config", config_factory)
    monkeypatch.setattr(inter_bot_delivery, "InterBotDeliveryStore", delivery_store)
    monkeypatch.setattr(turn_logs, "TurnLogStore", turn_store)

    for constructor in (inter_bot_delivery.InterBotDeliveryStore, turn_logs.TurnLogStore):
        with pytest.raises(pytest.fail.Exception, match=(
            "refusing to run integration DB test against operational database 'askllm'; "
            "set LLM_BAWT_POSTGRES_DATABASE=<x>_test explicitly"
        )):
            pg_isolation.isolated_pg_store(constructor)
        constructor.assert_not_called()
    assert config_factory.call_count == 2  # one Config per guarded construction


def test_dedicated_test_db_returns_validated_config(monkeypatch):
    config = SimpleNamespace(POSTGRES_DATABASE="llm_bawt_test")
    config_factory = Mock(return_value=config)
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.execute.return_value.scalar_one.return_value = "llm_bawt_test"
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", "llm_bawt_test")
    monkeypatch.setattr(pg_isolation, "Config", config_factory)
    monkeypatch.setattr(pg_isolation, "build_engine_from_config", Mock(return_value=engine))
    assert pg_isolation.require_test_db_config() is config
    engine.dispose.assert_called_once()
    config_factory.assert_called_once_with()


def test_dotenv_only_test_db_is_not_implicit_opt_in(monkeypatch):
    constructor = Mock()
    monkeypatch.delenv("LLM_BAWT_POSTGRES_DATABASE", raising=False)
    monkeypatch.setattr(pg_isolation, "Config", lambda: SimpleNamespace(POSTGRES_DATABASE="llm_bawt_test"))
    monkeypatch.setattr(pg_isolation, "build_engine_from_config", constructor)
    with pytest.raises(pytest.fail.Exception, match="explicitly"):
        pg_isolation.require_test_db_config()
    constructor.assert_not_called()


def test_mismatched_identity_fails_before_store_construction(monkeypatch):
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.execute.return_value.scalar_one.return_value = "askllm"
    store = Mock()
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", "llm_bawt_test")
    monkeypatch.setattr(pg_isolation, "Config", lambda: SimpleNamespace(POSTGRES_DATABASE="llm_bawt_test"))
    monkeypatch.setattr(pg_isolation, "build_engine_from_config", Mock(return_value=engine))
    with pytest.raises(pytest.fail.Exception, match="did not select"):
        pg_isolation.isolated_pg_store(store)
    store.assert_not_called()
    engine.dispose.assert_called_once()
