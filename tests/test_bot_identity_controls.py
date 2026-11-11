"""TASK-1373 identity controls (from Codex review probe)."""
from types import SimpleNamespace

import pytest

from llm_bawt.utils.bot_identity import (
    MissingBotIdentity,
    configured_default_bot_id,
    require_bot_id,
)


@pytest.mark.parametrize("value", [None, "", " \t"])
def test_missing_identity_rejected(value):
    with pytest.raises(MissingBotIdentity):
        require_bot_id(value, "probe")


def test_identity_normalized():
    assert require_bot_id(" ALPHA ", "probe") == "alpha"
    assert configured_default_bot_id(SimpleNamespace(DEFAULT_BOT=" ALPHA ")) == "alpha"


@pytest.mark.parametrize("owner", ["backend", "short_term", "history", "summarizer", "task"])
def test_namespace_owners_reject_omission(owner):
    from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager
    from llm_bawt.memory.summarization import HistorySummarizer
    from llm_bawt.service.tasks import Task, TaskType
    from llm_bawt.utils.history import HistoryManager

    constructors = {
        "backend": lambda: PostgreSQLMemoryBackend(None),
        "short_term": lambda: PostgreSQLShortTermManager(None),
        "history": lambda: HistoryManager(None, None),
        "summarizer": lambda: HistorySummarizer(None),
        "task": lambda: Task(task_type=next(iter(TaskType)), payload={}, user_id="system"),
    }
    with pytest.raises(MissingBotIdentity):
        constructors[owner]()


@pytest.mark.parametrize("explicit_bot", [True, False])
def test_remote_cli_without_local_db_reaches_service(monkeypatch, explicit_bot):
    import sys

    from llm_bawt import bots
    from llm_bawt.cli import app

    config = SimpleNamespace(
        DEFAULT_BOT="alpha",
        DEFAULT_USER="test-user",
        DEFAULT_MODEL_ALIAS=None,
        MODELS_CONFIG_PATH="unused",
        USE_SERVICE=True,
    )
    monkeypatch.setattr(bots, "_check_reload", lambda: None)
    monkeypatch.setattr(bots, "BUILTIN_BOTS", {})
    monkeypatch.setattr(app, "Config", lambda: config)
    monkeypatch.setattr(app, "ensure_default_user", lambda *_: None)
    monkeypatch.setattr(app, "_is_service_mode", lambda *_: True)
    monkeypatch.setattr(app.LogConfig, "configure", lambda **_: None)
    argv = ["llm", "--service", "--user", "test-user"]
    if explicit_bot:
        argv += ["--bot", "alpha"]
    monkeypatch.setattr(sys, "argv", [*argv, "hello"])
    calls = []
    monkeypatch.setattr(app, "run_app", lambda args, cfg, model: calls.append((cfg.DEFAULT_BOT, model)))
    with pytest.raises(SystemExit) as exit_info:
        app.main()
    assert exit_info.value.code == 0
    assert calls == [("alpha", None)]
