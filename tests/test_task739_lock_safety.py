from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _SearchConnection:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        self.engine.in_connection = True
        return self

    def __exit__(self, *_args):
        self.engine.in_connection = False

    def execute(self, _statement, _params):
        return _Rows(self.engine.rows)


class _SearchEngine:
    def __init__(self, rows):
        self.rows = rows
        self.in_connection = False

    def connect(self):
        return _SearchConnection(self)


def _message(message_id: str, source: str):
    return SimpleNamespace(
        id=message_id,
        role="assistant",
        content="locking regression",
        timestamp=1.0,
        session_id="session",
        author_entity_type="bot",
        author_entity_id=source,
        source=source,
        rank=0.9,
        total=2,
    )


def test_cross_bot_search_closes_select_before_author_hydration() -> None:
    from llm_bawt.mcp_server.message_search import CrossBotMessageSearcher

    engine = _SearchEngine([_message("one", "snark"), _message("two", "al")])
    hydrated_sources: list[str] = []

    def hydrate(rows, source):
        assert engine.in_connection is False
        hydrated_sources.append(source)
        return [{**row, "author": {"entity_id": source}} for row in rows]

    searcher = CrossBotMessageSearcher(engine, hydrate)
    results = searcher.search_fts(
        "locking regression",
        n_results=10,
        role_filter=None,
        sort_by="relevance",
        since=None,
        until=None,
        bot_id=None,
        excluded_bot_ids=set(),
    )

    assert [row["id"] for row in results] == ["one", "two"]
    assert hydrated_sources == ["snark", "al"]


def test_cold_history_read_constructs_manager_without_schema_ddl(monkeypatch) -> None:
    from llm_bawt.mcp_server import storage as storage_module
    from llm_bawt.memory import postgresql

    provision_values: list[bool] = []
    worker_threads: list[int] = []

    class Manager:
        def __init__(self, *, config, bot_id, provision_schema=True):
            provision_values.append(provision_schema)

        def get_messages(self, since_minutes=None):
            worker_threads.append(threading.get_ident())
            return []

        def ensure_schema(self):
            raise AssertionError("a GET attempted schema provisioning")

    monkeypatch.setattr(storage_module, "get_shared_engine", lambda _config: object())
    monkeypatch.setattr(postgresql, "PostgreSQLShortTermManager", Manager)
    storage = storage_module.MemoryStorage(config=SimpleNamespace())

    assert asyncio.run(storage.get_messages(bot_id="snark", limit=10)) == []
    assert provision_values == [False]
    assert len(worker_threads) == 1
    assert worker_threads[0] != threading.get_ident()


def test_async_cross_bot_search_offloads_synchronous_database_work(monkeypatch) -> None:
    from llm_bawt.mcp_server import message_search, storage as storage_module

    worker_threads: list[int] = []

    class Searcher:
        def __init__(self, _engine, _hydrator):
            pass

        def search_fts(self, _query, **_kwargs):
            worker_threads.append(threading.get_ident())
            return []

    monkeypatch.setattr(storage_module, "get_shared_engine", lambda _config: object())
    monkeypatch.setattr(message_search, "CrossBotMessageSearcher", Searcher)
    storage = storage_module.MemoryStorage(config=SimpleNamespace())

    assert asyncio.run(storage.search_all_messages("locking")) == []
    assert len(worker_threads) == 1
    assert worker_threads[0] != threading.get_ident()


def test_read_backend_can_be_promoted_once_for_a_later_write(monkeypatch) -> None:
    from llm_bawt.mcp_server import storage as storage_module

    constructed: list[bool] = []
    ensured: list[str] = []

    class Backend:
        def __init__(self, *, config, bot_id, provision_schema=True):
            constructed.append(provision_schema)
            self.bot_id = bot_id

        def ensure_schema(self):
            ensured.append(self.bot_id)

    monkeypatch.setattr(storage_module, "get_shared_engine", lambda _config: object())
    monkeypatch.setattr(storage_module, "PostgreSQLMemoryBackend", Backend)
    storage = storage_module.MemoryStorage(config=SimpleNamespace())

    first = storage.get_backend("snark")
    second = storage.get_backend("snark", provision_schema=True)

    assert first is second
    assert constructed == [False]
    assert ensured == ["snark"]


def test_existing_partitions_execute_catalog_reads_only() -> None:
    from llm_bawt.memory.postgresql import ensure_bot_partitions

    statements: list[str] = []

    class Result:
        def scalar(self):
            return "already_exists"

    class Connection:
        def execute(self, statement, _params=None):
            statements.append(str(statement))
            return Result()

    ensure_bot_partitions(Connection(), "snark")

    assert len(statements) == 3
    assert all("to_regclass" in statement for statement in statements)
    assert not any("CREATE" in statement or "ALTER" in statement for statement in statements)


def test_database_connections_receive_deadman_timeouts() -> None:
    from llm_bawt.utils.db import _configure_dbapi_connection

    executed: list[str] = []

    class Cursor:
        def execute(self, statement):
            executed.append(statement)

        def close(self):
            executed.append("CLOSE")

    connection = SimpleNamespace(cursor=lambda: Cursor())
    _configure_dbapi_connection(connection)

    assert executed == [
        "SET timezone = 'UTC'",
        "SET lock_timeout = '5s'",
        "SET statement_timeout = '120s'",
        "SET idle_in_transaction_session_timeout = '30s'",
        "CLOSE",
    ]


def test_codex_auth_broker_retry_uses_module_logger(monkeypatch, tmp_path: Path) -> None:
    import httpx
    from codex_bridge import auth_broker

    calls = 0

    def get(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("app is still starting")
        return SimpleNamespace(
            is_error=False,
            json=lambda: {"tokens": {"access_token": "test"}},
        )

    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app:8642")
    monkeypatch.setenv("CODEX_AUTH_BROKER_ATTEMPTS", "2")
    monkeypatch.setenv("CODEX_AUTH_BROKER_RETRY_SECONDS", "0")
    monkeypatch.setattr(httpx, "get", get)
    auth_file = tmp_path / "auth.json"

    assert auth_broker.materialize_auth_from_broker(auth_file) is True
    assert calls == 2
    assert auth_file.exists()
    assert auth_file.stat().st_mode & 0o777 == 0o600
