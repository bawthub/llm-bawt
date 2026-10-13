from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, text

from llm_bawt.utils.schema import SchemaBootstrapGuard


def test_schema_guard_is_single_flight_for_one_engine_and_scope() -> None:
    guard = SchemaBootstrapGuard()
    engine = create_engine("sqlite://")
    calls = 0
    calls_lock = threading.Lock()

    def bootstrap(_conn) -> None:
        nonlocal calls
        time.sleep(0.01)
        with calls_lock:
            calls += 1

    with ThreadPoolExecutor(max_workers=12) as pool:
        performed = list(
            pool.map(lambda _: guard.run(engine, "unit", bootstrap), range(24))
        )

    assert calls == 1
    assert performed.count(True) == 1
    assert performed.count(False) == 23


def test_schema_guard_retries_after_failure() -> None:
    guard = SchemaBootstrapGuard()
    engine = create_engine("sqlite://")
    attempts = 0

    def bootstrap(_conn) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first attempt fails")

    with pytest.raises(RuntimeError, match="first attempt fails"):
        guard.run(engine, "retry", bootstrap)

    assert guard.run(engine, "retry", bootstrap) is True
    assert guard.run(engine, "retry", bootstrap) is False
    assert attempts == 2


def test_schema_guard_initializes_a_second_engine_independently() -> None:
    guard = SchemaBootstrapGuard()
    first = create_engine("sqlite://")
    second = create_engine("sqlite://")
    calls: list[object] = []

    assert guard.run(first, "shared", lambda conn: calls.append(conn.engine)) is True
    assert guard.run(first, "shared", lambda conn: calls.append(conn.engine)) is False
    assert guard.run(second, "shared", lambda conn: calls.append(conn.engine)) is True

    assert calls == [first, second]


def test_steady_state_guard_calls_emit_zero_additional_ddl() -> None:
    guard = SchemaBootstrapGuard()
    engine = create_engine("sqlite://")
    ddl: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def record_ddl(_conn, _cursor, statement, _params, _context, _many) -> None:
        if statement.lstrip().upper().startswith(("CREATE ", "ALTER ", "DROP ")):
            ddl.append(statement)

    def bootstrap(conn) -> None:
        conn.execute(text("CREATE TABLE guarded_once (id INTEGER PRIMARY KEY)"))

    assert guard.run(engine, "ddl-count", bootstrap) is True
    for _ in range(10):
        assert guard.run(engine, "ddl-count", bootstrap) is False

    assert len(ddl) == 1


class FakeConnection:
    def __init__(self) -> None:
        self.dialect = SimpleNamespace(name="sqlite")
        self.statements: list[object] = []

    def execute(self, statement, *_args, **_kwargs):
        self.statements.append(statement)
        return SimpleNamespace()


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.commits = 0

    def begin(self):
        engine = self

        class Transaction:
            def __enter__(self):
                return engine.connection

            def __exit__(self, exc_type, _exc, _tb):
                if exc_type is None:
                    engine.commits += 1
                return False

        return Transaction()


def test_memory_schema_has_shared_and_per_sanitized_bot_scopes(monkeypatch) -> None:
    from llm_bawt.memory import postgresql

    engine = FakeEngine()
    parent_calls: list[int] = []
    partition_calls: list[str] = []
    monkeypatch.setattr(
        postgresql,
        "ensure_parent_tables",
        lambda _conn, dim: parent_calls.append(dim),
    )
    monkeypatch.setattr(
        postgresql,
        "ensure_bot_partitions",
        lambda _conn, bot_id: partition_calls.append(bot_id),
    )
    postgresql.PostgreSQLMemoryBackend._schema_guard.reset_for_tests()

    def backend(bot_id: str):
        value = postgresql.PostgreSQLMemoryBackend.__new__(
            postgresql.PostgreSQLMemoryBackend
        )
        value.engine = engine
        value.embedding_dim = 384
        value.bot_id = bot_id
        value.bot_id_sanitized = postgresql._sanitize_table_name(bot_id)
        return value

    backend("Alpha Bot")._ensure_tables_exist()
    backend("Alpha Bot")._ensure_tables_exist()
    backend("Beta Bot")._ensure_tables_exist()

    assert parent_calls == [384]
    assert partition_calls == ["alphabot", "betabot"]
    assert engine.commits == 3


def test_turn_log_orchestration_does_not_repeat_nested_schema(monkeypatch) -> None:
    from llm_bawt.service import changed_files_store, tool_call_store, turn_log_schema, turn_logs

    engine = FakeEngine()
    create_calls: list[tuple[str, ...]] = []

    def fake_create_all(*, bind, tables) -> None:
        assert bind is engine.connection
        create_calls.append(tuple(table.name for table in tables))

    monkeypatch.setattr(turn_log_schema.SQLModel.metadata, "create_all", fake_create_all)
    # The fake connection records DDL but is not an inspectable SQLAlchemy
    # connection. Report the current columns while testing bootstrap ordering.
    monkeypatch.setattr(
        changed_files_store,
        "inspect",
        lambda _conn: SimpleNamespace(get_columns=lambda _table: [
            {"name": name} for name in (
                "ignored_version", "in_repo", "commit_requested_at",
            )
        ]),
    )
    for store_type in (
        turn_logs.TurnLogStore,
        tool_call_store.ToolCallStore,
        changed_files_store.ChangedFilesStore,
    ):
        store_type._schema_guard.reset_for_tests()

    store = turn_logs.TurnLogStore.__new__(turn_logs.TurnLogStore)
    store.engine = engine
    store._ensure_tables_exist()
    store._ensure_tables_exist()

    assert create_calls == [
        ("tool_call_records", "tool_call_result_payloads"),
        ("turn_changed_files", "turn_diff_payloads"),
        ("turn_logs",),
    ]
