"""DB/Redis driver text never reaches a caller-visible frame or field (TASK-1611).

TASK-1565 sanitized catch-all HTTPException bodies. These cover the paths it
left: in-band SSE error frames, ``/v1/health/checks`` reasons and background
task results. A driver error anywhere in the exception chain is replaced by
generic text plus the type and logged in full; a domain error keeps its
message, which is written for the caller.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.service import health_checks
from llm_bawt.service.chat_response_adapter import completion_stream
from llm_bawt.service.chat_streaming import request_failed_events
from llm_bawt.service.http_errors import (
    client_error_text,
    names_infrastructure,
    safe_error_text,
)
from llm_bawt.service.runtime_health import LoopHealth
from llm_bawt.service.schemas import ChatCompletionRequest
from llm_bawt.service.turn_admission import AdmissionUnavailable

LEAK = 'connection to server at "db.internal" (192.0.2.7) failed for user "tenant_user"'
HTTP_ERRORS_LOGGER = "llm_bawt.service.http_errors"


class FakeDriverError(Exception):
    """Stands in for psycopg2.OperationalError without needing a live server."""


FakeDriverError.__module__ = "psycopg2.errors"


def _driver_error() -> FakeDriverError:
    return FakeDriverError(LEAK)


def _wrapped_driver_error() -> AdmissionUnavailable:
    """The real shape: turn admission wraps the driver error, text included."""
    try:
        try:
            raise _driver_error()
        except FakeDriverError as inner:
            raise AdmissionUnavailable(f"turn admission failed: {inner}") from inner
    except AdmissionUnavailable as outer:
        return outer


def _assert_clean(text: str) -> None:
    for fragment in ("db.internal", "192.0.2.7", "tenant_user"):
        assert fragment not in text


def test_real_driver_modules_are_recognised():
    sqlalchemy_exc = pytest.importorskip("sqlalchemy.exc")
    redis_exc = pytest.importorskip("redis.exceptions")
    assert names_infrastructure(sqlalchemy_exc.OperationalError("SELECT 1", {}, Exception(LEAK)))
    assert names_infrastructure(redis_exc.ConnectionError(LEAK))
    assert not names_infrastructure(ValueError("bad input"))


def test_wrapped_driver_error_is_hidden_and_logged(caplog):
    exc = _wrapped_driver_error()
    with caplog.at_level(logging.ERROR, logger=HTTP_ERRORS_LOGGER):
        text = client_error_text(exc, "Admission")

    assert re.fullmatch(r"Internal error \(AdmissionUnavailable, ref [0-9a-f]{8}\); see the app log", text)
    assert LEAK in caplog.records[-1].getMessage()


def test_domain_error_keeps_its_message(caplog):
    with caplog.at_level(logging.ERROR, logger=HTTP_ERRORS_LOGGER):
        assert client_error_text(ValueError("bot is busy"), "Admission") == "bot is busy"
    assert not caplog.records


def test_request_failed_sse_frame_hides_driver_text():
    events = request_failed_events("chatcmpl-x", 0, "model", _wrapped_driver_error())
    warning = json.loads(events[0][len("data: "):])

    [text] = warning["warnings"]
    assert re.fullmatch(
        r"request_failed: Internal error \(AdmissionUnavailable, ref [0-9a-f]{8}\); see the app log", text)
    _assert_clean("".join(events))


def test_request_failed_sse_frame_keeps_domain_refusal():
    events = request_failed_events("chatcmpl-x", 0, "model", ValueError("No user message found"))
    assert "request_failed: No user message found" in events[0]


def test_public_stream_error_frame_hides_driver_text():
    class Service:
        async def chat_completion_stream(self, _request):
            raise _driver_error()
            yield  # pragma: no cover - makes this an async generator

    async def collect():
        request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
        return [frame async for frame in completion_stream(Service(), request)]

    frames = asyncio.run(collect())
    error = json.loads(frames[-1][len("data: "):])["error"]
    assert re.fullmatch(r"Internal error \(FakeDriverError, ref [0-9a-f]{8}\); see the app log", error["message"])
    _assert_clean("".join(frames))


def test_database_health_reason_hides_driver_text(caplog):
    class Engine:
        def connect(self):
            raise _driver_error()

    with caplog.at_level(logging.ERROR, logger=HTTP_ERRORS_LOGGER):
        row = health_checks.check_database(Engine())

    assert row["status"] == health_checks.DOWN
    assert re.fullmatch(r"Database unreachable \(FakeDriverError, ref [0-9a-f]{8}\); see the app log", row["reason"])
    assert LEAK in caplog.records[-1].getMessage()


def test_redis_health_reason_hides_driver_text(monkeypatch):
    redis = pytest.importorskip("redis")

    def from_url(*_args, **_kwargs):
        raise redis.exceptions.ConnectionError(LEAK)

    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(from_url))
    row = health_checks.check_redis("redis://example:6379")

    assert re.fullmatch(r"Redis unreachable \(ConnectionError, ref [0-9a-f]{8}\); see the app log", row["reason"])


def test_failed_sweep_health_reason_hides_driver_text():
    health = LoopHealth()
    health.failed(_driver_error())
    row = health.check("scheduler.prompts", SimpleNamespace(done=lambda: False), stale_seconds=60)

    assert re.fullmatch(
        r"last sweep failed: FakeDriverError: Sweep failed \(FakeDriverError, ref [0-9a-f]{8}\); see the app log",
        row["reason"])


def test_background_task_failure_result_hides_driver_text():
    from llm_bawt.service.background_tasks import BackgroundTasksMixin
    from llm_bawt.service.tasks import Task, TaskStatus, TaskType

    class Processor(BackgroundTasksMixin):
        def _process_memory_extraction(self, _task):
            raise _driver_error()

    task = Task(task_type=TaskType.MEMORY_EXTRACTION, payload={}, bot_id="example", user_id="u")
    result = Processor()._run_task(task)

    assert result.status == TaskStatus.FAILED
    assert re.fullmatch(r"Task failed \(FakeDriverError, ref [0-9a-f]{8}\); see the app log", result.error)


# --- Round 2 (Loopy review cmw7ql3iw01zm0mojdgr1al34) -------------------------

REF_TEXT = re.compile(r"\(FakeDriverError, ref ([0-9a-f]{8})\); see the app log")


def test_sanitized_text_ref_matches_the_log_line(caplog):
    with caplog.at_level(logging.WARNING, logger=HTTP_ERRORS_LOGGER):
        text = safe_error_text(_driver_error(), "Database unreachable")

    ref = REF_TEXT.search(text).group(1)
    line = caplog.records[-1].getMessage()
    assert f"ref {ref}" in line and LEAK in line
    _assert_clean(text)


def test_safe_error_text_keeps_domain_message(caplog):
    with caplog.at_level(logging.WARNING, logger=HTTP_ERRORS_LOGGER):
        assert safe_error_text(ValueError("no such memory")) == "no such memory"
    assert not caplog.records


def _call_arg_sources(path: str, func: str) -> list[str]:
    tree = ast.parse(Path(path).read_text())
    return [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == func
    ]


def test_stream_failure_marker_never_formats_raw_exception():
    """The live marker and the persisted turn-log error both feed turn_failure_suffix."""
    calls = _call_arg_sources("src/llm_bawt/service/chat_streaming.py", "turn_failure_suffix")
    assert calls and all("str(" not in call for call in calls)
    worker = Path("src/llm_bawt/service/turn_stream_worker.py").read_text()
    assert not re.search(r"error_text\s*=\s*str\(", worker)


def test_event_stream_error_frame_hides_driver_text(monkeypatch):
    from llm_bawt.service.routes import event_stream

    class Subscriber:
        def __init__(self, _url):
            pass

        async def connect(self):
            return None

        def ui_stream(self, *_args, **_kwargs):
            class Reader:
                async def claim(self, resume=None):
                    raise _driver_error()

            return Reader()

        async def close(self):
            return None

    monkeypatch.setattr(event_stream, "RedisSubscriber", Subscriber)

    async def collect():
        return [c async for c in event_stream._unified_event_stream("redis://unused", ["caid"], "nick", "w")]

    frames = asyncio.run(collect())
    error = json.loads(frames[-1].strip().split("\n")[1].removeprefix("data: "))
    assert REF_TEXT.search(error["error"])
    _assert_clean("".join(frames))


def test_status_memory_error_hides_driver_text(monkeypatch):
    from llm_bawt.core import status
    from llm_bawt.memory import postgresql
    from llm_bawt.utils import config as config_mod

    def backend(*_args, **_kwargs):
        raise _driver_error()

    monkeypatch.setattr(config_mod, "has_database_credentials", lambda _c: True)
    monkeypatch.setattr(postgresql, "PostgreSQLMemoryBackend", backend)
    info = status._collect_memory_info(SimpleNamespace(POSTGRES_HOST="db.internal"), "example")

    assert info.postgres_connected is False
    assert REF_TEXT.search(info.postgres_error)
    _assert_clean(info.postgres_error)


def test_memory_maintenance_errors_hide_driver_text(monkeypatch):
    from llm_bawt.memory.maintenance import MemoryMaintenance

    def fail(self, _dry_run):
        raise _driver_error()

    for name in ("_run_consolidation", "_run_recurrence_detection", "_run_decay_pruning", "_run_orphan_cleanup"):
        monkeypatch.setattr(MemoryMaintenance, name, fail)
    result = MemoryMaintenance(backend=None).run(
        run_consolidation=True, run_recurrence_detection=True,
        run_decay_pruning=True, run_orphan_cleanup=True,
    )

    assert len(result.errors) == 4
    for error in result.errors:
        assert REF_TEXT.search(error)
        _assert_clean(error)


def test_profile_maintenance_error_hides_driver_text():
    from llm_bawt.memory.profile_maintenance import ProfileMaintenanceService

    class Failing:
        def __getattr__(self, _name):
            def call(*_args, **_kwargs):
                raise _driver_error()
            return call

    attr = SimpleNamespace(category="c", key="k", value="v")
    profiles = SimpleNamespace(get_all_attributes=lambda *_a: [attr])
    result = ProfileMaintenanceService(profiles, Failing()).run("u")

    assert REF_TEXT.search(result.error)
    _assert_clean(result.error)


# --- Round 3 (Loopy review cmw7r0x7c02000mojjjvjx01s) -------------------------


def test_summarize_thread_load_failure_hides_driver_text():
    from llm_bawt.memory.summarization import HistorySummarizer

    class Engine:
        def connect(self):
            raise _driver_error()

    summarizer = object.__new__(HistorySummarizer)
    summarizer._backend = SimpleNamespace(engine=Engine(), _messages_table_name="messages")
    result = summarizer.summarize_thread("session-1")

    assert result["success"] is False
    [error] = result["errors"]
    assert REF_TEXT.search(error)
    _assert_clean(error)


def test_sanitizer_imports_without_the_web_stack():
    """core.status and the memory jobs are used by the base install (no FastAPI)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('fastapi', 'starlette'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import llm_bawt.utils.error_text, llm_bawt.core.status, llm_bawt.memory.maintenance\n"
        "import llm_bawt.memory.profile_maintenance\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path.cwd(), check=False)
    assert done.returncode == 0, done.stderr[-2000:]
