"""No browser or network: PreToolUse validation of the exact tool arguments."""

from __future__ import annotations

import asyncio
import shutil
import time

import pytest

from claude_code_bridge import playwright_preflight as preflight


@pytest.fixture
def real_node(monkeypatch):
    """A real ``node`` parse whose verdict never depends on how fast it starts.

    Production keeps the 1s bound and fails open past it (covered below with a
    fake process). A test asserting the parser's verdict must not inherit that
    race on a cold, shared CI runner (TASK-1312); each such test spawns once.
    """
    assert shutil.which("node"), "the Playwright preflight parser tests need node on PATH"
    monkeypatch.setattr(preflight, "_PARSE_TIMEOUT_S", 30.0)


def test_only_playwright_functions_are_checked(monkeypatch):
    async def should_not_spawn(*args, **kwargs):
        raise AssertionError("parser should not run for other tools")

    monkeypatch.setattr(preflight.asyncio, "create_subprocess_exec", should_not_spawn)
    assert asyncio.run(preflight.playwright_syntax_error("Bash", {"code": "not JS"})) is None


@pytest.mark.parametrize(("tool", "field", "source"), [
    ("mcp__playwright__browser_evaluate", "function", "() => document.title"),
    ("mcp__playwright__browser_run_code_unsafe", "code", "async (page) => await page.title()"),
    ("mcp__playwright__browser_evaluate", "function", "() => document.title // comment"),
])
def test_valid_functions_pass_the_real_parser(real_node, tool, field, source):
    assert asyncio.run(preflight.playwright_syntax_error(tool, {field: source})) is None


def test_real_parser_reports_the_syntax_error_and_its_line(real_node):
    reason = asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_evaluate", {"function": "() => {\n  return 1;\n} stray_text"},
    ))
    assert "`function` at line 3: Unexpected identifier 'stray_text'" in reason
    assert "not executed" in reason and "NEW tool call" in reason


def test_runtime_reference_is_not_mistaken_for_syntax(real_node):
    assert asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_evaluate", {"function": "() => missingVariable"},
    )) is None


def test_editorial_marker_denied_without_node(monkeypatch):
    async def should_not_spawn(*args, **kwargs):
        raise AssertionError("parser should not run on known marker")

    monkeypatch.setattr(preflight.asyncio, "create_subprocess_exec", should_not_spawn)
    reason = asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_evaluate", {"function": "() => { return 1; } [sic]"},
    ))
    assert "stray trailing `[sic]`" in reason
    assert "not executed" in reason


def test_comment_marker_is_not_denied(real_node):
    assert asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_evaluate",
        {"function": "() => { return 1; } // annotation } [sic]"},
    )) is None


def test_unpaired_surrogate_defers_to_policy_and_tool():
    assert asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_evaluate", {"function": "() => 1\ud800"},
    )) is None


def test_missing_parser_defers_to_playwright(monkeypatch):
    async def missing_node(*args, **kwargs):
        raise FileNotFoundError("node")

    monkeypatch.setattr(preflight.asyncio, "create_subprocess_exec", missing_node)
    assert asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_evaluate", {"function": "() => { return ; }"},
    )) is None


def test_parser_timeout_kills_subprocess_and_defers(monkeypatch):
    class SlowProcess:
        returncode = None
        killed = False

        async def communicate(self, source=None):
            if source is not None:
                await asyncio.Event().wait()
            return b"", b""

        def kill(self):
            self.killed = True

    process = SlowProcess()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(preflight.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(preflight, "_PARSE_TIMEOUT_S", 0.01)
    assert asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_run_code_unsafe", {"code": "async (page) => 1"},
    )) is None
    assert process.killed


def test_parser_cleanup_is_bounded_when_communicate_hangs_after_kill(monkeypatch):
    class StuckProcess:
        returncode = None
        killed = False
        cleanup_cancelled = False

        async def communicate(self, source=None):
            if source is None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cleanup_cancelled = True
                    raise
            else:
                await asyncio.Event().wait()

        def kill(self):
            self.killed = True

    process = StuckProcess()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(preflight.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(preflight, "_PARSE_TIMEOUT_S", 0.01)
    start = time.monotonic()
    assert asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_evaluate", {"function": "() => 1"},
    )) is None
    assert time.monotonic() - start < 1.0
    assert process.killed and process.cleanup_cancelled


def test_parser_kill_races_process_exit(monkeypatch):
    class ExitingProcess:
        returncode = None
        kill_attempted = False
        drained = False

        async def communicate(self, source=None):
            if source is not None:
                await asyncio.Event().wait()
            self.drained = True
            return b"", b""

        def kill(self):
            self.kill_attempted = True
            self.returncode = 0
            raise ProcessLookupError("process exited before kill")

    process = ExitingProcess()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(preflight.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(preflight, "_PARSE_TIMEOUT_S", 0.01)
    assert asyncio.run(preflight.playwright_syntax_error(
        "mcp__playwright__browser_run_code_unsafe", {"code": "async (page) => 1"},
    )) is None
    assert process.kill_attempted and process.drained


def test_cancellation_kills_parser_without_swallowing_it(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    class HangingProcess:
        returncode = None
        killed = False

        async def communicate(self, source=None):
            if source is not None:
                started.set()
                await release.wait()
            return b"", b""

        def kill(self):
            self.killed = True
            release.set()

    process = HangingProcess()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(preflight.asyncio, "create_subprocess_exec", spawn)

    async def run():
        task = asyncio.create_task(preflight.playwright_syntax_error(
            "mcp__playwright__browser_evaluate", {"function": "() => 1"},
        ))
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation was swallowed")

    asyncio.run(run())
    assert process.killed
