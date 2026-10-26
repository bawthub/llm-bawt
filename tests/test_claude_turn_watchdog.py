import asyncio
import re

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    TaskProgressMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from claude_code_bridge.turn_watchdog import (
    DEFAULT_HEALTH_INTERVAL_SECONDS,
    DEFAULT_TOOL_GRACE_SECONDS,
    MAX_TOOL_TIMEOUT_SECONDS,
    TurnWatchdog,
    TurnWatchdogTimeout,
)
from tests import virtual_clock

# The async tests below run on the virtual clock (virtual_clock.run_timing) at
# the bridge's production intervals: asyncio timers jump, so a 400s tool takes
# no wall time, and the outcome cannot depend on how fast the runner is. The
# watchdog takes the loop's clock explicitly, so its deadlines and the
# stream's sleeps read the same time.
IDLE_TIMEOUT_S = 300.0
BASH_TIMEOUT_S = 500.0


def _watchdog(**kwargs) -> TurnWatchdog:
    return TurnWatchdog(
        kwargs.pop("idle_timeout", IDLE_TIMEOUT_S),
        clock=asyncio.get_running_loop().time,
        **kwargs,
    )


def _bash(tool_use_id: str, timeout_s: float) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=tool_use_id,
                name="Bash",
                input={"command": "sleep", "timeout": int(timeout_s * 1000)},
            )
        ],
        model="claude-test",
    )


async def _stream_with_long_bash():
    yield _bash("toolu_long", BASH_TIMEOUT_S)
    # Longer than the SDK idle timeout, inside the declared Bash timeout.
    await asyncio.sleep(IDLE_TIMEOUT_S + 100)
    yield UserMessage(
        content=[ToolResultBlock(tool_use_id="toolu_long", content="done")]
    )


def test_long_bash_outlives_normal_sdk_idle_timeout() -> None:
    async def run() -> float:
        stream = _stream_with_long_bash()
        health: list[dict] = []
        watchdog = _watchdog(on_health=health.append)

        start = await watchdog.receive_next(
            stream, request_id="req-long", session_key="snark:nick"
        )
        assert isinstance(start, AssistantMessage)
        assert watchdog.active_tools[0].timeout_seconds == BASH_TIMEOUT_S

        began = asyncio.get_running_loop().time()
        result = await watchdog.receive_next(
            stream, request_id="req-long", session_key="snark:nick"
        )
        assert isinstance(result, UserMessage)
        assert watchdog.active_tools == ()
        assert health
        assert health[-1]["phase"] == "tool_running"
        assert health[-1]["active_tools"][0]["name"] == "Bash"
        assert health[-1]["active_tools"][0]["timeout_seconds"] == BASH_TIMEOUT_S
        assert health[-1]["deadline_in_seconds"] > 0
        return asyncio.get_running_loop().time() - began

    waited = virtual_clock.run_timing(run())
    # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
    assert waited >= IDLE_TIMEOUT_S + 100


async def _silent_stream():
    await asyncio.sleep(IDLE_TIMEOUT_S * 2)
    yield UserMessage(content="too late")


def test_silent_sdk_still_hits_idle_watchdog(caplog) -> None:
    async def run() -> float:
        watchdog = _watchdog()
        began = asyncio.get_running_loop().time()

        with pytest.raises(
            TurnWatchdogTimeout,
            match=re.escape(f"No SDK messages for {IDLE_TIMEOUT_S}s"),
        ) as raised:
            await watchdog.receive_next(
                _silent_stream(), request_id="req-idle", session_key="loopy:nick"
            )
        assert raised.value.phase == "sdk_idle"
        assert raised.value.retry_fresh_session is True
        assert any(
            "turn_watchdog_failure request_id=req-idle" in record.message
            and "phase=sdk_idle" in record.message
            and "wait_elapsed_ms=" in record.message
            for record in caplog.records
        )
        return asyncio.get_running_loop().time() - began

    # One-sided: the full idle timeout was honored (per-step cost only adds).
    # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
    assert virtual_clock.run_timing(run()) >= IDLE_TIMEOUT_S


OVERDUE_BASH_S = 120.0


async def _stream_with_overdue_bash():
    yield _bash("toolu_overdue", OVERDUE_BASH_S)
    await asyncio.sleep(IDLE_TIMEOUT_S * 3)
    yield UserMessage(
        content=[ToolResultBlock(tool_use_id="toolu_overdue", content="done")]
    )


def test_active_tool_fails_after_declared_timeout_and_grace() -> None:
    async def run() -> float:
        stream = _stream_with_overdue_bash()
        watchdog = _watchdog()
        # Before the tool is observed, so elapsed covers its whole deadline.
        began = asyncio.get_running_loop().time()
        await watchdog.receive_next(
            stream, request_id="req-overdue", session_key="snark:nick"
        )

        with pytest.raises(TurnWatchdogTimeout, match="active-tool deadline") as raised:
            await watchdog.receive_next(
                stream, request_id="req-overdue", session_key="snark:nick"
            )
        # tool_running (not sdk_idle) is the Bash's own deadline firing.
        assert raised.value.phase == "tool_running"
        assert raised.value.retry_fresh_session is False
        return asyncio.get_running_loop().time() - began

    # One-sided: the Bash's timeout plus grace was honored.
    # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
    assert virtual_clock.run_timing(run()) >= OVERDUE_BASH_S + DEFAULT_TOOL_GRACE_SECONDS


def test_health_callback_failure_does_not_kill_turn() -> None:
    calls: list[dict] = []

    async def run() -> None:
        async def stream():
            # Several health intervals, inside the idle timeout.
            await asyncio.sleep(DEFAULT_HEALTH_INTERVAL_SECONDS * 3 + 1)
            yield UserMessage(content="alive")

        def broken_callback(health: dict) -> None:
            calls.append(health)
            raise RuntimeError("telemetry unavailable")

        watchdog = _watchdog(on_health=broken_callback)
        message = await watchdog.receive_next(
            stream(), request_id="req-health", session_key="loopy:nick"
        )
        assert isinstance(message, UserMessage)

    virtual_clock.run_timing(run())
    assert len(calls) == 3


def test_production_intervals_fit_the_rewritten_tests() -> None:
    """The constants above assume the bridge's defaults; fail loudly if they
    move so the long-Bash test still outlives the idle timeout."""
    assert IDLE_TIMEOUT_S + 100 < BASH_TIMEOUT_S <= MAX_TOOL_TIMEOUT_SECONDS
    assert OVERDUE_BASH_S + DEFAULT_TOOL_GRACE_SECONDS < IDLE_TIMEOUT_S


def test_parallel_tools_use_furthest_active_deadline() -> None:
    now = [100.0]
    watchdog = TurnWatchdog(
        idle_timeout=300,
        tool_grace=30,
        max_tool_timeout=600,
        clock=lambda: now[0],
    )
    watchdog.observe(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_short",
                    name="Bash",
                    input={"command": "short", "timeout": 120_000},
                ),
                ToolUseBlock(
                    id="toolu_long",
                    name="Bash",
                    input={"command": "long", "timeout": 500_000},
                ),
            ],
            model="claude-test",
        )
    )

    deadline, phase = watchdog._deadline(wait_started_at=100.0)
    assert phase == "tool_running"
    assert deadline == 630.0


def test_bash_timeout_is_capped_and_other_tools_keep_idle_limit() -> None:
    watchdog = TurnWatchdog(
        idle_timeout=300,
        tool_grace=30,
        max_tool_timeout=600,
    )
    watchdog.observe(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_bash",
                    name="Bash",
                    input={"command": "sleep 900", "timeout": 900_000},
                ),
                ToolUseBlock(
                    id="toolu_read",
                    name="Read",
                    input={"file_path": "/tmp/example"},
                ),
            ],
            model="claude-test",
        )
    )

    by_id = {tool.tool_use_id: tool for tool in watchdog.active_tools}
    assert by_id["toolu_bash"].timeout_seconds == 600
    assert by_id["toolu_read"].timeout_seconds == 300


def test_agent_child_activity_refreshes_only_its_inactivity_deadline() -> None:
    now = [100.0]
    watchdog = TurnWatchdog(
        idle_timeout=300,
        tool_grace=30,
        clock=lambda: now[0],
    )
    watchdog.observe(
        AssistantMessage(
            content=[ToolUseBlock(id="agent_1", name="Agent", input={})],
            model="claude-test",
        )
    )

    now[0] = 390.0
    watchdog.observe(
        AssistantMessage(
            content=[ToolUseBlock(id="read_1", name="Read", input={})],
            model="claude-test",
            parent_tool_use_id="agent_1",
        )
    )

    by_id = {tool.tool_use_id: tool for tool in watchdog.active_tools}
    assert by_id["agent_1"].started_at == 100.0
    assert by_id["agent_1"].last_activity_at == 390.0
    deadline, phase = watchdog._deadline(wait_started_at=390.0)
    assert phase == "tool_running"
    assert deadline == 720.0


def test_unrelated_activity_does_not_refresh_an_agent_deadline() -> None:
    now = [100.0]
    watchdog = TurnWatchdog(
        idle_timeout=300,
        tool_grace=30,
        clock=lambda: now[0],
    )
    watchdog.observe(
        AssistantMessage(
            content=[ToolUseBlock(id="agent_1", name="Agent", input={})],
            model="claude-test",
        )
    )

    now[0] = 390.0
    watchdog.observe(UserMessage(content="top-level progress"))

    deadline, phase = watchdog._deadline(wait_started_at=390.0)
    assert phase == "tool_running"
    assert deadline == 430.0


def test_bash_child_activity_does_not_extend_declared_runtime() -> None:
    now = [100.0]
    watchdog = TurnWatchdog(
        idle_timeout=300,
        tool_grace=30,
        max_tool_timeout=600,
        clock=lambda: now[0],
    )
    watchdog.observe(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="bash_1",
                    name="Bash",
                    input={"command": "sleep", "timeout": 120_000},
                )
            ],
            model="claude-test",
        )
    )

    now[0] = 200.0
    watchdog.observe(UserMessage(content="activity", parent_tool_use_id="bash_1"))

    deadline, phase = watchdog._deadline(wait_started_at=200.0)
    assert phase == "tool_running"
    assert deadline == 250.0


def test_task_progress_refreshes_workflow_inactivity_deadline() -> None:
    now = [100.0]
    watchdog = TurnWatchdog(
        idle_timeout=300,
        tool_grace=30,
        clock=lambda: now[0],
    )
    watchdog.observe(
        AssistantMessage(
            content=[ToolUseBlock(id="workflow_1", name="Workflow", input={})],
            model="claude-test",
        )
    )

    now[0] = 350.0
    watchdog.observe(
        TaskProgressMessage(
            subtype="task_progress",
            data={},
            task_id="task_1",
            description="still working",
            usage={},
            uuid="uuid-1",
            session_id="session-1",
            tool_use_id="workflow_1",
            last_tool_name="Read",
        )
    )

    deadline, phase = watchdog._deadline(wait_started_at=350.0)
    assert phase == "tool_running"
    assert deadline == 680.0
