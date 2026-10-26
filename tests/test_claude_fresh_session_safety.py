"""A silent tool-only turn must never be replayed in a fresh SDK session."""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from claude_agent_sdk import StreamEvent

from agent_bridge.events import AgentEventKind
from agent_bridge.run_contract import TeardownReport
from claude_code_bridge.bridge import ClaudeCodeBridge
from claude_code_bridge.turn_watchdog import TurnWatchdogTimeout
from tests import virtual_clock
from tests.test_claude_code_send_boundaries import SESSION, _SendHarness, _SDKStream

# Virtual-clock hang guard: a broken handler fails here instead of hanging.
HANG_GUARD_S = 30.0


@pytest.mark.parametrize("progress", [None, "tool_use", "thinking"])
@pytest.mark.parametrize("failure", ["watchdog", "timeout", "crash"])
def test_fresh_retry_only_before_model_progress(monkeypatch, progress, failure):
    async def run():
        harness = _SendHarness()
        harness._uses_claude_oauth_gateway = lambda: True
        attempts = []

        class SilentSDK(_SDKStream):
            async def receive_messages(self):
                if progress:
                    yield StreamEvent(uuid="progress", session_id="sdk-session", event={
                        "type": "content_block_start",
                        "content_block": {"type": progress, "id": "tool-1", "name": "Write"},
                    })
                    if progress == "thinking":
                        yield StreamEvent(uuid="thinking", session_id="sdk-session", event={
                            "type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "working"},
                        })
                if failure == "watchdog":
                    raise TurnWatchdogTimeout("idle after tools", phase="sdk_idle")
                if failure == "timeout":
                    raise TimeoutError("SDK timed out")
                raise RuntimeError("CLI subprocess failed with exit code 1")

        def client(**kwargs):
            sdk = SilentSDK()
            attempts.append(sdk)
            return sdk

        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", client)
        # Exercise the production handler/stream side-effect tracking and catch
        # block. No mock around the retry predicate itself.
        harness._build_agent_options = Mock(return_value=None)
        await harness.send()
        assert len(attempts) == (1 if progress else 2)
        assert all(sdk.disconnect_calls == 1 for sdk in attempts)
        assert len([event for event in harness.events if event.kind == AgentEventKind.ERROR]) == 1
        # Fake SDK clients have no CLI process, so no exit is proven.
        harness._publisher.publish_run_done.assert_called_once_with(
            "turn-request", teardown=TeardownReport("ok", "none", False))
    asyncio.run(run())


@pytest.mark.parametrize("steer_mode", ["now_accepted", "now_uncertain", "next_accepted"])
def test_steering_without_any_model_output_blocks_original_prompt_replay(monkeypatch, steer_mode):
    async def run():
        harness = _SendHarness()
        harness._uses_claude_oauth_gateway = lambda: True
        release_error = asyncio.Event()
        attempts = []

        class SilentSDK(_SDKStream):
            async def receive_messages(self):
                self.waiting.set()
                await release_error.wait()
                raise TurnWatchdogTimeout("silent after steering", phase="sdk_idle")
                yield  # Async iterator, deliberately emits no model blocks.

        sdk = SilentSDK()
        sdk.query = AsyncMock()
        async def interrupt():
            if steer_mode == "now_uncertain":
                await asyncio.Event().wait()
        sdk.interrupt = AsyncMock(side_effect=interrupt)
        def client(**kwargs):
            attempts.append(kwargs)
            return sdk
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", client)
        # The bridge's production request (idle) timeout, so it can't end the
        # run before the production steer timeout does.
        harness._request_timeout = ClaudeCodeBridge.DEFAULT_REQUEST_TIMEOUT
        send = harness.send()
        await asyncio.wait_for(sdk.waiting.wait(), timeout=HANG_GUARD_S)
        active = harness._session_queue.get_active_client(SESSION)
        # Production steer timeout: the virtual clock makes the uncertain
        # interrupt's full wait free, and a slow runner can't eat the budget
        # of an accepted one.
        loop = asyncio.get_running_loop()
        began = loop.time()
        if steer_mode == "next_accepted":
            active.lifecycle_supported = True
            await active.steer("continue without replay", priority="next")
        elif steer_mode == "now_uncertain":
            with pytest.raises(RuntimeError, match="steer_interrupt_timeout"):
                await active.steer("continue without replay", priority="now")
            sdk.query.assert_not_awaited()
            # One-sided: the full steer timeout was honored (per-step cost only
            # adds to it).
            # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
            assert loop.time() - began >= active._steer_timeout_s
        else:
            await active.steer("continue without replay", priority="now")
            sdk.query.assert_awaited_once()
        assert active.steering_attempted
        release_error.set()
        await asyncio.wait_for(send, timeout=HANG_GUARD_S)
        assert len(attempts) == 1, "steering must never become an original-prompt replay"
        # Fake SDK clients have no CLI process, so no exit is proven.
        harness._publisher.publish_run_done.assert_called_once_with(
            "turn-request", teardown=TeardownReport("ok", "none", False))
    virtual_clock.run_timing(run())
