"""Reconnect a deferred MCP catalog after it disappears within a live SDK turn."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from claude_code_bridge.active_run import ClaudeActiveRun
from tests.virtual_clock import run as virtual_run


class _Client:
    def __init__(self, statuses: list[dict[str, str]], *, fail_reconnect: int = 0):
        self.options = SimpleNamespace(mcp_servers={
            "bawthub": {"type": "http", "url": "http://app:8001/mcp"},
            "stream": {"type": "sse", "url": "http://stream.test/sse"},
            "local": {"type": "stdio", "command": "test-server"},
        })
        self.statuses = statuses
        self.fail_reconnect = fail_reconnect
        self.status_calls = 0
        self.reconnects: list[str] = []
        self.reconnected = asyncio.Event()
        self.checked = asyncio.Event()
        self.release = asyncio.Event()

    async def get_mcp_status(self):
        index = min(self.status_calls, len(self.statuses) - 1)
        self.status_calls += 1
        self.checked.set()
        return {"mcpServers": [
            {"name": name, "status": status}
            for name, status in self.statuses[index].items()
        ]}

    async def reconnect_mcp_server(self, name: str):
        self.reconnects.append(name)
        if len(self.reconnects) <= self.fail_reconnect:
            raise ConnectionError("server still unavailable")
        self.reconnected.set()

    async def receive_messages(self):
        yield object()
        await self.release.wait()


async def _drive(client: _Client) -> None:
    live = ClaudeActiveRun(client=client, request_id="req")
    messages = live.messages()
    await anext(messages)
    await client.reconnected.wait()
    live.mark_completed()
    await messages.aclose()


def test_failed_mcp_server_reconnects_without_resetting_the_active_run():
    client = _Client([{"bawthub": "failed", "local": "failed"}])
    virtual_run(_drive(client))
    assert client.status_calls >= 1
    assert client.reconnects == ["bawthub"]


def test_mcp_catalog_recovers_after_restart_during_one_long_turn():
    client = _Client([
        {"bawthub": "connected"},
        {"bawthub": "failed"},
    ])
    virtual_run(_drive(client))
    assert client.status_calls >= 2
    assert client.reconnects == ["bawthub"]


def test_failed_reconnect_is_retried_after_server_recovers():
    client = _Client([{"bawthub": "failed"}], fail_reconnect=1)
    virtual_run(_drive(client))
    assert client.status_calls >= 2
    assert client.reconnects == ["bawthub", "bawthub"]


def test_pending_catalog_without_tools_reconnects_after_two_checks():
    client = _Client([{"bawthub": "pending"}])

    async def main():
        live = ClaudeActiveRun(client=client, request_id="req")
        messages = live.messages()
        await anext(messages)
        try:
            await asyncio.wait_for(client.reconnected.wait(), timeout=46)  # virtual clock
        finally:
            live.mark_completed()
            await messages.aclose()

    virtual_run(main())
    assert client.status_calls >= 2
    assert client.reconnects == ["bawthub"]


def test_accepted_reconnect_gives_pending_catalog_another_grace_poll():
    client = _Client([{"bawthub": "pending"}])

    async def main():
        live = ClaudeActiveRun(client=client, request_id="req")
        messages = live.messages()
        await anext(messages)
        await client.reconnected.wait()
        await asyncio.sleep(16)  # virtual clock: next 15s status check
        live.mark_completed()
        await messages.aclose()

    virtual_run(main())
    assert client.status_calls >= 3
    assert client.reconnects == ["bawthub"]


def test_transient_pending_catalog_is_not_reconnected():
    client = _Client([{"bawthub": "pending"}, {"bawthub": "connected"}])

    async def main():
        live = ClaudeActiveRun(client=client, request_id="req")
        messages = live.messages()
        await anext(messages)
        await client.checked.wait()
        await asyncio.sleep(31)  # virtual clock: observe at least two checks
        live.mark_completed()
        await messages.aclose()

    virtual_run(main())
    assert client.status_calls >= 2
    assert client.reconnects == []


def test_sse_transport_reconnects_but_http_connected_server_is_unchanged():
    client = _Client([{"bawthub": "connected", "stream": "failed"}])
    virtual_run(_drive(client))
    assert client.reconnects == ["stream"]


def test_server_outage_backs_off_and_warns_once(caplog):
    client = _Client([{"bawthub": "failed"}], fail_reconnect=3)
    virtual_run(_drive(client))
    assert client.reconnects == ["bawthub"] * 4
    assert client.status_calls >= 8  # virtual 15s polls; retries at 15, 30, 60, 120
    assert sum("MCP server bawthub reconnect failed" in record.message
               for record in caplog.records) == 1


def test_monitor_stops_with_run_and_does_not_override_auth_or_local_mcp():
    client = _Client([{"bawthub": "needs-auth", "local": "failed"}])

    async def main():
        live = ClaudeActiveRun(client=client, request_id="req")
        messages = live.messages()
        await anext(messages)
        await client.checked.wait()
        live.mark_completed()
        checks_when_stopped = client.status_calls
        await asyncio.sleep(31)  # virtual clock: beyond two monitor intervals
        assert client.status_calls == checks_when_stopped
        await messages.aclose()

    virtual_run(main())
    assert client.reconnects == []
