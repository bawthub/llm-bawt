"""Open a fresh Codex thread that already holds the app's continuity seed (TASK-1579).

The app selects the seed and resolves its framing from the prompt registry;
this module only writes them into a new provider thread as one briefing turn
(user) and its reply (assistant) through app-server ``thread/inject_items``.
The items carry no system prompt: the send pastes the pinned prompt with the
first real turn, exactly as it does for an unseeded fresh thread.

Both transports resume the seeded thread by id, so exec bots seed through a
short-lived app-server process too. The app-server binary is version-pinned;
a binary without ``thread/inject_items`` fails the send loudly instead of
silently starting a blank thread.
"""
from __future__ import annotations

from typing import Any

from agent_bridge.continuity import SeedFraming, render_seed_briefing

from .app_server import APP_SERVER_VERSION, METHOD_NOT_FOUND, AppServerError, close_and_reap


class CodexSeedUnsupported(RuntimeError):
    """The installed codex app-server cannot inject seed items."""


def seed_items(messages: list[dict], framing: SeedFraming) -> list[dict]:
    """The two response items a seeded thread opens with."""
    return [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": render_seed_briefing(messages, framing.briefing)}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": framing.ack}]},
    ]


async def seed_thread(
    bridge: Any, codex: Any, overrides: tuple[str, ...], *,
    model: str, cwd: str | None, messages: list[dict], framing: SeedFraming,
    timeout: float | None,
) -> str:
    """Start a thread, inject the seed, and return the new provider thread id."""
    client = await bridge._open_app_server(codex, overrides)
    try:
        options: dict[str, Any] = {
            "model": model, "approvalPolicy": "never", "sandbox": "danger-full-access",
        }
        if cwd:
            options["cwd"] = cwd
        opened = await client.request("thread/start", options, timeout=timeout)
        thread_id = str(((opened or {}).get("thread") or {}).get("id") or "")
        if not thread_id:
            raise RuntimeError("codex app-server opened a seed thread without an id")
        try:
            await client.request(
                "thread/inject_items",
                {"threadId": thread_id, "items": seed_items(messages, framing)},
                timeout=timeout,
            )
        except AppServerError as exc:
            if exc.code == METHOD_NOT_FOUND:
                raise CodexSeedUnsupported(
                    f"codex app-server {APP_SERVER_VERSION} pin does not support "
                    "thread/inject_items; continuity seed was not applied"
                ) from exc
            raise
        return thread_id
    finally:
        await close_and_reap(client)
