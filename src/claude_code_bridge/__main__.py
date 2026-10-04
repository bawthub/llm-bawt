"""Standalone Claude Code bridge service.

Usage:
    python -m claude_code_bridge
"""

import asyncio
import logging
import os
import signal
import sys

from agent_bridge.publisher import RedisPublisher

from .bootstrap import bootstrap_claude_home
from .bridge import ClaudeCodeBridge
from .proxy import ProxyServer


async def _health_server(bridge: ClaudeCodeBridge, publisher: RedisPublisher, port: int) -> None:
    """Minimal TCP health check."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()
            redis_ok = publisher.connected
            status = "200 OK" if redis_ok else "503 Service Unavailable"
            body = f'{{"redis": {str(redis_ok).lower()}}}'
            resp = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n"
                f"{body}"
            )
            writer.write(resp.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "0.0.0.0", port)
    logger = logging.getLogger("claude_code_bridge.health")
    logger.info("Health check listening on :%d", port)
    async with server:
        await server.serve_forever()


def main() -> None:
    log_level = os.getenv("CLAUDE_CODE_BRIDGE_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy httpx request logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("claude_code_bridge")

    bootstrap_claude_home()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    publisher = RedisPublisher(redis_url, default_provider="claude-code")
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", redis_url)
        sys.exit(1)

    # Check for OAuth token (broker endpoint, legacy file, or env). The
    # Anthropic subscription token is still required for the default
    # (api.anthropic.com) path; the proxy path uses ChatGPT OAuth from
    # ~/.codex/auth.json but the SDK still needs *some* ANTHROPIC_AUTH_TOKEN
    # to send (the proxy ignores its value).
    from .bridge import _get_fresh_oauth_token
    if not _get_fresh_oauth_token():
        # TASK-835: warn, don't exit. A fresh tenant boots this stack BEFORE
        # anyone has run the setup wizard, so there is legitimately no Claude
        # credential yet. Exiting here turned that into a crash-loop with no
        # health endpoint to curl and no obvious cause — strictly worse to
        # diagnose than a running bridge that reports the problem.
        #
        # Nothing is lost by continuing: the token is re-fetched from the
        # broker on every send (it is deliberately uncached, see
        # _get_fresh_oauth_token), so the bridge starts working the moment the
        # provider is connected, with no restart. A turn attempted before then
        # fails with the broker's own error.
        logger.warning(
            "No OAuth token available yet — starting anyway; turns will fail "
            "until the Claude provider is connected. Ensure the app (llm-bawt) "
            "is running and Claude is connected (the bridge fetches tokens via "
            "the /v1/providers/claude/token broker endpoint, TASK-636), or set "
            "CLAUDE_CODE_OAUTH_TOKEN, or mount a legacy "
            "~/.claude/.credentials.json into the container."
        )

    add_dirs_raw = os.getenv("CLAUDE_CODE_ADD_DIRS", "")
    add_dirs = [d.strip() for d in add_dirs_raw.split(",") if d.strip()]

    if os.getenv("CLAUDE_CODE_MODEL"):
        logger.warning(
            "CLAUDE_CODE_MODEL env var is set but ignored: the bridge no longer "
            "accepts a default model. The model MUST be passed per-request, "
            "resolved from the bot's default_model catalog entry."
        )

    bridge = ClaudeCodeBridge(
        publisher=publisher,
        backend_name=os.getenv("CLAUDE_CODE_BACKEND_NAME", "claude-code"),
        app_api_url=os.getenv("LLM_BAWT_API_URL", ""),
        cwd=os.getenv("CLAUDE_CODE_CWD", "/app"),
        permission_mode=os.getenv("CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"),
        add_dirs=add_dirs,
        request_timeout=float(os.getenv("CLAUDE_CODE_REQUEST_TIMEOUT", "300")),
    )

    health_port = int(os.getenv("CLAUDE_CODE_BRIDGE_HEALTH_PORT", "8681"))

    # Anthropic-compatible proxy for non-Anthropic providers (OpenAI ChatGPT
    # subscription, future Grok/Kimi/etc.). Bound 127.0.0.1 only; ephemeral
    # port by default. Set CLAUDE_CODE_BRIDGE_PROXY_PORT to pin (e.g. 8691)
    # if you want to curl-test it.
    proxy_port = int(os.getenv("CLAUDE_CODE_BRIDGE_PROXY_PORT", "0"))
    proxy_disabled = os.getenv("CLAUDE_CODE_BRIDGE_PROXY_DISABLED", "").lower() in (
        "1", "true", "yes",
    )
    proxy_server: ProxyServer | None = None
    if not proxy_disabled:
        proxy_server = ProxyServer(
            host="127.0.0.1",
            port=proxy_port,
            status_callback=bridge.publish_proxy_status,
        )

    logger.info(
        "Starting Claude Code bridge (backend=%s)",
        os.getenv("CLAUDE_CODE_BACKEND_NAME", "claude-code"),
    )

    async def _run() -> None:
        shutdown_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)

        # Start the proxy first so its base_url is available when the bridge
        # spawns the SDK subprocess. Start failure is fatal — no point
        # running a bridge that can't route to its proxy when needed.
        if proxy_server is not None:
            await proxy_server.start()
            bridge.set_proxy_base_url(proxy_server.base_url)
            bridge.set_proxy_request_controls(
                proxy_server.cancel_request,
                proxy_server.clear_request,
            )
            logger.info("Anthropic-compat proxy active at %s", proxy_server.base_url)

        health_task = asyncio.create_task(
            _health_server(bridge, publisher, health_port)
        )
        bridge_task = asyncio.create_task(bridge.run_forever())

        await shutdown_event.wait()
        logger.info("Shutting down...")

        bridge_task.cancel()
        health_task.cancel()
        await bridge.stop()
        if proxy_server is not None:
            await proxy_server.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
