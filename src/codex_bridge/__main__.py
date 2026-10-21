"""Standalone Codex bridge service.

Usage:
    python -m codex_bridge

Environment:
    REDIS_URL                       — Redis connection (default redis://localhost:6379/0)
    LLM_BAWT_API_URL                — main app base URL for /v1/bots calls
    CODEX_AUTH_PATH                 — auth.json path (default /home/bridge/.codex/auth.json)
    CODEX_HOME                      — exported as CODEX_HOME for the SDK (default /home/bridge/.codex)
    CODEX_MODEL                     — default model when chat.send omits it
    CODEX_BACKEND_NAME              — backend filter on commands stream (default 'codex')
    CODEX_BRIDGE_HEALTH_PORT        — HTTP port for /health + /models (default 8682)
    CODEX_BRIDGE_LOG_LEVEL          — log level (default INFO)
    CODEX_BRIDGE_REQUEST_TIMEOUT    — per-call SDK timeout, seconds (default 300)
    CODEX_BRIDGE_CWD                — codex thread cwd (default /home/bridge/dev)
    CODEX_BIN                       — explicit codex binary path (default bundled)
    CODEX_LOCAL_PLUGINS_ENABLED     — stage repo-managed local plugins into ~/.agents + ~/plugins
    CODEX_LOCAL_PLUGINS_SRC         — source mapping root (default /home/bridge/dev/agent-skills/codex)
    CODEX_DEV_ROOT                  — dev root used to resolve repo-managed skills (default /home/bridge/dev)
    CODEX_BAWTHUB_MCP_URL           — BawtHub MCP endpoint for Codex tools (default http://app:8001/mcp)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from agent_bridge.publisher import RedisPublisher

from .auth_broker import ensure_auth_file
from .bridge import CodexBridge
from .exec_patch import install as install_exec_patch
from .local_plugins import install_repo_local_plugins
from .parser_patch import install as install_parser_patch
from .transport import auth_path, scrub_api_key_env, validate_auth_json

logger = logging.getLogger("codex_bridge")


# Fallback client version if the codex binary can't be probed. The backend
# hard-requires the ``client_version`` query param and HIDES models whose
# ``minimal_client_version`` exceeds it — so reporting a version below what the
# installed binary actually is would silently drop newer models (e.g. gpt-5.6
# needs >=0.144.0). We derive the real version from the binary at runtime so the
# listing always matches what turns can actually run; this is only the floor.
_CODEX_CLIENT_VERSION_FALLBACK = "0.144.1"


async def _codex_client_version() -> str:
    """Report the installed codex CLI version to the backend's /models gate.

    Order: explicit ``CODEX_CLIENT_VERSION`` env override → the real binary's
    ``--version`` (so a host ``npm update @openai/codex`` auto-unlocks newer
    models with no code/env change) → a sane fallback. Never fake a version the
    binary can't back up: the backend would list models the 5.5-era binary
    can't actually run.
    """
    override = os.getenv("CODEX_CLIENT_VERSION")
    if override:
        return override.strip()
    bin_path = os.getenv("CODEX_BIN") or "codex"
    try:
        proc = await asyncio.create_subprocess_exec(
            bin_path, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        # Output looks like "codex-cli 0.141.0" — take the trailing token.
        token = out.decode().strip().split()[-1]
        if token and token[0].isdigit():
            return token
    except Exception:  # binary missing, timeout, odd output
        pass
    return _CODEX_CLIENT_VERSION_FALLBACK


async def _fetch_codex_models() -> list[dict]:
    """Return live Codex models for the authenticated ChatGPT plan.

    Source of truth is the codex backend's own ``/models`` endpoint — the exact
    surface the codex CLI hits — reached over plain HTTP with the ChatGPT OAuth
    bearer. We deliberately do NOT use the SDK: the installed package is
    ``openai_codex_sdk`` and its ``Codex`` class exposes no ``models()`` method,
    so the old ``from openai_codex import AsyncCodex`` path always 503'd and the
    app silently fell back to a stale hardcoded catalog.

    Runs here — not in the app — because this container owns the OAuth bundle.
    The proxy's ``OpenAIChatGPTAdapter`` handles token load + auto-refresh and
    supplies the mandatory ``chatgpt-account-id`` header. Returns dicts carrying
    ``id`` and ``context_window`` (hidden models are filtered out).
    """
    import httpx

    from claude_code_bridge.proxy.adapters.openai_chatgpt import (
        OpenAIChatGPTAdapter,
    )

    adapter = OpenAIChatGPTAdapter()
    bearer, base_url = await adapter.authorize()
    headers = {"Authorization": f"Bearer {bearer}", **adapter.extra_headers({})}
    client_version = await _codex_client_version()

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{base_url.rstrip('/')}/models",
            params={"client_version": client_version},
            headers=headers,
        )
    resp.raise_for_status()

    out: list[dict] = []
    for m in resp.json().get("models", []) or []:
        if not isinstance(m, dict):
            continue
        if m.get("visibility") == "hide":
            continue
        slug = m.get("slug") or m.get("id")
        if not slug:
            continue
        out.append(
            {
                "id": slug,
                "context_window": m.get("context_window"),
                "description": m.get("description") or m.get("display_name") or "",
            }
        )
    return out


def _http_response(status: str, body: str) -> bytes:
    return (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
        f"{body}"
    ).encode()


async def _health_server(publisher: RedisPublisher, port: int) -> None:
    """Minimal HTTP server for docker healthchecks + model discovery.

    Routes:
      * ``/health``  — 200 when Redis is connected, 503 otherwise. Mirrors the
        shape of claude_code_bridge's health server so compose healthchecks
        across bridges look identical.
      * ``/models``  — live Codex model catalog as ``{"models": [{"id": ...}]}``,
        503 on SDK/auth failure. Lets the app discover models without importing
        the Codex SDK itself.
    """

    logger = logging.getLogger("codex_bridge.health")

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            try:
                path = request_line.decode("latin1").split(" ", 2)[1]
            except (IndexError, UnicodeDecodeError):
                path = "/"
            path = path.split("?", 1)[0]

            if path.startswith("/models"):
                try:
                    models = await _fetch_codex_models()
                    body = json.dumps({"models": models})
                    status = "200 OK"
                except Exception as e:  # auth expiry, offline, backend drift
                    logger.warning("Codex /models fetch failed: %s", e)
                    body = json.dumps({"error": str(e)})
                    status = "503 Service Unavailable"
            else:
                redis_ok = publisher.connected
                status = "200 OK" if redis_ok else "503 Service Unavailable"
                body = f'{{"redis": {str(redis_ok).lower()}}}'

            writer.write(_http_response(status, body))
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "0.0.0.0", port)
    logger.info("Health/models server listening on :%d", port)
    async with server:
        await server.serve_forever()


def _redact(s: str | None, *, keep: int = 6) -> str:
    if not s:
        return "<unset>"
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)


def main() -> None:
    log_level = os.getenv("CODEX_BRIDGE_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Patch the SDK's strict-literal parser so a single mismatched item
    # status doesn't kill the whole turn (see parser_patch.py).
    install_parser_patch()
    install_exec_patch()

    # --- TASK-204: scrub API key env vars before constructing the SDK ---
    scrubbed = scrub_api_key_env()
    if scrubbed:
        logger.warning(
            "Removed %s from environment — Codex bridge is OAuth-only.",
            ", ".join(scrubbed),
        )

    # --- TASK-204: validate auth.json before any SDK call ---
    auth_file = auth_path()

    # The app owns the refresh chain. A reconnect replaces its bundle but not
    # this container-local file, so fetch it even when auth.json already exists.
    # Preserve the last valid copy if the broker is transiently unavailable.
    if not ensure_auth_file(auth_file):
        logger.error(
            "auth.json missing at %s and broker unreachable — "
            "cannot start. Ensure the app is running and the ChatGPT "
            "provider is connected.",
            auth_file,
        )
        sys.exit(1)

    try:
        auth_data = validate_auth_json(auth_file)
    except RuntimeError as e:
        logger.error("Auth bootstrap failed: %s", e)
        sys.exit(1)

    tokens = auth_data.get("tokens") or {}
    logger.info(
        "Codex OAuth ok: path=%s auth_mode=%s account_id=%s last_refresh=%s",
        auth_file,
        (tokens.get("auth_mode") or auth_data.get("auth_mode") or "chatgpt"),
        _redact(tokens.get("account_id") or auth_data.get("account_id")),
        auth_data.get("last_refresh") or tokens.get("last_refresh") or "<unknown>",
    )

    # Make sure CODEX_HOME points at the dir containing auth.json so the SDK
    # picks up the same files we just validated.
    codex_home = os.getenv("CODEX_HOME") or str(auth_file.parent)
    os.environ["CODEX_HOME"] = codex_home
    install_repo_local_plugins(
        logger=logger,
        codex_home=Path(codex_home),
    )

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    publisher = RedisPublisher(
        redis_url,
        default_provider=os.getenv("CODEX_BACKEND_NAME", "codex"),
    )
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", redis_url)
        sys.exit(1)

    bridge = CodexBridge(
        publisher=publisher,
        backend_name=os.getenv("CODEX_BACKEND_NAME", "codex"),
        app_api_url=os.getenv("LLM_BAWT_API_URL", ""),
        default_model=os.getenv("CODEX_MODEL", "gpt-5.4"),
        cwd=os.getenv("CODEX_BRIDGE_CWD", "/home/bridge/dev"),
        codex_bin=os.getenv("CODEX_BIN") or None,
        request_timeout=float(os.getenv("CODEX_BRIDGE_REQUEST_TIMEOUT", "900")),
    )

    health_port = int(os.getenv("CODEX_BRIDGE_HEALTH_PORT", "8682"))

    logger.info(
        "Starting Codex bridge (backend=%s, model=%s, cwd=%s, codex_home=%s)",
        bridge.backend_name,
        bridge.default_model,
        bridge.cwd,
        codex_home,
    )

    async def _run() -> None:
        shutdown_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)

        health_task = asyncio.create_task(_health_server(publisher, health_port))
        bridge_task = asyncio.create_task(bridge.run_forever())

        await shutdown_event.wait()
        logger.info("Shutting down...")

        bridge_task.cancel()
        health_task.cancel()
        await bridge.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
