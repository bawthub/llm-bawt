"""Standalone local-model bridge service.

Usage:
    python -m local_model_bridge

Environment:
    REDIS_URL                          — Redis connection (default redis://localhost:6379/0)
    LLM_BAWT_API_URL                   — main app base URL for /v1/models calls
    LOCAL_MODEL_BACKEND_NAME           — backend filter on commands stream (default 'local')
    LOCAL_MODEL_BRIDGE_HEALTH_PORT     — TCP port for /health (default 8683)
    LOCAL_MODEL_BRIDGE_EMBED_PORT      — TCP port for the /embed API (default 8684)
    LOCAL_MODEL_BRIDGE_LOG_LEVEL       — log level (default INFO)
    LOCAL_MODEL_BRIDGE_REQUEST_TIMEOUT — per-call generation timeout, seconds (default 1800)
    LOCAL_MODEL_EMBED_MODEL            — sentence-transformers model (default all-MiniLM-L6-v2)
    LOCAL_MODEL_EMBED_DEVICE           — embed model device (default "cpu")
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from agent_bridge.publisher import RedisPublisher

from .bridge import LocalModelBridge


async def _health_server(publisher: RedisPublisher, port: int) -> None:
    """Minimal TCP health check.

    Returns 200 when Redis is connected, 503 otherwise. Mirrors the shape of
    the other bridges' health servers so docker compose healthchecks across
    bridges look identical.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()
            redis_ok = publisher.connected
            # Report embed readiness alongside Redis for observability. This port
            # (8683) stays gated on Redis only — the embed server has its own
            # authoritative healthcheck on :8684 — but surfacing embed here lets
            # an operator see both from one probe.
            try:
                from .embed_server import embed_ready

                embed_ok = embed_ready()
            except Exception:
                embed_ok = False
            status = "200 OK" if redis_ok else "503 Service Unavailable"
            body = (
                f'{{"redis": {str(redis_ok).lower()}, '
                f'"embed": {str(embed_ok).lower()}}}'
            )
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
    logger = logging.getLogger("local_model_bridge.health")
    logger.info("Health check listening on :%d", port)
    async with server:
        await server.serve_forever()


def _install_crash_handler() -> None:
    """Install a SIGABRT handler so CUDA abort() leaves a log trace.

    llama.cpp's CUDA backend calls ``GGML_CUDA_CHECK()`` which invokes
    ``abort()`` on any CUDA error.  The default SIGABRT handler just kills
    the process with zero diagnostics — Docker logs show only the CUDA
    restart banner.  This handler flushes one line of context before dying.
    """
    import traceback

    def _on_abort(signum, frame):
        # Force-write to fd 2 (real stderr) since Python's sys.stderr
        # might be in an inconsistent state after a C-level abort().
        try:
            msg = (
                f"\n!!! SIGABRT received (likely CUDA abort) !!!\n"
                f"Stack at abort:\n"
                f"{''.join(traceback.format_stack(frame))}\n"
            )
            os.write(2, msg.encode("utf-8", errors="replace"))
        except Exception:
            os.write(2, b"\n!!! SIGABRT -- could not format stack !!!\n")
        finally:
            # Re-raise with default handler so the container exits with
            # the right signal code (Docker restart policy sees it).
            signal.signal(signal.SIGABRT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGABRT)

    signal.signal(signal.SIGABRT, _on_abort)


def main() -> None:
    _install_crash_handler()

    log_level = os.getenv("LOCAL_MODEL_BRIDGE_LOG_LEVEL", "INFO")
    from agent_bridge.logfmt import configure_logging

    # The standard one-line format (TASK-2225); layout lives in logfmt.py.
    configure_logging(
        "local",
        level=getattr(logging, log_level.upper(), logging.INFO),
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("local_model_bridge")

    backend_name = os.getenv("LOCAL_MODEL_BACKEND_NAME", "local")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    publisher = RedisPublisher(redis_url, default_provider=backend_name)
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", redis_url)
        sys.exit(1)

    bridge = LocalModelBridge(
        publisher=publisher,
        backend_name=backend_name,
        app_api_url=os.getenv("LLM_BAWT_API_URL", ""),
        request_timeout=float(os.getenv("LOCAL_MODEL_BRIDGE_REQUEST_TIMEOUT", "1800")),
    )

    health_port = int(os.getenv("LOCAL_MODEL_BRIDGE_HEALTH_PORT", "8683"))
    embed_port = int(os.getenv("LOCAL_MODEL_BRIDGE_EMBED_PORT", "8684"))

    logger.info(
        "Starting local-model bridge (backend=%s, app_api_url=%s, health_port=%d, embed_port=%d)",
        bridge.backend_name,
        os.getenv("LLM_BAWT_API_URL", ""),
        health_port,
        embed_port,
    )

    async def _run() -> None:
        shutdown_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)

        # Embed API (TASK-277): MiniLM lives here now; the app POSTs to /embed.
        # Imported lazily so a sentence-transformers/fastapi import error can't
        # stop the chat bridge from starting.
        from .embed_server import serve_embed

        # Embed-only mode (shared GPU compute plane, TASK-345): when set, run the
        # standalone /embed endpoint (+ health) WITHOUT bridge.run_forever(), which
        # couples to the app/Redis for local inference. This lets one GPU host serve
        # embeddings to many tenants over HTTP with no app dependency.
        embed_only = os.getenv("LOCAL_MODEL_BRIDGE_EMBED_ONLY", "").lower() in (
            "1",
            "true",
            "yes",
        )

        # Supervise long-lived tasks: without this, an exception or an
        # unexpected return in any of them is silently swallowed by asyncio
        # while the process keeps waiting on shutdown_event — exactly how the
        # embed server sat dead for days while the container looked healthy.
        # This logs loudly so the failure is visible in the container logs; the
        # embed server's :8684 healthcheck is what actually marks the container
        # unhealthy for recovery.
        def _supervise(name: str):
            def _cb(task: asyncio.Task) -> None:
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    logger.error("%s task crashed", name, exc_info=exc)
                else:
                    logger.warning("%s task exited unexpectedly (no exception)", name)

            return _cb

        health_task = asyncio.create_task(_health_server(publisher, health_port))
        health_task.add_done_callback(_supervise("health"))
        bridge_task = (
            None if embed_only else asyncio.create_task(bridge.run_forever())
        )
        if bridge_task is not None:
            bridge_task.add_done_callback(_supervise("inference-bridge"))
        embed_task = asyncio.create_task(serve_embed(embed_port))
        embed_task.add_done_callback(_supervise("embed"))
        from .video_server import serve_video

        video_task = asyncio.create_task(serve_video(int(os.getenv("LOCAL_MODEL_VIDEO_PORT", "8685"))))
        video_task.add_done_callback(_supervise("video"))
        if embed_only:
            logger.info("EMBED-ONLY mode: app/Redis inference bridge NOT started")

        await shutdown_event.wait()
        logger.info("Shutting down...")

        if bridge_task is not None:
            bridge_task.cancel()
        health_task.cancel()
        embed_task.cancel()
        video_task.cancel()
        if not embed_only:
            await bridge.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
