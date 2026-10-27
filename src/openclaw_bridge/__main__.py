"""Standalone OpenClaw bridge service.

Usage:
    python -m openclaw_bridge
"""

import asyncio
import logging
import signal
import sys

from agent_bridge.publisher import RedisPublisher
from agent_bridge.store import EventStore, create_agent_event_tables
from .bridge import SessionBridge
from .config import BridgeConfig
from .gateway_token import gateway_token
from .ingest import EventIngestPipeline, IngestFilterConfig
from .ws_client import OpenClawWsClient, OpenClawWsConfig


async def _health_server(bridge: SessionBridge, publisher: RedisPublisher, port: int) -> None:
    """Minimal TCP health check on the given port.

    Returns HTTP 200 if WS + Redis are connected, 503 otherwise.
    Docker HEALTHCHECK can use: curl -f http://localhost:<port>/health
    """
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()  # consume request line
            ws_ok = bridge.connected
            redis_ok = publisher.connected
            ok = ws_ok and redis_ok
            status = "200 OK" if ok else "503 Service Unavailable"
            body = f'{{"ws": {str(ws_ok).lower()}, "redis": {str(redis_ok).lower()}}}'
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
    logger = logging.getLogger("openclaw_bridge.health")
    logger.info("Health check listening on :%d", port)
    async with server:
        await server.serve_forever()


def main() -> None:
    config = BridgeConfig.from_env()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("openclaw_bridge")

    if not config.ws_url:
        logger.error("OPENCLAW_WS_URL not set, cannot start bridge")
        sys.exit(1)

    # PostgreSQL — single shared pool for this process (TASK-202).
    #
    # The openclaw bridge runs in its own slim container that does NOT ship
    # llm_bawt, so we can't import ``utils.db.get_shared_engine``. There's
    # only one engine in this whole process, so we build it inline with the
    # same parameters as the main app's shared engine. The askllm role is
    # shared with the app under Postgres ``max_connections=100``; keep this
    # process's footprint within (pool_size + max_overflow).
    from sqlalchemy import create_engine
    engine = create_engine(
        config.postgres_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"application_name": "openclaw-bridge"},
    )
    create_agent_event_tables(engine)

    # Redis
    publisher = RedisPublisher(config.redis_url, default_provider="openclaw")
    if not publisher.connected:
        logger.error("Cannot connect to Redis at %s", config.redis_url)
        sys.exit(1)

    # Fetch session->bot mapping from main app API (source of truth for session keys)
    session_to_bot = config.fetch_session_to_bot()

    # WS client — session keys come from the bot mapping
    ws_config = OpenClawWsConfig(
        url=config.ws_url,
        token_provider=gateway_token,
        session_keys=list(session_to_bot.keys()),
        reconnect_max_delay=config.reconnect_max_delay,
    )
    ws_client = OpenClawWsClient(ws_config)

    # Ingest filters
    ingest_filter = IngestFilterConfig.from_env(
        drop_patterns_csv=config.ingest_drop_patterns,
        drop_events_csv=config.ingest_drop_events,
        drop_msg_types_csv=config.ingest_drop_msg_types,
    )

    # Assemble bridge
    bridge = SessionBridge(
        ws_client=ws_client,
        ingest=EventIngestPipeline(filter_config=ingest_filter),
        store=EventStore(engine),
        publisher=publisher,
        session_to_bot=session_to_bot,
    )

    logger.info(
        "Starting OpenClaw bridge (sessions=%s, bot_map=%s)",
        list(session_to_bot.keys()),
        session_to_bot,
    )

    loop = asyncio.new_event_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info("Received %s, shutting down...", sig.name)
        loop.create_task(bridge.stop())
        loop.call_later(2, loop.stop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    async def _refresh_session_map() -> None:
        """Periodically reload session→bot mapping from the app API.

        This covers the case where the bridge starts before the app is up
        (fetch_session_to_bot() returns {} at startup).  Once the app is
        reachable the correct mapping is loaded and commands are routed
        to the right bot_id.
        """
        _log = logging.getLogger("openclaw_bridge.config")
        try:
            while True:
                await asyncio.sleep(30)
                try:
                    updated = config.fetch_session_to_bot(retries=1, delay=0)
                    if updated:
                        bridge.update_session_map(updated)
                except Exception as exc:
                    _log.debug("Session map refresh failed: %s", exc)
        except asyncio.CancelledError:
            pass

    async def _run() -> None:
        # Start health check server
        health_task = asyncio.create_task(
            _health_server(bridge, publisher, config.health_port)
        )
        refresh_task = asyncio.create_task(_refresh_session_map())
        try:
            await bridge.run_forever()
        finally:
            health_task.cancel()
            refresh_task.cancel()
            # Await cancellations so the tasks finish cleanly before the event
            # loop is stopped (avoids "Task was destroyed but it is pending").
            await asyncio.gather(health_task, refresh_task, return_exceptions=True)

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(bridge.stop())
        loop.close()


if __name__ == "__main__":
    main()
