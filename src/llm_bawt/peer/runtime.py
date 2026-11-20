"""Peer networking settings and the opt-in listener (TASK-2517).

Off unless ``LLM_BAWT_PEER_PORT`` is set. Then the app also serves the peer
endpoint on that port (separate from ``/v1`` and the ``:8001`` MCP catalog).
``LLM_BAWT_PEER_PUBLIC_URL`` is the URL other installs use to reach it, on the
LAN or a VPN; it goes into invites. Publishing the port beyond that is a
deployment decision, not something this module does.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

log = logging.getLogger(__name__)

PORT_ENV = "LLM_BAWT_PEER_PORT"
URL_ENV = "LLM_BAWT_PEER_PUBLIC_URL"
LABEL_ENV = "LLM_BAWT_PEER_LABEL"

_schema_lock = threading.Lock()
_schema_ready = False


def enabled() -> bool:
    return bool(os.environ.get(PORT_ENV, "").strip())


def public_url() -> str:
    return os.environ.get(URL_ENV, "").strip()


def engine():
    """The shared engine, with the peer tables ensured once per process."""
    global _schema_ready
    from ..utils.config import Config
    from ..utils.db import get_shared_engine
    from . import identity, store

    eng = get_shared_engine(Config(), application_name="llm-bawt:peer")
    if eng is None:
        raise RuntimeError("peer networking needs the database")
    if not _schema_ready:
        with _schema_lock, eng.begin() as conn:
            if not _schema_ready:
                store.ensure_schema(conn)
                from . import messaging
                messaging.ensure_schema(conn)
                identity.load_or_create(conn, default_label=os.environ.get(LABEL_ENV) or "BawtHub")
                _schema_ready = True
    return eng


_pump_wake: asyncio.Event | None = None
PUMP_IDLE_SECONDS = 15


def wake_pump() -> None:
    """Nudge the outbox pump after an enqueue (same event loop as the app)."""
    if _pump_wake is not None:
        _pump_wake.set()


async def pump_forever() -> None:
    """Push queued outbound peer messages; runs beside the listener."""
    global _pump_wake
    from . import messaging

    _pump_wake = asyncio.Event()
    while True:
        try:
            await messaging.pump_once(engine())
        except Exception:  # noqa: BLE001 - keep pumping; rows stay QUEUED
            log.exception("peer outbox pump failed")
        try:
            await asyncio.wait_for(_pump_wake.wait(), PUMP_IDLE_SECONDS)
        except TimeoutError:
            pass
        _pump_wake.clear()


async def serve_forever() -> None:
    """Run the peer listener until cancelled; started from the app lifespan."""
    import uvicorn

    from .server import build_app

    port = int(os.environ[PORT_ENV])
    app = build_app(engine)
    server = _SecondaryServer(uvicorn.Config(app, host="0.0.0.0", port=port,
                                             log_level="warning", lifespan="on"))
    log.warning("Peer listener on :%d (public URL %s)", port, public_url() or "unset")
    await server.serve()


def _secondary_server_class():
    import contextlib

    import uvicorn

    class SecondaryServer(uvicorn.Server):
        """Never captures process signals: the main app server owns shutdown.

        The app's lifespan cancels the listener task on shutdown.
        """

        def capture_signals(self):
            return contextlib.nullcontext()

    return SecondaryServer


def _SecondaryServer(config):  # noqa: N802 - reads as a class at the call site
    return _secondary_server_class()(config)
