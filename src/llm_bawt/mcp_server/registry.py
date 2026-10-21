"""FastMCP server instance + host/origin allowlist (TASK-639 slice A).

Extracted from ``mcp_server/server.py`` so the FastMCP instance can be wrapped
by the approval-aware interceptor (TASK-639 Slice E) without dragging in the
full 900+ line tool-registration module. ``server.py`` re-exports ``mcp`` from
here, so existing ``from .server import mcp`` imports across the tool_modules
continue to work byte-compatibly.

DO NOT move tool registrations here. This module owns exactly the FastMCP
transport/security config; tool modules keep registering against the same
``mcp`` singleton via ``from .server import mcp`` (or, going forward,
``from .registry import mcp``).
"""

from __future__ import annotations

import logging
import os

# Suppress noisy MCP library session lifecycle logging BEFORE the FastMCP
# import so its own logger inherits the WARNING level.
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)

from mcp.server.transport_security import TransportSecuritySettings  # noqa: E402 — logging first

from .catalog import CatalogFastMCP  # noqa: E402 — logging first


# Allow localhost by default; add LAN hosts via LLM_BAWT_MCP_ALLOWED_HOSTS env var.
# The MCP library matches host:port patterns — use ":*" suffix to allow any port.
_allowed_hosts = [
    h.strip() for h in os.getenv(
        "LLM_BAWT_MCP_ALLOWED_HOSTS",
        "127.0.0.1:*,localhost:*",
    ).split(",")
]
_allowed_origins = [f"http://{h}" for h in _allowed_hosts]


mcp = CatalogFastMCP(
    "bawthub",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    ),
)

# Suppress uvicorn access logs by setting log_level to WARNING
# We log our own human-friendly MCP operation summaries via ServiceLogger
mcp.settings.log_level = "WARNING"


def ensure_tools_registered() -> None:
    """Import the tool modules so ``mcp`` can actually dispatch by name.

    The app and the MCP server are separate processes in one container. In the
    MCP process ``server.py`` is the entry point and its module-level imports
    register every tool. The FastAPI process never imports it — it only reaches
    for the ``mcp`` singleton when executing an *approved* MCP call server-side.
    Without this the singleton is bare and trusted execution fails with
    "Unknown tool: ops_run" after the user already approved it (TASK-639).

    Importing ``server`` is registration-only: ``run_server()`` runs solely
    under ``__main__``. The import is lazy (``registry`` is imported *by*
    ``server``, so a module-level import here would cycle) and idempotent via
    ``sys.modules``.

    The FastAPI process calls this once in its lifespan, before it serves
    (TASK-1220). It must never run at approval time: the tree is bind-mounted,
    so a late import loads whatever is on disk *now* into a process still
    holding modules from its start.
    """
    from . import server  # noqa: F401

    mcp.mark_tools_registered()


def tools_registration_complete() -> bool:
    """Whether :func:`ensure_tools_registered` finished in this process."""
    return mcp.tools_registered


__all__ = ["mcp", "ensure_tools_registered", "tools_registration_complete"]
