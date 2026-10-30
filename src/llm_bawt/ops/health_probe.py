"""Catalog-declared post-swap health probe of an image-family target. STANDARD LIBRARY ONLY (TASK-1801).

A digest deploy, its rollback and the frontend config/routing reconciles used
to verify the new container with a fixed in-container request to
``http://127.0.0.1:3000/api/health``: the frontend's own port and path, wired
into the shared executor. A second image target (pocket-tts) has neither, so
the probe is data on the catalog row instead:

* ``{"type": "http_exec", "port": 3000, "path": "/api/health"}``: the worker
  runs a fixed ``curl`` argv inside the target (it stays network-less) and the
  JSON reply must say ``status: healthy`` and carry the bound release block.
* ``{"type": "docker_health"}``: the target's own Docker healthcheck is the
  probe (the swap already waits for ``healthy``). Identity is then checked on
  the container itself: it must run the approved image ID and that image's
  release labels must equal the bound release.

The app accepts a row with or without a probe (staging: rows are patched
before the worker is repinned); the worker built with this module refuses an
image-family spec without one before it touches anything, and the app refuses
to hand a probe-carrying spec to a worker that lacks :data:`WORKER_LABEL`.
"""
from __future__ import annotations

import re

WORKER_LABEL = "llm-bawt.ops.health-probe"
HTTP_EXEC = "http_exec"
DOCKER_HEALTH = "docker_health"
_PATH = re.compile(r"/[A-Za-z0-9._~/-]{0,127}")


def validate_health_probe(value) -> dict:
    """The ``health_probe`` spec field; raises ValueError."""
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ValueError("health_probe must be an object with a type")
    kind = value["type"]
    if kind == DOCKER_HEALTH:
        if set(value) != {"type"}:
            raise ValueError("health_probe docker_health takes no other fields")
        return value
    if kind == HTTP_EXEC:
        if set(value) != {"type", "port", "path"}:
            raise ValueError("health_probe http_exec requires exactly type, port, path")
        if type(value["port"]) is not int or not 1 <= value["port"] <= 65535:
            raise ValueError("health_probe port must be an integer in 1..65535")
        path = value["path"]
        if not isinstance(path, str) or not _PATH.fullmatch(path) or ".." in path.split("/") or "//" in path:
            raise ValueError("health_probe path must be a clean absolute path")
        return value
    raise ValueError(f"health_probe type must be {HTTP_EXEC} or {DOCKER_HEALTH}")


def http_exec_argv(probe: dict) -> list[str]:
    """The fixed in-container request of an ``http_exec`` probe."""
    return ["curl", "-fsS", "--max-time", "10", f"http://127.0.0.1:{probe['port']}{probe['path']}"]


__all__ = ["DOCKER_HEALTH", "HTTP_EXEC", "WORKER_LABEL", "http_exec_argv", "validate_health_probe"]
