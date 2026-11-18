"""What harness this bridge process runs (TASK-2470).

Collected once per process and cached: ``/versions`` and the house versions
key serve the cache, so no request ever runs a subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = 1
HARNESS_PACKAGES = ("claude-agent-sdk", "openai-codex", "openai-codex-cli-bin", "openai-codex-sdk")
CLI_TIMEOUT_S = 10.0

_cache: dict | None = None
_lock = asyncio.Lock()


def house_versions_key(house: str, backend: str) -> str:
    return f"agent:house:{house}:versions:{backend}"


def mansion_versions_key(backend: str) -> str:
    return f"agent:mansion:versions:{backend}"


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        logger.debug("version lookup failed for %s", name, exc_info=True)
        return None


def bundled_claude_cli() -> Path | None:
    try:
        import claude_agent_sdk
    except Exception:
        return None
    path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return path if path.is_file() else None


def cli_version(path: Path | None) -> str | None:
    """``<cli> --version`` once; the first token (``2.1.3 (Claude Code)``)."""
    if path is None:
        return None
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                             timeout=CLI_TIMEOUT_S, check=False)
    except Exception:
        logger.warning("bundled CLI version check failed", exc_info=True)
        return None
    text = (out.stdout or "").strip()
    return text.split()[0] if out.returncode == 0 and text else None


def collect() -> dict:
    return {
        "schema": SCHEMA,
        "harness": {name: package_version(name) for name in HARNESS_PACKAGES},
        "claude_cli": cli_version(bundled_claude_cli()),
        "image": os.environ.get("BAWT_IMAGE_REF") or None,
        "started_at": datetime.now(UTC).isoformat(),
        "instance": f"{socket.gethostname()}:{uuid.uuid4().hex[:8]}",
    }


async def current() -> dict:
    """The cached record; the first call collects it off the event loop."""
    global _cache
    if _cache is None:
        async with _lock:
            if _cache is None:
                _cache = await asyncio.to_thread(collect)
    return _cache


async def publish_house(redis: Any, house: str, backend: str, epoch: str, ttl_s: int) -> None:
    """Bind this process's versions to the lease epoch it holds."""
    record = {"epoch": str(epoch), "versions": await current()}
    await redis.set(house_versions_key(house, backend), json.dumps(record), ex=ttl_s)


async def publish_mansion(redis: Any, backend: str) -> None:
    """Last report from the mansion bridge; no lease, so never shown as running."""
    await redis.set(mansion_versions_key(backend), json.dumps({"versions": await current()}))
