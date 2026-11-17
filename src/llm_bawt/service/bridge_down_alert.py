"""Tell the user when the Claude Code bridge stays down (TASK-2403).

Every agent watcher runs on that bridge, so its own outage cannot be noticed
from inside it. The app is a separate container: this loop asks the bridge's
health endpoint every minute and pushes one high-urgency notice once it has
failed for :data:`DOWN_AFTER_S`, again at most every :data:`RESEND_S` while it
stays down, and one notice when it recovers. A restart loop fails the same
probe. State is in memory: an app restart starts a fresh outage clock.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

CHECK_INTERVAL_S = 60.0
DOWN_AFTER_S = 300.0
RESEND_S = 3600.0
PROBE_TIMEOUT_S = 10.0
BRIDGE = "claude-code-bridge"
HEALTH_URL = "http://claude-code-bridge:8681/health"


async def probe_health(url: str = HEALTH_URL) -> bool:
    """Healthy only on HTTP 200 with ``redis: true`` (the reconcile's own contract)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as http:
            response = await http.get(url)
        return response.status_code == 200 and response.json().get("redis") is True
    except Exception:  # noqa: BLE001 - unreachable, refused, bad JSON: all down
        return False


def _minutes(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes} min" if minutes < 120 else f"{minutes // 60} h {minutes % 60} min"


class BridgeDownAlert:
    """``probe() -> bool`` and ``send(title, message, dedupe_key)``, both async."""

    def __init__(self, probe: Callable[[], Awaitable[bool]],
                 send: Callable[[str, str, str], Awaitable[object]], *, clock=time.monotonic):
        self.probe, self.send, self.clock = probe, send, clock
        self.down_since: float | None = None
        self.alerted_at: float | None = None
        self.alerts = 0
        self.outage_id = 0

    async def step(self) -> None:
        healthy = await self.probe()
        now = self.clock()
        if healthy:
            if self.alerted_at is not None:
                await self._send(f"{BRIDGE} recovered",
                                 f"{BRIDGE} is healthy again after about {_minutes(now - self.down_since)} "
                                 "down. Bots on it can work again; check for stalled tasks.", "recovered")
            self.down_since = self.alerted_at = None
            self.alerts = 0
            return
        if self.down_since is None:
            self.down_since = now
            self.outage_id = int(time.time())
        down_for = now - self.down_since
        if down_for < DOWN_AFTER_S or (self.alerted_at is not None and now - self.alerted_at < RESEND_S):
            return
        self.alerts += 1
        self.alerted_at = now
        await self._send(f"{BRIDGE} is down",
                         f"{BRIDGE} has failed its health check for about {_minutes(down_for)}. "
                         "Every Claude bot, and their watchers, are offline until it is fixed.",
                         f"down:{self.alerts}")

    async def _send(self, title: str, message: str, kind: str) -> None:
        key = f"bridge-down:{BRIDGE}:{self.outage_id}:{kind}"
        try:
            receipt = await self.send(title, message, key)
            result = receipt.get("result") if isinstance(receipt, dict) else receipt
            logger.warning("bridge down alert %s: %s", kind, result)
        except Exception:  # noqa: BLE001 - never kill the loop over one notice
            logger.exception("bridge down alert %s failed", kind)


async def run_bridge_down_alert(config) -> None:
    """App lifespan loop; cancelled at shutdown."""
    from ..house_fallback import push_system_notice

    user_id = str(getattr(config, "DEFAULT_USER", "") or "") or None

    async def send(title: str, message: str, dedupe_key: str):
        return await push_system_notice(bot_id="system", user_id=user_id, title=title,
                                        message=message, dedupe_key=dedupe_key)

    alert = BridgeDownAlert(probe_health, send)
    while True:
        try:
            await alert.step()
        except Exception:  # noqa: BLE001
            logger.exception("bridge down alert pass failed")
        await asyncio.sleep(CHECK_INTERVAL_S)
