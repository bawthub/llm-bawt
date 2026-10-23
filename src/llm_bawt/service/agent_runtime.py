"""Is the agent runtime that setup creates bots for reachable right now? (TASK-1297)

The setup wizard's first bot is a ``claude-code`` agent. Before creating one,
setup asks that bridge's live command consumer to answer a ``bridge.ping``
(:mod:`agent_bridge.ping`). The result is a preflight, not a promise: a bridge
that dies later surfaces through the normal turn-failure path.
"""

from __future__ import annotations

from agent_bridge.ping import PING_BUDGET_S, BridgeReadiness, probe_bridge

from ..utils.config import Config

#: The one backend setup provisions agents on (TASK-1293 route table).
SETUP_AGENT_BACKEND = "claude-code"


def _redis_factory(redis_url: str, budget_s: float):
    def factory():
        import redis.asyncio as aioredis

        # Reads block on XREAD; the probe's own deadline bounds everything, so
        # only the connect gets a socket-level bound.
        return aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=None,
            socket_connect_timeout=budget_s,
        )

    return factory


async def check_agent_runtime(
    config: Config,
    *,
    backend: str = SETUP_AGENT_BACKEND,
    budget_s: float = PING_BUDGET_S,
) -> BridgeReadiness:
    """Ping *backend*'s bridge within *budget_s*; never raises."""
    redis_url = (getattr(config, "REDIS_URL", "") or "").strip()
    if not redis_url:
        return BridgeReadiness(ready=False, backend=backend, instance=None, detail="Redis is not configured")
    return await probe_bridge(_redis_factory(redis_url, budget_s), backend, budget_s=budget_s)
