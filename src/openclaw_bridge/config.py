"""Bridge service configuration — all from env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class BridgeConfig:
    # OpenClaw Gateway
    gateway_url: str = ""  # HTTP base URL (e.g. http://gateway-host:18789)
    ws_url: str = ""       # WS URL (e.g. ws://gateway-host:18789/v1/ws)
    ws_token: str = ""     # Bearer token for both HTTP and WS auth
    reconnect_max_delay: int = 60

    # PostgreSQL (for event store)
    postgres_user: str = "llm_bawt"
    postgres_password: str = ""
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "llm_bawt"

    # Redis (for event fanout)
    redis_url: str = "redis://localhost:6379/0"

    # Main app API URL (for fetching bot config at startup)
    app_api_url: str = ""  # e.g. http://app:8642

    # Health check
    health_port: int = 8680

    # Ingest filters (comma-separated; merged with built-in defaults)
    ingest_drop_patterns: str = ""   # regex patterns to drop from user message content
    ingest_drop_events: str = ""     # additional gateway event names to drop
    ingest_drop_msg_types: str = ""  # additional top-level msg types to drop

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> BridgeConfig:
        return cls(
            ws_url=os.environ.get("OPENCLAW_WS_URL", ""),
            ws_token=os.environ.get("OPENCLAW_GATEWAY_TOKEN", ""),
            reconnect_max_delay=int(os.environ.get("OPENCLAW_WS_RECONNECT_MAX_DELAY", "60")),
            postgres_user=os.environ.get("POSTGRES_USER", "llm_bawt"),
            postgres_password=os.environ.get("POSTGRES_PASSWORD", ""),
            postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
            postgres_database=os.environ.get("POSTGRES_DATABASE", "llm_bawt"),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            app_api_url=os.environ.get("LLM_BAWT_API_URL", ""),
            health_port=int(os.environ.get("OPENCLAW_BRIDGE_HEALTH_PORT", "8680")),
            ingest_drop_patterns=os.environ.get("OPENCLAW_INGEST_DROP_PATTERNS", ""),
            ingest_drop_events=os.environ.get("OPENCLAW_INGEST_DROP_EVENTS", ""),
            ingest_drop_msg_types=os.environ.get("OPENCLAW_INGEST_DROP_MSG_TYPES", ""),
            log_level=os.environ.get("OPENCLAW_BRIDGE_LOG_LEVEL", "INFO"),
        )

    @property
    def postgres_url(self) -> str:
        # The slim shared bridge image installs psycopg2, not psycopg 3.
        # SQLAlchemy 2.1 defaults bare postgresql:// URLs to psycopg 3.
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )

    def fetch_session_to_bot(self, *, retries: int = 10, delay: float = 5.0) -> dict[str, str]:
        """Fetch session_key -> bot_id mapping from the main app's /v1/bots API.

        Retries on failure since the main app may still be starting up.
        """
        if not self.app_api_url:
            return {}
        import json
        import logging
        import time
        import urllib.request
        logger = logging.getLogger(__name__)
        url = f"{self.app_api_url.rstrip('/')}/v1/bots"
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read())
                mapping: dict[str, str] = {}
                for bot in data.get("data", []):
                    if bot.get("agent_backend") != "openclaw":
                        continue
                    bc = bot.get("agent_backend_config") or {}
                    sk = bc.get("session_key", "")
                    if sk:
                        mapping[sk] = bot["slug"]
                return mapping
            except Exception as e:
                if attempt < retries:
                    logger.info("Waiting for main app API (%s, attempt %d/%d)...", e, attempt, retries)
                    time.sleep(delay)
                else:
                    logger.warning("Failed to fetch bot mapping from %s after %d attempts: %s", url, retries, e)
        return {}
