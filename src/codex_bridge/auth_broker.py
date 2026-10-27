"""Materialize the app-owned Codex OAuth bundle for the native CLI."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import httpx

from agent_bridge import bridge_token

logger = logging.getLogger(__name__)


def ensure_auth_file(auth_file: Path) -> bool:
    """Sync on every startup, including when a previous auth file exists."""
    return materialize_auth_from_broker(auth_file) or auth_file.exists()


def materialize_auth_from_broker(auth_file: Path) -> bool:
    """Refresh the CLI's local copy without ever refreshing OAuth in the bridge."""
    api_url = (os.environ.get("LLM_BAWT_API_URL") or "").rstrip("/")
    if not api_url:
        logger.warning("LLM_BAWT_API_URL not set — cannot materialize auth.json from broker")
        return False
    headers = bridge_token.headers()
    attempts = max(1, int(os.getenv("CODEX_AUTH_BROKER_ATTEMPTS", "8")))
    delay = max(0.0, float(os.getenv("CODEX_AUTH_BROKER_RETRY_SECONDS", "1")))
    for attempt in range(1, attempts + 1):
        try:
            resp = httpx.get(
                f"{api_url}/v1/providers/openai_chatgpt/bundle",
                headers=headers,
                timeout=10.0,
            )
            if not resp.is_error:
                bundle = resp.json()
                auth_file.parent.mkdir(parents=True, exist_ok=True)
                # A concurrent CLI process must never observe a truncated bundle.
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=auth_file.parent, prefix=".auth-",
                    delete=False, encoding="utf-8",
                ) as temp:
                    temp_path = Path(temp.name)
                    try:
                        os.chmod(temp_path, 0o600)
                        json.dump(bundle, temp, indent=2)
                        temp.flush()
                        os.replace(temp_path, auth_file)
                    finally:
                        temp_path.unlink(missing_ok=True)
                logger.info("Materialized auth.json from app broker (path=%s)", auth_file)
                return True
            logger.warning(
                "Auth bundle broker returned %s on attempt %d/%d",
                resp.status_code, attempt, attempts,
            )
            if resp.status_code < 500:
                return False
        except Exception as exc:  # noqa: BLE001 — broker may be unavailable at startup
            logger.warning(
                "Auth bundle broker unreachable on attempt %d/%d: %s",
                attempt, attempts, exc,
            )
        if attempt < attempts:
            time.sleep(delay)
    return False
