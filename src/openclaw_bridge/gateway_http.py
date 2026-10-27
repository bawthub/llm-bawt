"""Lightweight HTTP client for the OpenClaw Gateway.

Used by the main app to send messages and fetch history directly via HTTP
instead of going through the WS bridge process.
"""

from __future__ import annotations

import logging
import uuid

import httpx

logger = logging.getLogger(__name__)


class GatewayHttpClient:
    def __init__(self, base_url: str, token: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=60.0,
        )

    @property
    def token(self) -> str:
        """The bearer this client was built with (callers rebuild on rotation)."""
        return self._token

    async def send_user_message(self, session_key: str, text: str) -> str:
        """Send a user message via the Gateway HTTP API. Returns run_id."""
        idempotency_key = f"idem_{uuid.uuid4().hex}"
        resp = await self._client.post(
            "/v1/chat/send",
            json={
                "sessionKey": session_key,
                "message": text,
                "idempotencyKey": idempotency_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("runId") or data.get("run_id") or idempotency_key)

    async def get_chat_history(self, session_key: str, *, limit: int = 50) -> list[dict]:
        """Fetch chat history for a session."""
        resp = await self._client.post(
            "/v1/chat/history",
            json={"sessionKey": session_key, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("messages") or data.get("history") or []

    async def close(self) -> None:
        await self._client.aclose()
