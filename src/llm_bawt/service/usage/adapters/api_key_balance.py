"""Pay-per-token API keys whose balance is a manual anchor (TASK-1461).

A connected API key bills per token. When the provider has no balance API that
the connected inference key can read, the balance is the amount the user enters,
which ``usage.balance`` counts down by each finished turn's recorded cost. A
subclass names the provider connection (``providers.registry`` id) whose key
pays for its catalog access paths. Nothing else differs, so a new pay-per-token
API-key provider is one data-only subclass registered in ``usage/__init__``.

Elsewhere: Moonshot reports its own balance (``moonshot.py``). z.ai and Kimi For
Coding keys are subscription plans with usage windows, not per-token credit.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from ..balance import BALANCE_ANCHOR
from ..base import UsageAdapter
from ..canonical import STATUS_OK, STATUS_UNAUTHORIZED


def utc_datetime(value: Any) -> datetime | None:
    """An aware UTC datetime from an ISO-8601 string; None if naive or invalid."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError):
        return None


class ApiKeyBalanceAdapter(UsageAdapter):
    """A connected pay-per-token API key: no plan windows, an anchored balance."""

    kind = "balance"
    balance_mode = BALANCE_ANCHOR
    connection_id: str = ""  # providers.registry id whose stored key pays these paths
    key_env_vars: tuple[str, ...] = ()  # legacy env keys the inference clients still accept

    def _connection(self):
        from ...dependencies import get_service
        from ...providers.base import CredentialStore

        return CredentialStore(get_service().config).load(self.connection_id)

    def _key_configured(self) -> bool:
        from ...providers.base import STATUS_CONNECTED

        record = self._connection()
        if record is not None and record.status == STATUS_CONNECTED:
            return True
        return any(os.getenv(name) for name in self.key_env_vars)

    def credential_connected_at(self) -> datetime | None:
        record = self._connection()
        return utc_datetime(record.connected_at) if record else None

    async def fetch(self):
        if not await asyncio.to_thread(self._key_configured):
            return self._base(available=False, status=STATUS_UNAUTHORIZED, limits=[],
                              error=f"No {self.display_name} key connected.")
        return self._base(available=True, status=STATUS_OK, limits=[])


class OpenAIApiUsageAdapter(ApiKeyBalanceAdapter):
    provider = "openai_api"
    display_name = "OpenAI API"
    access_path_keys = ("openai-api", "openai-responses")
    connection_id = "openai-api"
    key_env_vars = ("OPENAI_API_KEY",)


class OpenRouterUsageAdapter(ApiKeyBalanceAdapter):
    # OpenRouter's account credit endpoint (GET /api/v1/credits) requires a
    # management key; the connected key is an inference key, so it is anchored.
    provider = "openrouter"
    display_name = "OpenRouter"
    backend = "claude-code"  # reached via the claude-code bridge proxy
    access_path_keys = ("openrouter-anthropic",)
    connection_id = "openrouter"
