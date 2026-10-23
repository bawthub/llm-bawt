"""Bot-bound internal service identity, never minted from an MCP argument."""
from __future__ import annotations

import json
from dataclasses import dataclass

from agent_bridge.mcp_call_context import canonical_invocation_hash
from cryptography.fernet import InvalidToken

from llm_bawt.service.providers.crypto import _get_fernet
from .policy import BotAccess, BotAccessError, TrustedCaller

# These operations use a supplied user only as a read/close filter. Never add
# authorship/session-creation/memory writes here: their existing user pins apply.
SERVICE_SELF_TOOLS = frozenset({
    "messages_get", "sessions_get", "sessions_close", "sessions_list", "sessions_get_active",
})
SERVICE_SELF_STORAGE_OPS = frozenset({
    "get_messages", "get_session", "close_session", "list_sessions", "get_active_session",
})


def is_service_self(access: BotAccess, target: str) -> bool:
    """A verified user-less app principal may filter only its own bot's data."""
    caller = access.caller
    if caller is None:
        return False
    caller.validate()
    return caller.kind == "service" and caller.user_id is None and target == caller.bot_id


SERVICE_HEADER = "X-LLM-Bawt-MCP-Service"
SERVICE_TTL_SECONDS = 120


@dataclass(frozen=True)
class ServicePrincipal:
    bot_id: str
    user_id: str | None

    def caller(self) -> TrustedCaller:
        caller = TrustedCaller(self.bot_id, self.user_id, "service")
        caller.validate()
        return caller

    def mint(self, name: str, arguments: dict) -> str:
        caller = self.caller()
        payload = {
            "purpose": "mcp-service", "version": 1,
            "bot_id": caller.bot_id, "user_id": caller.user_id,
            "invocation": canonical_invocation_hash(name, arguments),
        }
        return _get_fernet().encrypt(json.dumps(payload, sort_keys=True).encode()).decode("ascii")


def open_service_capability(token: str, name: str, arguments: dict) -> TrustedCaller:
    try:
        signer = _get_fernet()
        raw = signer.decrypt(token.encode("ascii"), ttl=SERVICE_TTL_SECONDS)
        payload = json.loads(raw)
        if (not isinstance(payload, dict) or payload.get("purpose") != "mcp-service"
                or type(payload.get("version")) is not int or payload["version"] != 1
                or payload.get("invocation") != canonical_invocation_hash(name, arguments)):
            raise ValueError("invalid service capability")
        caller = TrustedCaller(payload["bot_id"], payload["user_id"], "service",
                               signer.extract_timestamp(token.encode("ascii")) + SERVICE_TTL_SECONDS)
        caller.validate()
        return caller
    except (InvalidToken, ValueError, TypeError, KeyError, UnicodeError, AttributeError):
        raise BotAccessError("caller_unavailable") from None
