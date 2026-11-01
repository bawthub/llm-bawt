"""Bot-bound internal service identity, never minted from an MCP argument."""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from agent_bridge.mcp_call_context import canonical_invocation_hash
from cryptography.fernet import InvalidToken

from llm_bawt.service.providers.crypto import _get_fernet
from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_TTL_SECONDS, open_task_turn_context
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
_runtime_turn: ContextVar[str | None] = ContextVar("mcp_runtime_service_turn", default=None)


@contextmanager
def bind_service_turn(capability: str | None):
    """Scope an EXISTING server-minted turn proof to runtime history writes.

    Never read this from MCP arguments or persist it on a cached MemoryClient.
    It travels inside the invocation-bound service envelope, NOT as an agent
    approval-context header: runtime persistence remains a service operation.
    """
    token = _runtime_turn.set(capability)
    try:
        yield
    finally:
        _runtime_turn.reset(token)


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
        raw = _runtime_turn.get()
        if name == "messages_add" and arguments.get("session_id") and raw is not None:
            # The proof was minted AFTER the runtime bound its committed
            # conversation. A raw session_id (even service-signed) is NOT proof.
            turn = open_task_turn_context(raw)
            if (turn.session_id == arguments["session_id"]
                    and turn.bot_id == caller.bot_id == arguments.get("bot_id")
                    and turn.user_id == caller.user_id == arguments.get("user_id")):
                payload["conversation_context"] = raw
        return _get_fernet().encrypt(json.dumps(payload, sort_keys=True).encode()).decode("ascii")


def _open_service(token: str, name: str, arguments: dict):
    try:
        signer = _get_fernet()
        raw = signer.decrypt(token.encode("ascii"), ttl=SERVICE_TTL_SECONDS)
        payload = json.loads(raw)
        if (not isinstance(payload, dict) or payload.get("purpose") != "mcp-service"
                or type(payload.get("version")) is not int or payload["version"] != 1
                or payload.get("invocation") != canonical_invocation_hash(name, arguments)):
            raise ValueError("invalid service capability")
        expires = signer.extract_timestamp(token.encode("ascii")) + SERVICE_TTL_SECONDS
        owner = None
        if "conversation_context" in payload:
            turn = open_task_turn_context(payload["conversation_context"])
            if (name != "messages_add" or turn.session_id != arguments.get("session_id")
                    or turn.bot_id != payload["bot_id"] or turn.user_id != payload["user_id"]
                    or turn.bot_id != arguments.get("bot_id")
                    or turn.user_id != arguments.get("user_id")):
                raise ValueError("invalid service conversation")
            expires = min(expires, turn.issued_at + TASK_TURN_CONTEXT_TTL_SECONDS)
            owner = (turn.bot_id, turn.user_id)
        caller = TrustedCaller(payload["bot_id"], payload["user_id"], "service", expires)
        caller.validate()
        return caller, owner
    except (InvalidToken, ValueError, TypeError, KeyError, UnicodeError, AttributeError):
        raise BotAccessError("caller_unavailable") from None


def open_service_capability(token: str, name: str, arguments: dict) -> TrustedCaller:
    return _open_service(token, name, arguments)[0]


def open_service_session_owner(token: str, name: str, arguments: dict):
    """Only the exact existing signed conversation, at BOTH authorization stages."""
    return _open_service(token, name, arguments)[1]
