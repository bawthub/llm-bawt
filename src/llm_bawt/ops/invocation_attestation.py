"""Scoped app verification handoff; ops never decrypts agent turn capabilities."""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Literal

from agent_bridge.mcp_call_context import canonical_invocation_hash
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .control_protocol import MAX_BODY_BYTES, control_token
from .validation import canonical_json

DOMAIN = "ops-approval-invocation/v1"
TTL_SECONDS = 120
CLOCK_SKEW_SECONDS = 5


class AttestationError(ValueError):
    """An app-verified invocation cannot be established; create nothing."""


class Caller(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_min_length=1, str_max_length=128)
    session_id: str
    turn_id: str
    trigger_message_id: str
    bot_id: str
    user_id: str
    issued_at: int = Field(ge=0)
    agent_request_id: str
    session_key: str
    backend: Literal["claude-code", "codex"]
    tool_use_id: str


class Invocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol: Literal["ops-approval-invocation/v1"] = DOMAIN
    arguments: dict[str, Any]
    invocation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    caller: Caller
    attested_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @property
    def request_id(self) -> str:
        identity = f"{self.caller.backend}\0{self.caller.tool_use_id}\0{self.invocation_hash}"
        return "mcp-appr-" + hashlib.sha256(identity.encode()).hexdigest()[:40]


def _mac(payload: dict, token_source) -> str:
    # Separate this attestation from bearer authentication and other MAC domains.
    message = (DOMAIN + "\0" + canonical_json(payload)).encode()
    return hmac.new((token_source or control_token)().encode(), message, hashlib.sha256).hexdigest()


def mint(arguments: dict, caller: dict, invocation_hash: str, *, now=None,
         token_source=None) -> dict:
    """App only: call after the existing turn AND exact-call verification."""
    issued = int(time.time() if now is None else now)
    body = Invocation(arguments=arguments, caller=Caller.model_validate(caller),
                      invocation_hash=invocation_hash, attested_at=issued,
                      expires_at=issued + TTL_SECONDS).model_dump()
    return {"payload": body, "signature": _mac(body, token_source)}


def verify(envelope: Any, *, now=None, token_source=None) -> Invocation:
    """Ops only: scoped export suffices; no master-key or capability fallback."""
    try:
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
            raise ValueError("envelope fields")
        if len(canonical_json(envelope).encode()) > MAX_BODY_BYTES:
            raise ValueError("envelope size")
        signature = envelope["signature"]
        if not isinstance(signature, str) or not hmac.compare_digest(
                signature, _mac(envelope["payload"], token_source)):
            raise ValueError("signature")
        invocation = Invocation.model_validate(envelope["payload"])
        clock = int(time.time() if now is None else now)
        if (invocation.attested_at > clock + CLOCK_SKEW_SECONDS
                or not 0 < invocation.expires_at - invocation.attested_at <= TTL_SECONDS
                or clock > invocation.expires_at):
            raise ValueError("expiry")
        if invocation.invocation_hash != canonical_invocation_hash("ops_run", invocation.arguments):
            raise ValueError("invocation hash")
        return invocation
    except (ValueError, TypeError, OverflowError, ValidationError) as exc:
        raise AttestationError("Invalid or expired app-verified ops invocation") from exc
