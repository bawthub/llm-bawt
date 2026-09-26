"""Durable native Responses reasoning carried through an SDK thinking signature.

The readable thinking text is display-only. This versioned envelope carries the
original replay item, not a reconstruction from that text. It survives JSONL
resume without a process-local lookup table. Foreign/legacy signatures remain
opaque and are never guessed to be OpenAI state.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

SIGNATURE_PREFIX = "bawt-responses-reasoning-v1:"
# Bound decoding of persisted/user-supplied envelopes, not history retention.
MAX_SIGNATURE_CHARS = 16 * 1024 * 1024
_ITEM_FIELDS = ("id", "type", "summary", "content", "encrypted_content", "status")


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _json_value(part) for key, part in value.items()}
    if isinstance(value, list):
        return [_json_value(part) for part in value]
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return value


@dataclass(frozen=True)
class ReasoningCodec:
    """Request-local replay compatibility, independent of context lifecycle.

    Exact-model matching is deliberately conservative until cross-model opaque
    state compatibility is verified. Account scope survives token refresh but
    excludes account/endpoint changes. Only the OpenAI adapter opts in today.
    """

    provider: str
    model: str
    scope: str

    @classmethod
    def for_account(cls, provider: str, model: str, base_url: str, account: str):
        scope = hashlib.sha256(
            json.dumps([base_url.rstrip("/"), account], separators=(",", ":")).encode()
        ).hexdigest()
        return cls(provider, model, scope)

    def encode(self, item: Any) -> str | None:
        raw = _json_value(item)
        if not isinstance(raw, dict):
            return None
        native = {key: raw[key] for key in _ITEM_FIELDS if key in raw}
        if not native.get("encrypted_content"):
            return None
        if not self._valid_item(native):
            raise ValueError("Malformed native reasoning item from upstream")
        envelope = {"provider": self.provider, "model": self.model,
                    "scope": self.scope, "item": native}
        signature = SIGNATURE_PREFIX + base64.b64encode(
            json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()
        ).decode("ascii")
        if len(signature) > MAX_SIGNATURE_CHARS:
            raise ValueError("Responses reasoning envelope exceeds transport limit")
        return signature

    def decode(self, signature: Any) -> dict | None:
        if not isinstance(signature, str) or not signature.startswith(SIGNATURE_PREFIX):
            return None
        if len(signature) > MAX_SIGNATURE_CHARS:
            raise ValueError("Responses reasoning envelope exceeds transport limit")
        try:
            envelope = json.loads(base64.b64decode(
                signature[len(SIGNATURE_PREFIX):], validate=True,
            ))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Malformed Responses reasoning envelope") from exc
        if not isinstance(envelope, dict):
            raise ValueError("Malformed Responses reasoning envelope")
        if any(envelope.get(key) != getattr(self, key)
               for key in ("provider", "model", "scope")):
            return None
        item = envelope.get("item")
        if not self._valid_item(item):
            raise ValueError("Malformed native reasoning item in envelope")
        return item

    @staticmethod
    def _valid_item(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and not (set(item) - set(_ITEM_FIELDS))
            and item.get("type") == "reasoning"
            and isinstance(item.get("id"), str) and bool(item["id"])
            and isinstance(item.get("encrypted_content"), str)
            and bool(item["encrypted_content"])
            and isinstance(item.get("summary"), list)
            and all(isinstance(part, dict) and part.get("type") == "summary_text"
                    and isinstance(part.get("text"), str) for part in item["summary"])
        )
