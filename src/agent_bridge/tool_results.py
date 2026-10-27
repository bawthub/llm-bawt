"""Lossless tool-result serialization shared by agent bridges and the app.

The provider-facing bridges may transiently carry the complete accepted result on
short-lived per-run streams. Public unified events and history responses use the
bounded preview plus metadata produced here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

TOOL_RESULT_PREVIEW_CHARS = 2_000
TOOL_RESULT_PAYLOAD_VERSION = 1
STRUCTURED_TOOL_RESULT_TYPE = "application/x-tool-blocks+json"

# Token accounting for tool results. tiktoken is a hard dependency; we use a
# modern OpenAI encoding (o200k_base) as a deterministic ESTIMATE of the text
# portion of a result — it is not the provider's exact tokenizer (Claude != o200k)
# and does not account for image blocks or prompt-cache framing. The encoder is
# resolved once per process (the first call may download+parse the vocab); any
# failure degrades to the repo-wide len//4 heuristic so a tool turn never dies
# over token counting.
_TOKEN_ENCODING_NAME = "o200k_base"
_TOKEN_ENCODER: Any = None
_TOKEN_ENCODER_RESOLVED = False


def _get_token_encoder() -> Any:
    global _TOKEN_ENCODER, _TOKEN_ENCODER_RESOLVED
    if _TOKEN_ENCODER_RESOLVED:
        return _TOKEN_ENCODER
    _TOKEN_ENCODER_RESOLVED = True
    try:
        import tiktoken

        _TOKEN_ENCODER = tiktoken.get_encoding(_TOKEN_ENCODING_NAME)
    except Exception:
        _TOKEN_ENCODER = None
    return _TOKEN_ENCODER


def count_result_tokens(content: str) -> int:
    """Estimate the token count of a tool-result string.

    Uses tiktoken when available; falls back to the len//4 heuristic. Never
    raises — tool results contain arbitrary text (including literal special-token
    sequences), so encoding disables special-token handling.
    """
    if not content:
        return 0
    encoder = _get_token_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(content, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(content) // 4)


# TASK-1585: provenance for token accounting. A producer that serialized the
# provider-delivered result itself (so inline media was stripped before
# counting) stamps this version; payloads without it — old producers, or the
# app's fallback from a bare result string — are "unverified" in reports.
TOOL_RESULT_ACCOUNTING_VERSION = 2


def _without_inline_images(value: Any, omitted: list[int] | None = None) -> Any:
    """Keep structured result shape without duplicating base64 image bytes.

    Screenshot bytes already have their own media-asset lifecycle. Retaining a
    small descriptor here keeps the result truthful and inspectable without
    storing the same multi-megabyte image twice. ``omitted`` (a one-item
    counter) receives the number of blocks replaced.
    """
    if isinstance(value, list):
        return [_without_inline_images(item, omitted) for item in value]
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        source = value.get("source")
        if block_type in {"image", "image_url"} or (
            isinstance(source, dict) and source.get("type") == "base64"
        ):
            if omitted is not None:
                omitted[0] += 1
            media_type = (
                source.get("media_type") if isinstance(source, dict) else value.get("mimeType")
            )
            return {"type": block_type or "image", "media_type": media_type, "omitted": True}
        return {str(key): _without_inline_images(item, omitted) for key, item in value.items()}
    return value


def _provenance_int(value: Any) -> int | None:
    """A provenance field is trusted only as a non-negative JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass(frozen=True)
class ToolResultPayload:
    version: int
    content: str
    content_type: str
    total_chars: int
    total_bytes: int
    total_tokens: int
    sha256: str
    complete: bool
    # TASK-1585 provenance (optional dict fields within payload version 1):
    # ``accounting`` is TOOL_RESULT_ACCOUNTING_VERSION only when the producer
    # serialized the provider value itself; ``media_blocks`` counts inline media
    # blocks it removed. None means unverified.
    media_blocks: int | None = None
    accounting: int | None = None

    @classmethod
    def from_producer(cls, value: Any, *, complete: bool = True) -> "ToolResultPayload":
        """Serialize a provider-delivered value and stamp accounting provenance.

        Only bridges/clients holding the provider's own value (not an already
        stringified copy) may call this.
        """
        return cls.from_value(value, complete=complete, _accounted=True)

    @classmethod
    def from_value(
        cls, value: Any, *, complete: bool = True, _accounted: bool = False,
    ) -> "ToolResultPayload":
        omitted = [0]
        if isinstance(value, str):
            content = value
            content_type = "text/plain"
        elif value is None:
            content = ""
            content_type = "text/plain"
        else:
            normalized = _without_inline_images(value, omitted)
            content = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            content_type = STRUCTURED_TOOL_RESULT_TYPE
        raw = content.encode("utf-8")
        return cls(
            version=TOOL_RESULT_PAYLOAD_VERSION,
            content=content,
            content_type=content_type,
            total_chars=len(content),
            total_bytes=len(raw),
            total_tokens=count_result_tokens(content),
            sha256=hashlib.sha256(raw).hexdigest(),
            complete=bool(complete),
            media_blocks=omitted[0] if _accounted else None,
            accounting=TOOL_RESULT_ACCOUNTING_VERSION if _accounted else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultPayload":
        if int(data.get("version") or 0) != TOOL_RESULT_PAYLOAD_VERSION:
            raise ValueError("unsupported tool-result payload version")
        content = data.get("content")
        if not isinstance(content, str):
            raise ValueError("tool-result payload content must be text")
        payload = cls.from_value(content, complete=bool(data.get("complete", False)))
        content_type = str(data.get("content_type") or "text/plain")
        accounting = _provenance_int(data.get("accounting"))
        if accounting != TOOL_RESULT_ACCOUNTING_VERSION:
            accounting = None
        media_blocks = _provenance_int(data.get("media_blocks")) if accounting else None
        return cls(
            version=TOOL_RESULT_PAYLOAD_VERSION,
            content=content,
            content_type=content_type,
            total_chars=len(content),
            total_bytes=len(content.encode("utf-8")),
            total_tokens=count_result_tokens(content),
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            complete=bool(data.get("complete", False)),
            media_blocks=media_blocks,
            accounting=accounting if media_blocks is not None else None,
        )

    @property
    def preview(self) -> str:
        return self.content[:TOOL_RESULT_PREVIEW_CHARS]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def result_meta(
        self,
        *,
        record_id: int | None = None,
        available: bool,
    ) -> dict[str, Any]:
        return {
            "record_id": record_id,
            "preview_chars": len(self.preview),
            "total_chars": self.total_chars,
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "sha256": self.sha256,
            "content_type": self.content_type,
            "complete": self.complete,
            "available": bool(available),
            "legacy": False,
        }


def payload_from_event(payload: Any, result: Any) -> ToolResultPayload:
    """Prefer a versioned producer payload; tolerate rolling-deploy producers."""
    if isinstance(payload, dict):
        try:
            return ToolResultPayload.from_dict(payload)
        except (TypeError, ValueError):
            pass
    return ToolResultPayload.from_value(result, complete=True)
