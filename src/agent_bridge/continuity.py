"""Shared continuity-seed mechanics for every agent bridge; never contains policy.

The app is the sole seed authority: it selects the messages, resolves the
framing text from its prompt registry and records the payload digest. Bridges
only render that payload into their provider's fresh session and report back.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

# Redis command field carrying the app-resolved ``{briefing, ack}`` framing.
SEED_FRAMING_FIELD = "inject_seed_framing"


def continuity_digest(messages: list[dict]) -> str:
    """Bind a bridge receipt to exactly the app-selected IDs and rendered text."""
    payload = [(m.get("source_id"), m.get("role"), m.get("content"), m.get("timestamp"))
               for m in messages]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class SeedFramingMissing(ValueError):
    """A non-empty seed arrived without the app's registry-resolved framing."""


@dataclass(frozen=True)
class SeedFraming:
    """Registry-owned text around a seed: the briefing header and the reply."""

    briefing: str
    ack: str

    def to_field(self) -> str:
        return json.dumps({"briefing": self.briefing, "ack": self.ack}, ensure_ascii=False)

    @classmethod
    def from_field(cls, raw) -> SeedFraming | None:
        """Decode the command field; anything malformed is absent, never a default."""
        if raw is None or raw == "":
            return None
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        briefing, ack = value.get("briefing"), value.get("ack")
        if not (isinstance(briefing, str) and briefing.strip() and isinstance(ack, str) and ack.strip()):
            return None
        return cls(briefing, ack)

    @staticmethod
    def require(framing: SeedFraming | None, messages: list | None) -> SeedFraming | None:
        """Refuse a non-empty seed without framing; bridges hold no fallback text."""
        if messages and framing is None:
            raise SeedFramingMissing("continuity seed arrived without its registry framing")
        return framing


def seed_receipt(messages: list | None) -> dict:
    """The ``new_session_ack`` fields binding a reset to the seed actually applied."""
    applied = list(messages or [])
    return {"seed_digest": continuity_digest(applied), "seeded_count": len(applied)}


def render_seed_briefing(messages: list[dict], briefing: str) -> str:
    """Flatten the seed into one labeled, text-only briefing block.

    Summaries keep their ``[Previous conversation X ago]`` headers; recent
    turns render as a transcript. No tool blocks, so a resumed session never
    wedges on an unmatched call.
    """
    lines = [briefing, ""]
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "summary":
            lines.append(content)
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant (you): {content}")
        else:
            lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()


def stats_from_messages(messages: list[dict]) -> dict:
    """Derive /new-ack stats from app-injected seed messages (TASK-501).

    approx_tokens is a cheap char/4 estimate; bridges have no tokenizer.
    """
    summary_count = sum(1 for m in messages if m.get("role") == "summary")
    convo_count = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
    approx_tokens = sum(len(m.get("content") or "") for m in messages) // 4
    ts = [float(m.get("timestamp") or 0.0) for m in messages if m.get("timestamp")]
    return {
        "summary_count": summary_count,
        "message_count": convo_count,
        "total_count": len(messages),
        "approx_tokens": approx_tokens,
        "oldest_timestamp": min(ts) if ts else None,
        "newest_timestamp": max(ts) if ts else None,
    }


def fmt_tokens(n: int | None) -> str:
    """Human-format a token count: 67288 -> '67.3k', 1486 -> '1.5k'."""
    if n is None:
        return "?"
    n = int(n)
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def format_seed_ack(stats: dict | None) -> str:
    """Human-facing /new acknowledgement, reporting seed stats when present."""
    base = "Session reset."
    if stats is None or stats.get("clean_start"):
        return f"{base} Ready for a new conversation."
    if not stats.get("seeded"):
        reason = stats.get("reason", "nothing to seed")
        return f"{base} History seeding on, but {reason}. Fresh start."
    summ = stats.get("summary_count", 0)
    msgs = stats.get("message_count", 0)
    toks = stats.get("approx_tokens", 0)
    span = ""
    oldest = stats.get("oldest_timestamp")
    newest = stats.get("newest_timestamp")
    if oldest and newest:
        try:
            o = datetime.fromtimestamp(oldest, timezone.utc).strftime("%Y-%m-%d")
            n = datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%d")
            span = f", spanning {o} → {n}" if o != n else f", from {o}"
        except Exception:
            span = ""
    return (
        f"{base} Seeded {summ} summary record{'s' if summ != 1 else ''} + "
        f"{msgs} recent message{'s' if msgs != 1 else ''} "
        f"(~{fmt_tokens(toks)} tokens){span}. Continuity restored."
    )


async def post_continuity_receipt(app_api_url: str | None, thread_id: str | None,
                                  bot_id: str, messages: list[dict], *, restored: bool,
                                  logger) -> None:
    """Confirm (or fail) the app's continuity manifest for this exact payload."""
    if not thread_id or not app_api_url:
        return
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(
                f"{app_api_url}/v1/sessions/{thread_id}/continuity-receipt",
                params={"bot_id": bot_id},
                json={"payload_digest": continuity_digest(messages), "restored": restored},
            )
            response.raise_for_status()
    except Exception:
        logger.warning("Continuity receipt unavailable for %s/%s", bot_id, thread_id, exc_info=True)
