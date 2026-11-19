"""Render a user's approved guidance into a prompt section (TASK-893 slice 6).

Dormant: the caller renders nothing unless the bot-resolvable setting
``guidance_prompt_feed`` is true. Only the registry's current approved version
is read, never pending proposals. The section holds the mandate verbatim, then
the statements of active preferences in document order; evidence, ids and
superseded rows never reach a prompt.

Everything fails closed: a protected or unknown bot, a malformed document or a
read error yields no section. Text is never truncated. If the heading plus
mandate alone exceeds the cap the whole section is omitted; otherwise whole
statements are dropped from the end until the section fits.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from .guidance import current_version, validate_doc
from .guidance_run import resolve_protected

SETTING = "guidance_prompt_feed"
MAX_CHARS = 4000
HEADING = "## The User's Approved Guidance"
PREFS_HEADING = "Preferences:"


@dataclass(frozen=True)
class Rendered:
    body: str | None
    provenance: dict[str, Any] = field(default_factory=dict)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _strict(doc: dict[str, Any]) -> tuple[str, list[str]]:
    """Mandate and active statements, refusing any non-text rendering value."""
    mandate = doc["mandate"]
    if not isinstance(mandate, str):
        raise ValueError("mandate must be text")
    active = []
    for p in doc["preferences"]:
        if not isinstance(p.get("statement"), str) or not isinstance(p.get("status"), str):
            raise ValueError("preference statement and status must be text")
        if p["status"] == "active":
            active.append(p["statement"])
    return mandate.strip(), [s.strip() for s in active if s.strip()]


def render(version: dict[str, Any] | None, max_chars: int = MAX_CHARS) -> Rendered:
    """Pure renderer over a ``current_version`` row. Raises on a malformed doc."""
    if version is None:
        return Rendered(None, {"reason": "no_guidance"})
    doc = validate_doc(json.loads(version["body"]))
    mandate, active = _strict(doc)
    prov: dict[str, Any] = {"version_row_id": version["row_id"],
                            "version": version["version"],
                            "source_sha256": version["sha256"],
                            "active_count": len(active)}
    head = HEADING + (f"\n{mandate}" if mandate else "")
    if len(head) > max_chars:
        return Rendered(None, {**prov, "reason": "mandate_over_cap", "dropped_count": len(active)})
    kept = list(active)

    def compose(items: list[str]) -> str:
        if not items:
            return head
        return head + f"\n\n{PREFS_HEADING}\n" + "\n".join(f"- {s}" for s in items)

    body = compose(kept)
    while len(body) > max_chars:
        kept.pop()
        body = compose(kept)
    if not mandate and not kept:
        return Rendered(None, {**prov, "reason": "empty", "dropped_count": len(active)})
    return Rendered(body, {**prov, "dropped_count": len(active) - len(kept),
                           "rendered_sha256": _sha(body)})


def load(engine, *, bot_id: str, user_id: str, max_chars: int = MAX_CHARS) -> Rendered:
    """Protection check, then the current approved version, then render."""
    with engine.connect() as conn:
        protected = resolve_protected(conn)
        if not protected:
            return Rendered(None, {"reason": "protection_unresolved"})
        if bot_id.lower() in protected:
            return Rendered(None, {"reason": "protected_bot"})
        known = conn.execute(text("SELECT 1 FROM bot_profiles WHERE slug = lower(:b)"),
                             {"b": bot_id}).first()
        if known is None:
            return Rendered(None, {"reason": "unknown_bot"})
        version = current_version(conn, user_id)
    return render(version, max_chars)
