"""MCP tool for agent self-recap.

Registers ``self_recap`` on the shared FastMCP server. When an agent (or the
user, via the agent) asks for a "self recap", this:

  1. Pulls the agent's last N hours of conversation (sent + received).
  2. Ships that transcript to Grok with a structured handoff prompt.
  3. Stores the result as a ``role='summary'`` record (the existing summary
     system) so it survives the session.
  4. Returns the recap text — which, as an MCP tool result, lands in the
     agent's SDK transcript and is replayed into context on the next
     ``resume=`` turn. That is the injection: we lean on the SDK's own
     transcript mechanism rather than fighting it.

Imported by server.py to trigger tool registration on startup.

The prompt has two co-equal priorities: an EXHAUSTIVE itemized TOPIC LOG (every
topic discussed, done or not) and the PENDING / INCOMPLETE work a cold-context
agent must finish. Claimed-vs-verified is enforced throughout.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from .recap_prompt import RECAP_SYSTEM_PROMPT
from .server import mcp

logger = logging.getLogger(__name__)

# Default Grok model for the recap. Override with LLM_BAWT_SELF_RECAP_MODEL.
# Must be a real xAI API model_id (see `model_definitions` rows, type='grok').
_DEFAULT_RECAP_MODEL = "grok-4-fast-non-reasoning"

# Roles that represent real conversation bubbles (what the user sees in the app).
# Everything else — notably role='summary' (prior recaps) — is excluded so the
# recap summarizes the actual transcript, not a pile of old summaries.
_RECAP_ROLES = {"user", "assistant"}

# Cap on transcript size shipped to Grok. Oldest messages are dropped first if
# exceeded, and the drop is reported (never silent) so the recap can't silently
# claim to have seen the whole window. Default 600k chars (~150k tokens) keeps a
# busy 24h intact while staying well under grok-4-fast's context. Override with
# LLM_BAWT_SELF_RECAP_MAX_CHARS for very heavy windows.
try:
    _MAX_TRANSCRIPT_CHARS = int(os.getenv("LLM_BAWT_SELF_RECAP_MAX_CHARS", "600000"))
except ValueError:
    _MAX_TRANSCRIPT_CHARS = 600_000

# Base URL of the llm-bawt app API. The MCP server runs in-process inside the
# app container, so localhost is the app itself. Overridable for tests / alt
# deployments. (server.py hardcodes this same host in three places — a shared
# constant would be the right consolidation, tracked separately.)
_APP_BASE_URL = os.getenv("LLM_BAWT_APP_BASE_URL", "http://localhost:8642").rstrip("/")


# The recap system prompt lives in the prompt registry (key 'self_recap.system')
# so it is editable from the prompts API/UI with version history, exactly like
# every other LLM prompt in the system. The canonical default body is defined in
# recap_prompt.py (a side-effect-free module the registry can import without
# triggering MCP tool registration). This alias is the last-resort fallback used
# only if the registry is unavailable.
_RECAP_SYSTEM_PROMPT = RECAP_SYSTEM_PROMPT


def _resolve_recap_system_prompt(bot_id: str, config) -> str:
    """Resolve the live recap system prompt: DB override → code default.

    Resolves the ``self_recap.system`` key from the prompt registry, preferring a
    bot-scoped override, then global, then the in-code default. Falls back to the
    code default on any registry error so the tool never breaks on a store hiccup.
    """
    try:
        from llm_bawt.prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver(config).resolve(
            "self_recap.system", scope_type="bot", scope_id=bot_id
        )
        if resolved and (resolved.body or "").strip():
            return resolved.body
    except Exception as e:
        logger.warning(
            "self_recap prompt registry resolve failed, using code default: %s", e
        )
    return RECAP_SYSTEM_PROMPT


def _format_transcript(messages: list[dict]) -> tuple[str, int, bool]:
    """Render messages into a readable transcript.

    Returns (transcript_text, used_count, truncated). Drops oldest messages
    first if the char cap is exceeded.
    """
    lines: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "?").upper()
        content = msg.get("content") or ""
        ts = msg.get("timestamp")
        when = ""
        if ts:
            try:
                when = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            except (ValueError, OSError, OverflowError):
                when = ""
        header = f"[{role}{(' @ ' + when) if when else ''}]"
        lines.append(f"{header}\n{content}")

    truncated = False
    # Trim from the oldest end until under the cap.
    while lines and sum(len(x) for x in lines) > _MAX_TRANSCRIPT_CHARS:
        lines.pop(0)
        truncated = True

    return "\n\n".join(lines), len(lines), truncated


@mcp.tool(name="self_recap")
async def self_recap(
    bot_id: str = "default",
    hours: int = 24,
    days: int = 0,
    model: str | None = None,
    store: bool = True,
) -> dict:
    """Summarize this agent's recent history into a cold-start continuation briefing.

    Pulls the last ``hours`` of conversation (sent + received), sends it to Grok
    with a structured handoff prompt, optionally stores the result as a summary
    record, and returns the recap. The briefing leads with an exhaustive itemized
    TOPIC LOG of everything discussed, then details the PENDING / INCOMPLETE work
    so an agent with completely clear context can pick up and continue.

    The returned recap text lands in this agent's SDK transcript (it's a tool
    result), so it is replayed into context on the next resumed turn.

    Args:
        bot_id: Bot namespace whose history to recap (pass your own slug).
        hours: Look-back window in hours (default 24).
        days: Additional look-back in days, stacked on top of ``hours``
            (e.g. days=7 → last week; days=7, hours=0 → also last week).
        model: Override the Grok model_id (default env LLM_BAWT_SELF_RECAP_MODEL
            or grok-4-fast-non-reasoning).
        store: Whether to persist the recap as a role='summary' record.

    Returns:
        Dict with: recap (text), stored (bool), summary_id, window_hours,
        messages_analyzed, model, truncated, and on failure, error.
    """
    # hours and days stack into one effective look-back window (min 1h).
    window_hours = max(1, int(hours) + int(days) * 24)
    window_label = (
        f"{days}d{f' {hours}h' if hours else ''}" if days else f"{hours}h"
    )
    logger.debug(
        "MCP tool invoked: self_recap bot_id=%s hours=%s days=%s window=%s",
        bot_id, hours, days, window_label,
    )

    from .server import _get_storage

    storage = _get_storage()
    since_seconds = window_hours * 3600

    # raw=True bypasses the summary-aware manager and reads the messages table
    # directly: real bubbles regardless of their `summarized` flag, with
    # role='summary' husks excluded. Without this, bubbles already rolled into a
    # session summary are invisible here and Grok ends up summarizing summaries.
    messages = await storage.get_messages(
        bot_id=bot_id, since_seconds=since_seconds, raw=True
    )

    # Keep only the conversation bubbles the user actually sees in the app
    # (raw mode already drops summaries; this also drops any system/tool rows).
    messages = [
        m for m in messages if (m.get("role") or "").lower() in _RECAP_ROLES
    ]

    if not messages:
        return {
            "recap": f"No messages found for '{bot_id}' in the last {window_label} — nothing to recap.",
            "stored": False,
            "summary_id": None,
            "window_hours": window_hours,
            "messages_analyzed": 0,
            "model": None,
            "truncated": False,
        }

    transcript, used_count, truncated = _format_transcript(messages)
    message_ids = [m["id"] for m in messages if m.get("id")]

    resolved_model = model or os.getenv("LLM_BAWT_SELF_RECAP_MODEL", _DEFAULT_RECAP_MODEL)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    user_message = (
        "=== CONTEXT ===\n"
        f"Agent being recapped : {bot_id}\n"
        f"Recap generated at   : {now}\n"
        f"Look-back window     : {window_label} ({used_count} messages"
        f"{', oldest dropped to fit size cap' if truncated else ''})\n\n"
        "=== RAW AGENT TRANSCRIPT ===\n"
        f"{transcript}"
    )

    try:
        from llm_bawt.clients.grok_client import GrokClient
        from llm_bawt.models.message import Message
        from llm_bawt.utils.config import Config

        config = Config()
        system_prompt = _resolve_recap_system_prompt(bot_id, config)
        client = GrokClient(model=resolved_model, config=config)

        # query() is synchronous and does network IO — keep it off the event loop.
        recap_text = await asyncio.to_thread(
            client.query,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_message),
            ],
            plaintext_output=True,
            stream=False,
        )
    except Exception as e:
        logger.error("self_recap Grok call failed: %s", e)
        return {
            "recap": None,
            "stored": False,
            "summary_id": None,
            "window_hours": window_hours,
            "messages_analyzed": used_count,
            "model": resolved_model,
            "truncated": truncated,
            "error": f"Grok call failed: {e}",
        }

    recap_text = (recap_text or "").strip()
    if not recap_text:
        return {
            "recap": None,
            "stored": False,
            "summary_id": None,
            "window_hours": window_hours,
            "messages_analyzed": used_count,
            "model": resolved_model,
            "truncated": truncated,
            "error": "Grok returned an empty recap.",
        }

    summary_id = None
    if store:
        summary_id = await storage.store_recap_summary(
            bot_id=bot_id,
            content=recap_text,
            window_start=float(messages[0].get("timestamp") or 0.0),
            window_end=float(messages[-1].get("timestamp") or 0.0),
            message_ids=message_ids,
            model=resolved_model,
        )

    return {
        "recap": recap_text,
        "stored": bool(summary_id),
        "summary_id": summary_id,
        "window_hours": window_hours,
        "messages_analyzed": used_count,
        "model": resolved_model,
        "truncated": truncated,
    }


async def _tail_bubbles(bot_id: str, count: int) -> tuple[list[dict], int]:
    """Return ``(last `count` real bubbles, total_available)`` for ``bot_id``.

    Shared retrieval behind both ``self_tail`` and ``self_fwd``. ``raw=True``
    reads the messages table directly so already-summarized bubbles are still
    visible (the summary-aware manager would hide them) and role='summary' husks
    are excluded. Filters to real user/assistant bubbles BEFORE slicing the last
    ``count``, so ``count`` always means N real bubbles.
    """
    count = max(1, int(count))
    from .server import _get_storage

    storage = _get_storage()
    messages = await storage.get_messages(bot_id=bot_id, raw=True)
    bubbles = [m for m in messages if (m.get("role") or "").lower() in _RECAP_ROLES]
    return bubbles[-count:], len(bubbles)


@mcp.tool(name="self_tail")
async def self_tail(
    bot_id: str = "default",
    count: int = 20,
) -> dict:
    """Reload the agent's last N raw conversation bubbles back INTO ITS CONTEXT.

    The lightweight sibling of ``self_recap``: where recap ships history to Grok
    for a summary, tail just hands back the most recent raw messages (exactly the
    user/assistant bubbles seen in the app). The PURPOSE is context restoration —
    the returned bubbles land in your SDK transcript (a tool result is replayed
    into context on your next turn), so calling this pulls recent conversation
    back into your working memory. Read-only; nothing is persisted.

    BEHAVIOR AFTER CALLING — DO NOT dump the returned messages/transcript back to
    the user. They asked you to *load* this context, not to have it printed at
    them; they can already scroll the same bubbles in the app. Silently absorb the
    content, then reply with a SHORT (1–3 sentence) summary of what you pulled in —
    e.g. how many bubbles, the span/topic, and any obvious open thread — plus a
    forward-moving offer if one is warranted. Only quote a specific bubble
    verbatim if the user explicitly asks to see the raw output.

    role='summary' rows are excluded, so ``count`` always means N real bubbles,
    not N rows that might be old summaries.

    Args:
        bot_id: Bot namespace whose history to tail (pass your own slug).
        count: Number of most-recent bubbles to return (default 20, min 1).

    Returns:
        Dict with: messages (list of {role, content, timestamp}), transcript
        (rendered text), count_returned, total_available, truncated. NOTE: this
        payload is for YOUR context, not for verbatim relay — summarize it back,
        don't reprint it (see BEHAVIOR AFTER CALLING above).
    """
    count = max(1, int(count))
    logger.debug("MCP tool invoked: self_tail bot_id=%s count=%s", bot_id, count)

    tail, total_available = await _tail_bubbles(bot_id, count)

    transcript, used_count, truncated = _format_transcript(tail)
    return {
        "messages": [
            {
                "role": m.get("role"),
                "content": m.get("content"),
                "timestamp": m.get("timestamp"),
            }
            for m in tail
        ],
        "transcript": transcript,
        "count_returned": used_count,
        "total_available": total_available,
        "truncated": truncated,
    }


@mcp.tool(name="self_fwd")
async def self_fwd(
    sender_bot_id: str = "default",
    target_bot_id: str = "",
    count: int = 10,
    note: str | None = None,
    force: bool = False,
    wait_for_reply: bool = False,
) -> dict:
    """Forward YOUR last N conversation bubbles to another bot as context.

    Combines ``self_tail`` (grab your most recent ``count`` real bubbles) with
    ``bots_send_message`` (deliver to another bot) in one hand-off: the target
    receives your recent transcript as a clearly-delimited FORWARDED CONTEXT
    block, then processes it like any inbound message. Async by default (the send
    returns immediately); pass ``wait_for_reply=True`` to block for the reply.

    IMPORTANT: pass YOUR OWN slug as ``sender_bot_id`` — it is the bot whose tail
    is forwarded and cannot be inferred. The literal "default" is rejected. So
    "self fwd vex 10" means: sender_bot_id=<your slug>, target_bot_id="vex",
    count=10.

    Args:
        sender_bot_id: YOUR slug — the bot whose recent tail is forwarded.
        target_bot_id: The bot slug to receive the forwarded context.
        count: How many of your most-recent bubbles to forward (default 10, min 1).
        note: Optional cover message prepended above the forwarded block.
        force: If True, deliver even if the target bot is mid-turn.
        wait_for_reply: If True, block up to the send's timeout and return the
            target's reply inside ``send_result``. Defaults to async.

    Returns:
        Dict with: success, sender, target, forwarded (bubbles sent),
        total_available, truncated, wait_for_reply, and send_result (the nested
        bots_send_message result — dispatched / in_turn / content). On failure,
        error.
    """
    sender = (sender_bot_id or "").strip().lower()
    target = (target_bot_id or "").strip().lower()
    count = max(1, int(count))
    logger.debug(
        "MCP tool invoked: self_fwd sender=%s target=%s count=%s wait=%s",
        sender, target, count, wait_for_reply,
    )

    if not sender or sender == "default":
        return {
            "success": False,
            "sender": sender,
            "target": target,
            "forwarded": 0,
            "error": "sender_bot_id is required — pass your own slug (not 'default').",
        }
    if not target:
        return {
            "success": False,
            "sender": sender,
            "target": target,
            "forwarded": 0,
            "error": "target_bot_id is required.",
        }
    if target == sender:
        return {
            "success": False,
            "sender": sender,
            "target": target,
            "forwarded": 0,
            "error": "Cannot forward to yourself.",
        }

    tail, total = await _tail_bubbles(sender, count)
    if not tail:
        return {
            "success": False,
            "sender": sender,
            "target": target,
            "forwarded": 0,
            "total_available": total,
            "error": f"No messages found for '{sender}' to forward.",
        }

    transcript, used_count, truncated = _format_transcript(tail)
    block = (
        f"=== FORWARDED CONTEXT from '{sender}' "
        f"(last {used_count} message{'s' if used_count != 1 else ''}) ===\n"
        f"{transcript}\n"
        "=== END FORWARDED CONTEXT ==="
    )
    message = f"{note.strip()}\n\n{block}" if note and note.strip() else block

    # Reuse the full inter-bot send tool (in-turn gating, stable message id,
    # async fire-and-forget bookkeeping) rather than re-implementing the HTTP
    # call. Local import mirrors the other `from .server import ...` calls in
    # this module and avoids a circular import at module load.
    from .server import send_message_to_bot

    send_result = await send_message_to_bot(
        target_bot_id=target,
        message=message,
        sender_bot_id=sender,
        force=force,
        wait_for_reply=wait_for_reply,
    )

    return {
        "success": bool(send_result.get("success")),
        "sender": sender,
        "target": target,
        "forwarded": used_count,
        "total_available": total,
        "truncated": truncated,
        "wait_for_reply": wait_for_reply,
        "send_result": send_result,
    }


async def _fetch_bot_profile(bot_id: str) -> dict:
    """GET a bot's DB profile via the app API. Raises on non-2xx / transport error."""
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_APP_BASE_URL}/v1/bots/{bot_id}/profile", timeout=15.0
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool(name="self_system_prompt")
async def self_system_prompt(
    bot_id: str = "default",
    action: str = "view",
    new_prompt: str | None = None,
) -> dict:
    """View or edit YOUR OWN persisted system prompt (the bot persona in the DB).

    This is the ``system_prompt`` stored on your ``bot_profiles`` row — the
    personality/instructions the platform sends as your system message on every
    turn (for agent bots it's pushed to the agent as the persona; for chat bots
    it's the LLM system message). It is the SINGLE source of truth for who you
    are, and it is the cacheable prefix of your context, so keep it byte-stable
    between deliberate edits.

    IMPORTANT: this is NOT the ephemeral per-turn harness wrapper (temporal
    lines, tool lists, user-profile injection). It is the durable persona only.

    Editing writes straight through the same path as the admin UI
    (``PATCH /v1/bots/{slug}/profile``): the DB row is upserted, the in-memory
    bot registry is reloaded, per-bot instance caches are invalidated, and for
    OpenClaw agents the new prompt is pushed to SOUL.md. It never rewrites the
    current in-flight context. A pinned agent session (Claude Code, TASK-1318)
    keeps its pinned prompt until the next session boundary: /new, reset, model
    switch, or the first turn after a manual /compact. Other backends pick the
    change up on the next turn.

    Args:
        bot_id: The bot whose system prompt to act on. Pass YOUR OWN slug
            (e.g. "caid"). The literal "default" is rejected for edits to avoid
            accidentally rewriting the wrong bot.
        action: "view" (default) to read the current prompt, or "edit" to
            replace it wholesale with ``new_prompt``.
        new_prompt: The full replacement system prompt (required for
            action="edit"). This is a FULL replace, not an append — send the
            complete desired prompt. The prior prompt is returned in the result
            (``old_prompt``) so it stays recoverable in your transcript.

    Returns:
        view: {bot_id, action, system_prompt, length, name, bot_type, updated_at}
        edit: {bot_id, action, updated, old_prompt, old_length, new_prompt,
               new_length, name, bot_type, updated_at}
        On failure: {bot_id, action, error, ...}.
    """
    import httpx

    bot_id = (bot_id or "").strip().lower()
    action = (action or "view").strip().lower()
    logger.debug("MCP tool invoked: self_system_prompt bot_id=%s action=%s", bot_id, action)

    if not bot_id:
        return {"bot_id": bot_id, "action": action, "error": "bot_id is required (pass your own slug)."}

    if action not in {"view", "edit"}:
        return {
            "bot_id": bot_id,
            "action": action,
            "error": f"Unknown action '{action}'. Valid: view, edit.",
        }

    # ---- VIEW -------------------------------------------------------------
    if action == "view":
        try:
            profile = await _fetch_bot_profile(bot_id)
        except httpx.HTTPStatusError as e:
            detail = "not found" if e.response.status_code == 404 else str(e)
            return {"bot_id": bot_id, "action": action, "error": f"Could not read profile: {detail}"}
        except Exception as e:
            return {"bot_id": bot_id, "action": action, "error": f"Profile fetch failed: {e}"}

        prompt = profile.get("system_prompt") or ""
        return {
            "bot_id": bot_id,
            "action": action,
            "system_prompt": prompt,
            "length": len(prompt),
            "name": profile.get("name"),
            "bot_type": profile.get("bot_type"),
            "updated_at": profile.get("updated_at"),
        }

    # ---- EDIT -------------------------------------------------------------
    if bot_id == "default":
        return {
            "bot_id": bot_id,
            "action": action,
            "error": "Refusing to edit the 'default' bot — pass your own explicit slug.",
        }
    if new_prompt is None or not str(new_prompt).strip():
        return {
            "bot_id": bot_id,
            "action": action,
            "error": "action='edit' requires a non-empty 'new_prompt' (full replacement text).",
        }

    # Read the current prompt first so the old value is recoverable in the
    # transcript and we can report a no-op.
    try:
        before = await _fetch_bot_profile(bot_id)
    except httpx.HTTPStatusError as e:
        detail = "not found" if e.response.status_code == 404 else str(e)
        return {"bot_id": bot_id, "action": action, "error": f"Could not read current profile: {detail}"}
    except Exception as e:
        return {"bot_id": bot_id, "action": action, "error": f"Profile fetch failed: {e}"}

    old_prompt = before.get("system_prompt") or ""

    # Write through the canonical PATCH path (upsert + registry reload + cache
    # invalidation + SOUL push all happen server-side — no logic duplicated here).
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{_APP_BASE_URL}/v1/bots/{bot_id}/profile",
                json={"system_prompt": new_prompt},
                timeout=30.0,
            )
            resp.raise_for_status()
            after = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "bot_id": bot_id,
            "action": action,
            "error": f"Update rejected ({e.response.status_code}): {e.response.text}",
            "old_prompt": old_prompt,
        }
    except Exception as e:
        return {"bot_id": bot_id, "action": action, "error": f"Update failed: {e}", "old_prompt": old_prompt}

    updated_prompt = after.get("system_prompt") or ""
    return {
        "bot_id": bot_id,
        "action": action,
        "updated": updated_prompt != old_prompt,
        "old_prompt": old_prompt,
        "old_length": len(old_prompt),
        "new_prompt": updated_prompt,
        "new_length": len(updated_prompt),
        "name": after.get("name"),
        "bot_type": after.get("bot_type"),
        "updated_at": after.get("updated_at"),
        # TASK-1318: pin-enabled agent sessions keep their pinned prompt until
        # the next session boundary (service/system_prompt_pin.py).
        "note": (
            "Not this turn. A pinned agent session (Claude Code) picks this up at its next "
            "session boundary: /new, reset, model switch, or the first turn after a manual "
            "/compact. Other backends pick it up on the next turn."
        ),
    }
