"""Tell an agent which task cards are assigned to it (TASK-2432).

Bots lose track of their own cards across session resets: parked WAITING work
is forgotten once the transcript that parked it is gone. Each agent turn reads
the live task board and builds a short list of the bot's open cards (short ID,
status, WAITING kind, title). Coordinators also get one fleet line: each bot's
active card and its waiting count.

The list rides the outbound user message, never the pinned system prompt, so
the cacheable prefix stays byte-stable. It is built every turn but sent only
when the transcript doesn't already hold the current list: on a new pin
episode (a session start or reset), after a compaction, or when the list
changed. ``sessions.session_metadata`` records what this session was last
told. Every failure is logged and the turn continues without the list.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

ASSIGNED_CARDS_KEY = "agents.assigned_cards"
#: ``sessions.session_metadata`` key: ``{"pin_turn_id", "compaction_gen", "digest"}``.
STATE_KEY = "assigned_cards"
#: The role capability whose holders get the fleet line.
FLEET_CAPABILITY = "coordinate"

#: Display order; any other status is left out.
STATUS_ORDER = ("IN_PROGRESS", "WAITING", "BLOCKED", "REVIEW", "QUEUED")
MAX_LINES = 12
TITLE_CHARS = 90
BOARD_TTL_SECONDS = 10.0
BOARD_TIMEOUT_SECONDS = 3.0

_board_lock = threading.Lock()
_board_cache: tuple[float, list[dict]] | None = None


# -- rendering (pure) ---------------------------------------------------------

def _title(row: dict) -> str:
    title = " ".join(str(row.get("title") or "").split())
    return title if len(title) <= TITLE_CHARS else title[: TITLE_CHARS - 1].rstrip() + "…"


def _status(row: dict) -> str:
    status = str(row.get("status") or "")
    waiting_on = row.get("waitingOn")
    return f"{status} {waiting_on}" if status == "WAITING" and waiting_on else status


def _rank(row: dict) -> tuple[int, int]:
    number = str(row.get("shortId") or "").removeprefix("TASK-")
    return STATUS_ORDER.index(row["status"]), int(number) if number.isdigit() else 0


def card_lines(rows: list[dict], bot_id: str) -> list[str]:
    """The bot's open cards, one line each, capped with a ``+N more`` line."""
    mine = sorted(
        (r for r in rows if r.get("agentBotId") == bot_id and r.get("status") in STATUS_ORDER
         and not r.get("archivedAt")),
        key=_rank,
    )
    lines = [f"- {r.get('shortId')} {_status(r)}: {_title(r)}" for r in mine[:MAX_LINES]]
    if len(mine) > MAX_LINES:
        lines.append(f"- +{len(mine) - MAX_LINES} more")
    return lines


def fleet_line(rows: list[dict]) -> str:
    """Per bot with open cards: its IN_PROGRESS card (or "idle") and WAITING count."""
    bots: dict[str, dict[str, Any]] = {}
    for r in rows:
        bot = r.get("agentBotId")
        if not bot or r.get("archivedAt") or r.get("status") not in STATUS_ORDER:
            continue
        entry = bots.setdefault(bot, {"active": [], "waiting": 0})
        if r["status"] == "IN_PROGRESS":
            entry["active"].append(str(r.get("shortId")))
        elif r["status"] == "WAITING":
            entry["waiting"] += 1
    parts = []
    for bot in sorted(bots):
        entry = bots[bot]
        active = "+".join(sorted(entry["active"])) or "idle"
        parts.append(f"{bot} {active} W{entry['waiting']}")
    return "; ".join(parts) if parts else "no open assigned cards"


def render(rows: list[dict], bot_id: str, coordinator: bool) -> tuple[str, str]:
    """``(cards, fleet)`` placeholder values; fleet is empty for non-coordinators."""
    lines = card_lines(rows, bot_id)
    cards = "\n".join(lines) if lines else "- none"
    fleet = f"\nFleet (bot, IN_PROGRESS card, W=waiting): {fleet_line(rows)}" if coordinator else ""
    return cards, fleet


def digest(cards: str, fleet: str) -> str:
    return hashlib.sha256(f"{cards}\n{fleet}".encode()).hexdigest()[:16]


def should_send(marker: Any, decision: Any, compaction_gen: int, current: str) -> bool:
    """Whether the transcript lacks the current list."""
    if not isinstance(marker, dict):
        return True
    return (marker.get("pin_turn_id") != decision.pin_turn_id
            or marker.get("compaction_gen") != compaction_gen
            or marker.get("digest") != current)


# -- reads ---------------------------------------------------------------------

def board_rows() -> list[dict]:
    """Open-lane rows of the whole board, cached briefly across turns."""
    global _board_cache
    with _board_lock:
        if _board_cache and time.monotonic() - _board_cache[0] < BOARD_TTL_SECONDS:
            return _board_cache[1]
    import httpx

    from .bawthub_auth import service_auth
    from .utils.config import DEFAULT_TASK_API_URL

    base = os.getenv("LLM_BAWT_TASK_API_URL", DEFAULT_TASK_API_URL)
    resp = httpx.get(
        f"{base}/api/tasks/board",
        params={"view": "status", "rows": "active", "updated": "all", "limit": "50"},
        auth=service_auth(), timeout=BOARD_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    rows = [row for lane in resp.json().get("lanes") or [] for row in lane.get("rows") or []
            if isinstance(row, dict)]
    with _board_lock:
        _board_cache = (time.monotonic(), rows)
    return rows


def is_coordinator(engine, bot_id: str) -> bool:
    from .bot_tags import grants

    with engine.connect() as conn:
        return grants.holds_capability(conn, bot_id, FLEET_CAPABILITY)


def render_block(bot_id: str, cards: str, fleet: str) -> str:
    from .prompt_registry import PromptResolver, get_prompt_resolver, render_template

    variables = {"cards": cards, "fleet": fleet}
    try:
        body = get_prompt_resolver().render(
            key=ASSIGNED_CARDS_KEY, variables=variables, scope_type="bot", scope_id=bot_id,
        )
        if body.strip():
            return body.rstrip()
    except Exception as error:  # a DB failure falls back to the registered default
        logger.warning("assigned cards prompt resolve failed, using its default: %s", error)
    definition = PromptResolver.definition_for(ASSIGNED_CARDS_KEY)
    if definition is None:
        raise RuntimeError(f"{ASSIGNED_CARDS_KEY} is not registered")
    return render_template(definition.default_body(), variables).rstrip()


def _claim(engine, session_id: str, decision: Any, current: str) -> bool:
    """Under the session row lock, record and report whether to send ``current``."""
    from .working_memory.session_state import compaction_gen_of, state_of

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT session_metadata FROM sessions WHERE id=:sid FOR UPDATE"),
            {"sid": session_id},
        ).first()
        if row is None:
            return False
        meta = row[0] if isinstance(row[0], dict) else {}
        gen = compaction_gen_of(state_of(meta))
        if not should_send(meta.get(STATE_KEY), decision, gen, current):
            return False
        mark = {"pin_turn_id": decision.pin_turn_id, "compaction_gen": gen, "digest": current}
        conn.execute(
            text("""
                UPDATE sessions SET session_metadata=jsonb_set(
                    COALESCE(session_metadata, '{}'::jsonb),
                    '{assigned_cards}', CAST(:mark AS jsonb)
                ) WHERE id=:sid
            """),
            {"mark": json.dumps(mark), "sid": session_id},
        )
    return True


def assigned_cards_notice(engine, *, bot_id: str | None, session_id: str | None,
                          decision: Any, eligible: bool = True) -> str | None:
    """The block for this turn, or None. Never raises."""
    if engine is None or not bot_id or not session_id or not eligible or decision is None:
        return None
    try:
        from .working_memory.policy import is_protected

        if is_protected(bot_id, engine):
            return None
        cards, fleet = render(board_rows(), bot_id, is_coordinator(engine, bot_id))
        if not _claim(engine, session_id, decision, digest(cards, fleet)):
            return None
        return render_block(bot_id, cards, fleet)
    except Exception:
        logger.warning("assigned cards block failed for %s session %s; the turn continues without it",
                       bot_id, session_id, exc_info=True)
        return None


def apply_assigned_cards(ctx: Any, messages: list, decision: Any) -> list:
    """``messages`` with the block on the last user message when it is due."""
    from .working_memory.turn import append_to_last_user, is_maintenance_command, scheduler_turn

    block = assigned_cards_notice(
        ctx.svc._turn_log_store.engine,
        bot_id=ctx.bot_id,
        session_id=str((ctx.thread_binding or {}).get("thread_session_id") or "").strip(),
        decision=decision,
        eligible=not is_maintenance_command(ctx.user_prompt) and scheduler_turn(ctx.svc, ctx.request) is False,
    )
    return append_to_last_user(messages, block) if block else messages


__all__ = [
    "ASSIGNED_CARDS_KEY", "STATE_KEY", "apply_assigned_cards", "assigned_cards_notice",
    "card_lines", "digest", "fleet_line", "render", "should_send",
]
