"""TASK-2064: a short generated title for each session.

After the first eligible user message lands, a background thread asks the
global maintenance model (bounded utility completion) for a title of about
eight words, using only that message, and stores it as
``title`` with ``title_source='generated'``. It never blocks the turn.

A name someone gave the session (a user rename or a schedule name) always
wins: the write is one conditional UPDATE that only fills a missing or
``auto`` title, so a rename that lands first is kept and a rename that lands
later replaces the generated title. Failure, refusal or timeout stores
nothing and the display falls back to the first message (TASK-2061).

The prompt lives in the prompt registry (``sessions.title.generate``).
Message text is never logged.
"""
from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

TITLE_PROMPT_KEY = "sessions.title.generate"
TITLE_SOURCE = "generated"
MAX_INPUT_CHARS = 2000
MAX_TITLE_CHARS = 80
MAX_TITLE_WORDS = 10
TIMEOUT_SECONDS = 30.0

TITLE_PROMPT_TEMPLATE = """\
Write a short title for a chat conversation that starts with the message below.

Rules:
- At most 8 words, in the language of the message.
- Summarize the topic, e.g. "Git rebase question", "Fix flaky login test", \
"Weekend trip packing list".
- No quotes, no ending punctuation, no emoji, no preamble.
- Output only the title.

Message:
{message}"""

_REFUSAL = re.compile(
    r"^(i\s*(can'?t|cannot|am unable|'m unable|won'?t)|sorry|as an ai|i'm sorry)\b", re.I)
_WS = re.compile(r"\s+")

# Sessions this process has already tried; one attempt per session per process.
_attempted: set[str] = set()
# Sessions that got another user message while their worker was running.
_pending: set[str] = set()
# Message ids each running worker answers for.
_triggers: dict[str, set[str]] = {}
_running: set[str] = set()
_attempted_lock = threading.Lock()


def clean_title(raw: str | None) -> str | None:
    """Normalize a model reply to a title, or None for empty/refusal output."""
    if not isinstance(raw, str):
        return None
    line = next((part.strip() for part in raw.strip().splitlines() if part.strip()), "")
    line = re.sub(r"^(title\s*:\s*)", "", line, flags=re.I)
    line = line.strip().strip("*#").strip()
    while len(line) >= 2 and line[0] in "\"'“‘`" and line[-1] in "\"'”’`":
        line = line[1:-1].strip()
    line = _WS.sub(" ", line).rstrip(" .,;:!")
    if not line or _REFUSAL.match(line):
        return None
    words = line.split(" ")
    if len(words) > MAX_TITLE_WORDS:
        line = " ".join(words[:MAX_TITLE_WORDS])
    if len(line) > MAX_TITLE_CHARS:
        cut = line[:MAX_TITLE_CHARS]
        line = cut[: cut.rfind(" ")] if " " in cut else cut
    return line.strip() or None


def _first_message(conn, session_id: str, bot_id: str, user_id: str | None):
    """The session's first eligible message id/text and whether it has a given name."""
    from sqlalchemy import text

    from .first_message import (
        eligible_sql, first_message_params, given_title_sql, storage_bot_sql,
    )

    row = conn.execute(text(f"""
        SELECT {given_title_sql('s')} AS given,
               fm.id AS fm_id, fm.content AS fm_text
        FROM sessions s
        LEFT JOIN LATERAL (SELECT m1.id, m1.content FROM messages m1
                WHERE m1.session_id = s.id AND m1.bot_id = {storage_bot_sql('s.bot_id')}
                  AND {eligible_sql('m1')}
                ORDER BY m1.timestamp ASC, m1.id ASC LIMIT 1) fm ON TRUE
        WHERE s.id = :id AND s.bot_id = :bot
          AND (CAST(:user AS TEXT) IS NULL OR s.user_id = :user)
          AND s.status <> 'deleted'
    """), {"id": session_id, "bot": bot_id, "user": user_id,
           **first_message_params()}).mappings().first()
    return row


def store_generated_title(conn, session_id: str, bot_id: str, user_id: str | None,
                          title: str) -> bool:
    """Fill the title only when nobody has named the session. True if stored."""
    import json

    from sqlalchemy import text

    from .first_message import first_message_params

    row = conn.execute(text("""
        UPDATE sessions AS s
        SET session_metadata = COALESCE(s.session_metadata, '{}'::jsonb) || CAST(:patch AS jsonb)
        WHERE s.id = :id AND s.bot_id = :bot
          AND (CAST(:user AS TEXT) IS NULL OR s.user_id = :user)
          AND jsonb_typeof(COALESCE(s.session_metadata, '{}'::jsonb)) = 'object'
          AND (
            COALESCE(s.session_metadata->>'title_source', '') = 'auto'
            OR jsonb_typeof(s.session_metadata->'title') IS DISTINCT FROM 'string'
            OR btrim(s.session_metadata->>'title', :fm_strip) = ''
          )
          AND COALESCE(s.session_metadata->>'title_source', '')
              IN ('', 'auto', 'default', 'generated')
        RETURNING s.id
    """), {"id": session_id, "bot": bot_id, "user": user_id,
           "patch": json.dumps({"title": title, "title_source": TITLE_SOURCE}),
           **first_message_params()}).first()
    return row is not None


NO_ELIGIBLE = object()
NOT_TRIGGERED = object()


def generate_session_title(engine, session_id: str, bot_id: str, user_id: str | None,
                           complete=None, trigger_ids=None):
    """Title the session from its first eligible message, read from the DB.

    ``trigger_ids`` are the message writes this worker answers for; when
    given, the first eligible message must be one of them, so a new message
    in an old unnamed chat never titles it from history (no backfill).

    Returns the stored title, ``None`` (named, refused or not stored),
    ``NO_ELIGIBLE`` when the session has no eligible message yet, or
    ``NOT_TRIGGERED`` when its first eligible message is not a trigger.
    """
    with engine.connect() as conn:
        row = _first_message(conn, session_id, bot_id, user_id)
    if row is None or row["given"]:
        return None
    if row["fm_id"] is None:
        return NO_ELIGIBLE
    if trigger_ids is not None and row["fm_id"] not in trigger_ids:
        return NOT_TRIGGERED
    content = row["fm_text"] or ""

    from ..prompt_registry import get_prompt_resolver
    from ..service.schemas import RawCompletionRequest

    prompt = get_prompt_resolver().render(
        key=TITLE_PROMPT_KEY, variables={"message": content.strip()[:MAX_INPUT_CHARS]})
    if complete is None:
        from ..service.utility_completion import complete_utility

        def complete(request):
            return complete_utility(request, timeout_seconds=TIMEOUT_SECONDS)

    result = complete(RawCompletionRequest(prompt=prompt, max_tokens=64, temperature=0.2))
    title = clean_title(result.content)
    if title is None:
        logger.info("session title: no usable title for %s", session_id)
        return None
    with engine.begin() as conn:
        stored = store_generated_title(conn, session_id, bot_id, user_id, title)
    logger.info("session title %s for %s", "stored" if stored else "skipped (named)", session_id)
    return title if stored else None


def _worker(engine, session_id: str, bot_id: str, user_id: str | None) -> None:
    """One worker per session, answering only for the writes that scheduled it.

    A write arriving while it runs adds its id and marks the session pending;
    a worker that found no eligible message, or a first message not yet among
    its trigger ids, re-reads when pending instead of stopping.
    """
    def finish(release: bool) -> None:  # caller holds _attempted_lock
        _triggers.pop(session_id, None)
        _running.discard(session_id)
        _pending.discard(session_id)
        if release:
            _attempted.discard(session_id)

    while True:
        with _attempted_lock:
            _pending.discard(session_id)
            ids = set(_triggers.get(session_id, ()))
        try:
            outcome = generate_session_title(engine, session_id, bot_id, user_id,
                                             trigger_ids=ids)
        except Exception as error:  # cosmetic; type only, never message text
            logger.warning("session title failed for %s (%s)", session_id,
                           type(error).__name__)
            outcome = None
        with _attempted_lock:
            if outcome is NO_ELIGIBLE or outcome is NOT_TRIGGERED:
                if session_id in _pending:
                    continue
            finish(release=outcome is NO_ELIGIBLE)
            return


def schedule_session_title(engine, *, session_id: str | None, bot_id: str,
                           user_id: str | None, message_id: str, content: str) -> None:
    """Start title generation in the background. Never raises, never blocks.

    Only the app service process generates titles (it owns the maintenance
    model). Its chat writes go through the MCP client, which does not know the
    server-resolved thread, so ``session_id`` may be None: the session is then
    read from the written message's row off the write path.
    """
    try:
        text_ = (content or "").strip()
        if not message_id or not text_ or text_.startswith("/"):
            return
        from ..service.dependencies import _service
        if _service is None:  # no app service in this process (MCP, CLI, tests)
            return
        if engine is None:
            from ..utils.db import get_shared_engine
            engine = get_shared_engine(_service.config)
            if engine is None:
                return
        # Every write, known thread or not, claims through one FIFO drainer,
        # so a later write can never claim (and be judged NOT_TRIGGERED)
        # before an earlier one that is still resolving its thread.
        with _resolve_lock:
            _resolve_queue.append((engine, session_id, bot_id, user_id, message_id))
            if _resolve_draining[0]:
                return
            _resolve_draining[0] = True
        threading.Thread(target=_drain_resolutions, daemon=True,
                         name="session-title").start()
    except Exception as error:
        logger.warning("session title scheduling failed (%s)", type(error).__name__)


_resolve_lock = threading.Lock()
_resolve_queue: list = []
_resolve_draining = [False]


def _drain_resolutions() -> None:
    while True:
        with _resolve_lock:
            if not _resolve_queue:
                _resolve_draining[0] = False
                return
            item = _resolve_queue.pop(0)
        _resolve_and_claim(*item)


def _resolve_and_claim(engine, session_id: str | None, bot_id: str,
                       user_id: str | None, message_id: str) -> None:
    try:
        if not session_id:
            from sqlalchemy import text
            with engine.connect() as conn:
                session_id = conn.execute(
                    text("SELECT session_id FROM messages WHERE id = :id"),
                    {"id": message_id}).scalar()
        if session_id:
            _claim(engine, str(session_id), bot_id, user_id, message_id)
    except Exception as error:
        logger.warning("session title scheduling failed (%s)", type(error).__name__)


def _claim(engine, session_id: str, bot_id: str, user_id: str | None,
           message_id: str) -> None:
    with _attempted_lock:
        if session_id in _attempted:
            if session_id in _running:  # hand the write to the live worker
                _triggers.setdefault(session_id, set()).add(message_id)
                _pending.add(session_id)
            return
        _attempted.add(session_id)
        _running.add(session_id)
        _triggers[session_id] = {message_id}
    threading.Thread(target=_worker, args=(engine, session_id, bot_id, user_id),
                     daemon=True, name="session-title").start()
