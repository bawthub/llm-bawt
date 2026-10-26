"""``messages_context``: expand around one message (TASK-1323 §5.2, §5.3).

The anchor is either a composite ``(bot, id)`` or a ``(handle, card_ref)``
naming a card a ``messages_find`` page actually served. A card reference
resolves from the handle's served record and never re-runs the find, so a
message inserted between the two calls can't change what ``c0.0`` means.

Either way the anchor is reauthorized under the current policy before
anything is read: it must pass the same scope predicate as ``messages_find``
(positive partition map, canonical session join with the user pin and live
status, effective-author class). A denied, vanished or unserved anchor is
``not_found``, the same answer as an id that never existed.

Neighbours come from the anchor's own (bot partition, session) in the stable
order ``(timestamp, id)``. Each one passes the scope predicate in the WHERE
clause, so a skipped neighbour (blank author, excluded author, another
session) never shows, and never changes ``more_before``/``more_after``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import OperationalError

from llm_bawt.bot_access import BotAccess, BotAccessError

from .find import (
    _SCOPE_FROM, FIND_HANDLES, TOKEN_MAX, FindContinuation, FindError, FindScope,
    ServedCard, _ReadSnapshot, _SPACE, failure, serialize, snippet,
)
from .handles import HandleExpired, HandleStore, Owner

logger = logging.getLogger(__name__)

SIDE_DEFAULT, SIDE_MAX = 3, 10
CHARS_DEFAULT, CHARS_MIN, CHARS_MAX = 4000, 1000, 8000
WIDE, NARROW = 200, 80
FOCUS_N = 6
# Width of a cursor key (token_urlsafe(32)); reserved before measuring.
_CURSOR_PLACEHOLDER = "x" * len(HandleStore.new_key())

Direction = Literal["before", "after"]


@dataclass(frozen=True)
class ContextRequest:
    bot: str | None = None
    id: str | None = None
    handle: str | None = None
    card_ref: str | None = None
    before: int = SIDE_DEFAULT
    after: int = SIDE_DEFAULT
    focus: tuple[str, ...] | None = None
    max_chars: int = CHARS_DEFAULT
    offset: int = 0
    cursor: str | None = None


@dataclass(frozen=True)
class ContextCursor:
    """Where a neighbour page resumes: the anchor (reauthorized on use), the
    last served position on one side, and the focus terms. No content."""

    bot: str
    id: str
    ts: float
    position: str
    direction: Direction
    focus: tuple[str, ...]


CONTEXT_CURSORS: HandleStore[ContextCursor] = HandleStore(per_owner=16)


class NotFound(LookupError):
    """Denied, vanished, unserved or unknown anchor: one indistinguishable answer."""


def _stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count(value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise FindError("invalid_request")
    return value


def _focus(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or len(value) > FOCUS_N:
        raise FindError("invalid_request")
    terms = []
    for term in value:
        if not isinstance(term, str) or len(term.strip()) > TOKEN_MAX:
            raise FindError("invalid_request")
        if term.strip():
            terms.append(term.strip())
    return tuple(terms)


def _neighbor(row: dict, focus: tuple[str, ...], width: int) -> dict:
    content = row["content"] or ""
    if len(content) <= width:
        body, truncated = _SPACE.sub(" ", content).strip(), False
    else:
        body, truncated = snippet(content, focus, width)[0], True
    return {"id": str(row["id"]), "speaker": row["speaker"], "ts": _stamp(row["timestamp"]),
            "snippet": body, "truncated": truncated}


def _anchor_text(content: str, offset: int, budget: int) -> tuple[str, int | None]:
    """The anchor's text from ``offset``, at most ``budget`` serialized chars."""
    piece = content[offset:offset + budget]
    while piece and len(serialize(piece)) > budget:
        piece = piece[:len(piece) - max(1, len(serialize(piece)) - budget)]
    end = offset + len(piece)
    return piece, (end if end < len(content) else None)


class MessageContext:
    """Runs one ``messages_context`` call under a caller's :class:`BotAccess`."""

    def __init__(self, engine: Engine,
                 finds: HandleStore[FindContinuation] = FIND_HANDLES,
                 cursors: HandleStore[ContextCursor] = CONTEXT_CURSORS):
        self.engine = engine
        self.finds = finds
        self.cursors = cursors

    def context(self, access: BotAccess, request: ContextRequest, *,
                trigger_id: str | None = None) -> str:
        """The serialized wire payload; failures are ``unavailable`` with a reason."""
        try:
            return self._context(access, request, trigger_id)
        except NotFound:
            return serialize({"status": "not_found"})
        except BotAccessError as error:
            return failure(error.code)
        except FindError as error:
            if error.code == "response_overflow":
                logger.error("messages_context response overflow")
            return failure(error.code)
        except HandleExpired:
            return failure("handle_expired")
        except OperationalError as error:
            code = getattr(getattr(error, "orig", None), "pgcode", None)
            logger.warning("messages_context query failed: %s", type(error).__name__)
            return failure("timeout" if code == "57014" else "source_unavailable")
        except Exception:
            logger.exception("messages_context failed")
            return failure("source_unavailable")

    # -- entry -------------------------------------------------------------

    def _context(self, access: BotAccess, request: ContextRequest,
                 trigger_id: str | None) -> str:
        if access.caller is None or not access.caller.user_id:
            raise BotAccessError("caller_unavailable")
        owner: Owner = (access.caller.bot_id, access.caller.user_id)
        before = _count(request.before, 0, SIDE_MAX)
        after = _count(request.after, 0, SIDE_MAX)
        max_chars = _count(request.max_chars, CHARS_MIN, CHARS_MAX)
        offset = _count(request.offset, 0, 2**31 - 1)
        focus = _focus(request.focus)
        by_id = request.bot is not None or request.id is not None
        by_card = request.handle is not None or request.card_ref is not None
        if request.cursor is not None:
            if by_id or by_card or offset:
                raise FindError("invalid_request")
            cursor = self.cursors.get(request.cursor, owner)
            n = before if cursor.direction == "before" else after
            return self._page(access, owner, cursor, n, max_chars, trigger_id,
                              cursor.focus if focus is None else focus)
        if by_id == by_card:
            raise FindError("invalid_request")
        if by_card:
            card, served = self._served_card(owner, request.handle, request.card_ref)
            bot, mid = card.bot, card.id
            if focus is None:
                focus = served.terms
        else:
            if not isinstance(request.bot, str) or not isinstance(request.id, str) or not request.id:
                raise FindError("invalid_request")
            bot, mid = request.bot, request.id
        return self._expand(access, owner, bot, mid, before, after, max_chars, offset,
                            focus or (), trigger_id)

    def _served_card(self, owner: Owner, handle: Any, card_ref: Any,
                     ) -> tuple[ServedCard, FindContinuation]:
        """The card a find page actually served under ``card_ref``, and that
        find's record (query, filters, plan, terms). Never re-runs the find;
        the caller still reauthorizes the anchor before reading it."""
        if not isinstance(card_ref, str):
            raise FindError("invalid_request")
        served = self.finds.get(handle, owner)  # unknown/expired/other owner: HandleExpired
        card = next((c for c in served.served if c.ref == card_ref), None)
        if card is None:
            raise NotFound()
        return card, served

    # -- SQL ---------------------------------------------------------------

    @staticmethod
    def _scope(access: BotAccess, bot: str, trigger_id: str | None) -> FindScope:
        scope = FindScope.build(access, bot, trigger_id)
        if not scope.partitions:  # excluded or unknown bot: same as a missing id
            raise NotFound()
        return scope

    @staticmethod
    def _anchor(conn: Connection, scope: FindScope, mid: str) -> dict:
        p = scope.params()
        p["anchor"] = mid
        clauses = ["m.bot_id = ANY(CAST(:f_parts AS text[]))", "c.speaker IS NOT NULL",
                   "m.id = :anchor"]
        if scope.trigger_id:
            clauses.append("m.id <> :trigger_id")
            p["trigger_id"] = scope.trigger_id
        row = conn.execute(text(f"""
            SELECT m.id, p.bot_id AS bot, m.session_id, m.timestamp, c.speaker, m.content
            {_SCOPE_FROM}
            WHERE {" AND ".join(clauses)}
            ORDER BY m.timestamp, m.id LIMIT 1
        """), p).mappings().first()
        if row is None:
            raise NotFound()
        return dict(row)

    @staticmethod
    def _side(conn: Connection, scope: FindScope, session: str, ts: float, position: str,
              direction: Direction, n: int) -> list[dict]:
        """Up to ``n + 1`` in-scope rows on one side of ``(ts, position)``,
        nearest first; the extra row only says whether more exist (so ``n=0``
        still answers ``more_*``)."""
        p = scope.params()
        p.update({"session": session, "ts": ts, "position": position, "limit": n + 1})
        compare, order = ("<", "DESC") if direction == "before" else (">", "ASC")
        clauses = ["m.bot_id = ANY(CAST(:f_parts AS text[]))", "c.speaker IS NOT NULL",
                   "m.session_id = :session",
                   f"(m.timestamp, m.id) {compare} (CAST(:ts AS double precision), CAST(:position AS text))"]
        if scope.trigger_id:
            clauses.append("m.id <> :trigger_id")
            p["trigger_id"] = scope.trigger_id
        rows = conn.execute(text(f"""
            SELECT m.id, m.timestamp, c.speaker, m.content
            {_SCOPE_FROM}
            WHERE {" AND ".join(clauses)}
            ORDER BY m.timestamp {order}, m.id {order} LIMIT :limit
        """), p).mappings().all()
        return [dict(row) for row in rows]

    # -- responses -----------------------------------------------------------

    def _cursor(self, owner: Owner, access: BotAccess, bot: str, mid: str, at: dict,
                direction: Direction, focus: tuple[str, ...]) -> str:
        """An owner-bound cursor resuming past position ``at`` on one side."""
        key = self.cursors.new_key()
        self.cursors.put(key, owner, ContextCursor(
            bot, mid, float(at["timestamp"]), str(at["id"]), direction, focus,
        ), capability_expires_at=access.caller.expires_at)
        return key

    def _expand(self, access: BotAccess, owner: Owner, bot: str, mid: str, before: int,
                after: int, max_chars: int, offset: int, focus: tuple[str, ...],
                trigger_id: str | None) -> str:
        scope = self._scope(access, bot, trigger_id)
        with _ReadSnapshot(self.engine) as conn:
            anchor = self._anchor(conn, scope, mid)
            pos = (anchor["session_id"], anchor["timestamp"], str(anchor["id"]))
            older = self._side(conn, scope, *pos, "before", before)
            newer = self._side(conn, scope, *pos, "after", after)
        content = anchor["content"] or ""
        if offset > len(content):
            raise FindError("invalid_request")
        piece, next_offset = _anchor_text(content, offset, max_chars // 2)
        older_rows = list(reversed(older[:before]))  # chronological, farthest first
        newer_rows = newer[:after]
        resp: dict[str, Any] = {
            "status": "ok",
            "anchor": {"bot": anchor["bot"], "id": str(anchor["id"]),
                       "session": anchor["session_id"], "speaker": anchor["speaker"],
                       "ts": _stamp(anchor["timestamp"]), "content": piece,
                       "truncated": next_offset is not None, "next_offset": next_offset},
            "neighbors": {"before": [], "after": []},
            # Reserved at full width; nulled after fitting where nothing is left.
            "more_before": _CURSOR_PLACEHOLDER, "more_after": _CURSOR_PLACEHOLDER,
        }

        def render(width: int) -> None:
            resp["neighbors"]["before"] = [_neighbor(r, focus, width) for r in older_rows]
            resp["neighbors"]["after"] = [_neighbor(r, focus, width) for r in newer_rows]

        render(WIDE)
        if len(serialize(resp)) > max_chars:
            render(NARROW)
        while len(serialize(resp)) > max_chars and (older_rows or newer_rows):
            # Farthest first; a tie drops the older side.
            if len(older_rows) >= len(newer_rows):
                older_rows.pop(0)
                resp["neighbors"]["before"].pop(0)
            else:
                newer_rows.pop()
                resp["neighbors"]["after"].pop()
        if len(serialize(resp)) > max_chars:
            raise FindError("response_overflow")
        bot, mid = anchor["bot"], str(anchor["id"])
        resp["more_before"] = (self._cursor(owner, access, bot, mid,
                                            older_rows[0] if older_rows else anchor, "before", focus)
                               if len(older) > len(older_rows) else None)
        resp["more_after"] = (self._cursor(owner, access, bot, mid,
                                           newer_rows[-1] if newer_rows else anchor, "after", focus)
                              if len(newer) > len(newer_rows) else None)
        return serialize(resp)

    def _page(self, access: BotAccess, owner: Owner, cursor: ContextCursor, n: int,
              max_chars: int, trigger_id: str | None, focus: tuple[str, ...]) -> str:
        """One more page of neighbours on the cursor's side; the anchor is
        reauthorized first and served as identities only."""
        scope = self._scope(access, cursor.bot, trigger_id)
        with _ReadSnapshot(self.engine) as conn:
            anchor = self._anchor(conn, scope, cursor.id)
            rows = self._side(conn, scope, anchor["session_id"], cursor.ts, cursor.position,
                              cursor.direction, n)
        served = rows[:n]
        more_key = f"more_{cursor.direction}"
        resp: dict[str, Any] = {
            "status": "ok",
            "anchor": {"bot": anchor["bot"], "id": str(anchor["id"]), "session": anchor["session_id"]},
            "neighbors": {cursor.direction: []},
            more_key: _CURSOR_PLACEHOLDER,
        }

        def chronological() -> list[dict]:
            return list(reversed(served)) if cursor.direction == "before" else list(served)

        def render(width: int) -> None:
            resp["neighbors"][cursor.direction] = [_neighbor(r, focus, width) for r in chronological()]

        render(WIDE)
        if len(serialize(resp)) > max_chars:
            render(NARROW)
        while len(serialize(resp)) > max_chars and served:
            served.pop()  # rows are nearest first: the farthest is last
            render(NARROW)
        if len(serialize(resp)) > max_chars:
            raise FindError("response_overflow")
        at = served[-1] if served else {"timestamp": cursor.ts, "id": cursor.position}
        resp[more_key] = (self._cursor(owner, access, cursor.bot, cursor.id, at,
                                       cursor.direction, focus)
                          if len(rows) > len(served) else None)
        return serialize(resp)
