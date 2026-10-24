"""``messages_find``: locate, compact, faceted (TASK-1323 §5.0–5.3).

Ported from the benchmarked r3 prototype (``find_proto3.find3``) onto the
TASK-1222 access layer. One call:

1. builds a :class:`FindScope` from a fresh :class:`BotAccess` (narrowing only);
2. parses the query into a fill-only rung ladder (:mod:`.find_query`);
3. in one REPEATABLE READ, READ ONLY transaction, runs the first rung with
   matches (the *primary*: it alone defines ``total`` and the facets), builds
   the response as served, fits it into :data:`BUDGET` and freezes it;
4. fills spare bytes with later rungs' cards and ``outside_filters``, which
   only ever replace placeholders reserved at maximum width.

Every count, facet, card, ``also`` id and snippet derives from rows that pass
the scope predicate, which is applied in the WHERE clause before anything is
matched. The serialized string this module returns is the wire payload, so
its length is what the budget measures.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import OperationalError

from llm_bawt.bot_access import BotAccess, BotAccessError, PartitionScope
from llm_bawt.message_authorship import EffectiveAuthorRule

from .find_query import InvalidQuery, Rung, ladder, parse
from .handles import HandleExpired, HandleStore, Owner

logger = logging.getLogger(__name__)

BEST_N, LATEST_N, ALSO_N = 6, 4, 3
PAGE_CARDS = BEST_N + LATEST_N
MAX_PAGES = 4
BUDGET = 3500
WIDTHS = (240, 200, 160, 120, 80, 60)
FACET_N, MATCHED_N, TOKEN_MAX, SLUG_MAX = 5, 6, 40, 32
# Sessions (not messages) per ordering: best takes BEST_N, latest takes
# LATEST_N after dropping best's sessions, so PAGE_CARDS always suffices.
SESSION_LIMIT = PAGE_CARDS
STATEMENT_MS = 3000
OUTSIDE_MS = 250
# Reservation placeholder: wider than any count or null this module emits.
MAX_INT = 9_999_999
SPEAKERS = frozenset({"user", "bot", "delivery"})
RELAXABLE = ("bot", "speaker", "since")
BROWSE = Rung("browse", (), (), "", 0)
_LEXEME = re.compile(r"[a-z0-9]+")
_SPACE = re.compile(r"\s+")


class FindError(RuntimeError):
    """A sanitized ``unavailable`` reason."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


# CallerScope (a protected caller's own partition, §5.0 self-access): no
# blank-author exception, and each class is tied to its role, so the speaker
# label is never a guess. This is narrower than HISTORY_AUTHORITY, which also
# admits author/role mismatches (an assistant row authored by the user); a
# narrower predicate can only omit rows.
_CALLER_CASE = """
    CASE
      WHEN m.role = 'assistant' AND m.author_entity_type = 'bot'
        AND m.author_entity_id = p.bot_id THEN 'bot'
      WHEN m.role = 'user' AND m.author_entity_type = 'user'
        AND m.author_entity_id = :authority_user THEN 'user'
      WHEN m.role = 'user' AND m.author_entity_type = 'bot'
        AND m.author_entity_id = ANY(CAST(:authority_authors AS text[])) THEN 'delivery'
    END
"""

# One row per (partition, message) in scope: positive partition map, canonical
# session join with the user pin and live status, then the effective-author
# class. c.speaker IS NULL means out of scope.
_SCOPE_FROM = f"""
    FROM messages m
    JOIN unnest(CAST(:f_bots AS text[]), CAST(:f_parts AS text[]), CAST(:f_auth AS text[]))
         AS p(bot_id, partition, authority) ON m.bot_id = p.partition
    JOIN sessions s ON s.id = m.session_id AND s.bot_id = p.bot_id
         AND s.user_id = :authority_user AND s.status IN ('active', 'archived')
    CROSS JOIN LATERAL (
        SELECT CASE WHEN p.authority = 'global'
                    THEN ({EffectiveAuthorRule.aggregate_case()})
                    ELSE ({_CALLER_CASE}) END AS speaker
    ) c
"""

_COVERAGE = ("cardinality(ARRAY(SELECT unnest(CAST(:lex AS text[])) "
             "INTERSECT SELECT unnest(tsvector_to_array(m.content_tsv))))")
_RANK = "ts_rank(m.content_tsv, to_tsquery('simple', :orq))"


def serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def failure(code: str) -> str:
    return serialize({"status": "unavailable", "reason": code})


def like_pattern(token: str) -> str:
    """A case-insensitive substring pattern; ``\\`` is ILIKE's default escape."""
    return "%" + token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


@dataclass(frozen=True)
class FindFilters:
    """Caller filters. They only narrow the scope; ``until`` is pinned at the
    first page's call time so later pages read the same window."""

    bot: str | None = None
    speaker: str | None = None
    since: float | None = None
    until: float | None = None

    def supplied(self) -> tuple[str, ...]:
        return tuple(k for k in RELAXABLE if getattr(self, k) is not None)


@dataclass(frozen=True)
class FindRequest:
    query: str | None = None
    bot: str | None = None
    speaker: str | None = None
    since: float | None = None
    until: float | None = None
    handle: str | None = None


@dataclass(frozen=True)
class ServedCard:
    """Identities of one served card; never content (§5.3)."""

    ref: str
    bot: str
    id: str
    session: str
    also: tuple[str, ...]
    kind: str  # "best" | "latest"
    rank: int
    rung: str
    page: int


@dataclass(frozen=True)
class FindContinuation:
    """Intent plus served identities. Each page re-runs under current policy."""

    query: str | None
    filters: FindFilters
    as_of: float
    plan: tuple[Rung, ...]
    primary: int
    terms: tuple[str, ...]
    served: tuple[ServedCard, ...]
    pages: int

    @property
    def served_sessions(self) -> frozenset[tuple[str, str]]:
        return frozenset((card.bot, card.session) for card in self.served)


FIND_HANDLES: HandleStore[FindContinuation] = HandleStore()


@dataclass(frozen=True)
class FindScope:
    """The partitions one call may read, and the authority values they bind.

    ``partitions`` honours the ``bot`` filter; ``everywhere`` is the caller's
    whole aggregate scope, which ``outside_filters`` uses to relax that
    filter. A ``bot`` that is excluded or unknown yields no partitions, so it
    answers with the same ``no_match`` as an empty in-scope bot.
    """

    partitions: tuple[PartitionScope, ...]
    everywhere: tuple[PartitionScope, ...]
    user_id: str
    public_authors: frozenset[str]
    trigger_id: str | None = None

    @classmethod
    def build(cls, access: BotAccess, bot: str | None,
              trigger_id: str | None = None) -> FindScope:
        everywhere = access.aggregate_scope(require_user=True)
        partitions = everywhere.partitions
        if bot is not None:
            try:
                partitions = access.aggregate_scope(bot, require_user=True).partitions
            except BotAccessError as error:
                if error.code != "access_denied":
                    raise
                partitions = ()
        return cls(partitions, everywhere.partitions, everywhere.user_id,
                   everywhere.public_authors, trigger_id)

    def params(self, *, everywhere: bool = False) -> dict[str, Any]:
        parts = self.everywhere if everywhere else self.partitions
        return {
            "f_bots": [p.bot_id for p in parts],
            "f_parts": [p.partition for p in parts],
            "f_auth": [p.authority for p in parts],
            "authority_user": self.user_id,
            "authority_authors": sorted(self.public_authors),
        }

    def partition_of(self, bot: str) -> str:
        for p in self.everywhere:
            if p.bot_id == bot:
                return p.partition
        raise BotAccessError("access_denied")


def _match(rung: Rung, p: dict[str, Any]) -> tuple[str, str, str, str]:
    """(match condition, exact-hit count, lexeme coverage, rank) for one rung."""
    coverage = rank = "0"
    if rung.lexemes:
        p["lex"] = list(rung.lexemes)
        p["orq"] = " | ".join(f"'{lexeme}'" for lexeme in rung.lexemes)
        coverage, rank = _COVERAGE, _RANK
    if rung.exact:
        names = [f"e{i}" for i in range(len(rung.exact))]
        p.update({name: like_pattern(token) for name, token in zip(names, rung.exact)})
        joiner = " AND " if rung.joiner == "all" else " OR "
        condition = "(" + joiner.join(f"m.content ILIKE :{n}" for n in names) + ")"
        hits = " + ".join(f"(m.content ILIKE :{n})::int" for n in names)
        return condition, hits, coverage, rank
    if rung.lexemes:
        p["need"] = rung.need
        return f"m.content_tsv @@ to_tsquery('simple', :orq) AND {coverage} >= :need", "0", coverage, rank
    return "TRUE", "0", coverage, rank


def _bounded_facet(counts: dict | None) -> dict[str, int]:
    items = sorted((counts or {}).items(), key=lambda kv: (-kv[1], kv[0]))
    out = {str(k)[:SLUG_MAX]: int(n) for k, n in items[:FACET_N]}
    rest = sum(int(n) for _, n in items[FACET_N:])
    if rest:
        out["_other"] = rest
    return out


def _collapse(rows: Iterable[dict] | None, n: int,
              skip: frozenset[tuple[str, str]] | set[tuple[str, str]]
              ) -> tuple[list[dict], set[tuple[str, str]]]:
    """Up to ``n`` cards from session rows (one per (bot, session), already
    collapsed and ordered in SQL), skipping sessions in ``skip``."""
    cards: list[dict] = []
    for row in rows or []:
        if len(cards) >= n:
            break
        if (row["bot"], row["session_id"]) in skip:
            continue
        cards.append({
            "ref": "", "id": str(row["id"]), "bot": str(row["bot"]),
            "session": row["session_id"],
            "ts": datetime.fromtimestamp(row["timestamp"], timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "speaker": row["speaker"], "hits": int(row["hits"]),
            "also": [str(i) for i in (row["also"] or [])], "snippet": "", "matched": [],
        })
    return cards, {(card["bot"], card["session"]) for card in cards}


def _probe(term: str) -> str:
    return term.lower()[:max(4, min(len(term), 6))]


def snippet(content: str, terms: tuple[str, ...], width: int) -> tuple[str, list[str]]:
    """``width`` chars around the first matched term (a third before it), with
    whitespace runs collapsed and ellipses where the content continues."""
    low = content.lower()
    positions = [p for p in (low.find(_probe(t)) for t in terms) if p >= 0]
    at = min(positions) if positions else 0
    start = max(0, at - width // 3)
    body = _SPACE.sub(" ", content[start:start + width]).strip()
    matched = [t[:TOKEN_MAX] for t in terms if _probe(t) in low][:MATCHED_N]
    return ("…" if start else "") + body + ("…" if start + width < len(content) else ""), matched


def _set_snippets(cards: list[dict], bodies: dict[tuple[str, str], str],
                  terms: tuple[str, ...], width: int) -> None:
    for card in cards:
        card["snippet"], card["matched"] = snippet(
            bodies.get((card["bot"], card["id"]), ""), terms, width)


def _fit(resp: dict, bodies: dict, terms: tuple[str, ...]) -> int:
    """Fit the primary response: shrink snippets, then drop ``latest`` cards
    from the end, then ``best`` cards from the end (never the first card)."""
    cards = resp["best"] + resp["latest"]
    for width in WIDTHS:
        _set_snippets(cards, bodies, terms, width)
        if len(serialize(resp)) <= BUDGET:
            return width
    keep_latest = 0 if resp["best"] else 1
    while len(serialize(resp)) > BUDGET and len(resp["latest"]) > keep_latest:
        resp["latest"].pop()
    while len(serialize(resp)) > BUDGET and len(resp["best"]) > 1:
        resp["best"].pop()
    return WIDTHS[-1]


class MessageFinder:
    """Runs one ``messages_find`` page under a caller's :class:`BotAccess`."""

    def __init__(self, engine: Engine, handles: HandleStore[FindContinuation] = FIND_HANDLES,
                 clock=time.time):
        self.engine = engine
        self.handles = handles
        self.clock = clock

    # -- entry -------------------------------------------------------------

    def find(self, access: BotAccess, request: FindRequest, *,
             trigger_id: str | None = None, fill: bool = True) -> str:
        """The serialized wire payload; failures are ``unavailable`` with a reason."""
        try:
            return self._find(access, request, trigger_id, fill)
        except BotAccessError as error:
            return failure(error.code)
        except (FindError, InvalidQuery) as error:
            if getattr(error, "code", None) == "response_overflow":
                logger.error("messages_find response overflow")
            return failure(getattr(error, "code", None) or str(error))
        except HandleExpired:
            return failure("handle_expired")
        except OperationalError as error:
            code = getattr(getattr(error, "orig", None), "pgcode", None)
            logger.warning("messages_find query failed: %s", type(error).__name__)
            return failure("timeout" if code == "57014" else "source_unavailable")
        except Exception:
            logger.exception("messages_find failed")
            return failure("source_unavailable")

    def _find(self, access: BotAccess, request: FindRequest,
              trigger_id: str | None, fill: bool) -> str:
        if access.caller is None or not access.caller.user_id:
            raise BotAccessError("caller_unavailable")
        owner: Owner = (access.caller.bot_id, access.caller.user_id)
        if request.handle is not None:
            if any(v is not None for v in (request.query, request.bot, request.speaker,
                                           request.since, request.until)):
                raise FindError("invalid_request")
            return self._next_page(access, owner, request.handle, trigger_id, fill)
        if request.speaker is not None and request.speaker not in SPEAKERS:
            raise FindError("invalid_request")
        query = request.query.strip() if isinstance(request.query, str) else request.query
        parsed = parse(query)  # length and type checks raise invalid_query
        as_of = self.clock()
        until = as_of if request.until is None else min(request.until, as_of)
        filters = FindFilters(request.bot, request.speaker, request.since, until)
        scope = FindScope.build(access, request.bot, trigger_id)
        with self._transaction() as conn:
            if not query:
                plan, terms = (BROWSE,), ()
            else:
                word_lex, part_lex = self._lexemes(conn, parsed.words, parsed.parts + parsed.words)
                plan, terms = ladder(parsed, word_lex, part_lex if parsed.exact else ()), parsed.terms
                if not plan:
                    return serialize({"status": "no_match", "reason": "no_searchable_terms"})
            key = self.handles.new_key()
            resp, served, primary = self._page(conn, scope, filters, as_of, plan, None, terms,
                                                frozenset(), 0, key, fill)
        if resp["status"] != "ok":
            return serialize(resp)  # no_match carries no handle and stores none
        self.handles.put(key, owner, FindContinuation(
            query, filters, as_of, plan, primary, terms, served, 1,
        ), capability_expires_at=access.caller.expires_at)
        return serialize(resp)

    def _next_page(self, access: BotAccess, owner: Owner, key: str,
                   trigger_id: str | None, fill: bool) -> str:
        cont = self.handles.get(key, owner)
        if cont.pages >= MAX_PAGES:
            return serialize({"status": "no_match", "reason": "page_limit"})
        scope = FindScope.build(access, cont.filters.bot, trigger_id)
        with self._transaction() as conn:
            resp, served, _ = self._page(conn, scope, cont.filters, cont.as_of, cont.plan,
                                         cont.primary, cont.terms, cont.served_sessions,
                                         cont.pages, key, fill)
        if resp["status"] != "ok":
            return serialize({"status": "no_match", "reason": "exhausted"})
        if not self.handles.replace(key, owner, cont, replace(
                cont, served=cont.served + served, pages=cont.pages + 1)):
            raise FindError("handle_busy")
        return serialize(resp)

    # -- SQL ---------------------------------------------------------------

    def _transaction(self):
        return _ReadSnapshot(self.engine)

    @staticmethod
    def _lexemes(conn: Connection, words: tuple[str, ...],
                 part_words: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """English-config lexemes, from the same parser that built ``content_tsv``."""
        row = conn.execute(text(
            "SELECT tsvector_to_array(to_tsvector('english', :w)) AS w, "
            "tsvector_to_array(to_tsvector('english', :pw)) AS pw"
        ), {"w": " ".join(words), "pw": " ".join(part_words)}).mappings().one()

        def clean(values) -> tuple[str, ...]:
            return tuple(v for v in (values or []) if _LEXEME.fullmatch(v))
        return clean(row["w"]), clean(row["pw"])

    @staticmethod
    def _where(scope: FindScope, filters: FindFilters, p: dict, drop: tuple[str, ...]) -> str:
        clauses = ["m.bot_id = ANY(CAST(:f_parts AS text[]))", "c.speaker IS NOT NULL",
                   "m.timestamp <= :until"]
        p["until"] = filters.until
        if filters.speaker is not None and "speaker" not in drop:
            clauses.append("c.speaker = :speaker")
            p["speaker"] = filters.speaker
        if filters.since is not None and "since" not in drop:
            clauses.append("m.timestamp >= :since")
            p["since"] = filters.since
        if scope.trigger_id:
            clauses.append("m.id <> :trigger_id")
            p["trigger_id"] = scope.trigger_id
        return " AND ".join(clauses)

    def _count(self, conn: Connection, scope: FindScope, filters: FindFilters,
               rung: Rung, drop: tuple[str, ...]) -> int:
        p = scope.params(everywhere="bot" in drop)
        condition, _, _, _ = _match(rung, p)
        where = self._where(scope, filters, p, drop)
        return int(conn.execute(text(
            f"SELECT count(*) {_SCOPE_FROM} WHERE {where} AND {condition}"), p).scalar())

    def _run(self, conn: Connection, scope: FindScope, filters: FindFilters,
             as_of: float, rung: Rung, exclude: Iterable[tuple[str, str]] = ()) -> dict:
        """Totals and facets over every match; ``best``/``latest`` are session
        rows (representative message, exact ``hits``, next ``also`` ids) from
        sessions not in ``exclude``, so a dense or already-served session can
        never use up the limit and hide another session."""
        p = scope.params()
        condition, hits, coverage, rank = _match(rung, p)
        where = self._where(scope, filters, p, ())
        skip = sorted(set(exclude))
        p.update({"at": as_of, "session_limit": SESSION_LIMIT, "also_end": ALSO_N + 1,
                  "x_bots": [b for b, _ in skip], "x_sessions": [s for _, s in skip]})
        statement = f"""
            WITH mm AS MATERIALIZED (
              SELECT m.id, p.bot_id AS bot, m.session_id, m.timestamp, c.speaker,
                     {hits} AS eh, {coverage} AS wc, {rank} AS r
              {_SCOPE_FROM}
              WHERE {where} AND {condition}
            ), lbl AS (
              SELECT *, CASE WHEN :at - timestamp <= 86400 THEN '24h'
                             WHEN :at - timestamp <= 604800 THEN '7d'
                             WHEN :at - timestamp <= 2592000 THEN '30d'
                             ELSE 'older' END AS age
              FROM mm
            ), ranked AS MATERIALIZED (
              SELECT *,
                row_number() OVER (PARTITION BY bot, session_id
                                   ORDER BY eh DESC, wc DESC, r DESC, timestamp DESC, id) AS brn,
                row_number() OVER (PARTITION BY bot, session_id
                                   ORDER BY timestamp DESC, id) AS lrn,
                count(*) OVER (PARTITION BY bot, session_id) AS hits
              FROM mm
              WHERE NOT EXISTS (
                SELECT 1 FROM unnest(CAST(:x_bots AS text[]), CAST(:x_sessions AS text[]))
                     AS x(bot, session_id)
                WHERE x.bot = mm.bot AND x.session_id = mm.session_id)
            ), extra AS (
              SELECT bot, session_id,
                array_agg(id ORDER BY brn) FILTER (WHERE brn BETWEEN 2 AND :also_end) AS also_best,
                array_agg(id ORDER BY lrn) FILTER (WHERE lrn BETWEEN 2 AND :also_end) AS also_latest
              FROM ranked WHERE brn <= :also_end OR lrn <= :also_end
              GROUP BY 1, 2
            )
            SELECT (SELECT count(*) FROM mm) AS total,
              (SELECT json_object_agg(k, n) FROM (SELECT bot k, count(*) n FROM mm GROUP BY 1) x) AS f_bot,
              (SELECT json_object_agg(k, n) FROM (SELECT age k, count(*) n FROM lbl GROUP BY 1) x) AS f_age,
              (SELECT json_object_agg(k, n) FROM (SELECT speaker k, count(*) n FROM mm GROUP BY 1) x) AS f_speaker,
              (SELECT json_agg(b) FROM (
                   SELECT k.id, k.bot, k.session_id, k.timestamp, k.speaker, k.hits,
                          e.also_best AS also
                   FROM ranked k JOIN extra e USING (bot, session_id) WHERE k.brn = 1
                   ORDER BY k.eh DESC, k.wc DESC, k.r DESC, k.timestamp DESC, k.bot, k.id
                   LIMIT :session_limit) b) AS best,
              (SELECT json_agg(b) FROM (
                   SELECT k.id, k.bot, k.session_id, k.timestamp, k.speaker, k.hits,
                          e.also_latest AS also
                   FROM ranked k JOIN extra e USING (bot, session_id) WHERE k.lrn = 1
                   ORDER BY k.timestamp DESC, k.bot, k.id
                   LIMIT :session_limit) b) AS latest
        """
        return dict(conn.execute(text(statement), p).mappings().one())

    @staticmethod
    def _bodies(conn: Connection, scope: FindScope, cards: list[dict]) -> dict[tuple[str, str], str]:
        """Content of cards already in scope, read in the same snapshot."""
        if not cards:
            return {}
        bots = {p.partition: p.bot_id for p in scope.everywhere}
        parts = [scope.partition_of(card["bot"]) for card in cards]
        rows = conn.execute(text("""
            SELECT m.bot_id, m.id, m.content FROM messages m
            JOIN unnest(CAST(:parts AS text[]), CAST(:ids AS text[])) AS k(partition, id)
              ON m.bot_id = k.partition AND m.id = k.id
            WHERE m.bot_id = ANY(CAST(:parts AS text[]))
        """), {"parts": parts, "ids": [card["id"] for card in cards]})
        return {(bots[row[0]], str(row[1])): row[2] or "" for row in rows}

    def _outside(self, conn: Connection, scope: FindScope, filters: FindFilters,
                 rung: Rung, total: int) -> dict[str, int | None]:
        """Per supplied filter, the extra matches from dropping only that one.

        Each count gets its own 250 ms budget inside a savepoint; a timeout
        reports null and the transaction's own timeout is restored.
        """
        out: dict[str, int | None] = {}
        previous = conn.execute(text("SHOW statement_timeout")).scalar()
        for name in filters.supplied():
            try:
                with conn.begin_nested():
                    conn.execute(text(f"SET LOCAL statement_timeout = '{OUTSIDE_MS}ms'"))
                    out[name] = max(0, self._count(conn, scope, filters, rung, (name,)) - total)
            except OperationalError:
                out[name] = None
            conn.execute(text("SELECT set_config('statement_timeout', :v, true)"),
                         {"v": previous})
        return out

    # -- one page ------------------------------------------------------------

    def _page(self, conn: Connection, scope: FindScope, filters: FindFilters, as_of: float,
              plan: tuple[Rung, ...], primary: int | None, terms: tuple[str, ...],
              served: frozenset[tuple[str, str]], page: int, key: str, fill: bool,
              ) -> tuple[dict, tuple[ServedCard, ...], int]:
        tried: list[list] = []
        if primary is None:
            for i, rung in enumerate(plan):
                row = self._run(conn, scope, filters, as_of, rung, served)
                tried.append([rung.name, row["total"]])
                if row["total"]:
                    primary = i
                    break
            if primary is None:
                absent = [token[:TOKEN_MAX] for token in plan[0].exact]
                return {"status": "no_match", "rungs": tried, "absent": absent}, (), -1
        else:
            row = self._run(conn, scope, filters, as_of, plan[primary], served)
            tried = [[r.name, None] for r in plan[:primary]] + [[plan[primary].name, row["total"]]]
        rung = plan[primary]
        browse = rung is BROWSE or rung.name == BROWSE.name
        best_n, latest_n = (0, PAGE_CARDS) if browse else (BEST_N, LATEST_N)
        best, seen = _collapse(row["best"], best_n, served) if best_n else ([], set())
        latest, more = _collapse(row["latest"], latest_n, served | seen)
        seen |= more
        if not best and not latest:
            return {"status": "no_match"}, (), primary
        n_best, n_latest = len(best), len(latest)
        for i, card in enumerate(best + latest):
            card["ref"] = f"c{page}.{i}"
        next_ref = n_best + n_latest
        bodies = self._bodies(conn, scope, best + latest)
        supplied = filters.supplied()
        # Reserve: the response as served, every fill-dependent field at max width.
        resp: dict[str, Any] = {
            "status": "ok", "rung": rung.name, "total": row["total"],
            "rungs": [[r.name, MAX_INT] for r in plan],
            "facets": {"bot": _bounded_facet(row["f_bot"]), "age": _bounded_facet(row["f_age"]),
                       "speaker": _bounded_facet(row["f_speaker"])},
            "best": best, "latest": latest, "omitted": [MAX_INT, MAX_INT],
        }
        if supplied:
            resp["outside_filters"] = {name: MAX_INT for name in supplied}
        resp["handle"] = key
        width = _fit(resp, bodies, terms)
        frozen = len(serialize(resp))
        if frozen > BUDGET:
            raise FindError("response_overflow")
        # Frozen. Fill touches only reserved fields and spare bytes from here.
        rungs = tried + [[r.name, None] for r in plan[len(tried):]]
        underfilled = (n_latest < PAGE_CARDS) if browse else (n_best < BEST_N)
        fallback: list[dict] = []
        if fill and underfilled and not browse:
            for j in range(primary + 1, len(plan)):
                frow = self._run(conn, scope, filters, as_of, plan[j], served | seen)
                rungs[j][1] = frow["total"]
                if frow["total"]:
                    cards, more = _collapse(frow["best"], BEST_N - n_best - len(fallback), served | seen)
                    seen |= more
                    for card in cards:
                        card["ref"] = f"c{page}.{next_ref}"
                        card["rung"] = plan[j].name
                        next_ref += 1
                    fallback += cards
                if n_best + len(fallback) >= BEST_N:
                    break
        outside = None
        if fill and supplied and underfilled:
            outside = self._outside(conn, scope, filters, rung, row["total"])
        resp["rungs"] = rungs
        resp["omitted"] = [n_best - len(resp["best"]), n_latest - len(resp["latest"])]
        if supplied:
            if outside is None:
                del resp["outside_filters"]
            else:
                resp["outside_filters"] = outside
        if len(serialize(resp)) > frozen:
            raise FindError("response_overflow")
        if fallback:
            bodies.update(self._bodies(conn, scope, fallback))
            _set_snippets(fallback, bodies, terms, width)
            for card in fallback:
                resp["best"].append(card)
                if len(serialize(resp)) > BUDGET:
                    resp["best"].pop()
                    break
        if len(serialize(resp)) > BUDGET:
            raise FindError("response_overflow")
        records = tuple(
            ServedCard(card["ref"], card["bot"], card["id"], card["session"], tuple(card["also"]),
                       kind, rank, card.get("rung", rung.name), page)
            for kind, cards in (("best", resp["best"]), ("latest", resp["latest"]))
            for rank, card in enumerate(cards)
        )
        return resp, records, primary


class _ReadSnapshot:
    """One REPEATABLE READ, READ ONLY transaction with a statement budget, so
    totals, facets, cards, bodies and outside counts all read one snapshot."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._conn: Connection | None = None
        self._txn = None

    def __enter__(self) -> Connection:
        self._conn = self.engine.connect()
        try:
            self._txn = self._conn.begin()
            self._conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            self._conn.execute(text(f"SET LOCAL statement_timeout = '{STATEMENT_MS}ms'"))
            self._conn.execute(text("SET LOCAL lock_timeout = '500ms'"))
        except BaseException:
            self._conn.close()
            raise
        return self._conn

    def __exit__(self, *exc) -> None:
        try:
            if self._txn is not None and self._txn.is_active:
                self._txn.rollback()  # read only: nothing to commit
        finally:
            self._conn.close()
