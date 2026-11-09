"""TASK-2061: a session's first message, its given name and who started it.

A session label is either a name someone gave it (a user rename or a
schedule's name, ``title_source`` other than ``auto``) or the session's first
eligible message, read from ``messages`` every time and never stored or cut.
The removed auto-title writer kept a 60-character copy of that message as
the ``title``; search over the copy missed everything past the cut. Existing
``title_source='auto'`` rows are ignored by every reader until cleanup.

Eligibility (one SQL rule shared by every reader): a ``user``-role row whose
content, stripped of Python whitespace, is non-empty, is not a slash command,
and still has text once code-fence markers, backticks and markdown
heading/quote/list markers are removed. Ties break on ``(timestamp, id)``.

Every query using these fragments must bind ``:fm_strip`` via
``first_message_params()``.
"""
from __future__ import annotations

import re

# Every character Python's str.strip() removes. PostgreSQL btrim's default
# trims only ASCII space; eligibility follows the Python rule.
PYTHON_STRIP_CHARS = "".join(c for c in map(chr, range(0x110000)) if c.isspace())


# Every character Python's str.splitlines() breaks on.
PYTHON_LINE_BREAKS = "".join(
    c for c in map(chr, range(0x110000)) if len(f"a{c}b".splitlines()) == 2)


def first_message_params() -> dict:
    return {"fm_strip": PYTHON_STRIP_CHARS, "fm_lines": f"[{PYTHON_LINE_BREAKS}]"}


def eligible_sql(alias: str) -> str:
    """SQL predicate: ``alias`` is an eligible first-message row.

    Step for step the retired writer's rule: strip (Python whitespace); not
    empty and not a slash command; code-fence openers and backticks become
    spaces; some line (split as ``str.splitlines``) is non-empty after
    ``strip().lstrip("#>*- ").strip()``. Per line, because a marker after
    non-ASCII whitespace (``"#\\u00a0#"``) is content, and a line made only
    of markers and such whitespace is not.
    """
    content = f"btrim({alias}.content, :fm_strip)"
    return f"""(
        {alias}.role = 'user'
        AND {content} <> ''
        AND left({content}, 1) <> '/'
        AND EXISTS (
            SELECT 1 FROM regexp_split_to_table(
                replace(regexp_replace({content}, '```[a-zA-Z0-9_-]*', ' ', 'g'), '`', ' '),
                :fm_lines) AS line
            WHERE btrim(ltrim(btrim(line, :fm_strip), '#>*- '), :fm_strip) <> ''
        )
    )"""


def storage_bot_sql(slug_expr: str) -> str:
    """SQL twin of ``utils.bot_partition.sanitize_table_name``.

    ``messages.bot_id`` holds the partition key (``task-1946`` -> ``task1946``),
    not the session's slug; this keeps the lateral on one partition.
    """
    return f"COALESCE(NULLIF(regexp_replace(lower({slug_expr}), '[^a-z0-9_]', '', 'g'), ''), 'default')"


def given_title_sql(session_alias: str = "s") -> str:
    """SQL text: a name someone gave the session, else ''. Auto copies excluded."""
    meta = f"{session_alias}.session_metadata"
    return f"""(CASE
        WHEN jsonb_typeof({meta}->'title') = 'string'
             AND COALESCE({meta}->>'title_source', '') <> 'auto'
        THEN btrim({meta}->>'title', :fm_strip) ELSE '' END)"""


def first_message_lateral(messages_table: str, session_alias: str = "s", alias: str = "fm") -> str:
    """LEFT JOIN LATERAL selecting the first eligible message and its origin.

    Exposes ``{alias}_id``, ``{alias}_text``, ``{alias}_author_type``,
    ``{alias}_author_id`` and ``{alias}_schedule_id``. A schedule origin comes
    only from the trusted occurrence -> delivery -> message join (the same
    rule as ``routes/history_scheduler.py``), never from session metadata.
    """
    s = session_alias
    return f"""
        LEFT JOIN LATERAL (
            SELECT m1.id AS {alias}_id, m1.content AS {alias}_text,
                   m1.author_entity_type AS {alias}_author_type,
                   m1.author_entity_id AS {alias}_author_id,
                   (SELECT o.job_id FROM inter_bot_deliveries d
                    JOIN prompt_occurrences o ON o.delivery_id = d.id
                    JOIN prompt_schedules p ON p.job_id = o.job_id
                    WHERE d.user_message_id = m1.id AND d.target_bot_id = {s}.bot_id
                      AND o.session_id = {s}.id AND p.owner_user_id = {s}.user_id
                      AND ((d.author_entity_type = 'user' AND d.author_entity_id = {s}.user_id)
                           OR (d.author_entity_type = 'bot'
                               AND d.author_entity_id = p.creator_bot_id))
                    LIMIT 1) AS {alias}_schedule_id
            FROM {messages_table} m1
            WHERE m1.session_id = {s}.id AND m1.bot_id = {storage_bot_sql(s + '.bot_id')}
              AND {eligible_sql('m1')}
            ORDER BY m1.timestamp ASC, m1.id ASC
            LIMIT 1
        ) {alias} ON TRUE"""


INITIATOR_KINDS = ("user", "bot", "schedule", "system", "unknown")


def initiator_kind_sql(alias: str = "fm") -> str:
    """SQL twin of ``initiated_by(...)['kind']``; NULL without a first message."""
    return f"""(CASE
        WHEN {alias}_id IS NULL THEN NULL
        WHEN {alias}_schedule_id IS NOT NULL THEN 'schedule'
        WHEN {alias}_author_type IN ('user', 'bot', 'system')
             AND {alias}_author_id IS NOT NULL THEN {alias}_author_type
        ELSE 'unknown' END)"""


def initiated_by(author_type: str | None, author_id: str | None,
                 schedule_id: str | None) -> dict | None:
    """Who started the session, from its first eligible message.

    ``None`` when the session has no eligible message. A missing author is
    ``unknown`` rather than a guess.
    """
    if schedule_id:
        return {"kind": "schedule", "id": str(schedule_id)}
    if author_type in ("user", "bot", "system") and author_id:
        return {"kind": author_type, "id": str(author_id)}
    return {"kind": "unknown", "id": None}


def first_message_fields(row) -> dict:
    """``first_message_text`` / ``initiated_by`` from a row using the lateral."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    if get("fm_id") is None:
        return {"first_message_text": None, "initiated_by": None}
    return {
        "first_message_text": get("fm_text"),
        "initiated_by": initiated_by(get("fm_author_type"), get("fm_author_id"),
                                     get("fm_schedule_id")),
    }


def fetch_first_messages(conn, sessions: list[dict], *,
                         messages_table: str = "messages") -> dict[str, dict]:
    """One batched read: ``{session_id: first_message_fields}`` for given rows.

    ``sessions`` are already-authorized rows carrying ``id``, ``bot_id`` and
    ``user_id``; the lookup is keyed by all three and adds no authority.
    A bot-scoped caller may pass its internally resolved partition identifier
    to avoid planning every partition of the parent. Never pass request SQL;
    only simple identifiers are accepted, and the bot/owner checks still apply.
    """
    from sqlalchemy import text

    if not re.fullmatch(r"[a-z_][a-z0-9_]*", messages_table):
        raise ValueError("Expected an internal messages table identifier")
    keyed = [s for s in sessions if s.get("id") and s.get("bot_id")]
    if not keyed:
        return {}
    # Join the registry on the full (id, bot, user) key: a caller's stale or
    # foreign tuple yields nothing rather than another owner's message.
    sql = text(f"""
        SELECT s.id, fm.*
        FROM unnest(CAST(:ids AS text[]), CAST(:bots AS text[]), CAST(:users AS text[]))
             AS k(id, bot_id, user_id)
        JOIN sessions s ON s.id = k.id AND s.bot_id = k.bot_id
                       AND s.user_id IS NOT DISTINCT FROM k.user_id
        {first_message_lateral(messages_table)}
    """)
    rows = conn.execute(sql, {
        "ids": [str(s["id"]) for s in keyed],
        "bots": [str(s["bot_id"]) for s in keyed],
        "users": [s.get("user_id") for s in keyed],
        **first_message_params(),
    }).mappings().all()
    return {str(row["id"]): first_message_fields(row) for row in rows}
