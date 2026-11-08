"""Caller-safe exception text (TASK-1565, TASK-1611).

An exception's text can name infrastructure: a psycopg2 connect error carries
the DB host, IP and user. Text that reaches a caller (HTTP body, SSE frame,
task result, status field) uses these helpers: a DB/Redis driver error is
replaced by generic text plus the exception type and a short ``ref``, and the
full text is logged under the same ref so an operator can match the two.
Domain errors keep their message, which is written for the caller.

Standard library only, so CLI and memory modules can use it without the
optional web service dependencies.
"""

from __future__ import annotations

import logging
import secrets

logger = logging.getLogger("llm_bawt.service.http_errors")

SEE_LOG = "see the app log"

# Driver packages whose exception text names the DB/Redis host, IP or user.
# Matched on the top-level module of every exception in the chain, so a
# domain error raised ``from`` a driver error is caught too.
_INFRASTRUCTURE_MODULES = frozenset({"psycopg", "psycopg2", "psycopg_pool", "sqlalchemy", "asyncpg", "redis"})


def new_error_ref() -> str:
    """Short correlation id shared by a sanitized message and its log line."""
    return secrets.token_hex(4)


def internal_error_detail(exc: BaseException, public: str = "Internal error", ref: str | None = None) -> str:
    """Caller-safe text: ``public`` plus the exception type (and ``ref``), never its message."""
    tag = f"{type(exc).__name__}, ref {ref}" if ref else type(exc).__name__
    return f"{public} ({tag}); {SEE_LOG}"


def names_infrastructure(exc: BaseException) -> bool:
    """True if ``exc`` or anything in its cause/context chain is a DB/Redis driver error."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if type(current).__module__.split(".", 1)[0] in _INFRASTRUCTURE_MODULES:
            return True
        pending.extend((current.__cause__, current.__context__))
    return False


def safe_error_text(exc: BaseException, public: str = "Internal error") -> str:
    """:func:`client_error_text` for callers that already log the traceback.

    Use where a caught exception becomes a string stored in a result, status
    or history row that a caller can later read. Logs one line tying the ref
    to the full text, without a second traceback.
    """
    if not names_infrastructure(exc):
        return str(exc)
    ref = new_error_ref()
    logger.warning("%s [ref %s]: %s", public, ref, exc)
    return internal_error_detail(exc, public, ref)


def client_error_text(exc: BaseException, context: str, public: str = "Internal error") -> str:
    """Text for a caller-visible error frame or detail field.

    Domain errors keep their message; a DB/Redis driver error anywhere in the
    chain is logged in full under ``context`` and replaced by
    :func:`internal_error_detail`.
    """
    if not names_infrastructure(exc):
        return str(exc)
    ref = new_error_ref()
    logger.error("%s [ref %s]: %s", context, ref, exc, exc_info=exc)
    return internal_error_detail(exc, public, ref)
