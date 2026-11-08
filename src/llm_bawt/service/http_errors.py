"""HTTP error bodies for unexpected exceptions (TASK-1565).

An unexpected exception's text can name infrastructure: a psycopg2 connect
error carries the DB host, IP and user, and tenants publish the app port. So a
route that catches *any* exception logs the full text and traceback here and
answers with a generic message plus the exception type and a correlation ref.

The text helpers live in :mod:`llm_bawt.utils.error_text` (no web
dependencies) and are re-exported here for service code.

Deliberate client-facing errors (validation ``ValueError``s, domain errors
whose message is written for the caller) keep their own text; this helper is
for catch-all handlers only. ``tests/test_http_error_bodies.py`` ratchets the
number of catch-all handlers that still echo exception text.
"""

from __future__ import annotations

from fastapi import HTTPException

from ..utils.error_text import (
    SEE_LOG,
    client_error_text,
    internal_error_detail,
    logger,
    names_infrastructure,
    new_error_ref,
    safe_error_text,
)

__all__ = [
    "SEE_LOG",
    "client_error_text",
    "internal_error",
    "internal_error_detail",
    "names_infrastructure",
    "new_error_ref",
    "safe_error_text",
]


def internal_error(
    exc: BaseException,
    context: str,
    *,
    status_code: int = 500,
    public: str = "Internal error",
) -> HTTPException:
    """Log ``exc`` in full under ``context`` and return a sanitized HTTPException.

    Use as ``raise internal_error(e, "Failed to list memories") from e``.
    """
    ref = new_error_ref()
    logger.error("%s [ref %s]: %s", context, ref, exc, exc_info=exc)
    return HTTPException(status_code=status_code, detail=internal_error_detail(exc, public, ref))
