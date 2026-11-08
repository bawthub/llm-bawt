"""HTTP error bodies for unexpected exceptions (TASK-1565).

An unexpected exception's text can name infrastructure: a psycopg2 connect
error carries the DB host, IP and user, and tenants publish the app port. So a
route that catches *any* exception logs the full text and traceback here and
answers with a generic message plus the exception type only.

Deliberate client-facing errors (validation ``ValueError``s, domain errors
whose message is written for the caller) keep their own text; this helper is
for catch-all handlers only. ``tests/test_http_error_bodies.py`` ratchets the
number of catch-all handlers that still echo exception text.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)

SEE_LOG = "see the app log"


def internal_error_detail(exc: BaseException, public: str = "Internal error") -> str:
    """Caller-safe text: ``public`` plus the exception type, never its message."""
    return f"{public} ({type(exc).__name__}); {SEE_LOG}"


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
    logger.error("%s: %s", context, exc, exc_info=exc)
    return HTTPException(status_code=status_code, detail=internal_error_detail(exc, public))
