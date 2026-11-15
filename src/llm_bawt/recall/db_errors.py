"""Classify recall database failures as transient (worth one retry) or permanent."""
from __future__ import annotations

from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout

# Failures a second attempt can plausibly clear: statement timeout (a cold
# cache), lock wait, serialization/deadlock, server overload or restart.
# Connection errors (class 08) are transient too.
TRANSIENT_SQLSTATES = frozenset({"57014", "55P03", "40001", "40P01", "53300", "57P01", "57P02", "57P03"})


def db_sqlstate(error: BaseException | None) -> str | None:
    if error is None:
        return None
    original = getattr(error, "orig", None) or error
    return getattr(original, "pgcode", None) or getattr(original, "sqlstate", None)


def is_transient_db_error(error: BaseException) -> bool:
    """Transient if a retry may succeed; SQL, schema, data and auth errors are permanent."""
    if isinstance(error, PoolTimeout):
        return True
    if isinstance(error, DBAPIError) and error.connection_invalidated:
        return True
    code = db_sqlstate(error)
    if code:
        return code in TRANSIENT_SQLSTATES or code.startswith("08")
    # psycopg2 reports a dropped connection as OperationalError with no SQLSTATE.
    return isinstance(error, OperationalError)
