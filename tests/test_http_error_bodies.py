"""Catch-all HTTP handlers never echo exception text (TASK-1565).

An unexpected exception can name infrastructure (a psycopg2 connect error
carries the DB host, IP and user), and tenants publish the app port. Catch-all
handlers answer via ``internal_error``: generic text plus the exception type,
with the full text in the log. Deliberate client-facing errors caught by a
specific type keep their message and are out of scope here.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.service.http_errors import internal_error
from llm_bawt.service.routes.memory import get_memory_stats

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "src" / "llm_bawt" / "service"
LEAK = 'connection to server at "db.internal" (192.0.2.7) failed for user "tenant_user"'

# Ratchet: catch-all handlers that still put exception text in an
# HTTPException detail. Only allowed to go down.
MAX_CATCH_ALL_ECHO_SITES = 0


def _narrowed(node: ast.AST, name: str) -> bool:
    """``if isinstance(<name>, SomeError):`` -- the body handles one specific type."""
    return (
        isinstance(node, ast.If) and isinstance(node.test, ast.Call)
        and ast.unparse(node.test.func) == "isinstance"
        and len(node.test.args) == 2 and isinstance(node.test.args[0], ast.Name)
        and node.test.args[0].id == name
        and ast.unparse(node.test.args[1]) not in {"Exception", "BaseException"}
    )


def _unnarrowed_nodes(node: ast.AST, name: str):
    """Walk ``node`` but skip bodies of isinstance-narrowed branches (their else still counts)."""
    yield node
    if _narrowed(node, name):
        children = node.orelse
    else:
        children = list(ast.iter_child_nodes(node))
    for child in children:
        yield from _unnarrowed_nodes(child, name)


def _catch_all_echo_sites(root: Path = SERVICE_ROOT) -> list[str]:
    """``except Exception``/bare handlers raising HTTPException(detail=<uses exc text>)."""
    sites = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler) or not handler.name:
                continue
            caught = ast.unparse(handler.type) if handler.type else "BaseException"
            if caught not in {"Exception", "BaseException"}:
                continue
            for node in _unnarrowed_nodes(handler, handler.name):
                if not (isinstance(node, ast.Call) and ast.unparse(node.func).endswith("HTTPException")):
                    continue
                detail = next((k.value for k in node.keywords if k.arg == "detail"), None)
                if detail is None:
                    continue
                # ``type(exc).__name__`` is safe; any other use of the bound name is not.
                uses = [
                    n for n in ast.walk(detail)
                    if isinstance(n, ast.Name) and n.id == handler.name
                ]
                type_only = sum(
                    1 for n in ast.walk(detail)
                    if isinstance(n, ast.Call) and ast.unparse(n.func) == "type"
                    and len(n.args) == 1 and isinstance(n.args[0], ast.Name)
                    and n.args[0].id == handler.name
                )
                if len(uses) > type_only:
                    sites.append(f"{path.relative_to(root)}:{node.lineno}")
    return sites


def test_catch_all_handlers_do_not_echo_exception_text():
    sites = _catch_all_echo_sites()
    assert len(sites) <= MAX_CATCH_ALL_ECHO_SITES, (
        "Catch-all handler echoes exception text in an HTTP body; use "
        f"llm_bawt.service.http_errors.internal_error instead: {sites}"
    )


def test_ratchet_scanner_flags_raw_echo(tmp_path):
    bad = tmp_path / "routes.py"
    bad.write_text(
        "def f():\n"
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        raise HTTPException(status_code=500, detail=f'boom: {e}')\n"
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        raise HTTPException(status_code=500, detail=f'x ({type(e).__name__})')\n"
        "    try:\n        pass\n"
        "    except ValueError as e:\n"
        "        raise HTTPException(status_code=400, detail=str(e))\n"
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        if isinstance(e, KeyError):\n"
        "            raise HTTPException(status_code=400, detail=str(e))\n"
        "        raise HTTPException(status_code=500, detail=str(e))\n",
        encoding="utf-8",
    )
    assert _catch_all_echo_sites(tmp_path) == ["routes.py:5", "routes.py:19"]


def test_internal_error_hides_text_and_logs_it(caplog):
    exc = RuntimeError(LEAK)
    with caplog.at_level(logging.ERROR, logger="llm_bawt.service.http_errors"):
        err = internal_error(exc, "Failed to do the thing", status_code=409, public="Thing rejected")

    assert isinstance(err, HTTPException)
    assert err.status_code == 409
    assert err.detail == "Thing rejected (RuntimeError); see the app log"
    record = caplog.records[-1]
    assert "Failed to do the thing" in record.getMessage() and LEAK in record.getMessage()
    assert record.exc_info and record.exc_info[1] is exc


def test_memory_stats_db_failure_keeps_500_without_db_details():
    def stats():
        raise RuntimeError(LEAK)

    with pytest.raises(HTTPException) as caught:
        get_memory_stats(bot_id="example", client=SimpleNamespace(stats=stats))

    assert caught.value.status_code == 500
    assert caught.value.detail == "Internal error (RuntimeError); see the app log"
    assert "192.0.2.7" not in str(caught.value.detail)
