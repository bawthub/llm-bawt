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
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.service.http_errors import internal_error
from llm_bawt.service.routes import history_search
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


BROAD = {"Exception", "BaseException"}


def _broad_scopes(tree: ast.AST):
    """(scope, name) pairs where ``name`` may hold any exception.

    Catch-all ``except`` handlers, plus helpers taking an ``Exception``-typed
    parameter, since a route can delegate its catch-all to one of those.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.name:
            if (ast.unparse(node.type) if node.type else "BaseException") in BROAD:
                yield node, node.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if arg.annotation is not None and ast.unparse(arg.annotation) in BROAD:
                    yield node, arg.arg


def _catch_all_echo_sites(root: Path = SERVICE_ROOT) -> list[str]:
    """Catch-all scopes raising HTTPException(detail=<uses the exception's text>)."""
    sites = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for scope, name in _broad_scopes(tree):
            for node in _unnarrowed_nodes(scope, name):
                if not (isinstance(node, ast.Call) and ast.unparse(node.func).endswith("HTTPException")):
                    continue
                detail = next((k.value for k in node.keywords if k.arg == "detail"), None)
                if detail is None:
                    continue
                # ``type(exc)`` and the type-only ``internal_error_detail(exc)`` are
                # safe; any other use of the bound name is not.
                uses = [
                    n for n in ast.walk(detail)
                    if isinstance(n, ast.Name) and n.id == name
                ]
                type_only = sum(
                    1 for n in ast.walk(detail)
                    if isinstance(n, ast.Call)
                    and ast.unparse(n.func) in {"type", "internal_error_detail"}
                    and n.args and isinstance(n.args[0], ast.Name)
                    and n.args[0].id == name
                )
                if len(uses) > type_only:
                    sites.add(f"{path.relative_to(root)}:{node.lineno}")
    return sorted(sites, key=lambda s: (s.split(":")[0], int(s.split(":")[1])))


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
        "        raise HTTPException(status_code=500, detail=str(e))\n"
        "def handle(query: str, error: Exception) -> None:\n"
        "    raise HTTPException(status_code=500, detail=str(error))\n",
        encoding="utf-8",
    )
    assert _catch_all_echo_sites(tmp_path) == ["routes.py:5", "routes.py:19", "routes.py:21"]


def test_internal_error_hides_text_and_logs_it(caplog):
    exc = RuntimeError(LEAK)
    with caplog.at_level(logging.ERROR, logger="llm_bawt.service.http_errors"):
        err = internal_error(exc, "Failed to do the thing", status_code=409, public="Thing rejected")

    assert isinstance(err, HTTPException)
    assert err.status_code == 409
    assert re.fullmatch(r"Thing rejected \(RuntimeError, ref [0-9a-f]{8}\); see the app log", err.detail)
    record = caplog.records[-1]
    assert "Failed to do the thing" in record.getMessage() and LEAK in record.getMessage()
    assert record.exc_info and record.exc_info[1] is exc


@pytest.mark.parametrize("route", [history_search.search_history, history_search.search_history_get])
def test_history_search_delegated_failure_hides_db_details(route, monkeypatch, caplog):
    """Both methods hand their catch-all to a shared helper (Codex, TASK-1565 review)."""
    def fail(**_kwargs):
        raise RuntimeError(LEAK)

    monkeypatch.setattr(history_search, "_run_history_search_route", fail)
    with caplog.at_level(logging.ERROR, logger="llm_bawt.service.http_errors"):
        with pytest.raises(HTTPException) as caught:
            route(query="needle", bot_id="example", limit=5, before=None, since=None, until=None)

    assert caught.value.status_code == 500
    assert re.fullmatch(r"Internal error \(RuntimeError, ref [0-9a-f]{8}\); see the app log", caught.value.detail)
    assert "needle" in caplog.records[-1].getMessage() and LEAK in caplog.records[-1].getMessage()


def test_memory_stats_db_failure_keeps_500_without_db_details(monkeypatch):
    def stats():
        raise RuntimeError(LEAK)

    client = SimpleNamespace(stats=stats)
    monkeypatch.setattr(
        "llm_bawt.service.routes.memory._scoped_client", lambda _bot_id, _user_id: client)
    with pytest.raises(HTTPException) as caught:
        get_memory_stats(user_id=None, bot_id="example")

    assert caught.value.status_code == 500
    assert re.fullmatch(r"Internal error \(RuntimeError, ref [0-9a-f]{8}\); see the app log", caught.value.detail)
    assert "192.0.2.7" not in str(caught.value.detail)
