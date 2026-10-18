"""Hermetic regression proofs for the collection-time DBAPI/CLI boundary."""

import importlib.util
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import create_engine

from tests import pg_connection_guard as guard
from tests import test_cli_commands as cli


def test_operational_engine_rejected_before_dbapi_connect(monkeypatch):
    original = Mock(side_effect=AssertionError("network connect was attempted"))
    monkeypatch.setattr(guard, "_original_connect", original)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", "askllm")
    engine = create_engine("postgresql+psycopg2://test_user:fake@127.0.0.1/askllm")
    try:
        with pytest.raises(RuntimeError, match="refuses PostgreSQL target"):
            engine.connect()
        original.assert_not_called()
    finally:
        engine.dispose()


def test_import_time_connect_rejected_before_network(monkeypatch, tmp_path):
    original = Mock(side_effect=AssertionError("network connect was attempted"))
    monkeypatch.setattr(guard, "_original_connect", original)
    module_path = tmp_path / "import_db.py"
    module_path.write_text("import psycopg2\npsycopg2.connect(dbname='askllm', host='127.0.0.1')\n")
    spec = importlib.util.spec_from_file_location("import_db_probe", module_path)
    with pytest.raises(RuntimeError, match="refuses PostgreSQL target"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    original.assert_not_called()


@pytest.mark.parametrize("dsn,kwargs", [
    ("dbname=askllm", {"dbname": "llm_bawt_test", "service": "operational"}),
    ("service=operational dbname=llm_bawt_test", {}),
    ("host=/tmp/pytest-private-socket hostaddr=10.0.2.32 dbname=llm_bawt_test", {}),
    ("dbname=askllm", {}),
    ("dbname=llm_bawt_test", {"dbname": "askllm"}),
    (None, {}),
])
def test_ambiguous_or_operational_routing_refused_without_connect(monkeypatch, dsn, kwargs):
    original = Mock(side_effect=AssertionError("network connect was attempted"))
    monkeypatch.setattr(guard, "_original_connect", original)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", "llm_bawt_test")
    with pytest.raises(RuntimeError, match="refuses"):
        guard.checked_connect(*([dsn] if dsn is not None else []), **kwargs)
    original.assert_not_called()


def test_private_socket_allowed_only_when_registered_and_identity_matches(monkeypatch):
    original = Mock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = ("postgres", "55439", None)
    original.return_value = conn
    monkeypatch.setattr(guard, "_original_connect", original)
    params = dict(dbname="postgres", user="test_user", host="/tmp/pytest-private-socket", port="55439")
    with pytest.raises(RuntimeError, match="refuses PostgreSQL target"):
        guard.checked_connect(**params)
    original.assert_not_called()
    monkeypatch.setattr(guard, "_private_sockets", {(params["host"], params["port"])})
    assert guard.checked_connect(**params) is conn
    conn.rollback.assert_called_once()


def test_cli_session_probe_is_blocked_before_http(monkeypatch):
    probe = Mock(side_effect=AssertionError("live service probed"))
    monkeypatch.setattr(cli.urllib.request, "urlopen", probe)
    with pytest.raises(pytest.fail.Exception, match="separately verified isolated service"):
        cli.service_up.__wrapped__()
    probe.assert_not_called()


def test_cli_is_blocked_before_subprocess_even_with_inherited_credentials(monkeypatch):
    launch = Mock(side_effect=AssertionError("CLI launched"))
    monkeypatch.setattr(cli.subprocess, "run", launch)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PASSWORD", "inherited-secret")
    with pytest.raises(pytest.fail.Exception, match="separately verified isolated service"):
        cli.run("llm-memory clear --bot spark")
    launch.assert_not_called()
