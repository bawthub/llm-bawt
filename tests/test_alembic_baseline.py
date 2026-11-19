"""TASK-631 slice 1: the Alembic baseline is a version marker with no DDL."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

alembic = pytest.importorskip("alembic")
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from alembic.script import ScriptDirectory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _config(buffer: io.StringIO | None = None) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"), output_buffer=buffer)
    return cfg


def _offline_sql(monkeypatch: pytest.MonkeyPatch, action) -> str:
    # Offline mode must never need DB settings or a connection.
    for key in ("POSTGRES_USER", "POSTGRES_DATABASE", "POSTGRES_HOST", "LLM_BAWT_ALEMBIC_URL"):
        monkeypatch.delenv(key, raising=False)
    buf = io.StringIO()
    action(_config(buf))
    return buf.getvalue()


def _statements(sql: str) -> list[str]:
    body = "\n".join(line for line in sql.splitlines() if line.strip() and not line.startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def test_single_baseline_head() -> None:
    script = ScriptDirectory.from_config(_config())
    assert script.get_heads() == ["0004_gates"]
    base = script.get_revision("0001_baseline")
    assert base.down_revision is None


def test_baseline_upgrade_touches_only_version_table(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _offline_sql(monkeypatch, lambda cfg: command.upgrade(cfg, "0001_baseline", sql=True))
    stmts = _statements(sql)
    assert stmts, sql
    for stmt in stmts:
        assert re.match(r"^(BEGIN|COMMIT)$", stmt) or "alembic_version" in stmt, stmt


def test_baseline_downgrade_touches_only_version_table(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _offline_sql(monkeypatch, lambda cfg: command.downgrade(cfg, "0001_baseline:base", sql=True))
    for stmt in _statements(sql):
        assert re.match(r"^(BEGIN|COMMIT)$", stmt) or "alembic_version" in stmt, stmt


def test_env_does_not_import_runtime_bootstrap() -> None:
    env = (ROOT / "migrations" / "env.py").read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", env, re.MULTILINE)
    allowed = {"__future__", "os", "alembic", "sqlalchemy", "llm_bawt.utils.config", "llm_bawt.utils.db"}
    assert set(imports) <= allowed, imports
    assert "target_metadata = None" in env


def test_app_images_ship_alembic_files() -> None:
    cpu = (ROOT / "Dockerfile.app-cpu").read_text()
    assert "COPY alembic.ini ./" in cpu and "COPY migrations/ ./migrations/" in cpu
    gpu = (ROOT / "Dockerfile").read_text()
    assert "COPY alembic.ini ./" in gpu and "COPY migrations/ ./migrations/" in gpu
    assert "COPY --from=base /app/alembic.ini /app/alembic.ini" in gpu
    assert "COPY --from=base /app/migrations /app/migrations" in gpu
    dev = (ROOT / "docker-compose.dev.yml").read_text()
    assert "./alembic.ini:/app/alembic.ini:ro" in dev and "./migrations:/app/migrations:ro" in dev
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert not any(line.strip().rstrip("/") in {"migrations", "alembic.ini"} for line in ignore)


def _run_snapshot(tmp_path: Path, fake_body: str) -> tuple[int, str]:
    import subprocess

    fake = tmp_path / "fake_pg_dump"
    fake.write_text("#!/usr/bin/env bash\n" + fake_body)
    fake.chmod(0o755)
    target = tmp_path / "baseline_schema.sql"
    target.write_text("ORIGINAL\n")
    proc = subprocess.run(
        [str(ROOT / "scripts" / "alembic_schema_snapshot.sh"), str(target)],
        env={"PATH": "/usr/bin:/bin", "PG_DUMP": str(fake)},
        capture_output=True,
        text=True,
    )
    leftovers = [p.name for p in tmp_path.iterdir() if p.name not in {"fake_pg_dump", "baseline_schema.sql"}]
    assert leftovers == []
    return proc.returncode, target.read_text()


def test_snapshot_failure_preserves_existing_reference(tmp_path: Path) -> None:
    rc, content = _run_snapshot(tmp_path, "echo partial\nexit 1\n")
    assert rc != 0
    assert content == "ORIGINAL\n"
    rc, content = _run_snapshot(tmp_path, "echo '-- empty dump'\n")
    assert rc != 0
    assert content == "ORIGINAL\n"


def test_snapshot_success_normalizes(tmp_path: Path) -> None:
    body = (
        "printf '%s\\n' '\\restrict abc' '-- Dumped from database version 16' "
        "'SET transaction_timeout = 0;' 'CREATE TABLE public.t (id int);' '\\unrestrict abc'\n"
    )
    rc, content = _run_snapshot(tmp_path, body)
    assert rc == 0
    assert "CREATE TABLE public.t" in content
    for volatile in ("restrict", "Dumped from", "transaction_timeout"):
        assert volatile not in content
