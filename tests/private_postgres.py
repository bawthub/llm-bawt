"""Test-owned pgvector PostgreSQL in an ephemeral Docker container.

No host ports, network, production volumes, or inherited DB credentials. All
traffic uses the container's Unix socket in a pytest-owned temporary directory.
A unique container ID is retained and removed in fixture teardown.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from tests.pg_connection_guard import private_cluster_socket


IMAGE = "pgvector/pgvector@sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b"
PORT = "55439"
# The stress step (scripts/test_suite.py) pins the tests to a few CPUs. A
# container doesn't inherit that affinity, so the cpuset arrives here instead.
CPUSET_ENV = "LLM_BAWT_TEST_PG_CPUSET"
_CPUSET = re.compile(r"\d+(?:,\d+)*")
_GATE_ENV = "OPS_GATE_SANDBOX"
_BINDING_ENV = "OPS_GATE_PG_BINDING"
_BINDING_KEYS = {"schema", "job_id", "repo", "sha", "mode", "jobs"}
_PG_KEYS = {"image", "socket_dir", "port", "user", "database"}


def isolated_gate_socket() -> Path | None:
    """Accept only this checked-out commit's worker-owned private PG binding.

    The ordinary gate still owns its Docker container. In an isolated ops gate,
    the trusted worker owns PostgreSQL and the test process has no Docker socket.
    Both environment fields must be present together; never fall back or skip.
    """
    marker = os.environ.get(_GATE_ENV)
    binding = os.environ.get(_BINDING_ENV)
    if marker is None and binding is None:
        return None
    if marker != "1" or not binding:
        raise RuntimeError("private PostgreSQL binding requires the isolated gate marker and file")
    path = Path(binding)
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("private PostgreSQL binding path is not an absolute regular file")
    try:
        if not path.is_file() or path.stat().st_size > 8192:
            raise ValueError("not a bounded regular file")
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("private PostgreSQL binding file is missing or invalid") from exc
    if not isinstance(data, dict) or set(data) != _BINDING_KEYS:
        raise RuntimeError("private PostgreSQL binding has an invalid schema")
    jobs = data["jobs"]
    job = jobs.get("private-pg") if isinstance(jobs, dict) and set(jobs) == {"private-pg"} else None
    if (data["schema"] != "ops.gate.pg/v1" or data["repo"] != "llm-bawt"
            or data["mode"] != "unix"
            or not isinstance(data["job_id"], str)
            or re.fullmatch(r"[0-9a-f]{32}", data["job_id"]) is None
            or not isinstance(data["sha"], str)
            or re.fullmatch(r"[0-9a-f]{40}", data["sha"]) is None
            or not isinstance(job, dict) or set(job) != _PG_KEYS
            or job["image"] != IMAGE or job["port"] != PORT
            or job["user"] != "test_user" or job["database"] != "llm_bawt_test"
            or not isinstance(job["socket_dir"], str)):
        raise RuntimeError("private PostgreSQL binding has an invalid sidecar identity")
    try:
        commit = subprocess.run(["git", "rev-parse", "--verify", "HEAD^{commit}"],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("private PostgreSQL binding could not verify the checked-out commit") from exc
    if data["sha"] != commit:
        raise RuntimeError("private PostgreSQL binding does not match the checked-out commit")
    socket = Path(job["socket_dir"])
    if not socket.is_absolute() or not socket.is_dir():
        raise RuntimeError("private PostgreSQL binding has no private Unix socket directory")
    return socket.resolve(strict=True)


@contextmanager
def verified_socket(socket: Path):
    with private_cluster_socket(socket, None, PORT):
        # This connection is checked by the suite-wide DBAPI boundary.
        engine = create_engine(private_url(socket, "postgres"))
        try:
            with engine.connect() as conn:
                assert conn.exec_driver_sql("SELECT current_database()").scalar_one() == "postgres"
            yield
        finally:
            engine.dispose()


def cpuset_args() -> list[str]:
    cpuset = os.environ.get(CPUSET_ENV, "").strip()
    if not cpuset:
        return []
    if not _CPUSET.fullmatch(cpuset):
        raise RuntimeError(f"{CPUSET_ENV} must be a comma-separated CPU list, got {cpuset!r}")
    return ["--cpuset-cpus", cpuset]


def private_url(socket: Path, database: str) -> URL:
    return URL.create("postgresql+psycopg2", username="test_user", database=database,
                      query={"host": str(socket), "port": PORT})


@pytest.fixture(scope="session")
def private_pg(tmp_path_factory):
    external_socket = isolated_gate_socket()
    if external_socket is not None:
        if not (external_socket / f".s.PGSQL.{PORT}").exists():
            raise RuntimeError("private PostgreSQL cluster socket is not running")
        with verified_socket(external_socket):
            yield external_socket
        return
    if shutil.which("docker") is None:
        pytest.skip("private pgvector acceptance needs Docker on this test runner")
    root = tmp_path_factory.mktemp("private-pgvector")
    socket = root / "socket"
    socket.mkdir(mode=0o777)
    owner = uuid.uuid4().hex
    name = f"llm-bawt-pgtest-{owner[:12]}"
    # No -p and --network none: neither an operational server nor the LAN is
    # reachable. The container's writable data layer is deleted by --rm.
    command = ["docker", "run", "--detach", "--rm", "--network", "none",
               "--name", name, "--label", "llm-bawt.pg-acceptance=true",
               "--label", f"llm-bawt.pg-acceptance.owner={owner}",
               "--mount", f"type=bind,source={socket},target=/var/run/postgresql",
               "--env", "POSTGRES_HOST_AUTH_METHOD=trust",
               "--env", "POSTGRES_USER=test_user",
               "--env", "POSTGRES_DB=llm_bawt_test",
               *cpuset_args(),
               IMAGE, "-c", "listen_addresses=", "-c", f"port={PORT}"]
    container_id = None
    try:
        started = subprocess.run(command, capture_output=True, text=True, check=True)
        container_id = started.stdout.strip()
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            ready = subprocess.run(
                ["docker", "exec", container_id, "pg_isready", "-q",
                 "-h", "/var/run/postgresql", "-p", PORT,
                 "-U", "test_user", "-d", "llm_bawt_test"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.2)
        else:
            pytest.fail("private pgvector container did not become ready")
        with verified_socket(socket):
            yield socket
    finally:
        # SIGTERM lets PostgreSQL unlink its socket/lock; force-removal leaves
        # root-owned socket files in pytest's temp tree and breaks its cleanup.
        # An interrupted docker run may have started a container without
        # returning its ID. Inspect the unique ownership label before stopping
        # by name; a failed launch must never stop an unrelated container.
        if not container_id:
            probe = subprocess.run(
                ["docker", "inspect", "--format",
                 '{{ index .Config.Labels "llm-bawt.pg-acceptance.owner" }}', name],
                capture_output=True, text=True,
            )
            owned = probe.returncode == 0 and probe.stdout.strip() == owner
        else:
            owned = True
        if owned:
            stopped = subprocess.run(["docker", "stop", "--time", "10", container_id or name],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if stopped.returncode:
                subprocess.run(["docker", "rm", "--force", container_id or name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        socket.rmdir()


@pytest.fixture(scope="module")
def private_module_db(private_pg):
    """Give each module a fresh database inside the session's private cluster."""
    from sqlalchemy import text

    name = f"pg_{uuid.uuid4().hex}_test"
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    patched = pytest.MonkeyPatch()
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        patched.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
        patched.setenv("LLM_BAWT_POSTGRES_HOST", str(private_pg))
        patched.setenv("LLM_BAWT_POSTGRES_PORT", PORT)
        patched.setenv("LLM_BAWT_POSTGRES_USER", "test_user")
        yield name
    finally:
        patched.undo()
        try:
            with admin.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        finally:
            admin.dispose()


@pytest.fixture(scope="module")
def private_app_pg(private_pg, private_module_db):
    """Select the module's private DB, verify identity, then app-bootstrap."""
    from llm_bawt import prompt_registry
    from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
    from llm_bawt.service.turn_logs import TurnLogStore
    from llm_bawt.utils import db
    from tests.pg_isolation import require_test_db_config

    socket = private_pg
    patched = pytest.MonkeyPatch()
    url = private_url(socket, private_module_db)
    # Production's URL builder is for TCP hosts, not Unix-socket test paths.
    # Patch only in this pytest module's lifetime; no production code changes.
    patched.setattr(db, "build_postgres_url", lambda config: url)
    # get_prompt_resolver() caches one resolver per process, bound to the
    # Config (and module DB) of whichever module called it first. Under xdist
    # a later module on the same worker would read a dropped database.
    patched.setattr(prompt_registry, "_DEFAULT_PROMPT_RESOLVER", None)
    patched.setenv("LLM_BAWT_POSTGRES_HOST", str(socket))
    patched.setenv("LLM_BAWT_POSTGRES_PORT", PORT)
    patched.setenv("LLM_BAWT_POSTGRES_USER", "test_user")
    patched.setenv("LLM_BAWT_POSTGRES_DATABASE", private_module_db)
    patched.setenv("LLM_BAWT_POSTGRES_PASSWORD", uuid.uuid4().hex)
    try:
        config = require_test_db_config()
        # App-owned schema paths, never handcrafted DDL or shared test reset.
        PostgreSQLMemoryBackend(config, bot_id=f"pgaccept{uuid.uuid4().hex[:12]}")
        TurnLogStore(config)
        yield config
    finally:
        db.reset_shared_engines()
        patched.undo()
