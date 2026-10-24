"""Test-owned pgvector PostgreSQL in an ephemeral Docker container.

No host ports, network, production volumes, or inherited DB credentials. All
traffic uses the container's Unix socket in a pytest-owned temporary directory.
A unique container ID is retained and removed in fixture teardown.
"""

from __future__ import annotations

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


def private_url(socket: Path, database: str) -> URL:
    return URL.create("postgresql+psycopg2", username="test_user", database=database,
                      query={"host": str(socket), "port": PORT})


@pytest.fixture(scope="session")
def private_pg(tmp_path_factory):
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
        with private_cluster_socket(socket, None, PORT):
            # This connection is checked by the suite-wide DBAPI boundary.
            engine = create_engine(private_url(socket, "postgres"))
            try:
                with engine.connect() as conn:
                    assert conn.exec_driver_sql("SELECT current_database()").scalar_one() == "postgres"
                yield socket
            finally:
                engine.dispose()
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
