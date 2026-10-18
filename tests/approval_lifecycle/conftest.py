"""TASK-861 approval tests get an additional per-test PostgreSQL block.

The root conftest installs the DBAPI boundary before collection. Do not mutate
process-wide environment or psycopg2.connect at import time: pytest also
collects unrelated modules in this process, including private-cluster proofs.
"""

import os

import psycopg2
import pytest


def _forbid_connect(*args, **kwargs):
    raise RuntimeError("PostgreSQL access forbidden by TASK-861 isolated tests")


@pytest.fixture(autouse=True)
def isolate_approval_lifecycle(monkeypatch):
    for key in os.environ:
        upper = key.upper()
        if (upper.startswith("PG") or any(part in upper for part in
                ("POSTGRES", "DATABASE", "DB_HOST", "DB_PORT", "DB_USER", "DB_PASS", "DB_NAME", "DB_URL"))):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    monkeypatch.setattr(psycopg2, "connect", _forbid_connect)
