"""PostgreSQL approval locks must use a text-safe key."""
from types import SimpleNamespace

from llm_bawt.approval_mcp_store import _lock_mcp_invocation


def test_mcp_approval_advisory_lock_uses_postgres_safe_distinct_keys():
    seen = []

    class Session:
        def get_bind(self):
            return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        def execute(self, statement, params):
            seen.append((str(statement), params["key"]))

    session = Session()
    _lock_mcp_invocation(session, "loopy", "nick", "abc")
    _lock_mcp_invocation(session, "loopy", "nick", "def")
    assert all("pg_advisory_xact_lock" in statement for statement, _ in seen)
    assert all("\x00" not in key for _, key in seen)
    assert seen[0][1] != seen[1][1]
    assert seen[0][1].split("\x1f") == ["mcp-approval", "loopy", "nick", "abc"]
