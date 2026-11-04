"""TASK-1151: ops_list_operations lists compact rows; full detail by slug."""

import asyncio

from llm_bawt.mcp_server import ops_tools

ROWS = [
    {"slug": "llm-bawt.restart-bridge", "enabled": True, "title": "Restart bridge",
     "description": "Restart one agent bridge. Kills active turns.", "risk_level": "high",
     "category": "service", "target_host": "echo", "timeout_seconds": 120,
     "args_schema": {"type": "object", "properties": {"bridge": {"type": "string"}, "x": {}},
                     "required": ["bridge"]}, "args_defaults": {}},
    {"slug": "bawthub.restart-frontend", "enabled": False, "title": "Restart frontend",
     "description": "x" * 300, "risk_level": "low", "category": "service",
     "target_host": "echo", "timeout_seconds": 60,
     "args_schema": {"type": "object", "properties": {}}, "args_defaults": {}},
]


class _Ops:
    def list_operations_for_agent(self, include_disabled=False):
        return ROWS


def _run(monkeypatch, **kwargs):
    monkeypatch.setattr(ops_tools, "_get_ops_service", lambda: _Ops())
    return asyncio.run(ops_tools.ops_list_operations(**kwargs))


def test_list_rows_are_compact(monkeypatch):
    out = _run(monkeypatch)
    first, second = out["operations"]
    assert out["total"] == 2 and "slug=" in out["detail"]
    assert first == {"slug": "llm-bawt.restart-bridge", "title": "Restart bridge",
                     "summary": "Restart one agent bridge", "risk_level": "high",
                     "category": "service", "args": ["bridge*", "x"]}
    assert "args" not in second and second["enabled"] is False
    assert len(second["summary"]) == 140 and second["summary"].endswith("…")


def test_slug_returns_full_row_or_error(monkeypatch):
    assert _run(monkeypatch, slug="llm-bawt.restart-bridge") == {"operation": ROWS[0]}
    assert "error" in _run(monkeypatch, slug="nope")
