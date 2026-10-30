"""Server-mode MemoryClient calls must use names actually registered by MCP."""

import ast
from pathlib import Path


MCP_DIR = Path(__file__).resolve().parents[1] / "src/llm_bawt/mcp_server"


def _registered_tool_names() -> set[str]:
    names = set()
    for path in MCP_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                if not (isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "mcp" and decorator.func.attr == "tool"):
                    continue
                for keyword in decorator.keywords:
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                        assert isinstance(keyword.value.value, str), (path, node.lineno)
                        names.add(keyword.value.value)
    return names


def _client_calls() -> list[tuple[str, int, str]]:
    calls = []
    for path in MCP_DIR.glob("client*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_call_server"):
                continue
            assert node.args and isinstance(node.args[0], ast.Constant), (
                f"{path.name}:{node.lineno} must use a literal registered MCP tool name"
            )
            name = node.args[0].value
            assert isinstance(name, str), (path, node.lineno)
            calls.append((path.name, node.lineno, name))
    return calls


def test_every_memory_client_tool_call_is_registered():
    """AST traverses multi-line calls, not just grep matches on one line."""
    calls = _client_calls()
    assert len(calls) >= 25, "memory client call scan unexpectedly shrank"
    registered = _registered_tool_names()
    missing = [(file, line, name) for file, line, name in calls if name not in registered]
    assert not missing, f"MemoryClient calls unknown MCP tools: {missing}"
    assert {"memory_list_high_importance", "memory_list_recent", "memory_supersede",
            "messages_get_for_summary", "messages_mark_recalled",
            "messages_preview_since_minutes"} <= {name for _, _, name in calls}
