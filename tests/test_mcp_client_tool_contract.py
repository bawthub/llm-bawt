"""Server-mode MemoryClient calls must use names actually registered by MCP."""

import ast
from pathlib import Path


MCP_DIR = Path(__file__).resolve().parents[1] / "src/llm_bawt/mcp_server"


class _Signature:
    """Keyword names one registered MCP tool function accepts."""

    def __init__(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        args = node.args
        positional = args.posonlyargs + args.args
        defaulted = len(args.defaults)
        self.params = {a.arg for a in positional + args.kwonlyargs}
        self.required = {a.arg for a in positional[:len(positional) - defaulted]}
        self.required |= {a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is None}
        self.var_keyword = args.kwarg is not None


def _tool_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Name from ``@mcp.tool(name=...)`` or a bare ``@mcp.tool()`` (function name)."""
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        func = call.func if call else decorator
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "mcp" and func.attr == "tool"):
            continue
        for keyword in call.keywords if call else []:
            if keyword.arg == "name":
                assert isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str), (
                    node.name, node.lineno)
                return keyword.value.value
        return node.name
    return None


def _registered_tools() -> dict[str, _Signature]:
    tools = {}
    for path in MCP_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (name := _tool_name(node)):
                tools[name] = _Signature(node)
    return tools


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
    return [(file, line, name) for file, line, name, *_ in _client_call_nodes()]


def _client_call_nodes() -> list[tuple[str, int, str, ast.Call, ast.AST]]:
    calls = []
    for path in MCP_DIR.glob("client*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scopes = {child: fn for fn in ast.walk(tree)
                  if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) for child in ast.walk(fn)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_call_server"):
                continue
            assert node.args and isinstance(node.args[0], ast.Constant), (
                f"{path.name}:{node.lineno} must use a literal registered MCP tool name"
            )
            name = node.args[0].value
            assert isinstance(name, str), (path, node.lineno)
            calls.append((path.name, node.lineno, name, node, scopes.get(node, tree)))
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


def _local_dict(name: str, scope: ast.AST) -> tuple[list[ast.expr | None], set[str]] | None:
    """Keys of ``name = {...}`` in the calling function plus ``name["k"] = ...`` additions."""
    keys, conditional, found = [], set(), False
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                if found or not isinstance(node.value, ast.Dict):
                    return None
                found, keys = True, list(node.value.keys)
            elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                  and target.value.id == name):
                if not (isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)):
                    return None
                conditional.add(target.slice.value)
    return (keys, conditional) if found else None


def _argument_problems(name: str, call: ast.Call, signature: _Signature,
                       scope: ast.AST | None = None) -> list[str]:
    """Literal keys must be accepted; required ones must be sent unless a ``**`` spread may supply them."""
    if len(call.args) < 2:
        return [] if not signature.required else [f"missing required {sorted(signature.required)}"]
    params = call.args[1]
    added: set[str] = set()
    if isinstance(params, ast.Dict):
        dict_keys = list(params.keys)
    elif isinstance(params, ast.Name) and scope is not None and (local := _local_dict(params.id, scope)):
        dict_keys, added = local
    else:
        return ["arguments must be a dict literal so their names can be checked"]
    keys, spread = set(), False
    if not signature.var_keyword and (unknown := added - signature.params):
        return [f"unknown {sorted(unknown)}"]
    for key in dict_keys:
        if key is None:
            spread = True
        elif isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
        else:
            return ["argument names must be string literals"]
    problems = []
    if not signature.var_keyword and (unknown := keys - signature.params):
        problems.append(f"unknown {sorted(unknown)}")
    if not spread and (missing := signature.required - keys):
        problems.append(f"missing required {sorted(missing)}")
    return problems


def test_every_memory_client_tool_call_uses_registered_argument_names():
    """TASK-1809: a server-side parameter rename must break this test, not production calls."""
    tools = _registered_tools()
    problems = [
        f"{file}:{line} {name}: {problem}"
        for file, line, name, call, scope in _client_call_nodes() if name in tools
        for problem in _argument_problems(name, call, tools[name], scope)
    ]
    assert not problems, "MemoryClient argument names drift from MCP tools:\n" + "\n".join(problems)


def test_argument_guard_rejects_renamed_and_missing_parameters():
    tool = ast.parse("def t(bot_id: str, limit: int = 5, *, scope: str | None = None): ...").body[0]
    signature = _Signature(tool)
    call = lambda src: ast.parse(src).body[0].value  # noqa: E731
    assert _argument_problems("t", call('c._call_server("t", {"bot_id": b, "limit": 1})'), signature) == []
    assert _argument_problems("t", call('c._call_server("t", {"bot_id": b, "max": 1})'), signature) == [
        "unknown ['max']"]
    assert _argument_problems("t", call('c._call_server("t", {"limit": 1})'), signature) == [
        "missing required ['bot_id']"]
    assert _argument_problems("t", call('c._call_server("t", {"limit": 1, **extra})'), signature) == []
    assert _argument_problems("t", call('c._call_server("t", params)'), signature) == [
        "arguments must be a dict literal so their names can be checked"]
    fn = ast.parse('def f():\n    a = {"bot_id": b}\n    a["scope"] = s\n    c._call_server("t", a)').body[0]
    local_call = fn.body[-1].value
    assert _argument_problems("t", local_call, signature, fn) == []
    fn = ast.parse('def f():\n    a = {"bot_id": b}\n    a["max"] = s\n    c._call_server("t", a)').body[0]
    assert _argument_problems("t", fn.body[-1].value, signature, fn) == ["unknown ['max']"]
    bare = ast.parse("@mcp.tool()\ndef bare_tool(x): ...").body[0]
    assert _tool_name(bare) == "bare_tool"
