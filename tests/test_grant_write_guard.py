"""No MCP tool writes what decides who manages grants (TASK-1609).

A bot that could edit role capabilities, the runtime settings or grant tags
directly could make itself a grant manager or ordain a peer outside the audited
grant route. The only grant writer an MCP tool may reach is
``PUT /v1/bot-grants`` through ``bots_grant_set``; tag changes through
``bots_profile_patch`` are refused for grant tags (the app) and manager roles
(the tool), covered in ``test_bot_profile_tools.py`` and
``test_bot_grants_pg.py``. This is a source ratchet over ``mcp_server``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import llm_bawt.mcp_server as mcp_package

PACKAGE = Path(mcp_package.__file__).parent
GRANT_TOOL_MODULE = "bot_grant_tools.py"

#: Store and catalog methods that write settings, role tags or assignments.
WRITE_METHODS = frozenset({
    "set_value", "delete_value",  # RuntimeSettingsStore
    "create_role", "update_role", "rename_role", "delete_role",  # BotTagStore
    "save_instructions", "clear_instructions", "unassign",
    "assign", "replace_assignments",  # bot_tags.catalog
    "write_grant",  # bot_tags.grants
})
#: App routes that write settings or role tags.
WRITE_ROUTES = ("/v1/bot-tags", "/v1/settings")
WRITE_SQL = re.compile(
    r"\b(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(runtime_settings|bot_tag_catalog|bot_profile_tags)\b",
    re.IGNORECASE)


def _modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path, ast.parse(path.read_text(), filename=str(path))


def _strings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def test_the_package_was_found():
    assert (PACKAGE / GRANT_TOOL_MODULE).is_file()


def test_no_mcp_module_calls_a_settings_or_tag_write_method():
    calls = [
        f"{path.name}:{node.lineno} .{node.func.attr}()"
        for path, tree in _modules() for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in WRITE_METHODS
    ]
    assert calls == []


def test_no_mcp_module_names_a_settings_or_tag_write_route():
    found = [f"{path.name}: {value!r}" for path, tree in _modules()
             for value in _strings(tree) if any(route in value for route in WRITE_ROUTES)]
    assert found == []


def test_no_mcp_module_writes_the_settings_or_tag_tables():
    found = [f"{path.name}: {WRITE_SQL.search(value).group(0)!r}" for path, tree in _modules()
             for value in _strings(tree) if WRITE_SQL.search(value)]
    assert found == []


def test_only_the_grant_tool_reaches_the_grant_route():
    found = {path.name for path, tree in _modules()
             for value in _strings(tree) if "/v1/bot-grants" in value}
    assert found == {GRANT_TOOL_MODULE}
