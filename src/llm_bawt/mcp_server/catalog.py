"""Compact discovery metadata without changing dispatch or validation.

FastMCP still owns callable signatures and validation. The public list_tools seam
only replaces model-facing descriptions and removes generated schema titles.
Titles are annotations, not field names; never recursively strip arbitrary keys.

TASK-1555: call_tool also re-emits JSON result text without whitespace.
FastMCP renders a tool's dict/list result with ``indent=2``, and every
newline and indent is a model token: the same payload costs ~45% more.
"""

import json
from copy import deepcopy
from typing import Any

from mcp.types import TextContent, Tool

from .approval_interceptor import ApprovalAwareFastMCP
from .catalog_contracts import TOOL_SUMMARIES, tool_description
from .bot_access import ToolBotAccess, invocation_context
from agent_bridge.mcp_call_context import MCP_CALL_CONTEXT_KEY

# JSON Schema locations that hold schemas (not arbitrary user data/defaults).
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_SINGLES = (
    "additionalProperties", "unevaluatedProperties", "propertyNames", "items",
    "contains", "not", "if", "then", "else", "additionalItems", "unevaluatedItems",
)


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a schema, dropping only presentation titles at schema nodes."""
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node.pop("title", None)
        for key in _SCHEMA_MAPS:
            children = node.get(key)
            if isinstance(children, dict):
                for child in children.values():
                    visit(child)
        for key in _SCHEMA_LISTS:
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    visit(child)
        for key in _SCHEMA_SINGLES:
            child = node.get(key)
            if isinstance(child, list):  # draft-07 tuple items
                for entry in child:
                    visit(entry)
            else:
                visit(child)

    visit(result)
    return result


def compact_json_text(block: Any) -> Any:
    """``block`` with JSON text re-serialized without whitespace; else as-is.

    Only text that parses as a JSON object or array changes, and it decodes to
    the same value: nothing is dropped or reordered. Non-ASCII stays literal,
    as FastMCP wrote it.
    """
    if not isinstance(block, TextContent) or block.text[:1] not in ("{", "["):
        return block
    try:
        value = json.loads(block.text)
    except ValueError:
        return block
    compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return block if compact == block.text else block.model_copy(update={"text": compact})


def compact_call_result(result: Any) -> Any:
    """A FastMCP call_tool result with every JSON text block compacted.

    The result is content blocks, or (content blocks, structured content) for a
    tool with an output schema. Structured content and any other shape (an
    approval error dict) pass through unchanged.
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
        return ([compact_json_text(block) for block in result[0]], result[1])
    if isinstance(result, list):
        return [compact_json_text(block) for block in result]
    return result


class CatalogFastMCP(ApprovalAwareFastMCP):
    """Approval-aware server with a compact, skill-linked discovery catalog."""

    def __init__(self, *args, bot_policy_resolver=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot_access = ToolBotAccess(bot_policy_resolver)

    def add_tool(self, fn, name=None, *args, **kwargs):
        wrapped = self.bot_access.register(name or fn.__name__, fn)
        return super().add_tool(wrapped, name, *args, **kwargs)

    async def call_tool(self, name, arguments):
        public = dict(arguments or {})
        public.pop(MCP_CALL_CONTEXT_KEY, None)
        with invocation_context(name, public):
            # Avoid doomed ASK cards. The registered callable checks again at
            # execution, including approved execution's bypass of call_tool.
            await self.bot_access.preflight(name, public)
            return compact_call_result(await super().call_tool(name, arguments))

    async def call_approved_tool(self, name, arguments, **kwargs):
        with invocation_context(name, arguments):
            return await super().call_approved_tool(name, arguments, **kwargs)

    async def list_tools(self) -> list[Tool]:
        tools = await super().list_tools()
        return [
            tool.model_copy(update={
                # Unmigrated/new tools remain callable and discoverable. Tests
                # require explicit coverage before they can land in this repo.
                "description": tool_description(tool.name)
                if tool.name in TOOL_SUMMARIES else tool.description,
                "inputSchema": compact_schema(tool.inputSchema),
            })
            for tool in tools
        ]
